"""
Streamlit UI for the Voice Assistant Pipeline
  Mic → Parakeet ASR → Gemini LLM → Piper TTS

Run:
  streamlit run app.py
"""

import io
import os
import wave
import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Voice Assistant",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ Voice Assistant")
st.caption("Parakeet ASR  ·  Gemini LLM  ·  Piper TTS")

# ──────────────────────────────────────────────
# Model loaders (cached — load once per session)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Parakeet ASR model …")
def load_asr():
    from nemo.collections.asr.models import ASRModel
    model = ASRModel.from_pretrained("nvidia/parakeet_realtime_eou_120m-v1", map_location="cpu")
    model.eval()
    return model


@st.cache_resource(show_spinner="Initialising Gemini client …")
def load_gemini():
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("GEMINI_API_KEY not set. Add it to your .env file.")
        st.stop()
    return genai.Client(api_key=api_key)


@st.cache_resource(show_spinner="Loading Piper TTS voice …")
def load_tts():
    from pathlib import Path
    from piper import PiperVoice
    onnx = Path("voices/en_US-lessac-medium.onnx")
    if not onnx.exists():
        st.error("Piper voice model not found. Run:  python pipeline.py --download-voice")
        st.stop()
    return PiperVoice.load(str(onnx))


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────
GEMINI_MODEL = "gemini-3.5-flash"

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep your responses concise and "
    "conversational — 2 to 4 sentences. Avoid markdown, bullet points, "
    "or any formatting since your response will be spoken aloud."
)


def audio_bytes_to_numpy(audio_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    """Convert WAV bytes (from st.audio_input) to float32 numpy at target_sr."""
    import soundfile as sf
    import scipy.signal as sps

    buf = io.BytesIO(audio_bytes)
    audio, sr = sf.read(buf, dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        num_samples = int(len(audio) * target_sr / sr)
        audio = sps.resample(audio, num_samples)

    return audio.astype(np.float32)


def transcribe(model, audio: np.ndarray) -> str:
    results = model.transcribe(audio=[audio], batch_size=1)
    text = results[0] if isinstance(results[0], str) else results[0].text
    return text.replace("<EOU>", "").replace("<EOB>", "").strip()


def ask_gemini(client, history: list, user_text: str) -> str:
    contents = []
    for turn in history:
        # Gemini uses "model" for assistant turns
        role = "model" if turn["role"] == "assistant" else turn["role"]
        contents.append({"role": role, "parts": [{"text": turn["text"]}]})
    contents.append({"role": "user", "parts": [{"text": user_text}]})

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "thinking_config": {"thinking_budget": -1, "include_thoughts": False},
        },
    )
    return response.text.strip()


def synthesize(voice, text: str) -> bytes:
    """Return WAV bytes for the given text using Piper v1.4+ API."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize_wav(text, wf)
    return buf.getvalue()


# ──────────────────────────────────────────────
# Session state init
# ──────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of {"role": "user"|"assistant", "text": str, "audio": bytes|None}
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None  # prevents re-processing same recording on rerun

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("**ASR:** `parakeet_realtime_eou_120m-v1`")
    st.markdown("**LLM:** `gemini-3.5-flash` (thinking: medium)")
    st.markdown("**TTS:** `Piper en_US-lessac-medium`")

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.caption("Models load once and stay cached.")

# ──────────────────────────────────────────────
# Load models
# ──────────────────────────────────────────────
asr_model  = load_asr()
gemini     = load_gemini()
tts_voice  = load_tts()

# ──────────────────────────────────────────────
# Conversation display
# ──────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    last_idx = len(st.session_state.history) - 1
    for i, turn in enumerate(st.session_state.history):
        if turn["role"] == "user":
            with st.chat_message("user"):
                st.write(turn["text"])
        else:
            with st.chat_message("assistant"):
                st.write(turn["text"])
                if turn.get("audio"):
                    # autoplay only the most recent reply
                    st.audio(turn["audio"], format="audio/wav", autoplay=(i == last_idx))

# ──────────────────────────────────────────────
# Voice input
# ──────────────────────────────────────────────
st.divider()
st.subheader("🎤 Tap the mic, speak, then tap again to stop")
audio_input = st.audio_input("Record", label_visibility="collapsed")

# ──────────────────────────────────────────────
# Process audio input
# ──────────────────────────────────────────────
if audio_input is not None and audio_input.file_id != st.session_state.last_audio_id:
    st.session_state.last_audio_id = audio_input.file_id
    with st.status("Processing …", expanded=True) as status:
        st.write("🎙️ Transcribing with Parakeet …")
        audio_np = audio_bytes_to_numpy(audio_input.getvalue())
        user_text = transcribe(asr_model, audio_np)

        if not user_text:
            status.update(label="Nothing detected — try again.", state="error")
            st.stop()

        st.write(f"📝 **You said:** {user_text}")

        st.write("🤖 Asking Gemini …")
        reply = ask_gemini(gemini, [
            {"role": t["role"], "text": t["text"]} for t in st.session_state.history
        ], user_text)
        st.write(f"💬 **Gemini:** {reply}")

        st.write("🔊 Synthesising with Piper …")
        reply_audio = synthesize(tts_voice, reply)

        status.update(label="Done!", state="complete")

    st.session_state.history.append({"role": "user",      "text": user_text,  "audio": None})
    st.session_state.history.append({"role": "assistant",  "text": reply,      "audio": reply_audio})
    st.rerun()
