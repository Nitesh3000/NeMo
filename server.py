"""
Voice Assistant Server  (Parakeet ASR · Gemini LLM · Piper TTS)
Browser does continuous mic + VAD, server runs the model pipeline.

Run:
  uvicorn server:app --host 127.0.0.1 --port 8000
or:
  python server.py
"""

import io
import os
import wave
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-3.5-flash"
PIPER_VOICE = Path("voices/en_US-lessac-medium.onnx")
TARGET_SR = 16000

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep responses concise and "
    "conversational — 1 to 3 sentences. Avoid markdown, bullet points, "
    "or any formatting because your response will be spoken aloud."
)

# ────────────────────────────────────────────────────────────
# Load models once at startup
# ────────────────────────────────────────────────────────────
print("[init] Loading Parakeet ASR …")
from nemo.collections.asr.models import ASRModel
asr_model = ASRModel.from_pretrained("nvidia/parakeet_realtime_eou_120m-v1", map_location="cpu")
asr_model.eval()

print("[init] Loading Piper voice …")
from piper import PiperVoice
tts_voice = PiperVoice.load(str(PIPER_VOICE))

print("[init] Initialising Gemini client …")
from google import genai
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

print("[init] Ready.")

# Per-session conversation history (single-user demo — no auth)
conversation_history: List[dict] = []


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────
def wav_bytes_to_numpy(audio_bytes: bytes) -> np.ndarray:
    import soundfile as sf
    import scipy.signal as sps

    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = sps.resample(audio, int(len(audio) * TARGET_SR / sr))
    return audio.astype(np.float32)


def transcribe(audio: np.ndarray) -> str:
    if len(audio) < TARGET_SR * 0.2:
        return ""
    results = asr_model.transcribe(audio=[audio], batch_size=1)
    text = results[0] if isinstance(results[0], str) else results[0].text
    return text.replace("<EOU>", "").replace("<EOB>", "").strip()


def ask_gemini(user_text: str) -> str:
    contents = []
    for turn in conversation_history:
        role = "model" if turn["role"] == "assistant" else turn["role"]
        contents.append({"role": role, "parts": [{"text": turn["text"]}]})
    contents.append({"role": "user", "parts": [{"text": user_text}]})

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "thinking_config": {"thinking_budget": -1, "include_thoughts": False},
        },
    )
    return response.text.strip()


def synthesize_wav(text: str) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        tts_voice.synthesize_wav(text, wf)
    return buf.getvalue()


# ────────────────────────────────────────────────────────────
# FastAPI app
# ────────────────────────────────────────────────────────────
app = FastAPI(title="Voice Assistant")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.post("/api/reset")
def reset():
    conversation_history.clear()
    return {"ok": True}


@app.get("/api/history")
def history():
    return JSONResponse(conversation_history)


class ProcessResponse(BaseModel):
    user_text: str
    reply_text: str


@app.post("/api/process")
async def process_audio(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    try:
        audio_np = wav_bytes_to_numpy(audio_bytes)
    except Exception as e:
        raise HTTPException(400, f"Audio decode failed: {e}")

    user_text = transcribe(audio_np)
    if not user_text:
        return JSONResponse({"user_text": "", "reply_text": "", "audio_b64": "", "skip": True})

    reply_text = ask_gemini(user_text)
    wav_bytes = synthesize_wav(reply_text)

    conversation_history.append({"role": "user", "text": user_text})
    conversation_history.append({"role": "assistant", "text": reply_text})

    import base64
    return JSONResponse({
        "user_text": user_text,
        "reply_text": reply_text,
        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
        "skip": False,
    })


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
