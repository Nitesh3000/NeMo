"""
Voice Assistant Pipeline
  Mic → Parakeet ASR (EOU) → Gemini LLM → Piper TTS → Speaker

Setup:
  1. Copy .env.example to .env and fill in your GEMINI_API_KEY
  2. Download a Piper voice model:
       python pipeline.py --download-voice
  3. Run:
       python pipeline.py
"""

import io
import os
import sys
import time
import wave
import argparse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # loads .env from the current directory

import numpy as np
import sounddevice as sd

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
SAMPLE_RATE = 16000          # Parakeet expects 16kHz
CHANNELS = 1
CHUNK_DURATION = 0.1         # seconds per audio chunk
SILENCE_THRESHOLD = 0.01     # RMS energy below this = silence
SILENCE_DURATION = 1.2       # seconds of silence before EOU
MAX_RECORD_SECS = 30         # safety cap on recording length

GEMINI_MODEL = "gemini-3.5-flash"
PIPER_VOICE_DIR = Path(__file__).parent / "voices"
PIPER_VOICE_NAME = "en_US-lessac-medium"
PIPER_VOICE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main"
    "/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
)
PIPER_VOICE_CONFIG_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main"
    "/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
)

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep your responses concise and "
    "conversational — 2 to 4 sentences. Avoid markdown, bullet points, "
    "or any formatting since your response will be spoken aloud."
)


# ──────────────────────────────────────────────
# Voice model download
# ──────────────────────────────────────────────
def download_voice():
    PIPER_VOICE_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = PIPER_VOICE_DIR / f"{PIPER_VOICE_NAME}.onnx"
    json_path = PIPER_VOICE_DIR / f"{PIPER_VOICE_NAME}.onnx.json"

    for url, path in [(PIPER_VOICE_URL, onnx_path), (PIPER_VOICE_CONFIG_URL, json_path)]:
        if path.exists():
            print(f"  Already exists: {path.name}")
            continue
        print(f"  Downloading {path.name} ...")
        urllib.request.urlretrieve(url, path)
        print(f"  Saved to {path}")


# ──────────────────────────────────────────────
# ASR — Parakeet EOU
# ──────────────────────────────────────────────
def load_asr_model():
    print("[ASR] Loading parakeet_realtime_eou_120m-v1 ...")
    from nemo.collections.asr.models import ASRModel
    model = ASRModel.from_pretrained("nvidia/parakeet_realtime_eou_120m-v1", map_location="cpu")
    model.eval()
    print("[ASR] Model ready.")
    return model


def record_until_silence() -> np.ndarray:
    """Record mic audio until the user stops speaking."""
    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)
    max_chunks = int(MAX_RECORD_SECS / CHUNK_DURATION)
    silent_chunks_needed = int(SILENCE_DURATION / CHUNK_DURATION)

    print("\n[Listening] Speak now... (silence stops recording)")
    audio_chunks = []
    silent_streak = 0
    started_speaking = False

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32") as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_samples)
            chunk = chunk.flatten()
            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms > SILENCE_THRESHOLD:
                started_speaking = True
                silent_streak = 0
                audio_chunks.append(chunk)
            elif started_speaking:
                audio_chunks.append(chunk)
                silent_streak += 1
                if silent_streak >= silent_chunks_needed:
                    break

    if not audio_chunks:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(audio_chunks)
    print(f"[Listening] Captured {len(audio) / SAMPLE_RATE:.1f}s of audio.")
    return audio


def transcribe(model, audio: np.ndarray) -> str:
    if len(audio) < SAMPLE_RATE * 0.3:  # ignore very short clips
        return ""
    print("[ASR] Transcribing ...")
    results = model.transcribe(audio=[audio], batch_size=1)
    text = results[0] if isinstance(results[0], str) else results[0].text
    # Strip EOU special tokens if present
    text = text.replace("<EOU>", "").replace("<EOB>", "").strip()
    print(f"[ASR] → \"{text}\"")
    return text


# ──────────────────────────────────────────────
# LLM — Gemini
# ──────────────────────────────────────────────
def load_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] Set the GEMINI_API_KEY environment variable first.")
        sys.exit(1)
    from google import genai
    client = genai.Client(api_key=api_key)
    print("[LLM] Gemini client ready.")
    return client


def ask_gemini(client, history: list, user_text: str) -> str:
    from google.genai import types

    history.append({"role": "user", "parts": [{"text": user_text}]})

    print("[LLM] Sending to Gemini ...")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=history,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "thinking_config": {"thinking_budget": -1, "include_thoughts": False},
        },
    )
    reply = response.text.strip()
    history.append({"role": "model", "parts": [{"text": reply}]})
    print(f"[LLM] → \"{reply}\"")
    return reply


# ──────────────────────────────────────────────
# TTS — Piper
# ──────────────────────────────────────────────
def load_tts_voice():
    onnx_path = PIPER_VOICE_DIR / f"{PIPER_VOICE_NAME}.onnx"
    if not onnx_path.exists():
        print(f"[TTS] Voice model not found. Run:  python pipeline.py --download-voice")
        sys.exit(1)
    from piper import PiperVoice
    voice = PiperVoice.load(str(onnx_path))
    print("[TTS] Piper voice ready.")
    return voice


def speak(voice, text: str):
    print("[TTS] Synthesizing ...")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)

    buf.seek(0)
    with wave.open(buf, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        raw = wav_file.readframes(wav_file.getnframes())

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)

    print("[TTS] Playing ...")
    sd.play(audio, samplerate=sample_rate)
    sd.wait()


# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Voice assistant pipeline")
    parser.add_argument("--download-voice", action="store_true", help="Download Piper voice model and exit")
    args = parser.parse_args()

    if args.download_voice:
        print("Downloading Piper voice model ...")
        download_voice()
        print("Done.")
        return

    print("=" * 50)
    print("  Voice Assistant  |  Parakeet + Gemini + Piper")
    print("=" * 50)

    asr_model = load_asr_model()
    gemini_client = load_gemini_client()
    tts_voice = load_tts_voice()

    conversation_history = []

    print("\nReady! Press Ctrl+C to quit.\n")

    while True:
        try:
            audio = record_until_silence()
            if len(audio) == 0:
                continue

            user_text = transcribe(asr_model, audio)
            if not user_text:
                print("[Info] Nothing detected, listening again ...")
                continue

            reply = ask_gemini(gemini_client, conversation_history, user_text)
            speak(tts_voice, reply)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"[Error] {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()
