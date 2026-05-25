# Hackathon Voice Assistant

Real-time voice conversation app powered by:

| Component | Model |
|---|---|
| **ASR** | `nvidia/parakeet_realtime_eou_120m-v1` (NeMo) |
| **LLM** | `gemini-3.5-flash` (with thinking enabled) |
| **TTS** | `Piper` (`en_US-lessac-medium`) |

Talk → It listens → Replies → Auto-listens again. No clicking.

---

## Quick start (Windows)

### 1. Prerequisites
- **Python 3.13** ([python.org](https://www.python.org/downloads/release/python-3135/))
- **Git** ([git-scm.com](https://git-scm.com/))
- A **Gemini API key** ([aistudio.google.com/apikey](https://aistudio.google.com/apikey))

### 2. Clone
```powershell
git clone https://github.com/Nitesh3000/NeMo.git hackathon
cd hackathon
```

> The project files (`server.py`, `static/`, `setup.ps1`, etc.) live alongside the cloned `NeMo/` directory in the parent repo.

### 3. Run the one-shot setup
```powershell
.\setup.ps1
```
This will (takes ~10 minutes the first time):
- Create the Python 3.13 venv
- Clone NeMo source (if missing)
- Install PyTorch CPU + all dependencies
- Apply Windows compatibility patches to NeMo source
- Install stubs for unavailable packages (`editdistance`, `nv_one_logger`)
- Download the Piper voice model
- Create your `.env` file

### 4. Add your Gemini API key
Open `.env` and replace `your-gemini-api-key-here`:
```
GEMINI_API_KEY=AIza...
```

### 5. Run
```powershell
.\venv313\Scripts\python.exe server.py
```
Open **http://127.0.0.1:8000**, click **"Start conversation"** once to grant mic permission, then just talk.

---

## What's in the box

| File | Purpose |
|---|---|
| `server.py` | FastAPI backend — loads models, runs the pipeline |
| `static/index.html` | Browser-based UI with continuous mic + VAD |
| `app.py` | Streamlit alternative UI (click-to-record) |
| `pipeline.py` | Pure-CLI version (speak in terminal) |
| `setup.ps1` | One-shot Windows install script |
| `requirements.txt` | Curated direct dependencies |
| `voices/` | Piper voice model files |
| `.env` | Your Gemini API key (gitignored) |

---

## Architecture

```
┌─ Browser (continuous mic + JS VAD) ─┐
│                                     │
│   1. Captures audio at 16kHz        │
│   2. Detects ~1s of silence         │
│   3. POSTs WAV to /api/process      │
│   4. Plays returned TTS audio       │
│   5. Loops automatically            │
└─────────────────────────────────────┘
                │
                ▼
┌─ FastAPI server (server.py) ─────────┐
│                                      │
│   Parakeet ASR  →  Gemini LLM        │
│        ↓                  ↓          │
│        └──────  Piper TTS  ←─────────│
└──────────────────────────────────────┘
```

---

## Troubleshooting

**"GEMINI_API_KEY not set"** — open `.env` and paste your key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

**Mic permission denied** — check browser site settings → allow microphone for `127.0.0.1:8000`.

**"Listening..." but nothing happens when I speak** — open browser console (F12), check for errors. Try lowering `SILENCE_THRESHOLD` in `static/index.html` (line ~165) to `0.005` if your mic is quiet, or raising it if there's background noise.

**Model takes forever to load** — first run downloads ~500MB from HuggingFace. Subsequent runs use the cache (`~/.cache/huggingface/`).

**CPU is slow** — yes, on CPU expect ~1-2s latency per turn. GPU would be near-instant; we can switch later.

---

## Collaboration

This fork lives at [github.com/Nitesh3000/NeMo](https://github.com/Nitesh3000/NeMo). To add teammates:
**Settings** → **Collaborators** → **Add people**.

Standard branch workflow: create a feature branch, commit, push, open a PR on this fork.
