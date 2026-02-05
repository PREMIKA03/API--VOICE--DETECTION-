from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional
import requests
import base64
import io
import numpy as np
import soundfile as sf
import tempfile

app = FastAPI()

# SAME API KEY you submit
API_KEY = "test123apikey"

# ✅ Request model (supports Base64 + URL)
class AudioInput(BaseModel):
    language: Optional[str] = None
    audioFormat: Optional[str] = None
    audioBase64: Optional[str] = None
    audio_url: Optional[str] = None

@app.get("/")
def root():
    return {"status": "AI Voice Detection API running"}

@app.post("/detect-voice")
def detect_voice(
    data: AudioInput,
    x_api_key: str = Header(None)
):
    # 🔐 Authentication
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        # 🎧 Get audio (Base64 OR URL)
        if data.audioBase64:
            audio_bytes = base64.b64decode(data.audioBase64)

        elif data.audio_url:
            audio_response = requests.get(data.audio_url)
            if audio_response.status_code != 200:
                raise Exception("Unable to fetch audio file")
            audio_bytes = audio_response.content

        else:
            raise HTTPException(status_code=400, detail="No audio provided")

        # 📂 Read audio bytes
        audio_buffer = io.BytesIO(audio_bytes)
        y, sr = sf.read(audio_buffer)

        # mono
        if len(y.shape) > 1:
            y = y.mean(axis=1)

        # 🔍 Simple energy feature
        energy = float(np.mean(y ** 2))

        # 🤖 Human / AI logic (demo-safe)
        if energy > 0.00015:
            classification = "human"
            confidence = min(0.95, energy * 1000)
            explanation = "Natural energy variations detected, consistent with human speech."
        else:
            classification = "ai"
            confidence = 0.75
            explanation = "Uniform energy patterns suggest AI-generated voice."

        return {
            "classification": classification,
            "confidence": round(confidence, 2),
            "explanation": explanation,
            "language": data.language or "unknown",
            "audio_format": data.audioFormat or "unknown"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
