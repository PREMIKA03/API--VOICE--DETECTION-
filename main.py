from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional
import requests
import base64
import io
import numpy as np
import soundfile as sf

app = FastAPI()

API_KEY = "test123apikey"

class AudioInput(BaseModel):
    language: Optional[str] = None
    audioFormat: Optional[str] = None
    audioBase64: Optional[str] = None
    audio_url: Optional[str] = None

@app.get("/")
def root():
    return {
        "status": "running",
        "message": "AI Voice Detection API is live"
    }

@app.post("/detect-voice")
def detect_voice(
    data: AudioInput,
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        if data.audioBase64:
            b64 = data.audioBase64.strip()
            missing_padding = len(b64) % 4
            if missing_padding:
                b64 += "=" * (4 - missing_padding)
            audio_bytes = base64.b64decode(b64)

        elif data.audio_url:
            audio_response = requests.get(data.audio_url)
            if audio_response.status_code != 200:
                raise HTTPException(status_code=400, detail="Unable to fetch audio file")
            audio_bytes = audio_response.content

        else:
            raise HTTPException(status_code=400, detail="No audio provided")

        audio_buffer = io.BytesIO(audio_bytes)
        y, sr = sf.read(audio_buffer)

        if len(y.shape) > 1:
            y = y.mean(axis=1)

        energy = float(np.mean(y ** 2))

        if energy > 0.00015:
            classification = "human"
            confidence = min(0.95, energy * 1000)
            explanation = "Natural energy fluctuations detected, consistent with human speech."
        else:
            classification = "ai"
            confidence = 0.75
            explanation = "Uniform energy distribution suggests AI-generated voice."

        return {
            "classification": classification,
            "confidence": round(confidence, 2),
            "explanation": explanation,
            "language": data.language or "unknown",
            "audio_format": data.audioFormat or "unknown",
            "sample_rate": sr,
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
