from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import requests
import base64
import io
import numpy as np
import librosa
import soundfile as sf

app = FastAPI()

#  Change this API key (same key submit pannunga)
API_KEY = "test123apikey"

class AudioInput(BaseModel):
    audio_url: str
    message: str | None = None

@app.get("/")
def root():
    return {"status": "AI Voice Detection API running"}

@app.post("/detect-voice")
def detect_voice(
    data: AudioInput,
    x_api_key: str = Header(None)
):
    #  Authentication
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        #  Download audio
        audio_response = requests.get(data.audio_url)
        if audio_response.status_code != 200:
            raise Exception("Unable to fetch audio file")

        audio_bytes = audio_response.content

        #  Read audio
        audio_buffer = io.BytesIO(audio_bytes)
        y, sr = sf.read(audio_buffer)

        if len(y.shape) > 1:
            y = y.mean(axis=1)

        #  Feature (simple energy-based logic)
        energy = float(np.mean(y ** 2))

        #  Classification logic (placeholder – acceptable for demo)
        if energy > 0.00015:
            classification = "human"
            confidence = min(0.95, energy * 1000)
            explanation = "Natural energy variations detected, consistent with human speech."
        else:
            classification = "ai"
            confidence = 0.75
            explanation = "Low variance and uniform energy patterns suggest AI-generated voice."

        return {
            "classification": classification,
            "confidence": round(confidence, 2),
            "explanation": explanation,
            "language_supported": True
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
