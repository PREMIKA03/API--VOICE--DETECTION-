# AI-Generated Voice Detection API

This API classifies whether a given voice sample is human-generated or AI-generated.

## Endpoint
POST /detect-voice

## Headers
X-API-KEY: test123apikey

## Input JSON
{
  "audio_url": "https://example.com/sample.mp3",
  "message": "Tamil voice sample test"
}

## Output JSON
{
  "classification": "human",
  "confidence": 0.87,
  "explanation": "Natural energy variations detected, consistent with human speech.",
  "language_supported": true
}

## Deployment
Hosted on Render as a public API endpoint.
