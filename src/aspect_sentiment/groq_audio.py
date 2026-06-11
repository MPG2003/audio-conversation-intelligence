from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx

from src.aspect_sentiment.audio import TranscriptionResult

logger = logging.getLogger(__name__)

class GroqCloudTranscriber:
    """
    A drop-in replacement for WhisperTranscriber that offloads audio processing 
    to Groq's lightning-fast Whisper API, using 0 local RAM.
    """
    def __init__(self, model_size: str | None = None, device: str | None = None) -> None:
        self.model_size = "whisper-large-v3-turbo"
        self.device = "cloud"
        self.api_key = os.getenv("LLAMA_API_KEY") or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("No LLAMA_API_KEY found. GroqCloudTranscriber will fail.")

    def transcribe(self, audio_path: Path | str, language_hint: str | None = None) -> TranscriptionResult:
        if not self.api_key:
            raise ValueError("LLAMA_API_KEY is not set. Cannot use Groq Whisper.")

        audio_path = Path(audio_path)
        logger.info("Transcribing audio via Groq Cloud API (whisper-large-v3-turbo)...")
        
        # Determine content type based on extension (fallback to mpeg)
        ext = audio_path.suffix.lower().replace(".", "")
        content_type = f"audio/{ext}" if ext in ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"] else "audio/mpeg"

        with open(audio_path, "rb") as file:
            files = {"file": (audio_path.name, file, content_type)}
            data = {
                "model": "whisper-large-v3-turbo",
                "response_format": "verbose_json",
            }
            if language_hint:
                data["language"] = language_hint

            response = httpx.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=files,
                data=data,
                timeout=120.0,
            )
            
            if response.status_code != 200:
                logger.error(f"Groq Whisper API failed: {response.text}")
                response.raise_for_status()
                
            result = response.json()

        text = result.get("text", "").strip()
        segments = result.get("segments", [])
        
        # If segments aren't provided by the API but we have text, mock one
        if not segments and text:
            segments = [{"start": 0.0, "end": result.get("duration", 1.0), "text": text}]
            
        duration = result.get("duration", 0.0)
        if not duration and segments:
            duration = float(segments[-1].get("end", 0.0))

        logger.info(f"Groq Transcription complete. Text length: {len(text)} chars.")

        return TranscriptionResult(
            text=text,
            segments=segments,
            language=result.get("language"),
            confidence=0.99,  # Mocked high confidence for cloud API
            duration_seconds=duration,
        )
