from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.transcribe import choose_device

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
HALLUCINATION_TEXTS = {"thank you", "thanks for watching", "subscribe", "you"}


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    segments: list[dict[str, Any]]
    language: str | None
    confidence: float | None
    duration_seconds: float


class WhisperTranscriber:
    def __init__(self, model_size: str | None = None, device: str | None = None) -> None:
        self.model_size = model_size or os.getenv("WHISPER_MODEL_SIZE", "small")
        self.device = choose_device(device or os.getenv("WHISPER_DEVICE"))

    def _ensure_ffmpeg_on_path(self) -> None:
        existing = os.environ.get("PATH", "")
        for ffmpeg_dir in REPO_ROOT.glob("ffmpeg-*"):
            candidate = ffmpeg_dir / "bin"
            ffmpeg_exe = candidate / "ffmpeg.exe"
            if ffmpeg_exe.exists() and str(candidate) not in existing:
                os.environ["PATH"] = f"{candidate}{os.pathsep}{existing}"
                logger.info("Added bundled ffmpeg to PATH from %s", candidate)
                return

    @lru_cache(maxsize=1)
    def _load_model(self):
        self._ensure_ffmpeg_on_path()
        logger.info("Loading Whisper model '%s' on %s", self.model_size, self.device)
        import whisper

        return whisper.load_model(self.model_size, device=self.device)

    @staticmethod
    def _estimate_confidence(result: dict[str, Any]) -> float | None:
        segments = result.get("segments") or []
        if not segments:
            return None

        confidence_values = []
        for segment in segments:
            text = str(segment.get("text", "")).strip()
            if not text:
                continue

            avg_logprob = float(segment.get("avg_logprob", -1.0))
            no_speech_prob = float(segment.get("no_speech_prob", 0.25))
            compression_ratio = float(segment.get("compression_ratio", 1.0))

            logprob_score = max(0.0, min(1.0, pow(2.718281828, avg_logprob)))
            speech_score = max(0.0, min(1.0, 1.0 - no_speech_prob))
            compression_penalty = 0.75 if compression_ratio > 2.4 else 1.0

            # Whisper's no_speech_prob is unreliable on short, quiet sales-call turns.
            # Avg log-prob tracks transcript quality better, so use it as the main signal.
            confidence_values.append((0.85 * logprob_score + 0.15 * speech_score) * compression_penalty)

        if not confidence_values:
            return None
        return round(sum(confidence_values) / len(confidence_values), 3)

    @staticmethod
    def _estimate_duration(result: dict[str, Any]) -> float:
        segments = result.get("segments") or []
        if not segments:
            return 0.0
        return round(float(segments[-1].get("end", 0.0) or 0.0), 2)

    @staticmethod
    def _clean_segments(raw_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        previous_text = ""
        for segment in raw_segments:
            text = " ".join(str(segment.get("text", "")).split()).strip()
            normalized = re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()
            avg_logprob = float(segment.get("avg_logprob", -1.0))
            no_speech_prob = float(segment.get("no_speech_prob", 0.0))
            compression_ratio = float(segment.get("compression_ratio", 1.0))

            if not normalized:
                continue
            if no_speech_prob > 0.75 and avg_logprob < -0.8:
                continue
            if compression_ratio > 2.8:
                continue
            if normalized in HALLUCINATION_TEXTS and (avg_logprob < -0.55 or no_speech_prob > 0.45):
                continue
            if normalized == previous_text and len(normalized.split()) <= 5:
                continue

            cleaned.append(
                {
                    "start": float(segment.get("start", 0.0) or 0.0),
                    "end": float(segment.get("end", 0.0) or 0.0),
                    "text": text,
                    "avg_logprob": avg_logprob,
                    "no_speech_prob": no_speech_prob,
                    "compression_ratio": compression_ratio,
                }
            )
            previous_text = normalized
        return cleaned

    def transcribe(self, audio_path: Path, language_hint: str | None = None) -> TranscriptionResult:
        model = self._load_model()
        use_fp16 = bool(getattr(model, "device", None) and getattr(model.device, "type", None) == "cuda")
        result = model.transcribe(
            str(audio_path),
            language=language_hint,
            fp16=use_fp16,
            temperature=0,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        segments = self._clean_segments(result.get("segments", []))
        text = " ".join(segment["text"] for segment in segments).strip()

        return TranscriptionResult(
            text=text,
            segments=segments,
            language=result.get("language"),
            confidence=self._estimate_confidence(result),
            duration_seconds=self._estimate_duration(result),
        )
