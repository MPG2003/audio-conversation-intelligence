from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_RATE = 16000

SPEAKER_LABEL_RX = re.compile(
    r"(?i)(?<![A-Za-z])\[?\s*(customer|agent|speaker\s*[ab]|speaker_[ab])\s*\]?\s*:"
)
SENTENCE_SPLIT_RX = re.compile(r"(?<=[.!?])\s+")
AGENT_TERMS = {
    "good morning sir",
    "good afternoon sir",
    "how are you",
    "what about your name",
    "what about your job",
    "features you need",
    "good to know",
    "we have",
    "i can suggest",
    "i will share",
    "emi offer",
    "offer available",
    "available",
    "recommend",
    "let me",
    "our",
}
CUSTOMER_TERMS = {
    "my name",
    "my budget",
    "i am earning",
    "i am a",
    "i need",
    "i want",
    "my budget",
    "i mostly",
    "i may",
    "i think",
    "not sure",
    "under",
    "looking for",
    "can you",
}


@dataclass(slots=True)
class TranscriptTurn:
    speaker: str
    text: str
    start: float | None = None
    end: float | None = None
    raw_speaker: str | None = None


@dataclass(slots=True)
class DiarizationResult:
    turns: list[TranscriptTurn]
    speaker_map: dict[str, str] = field(default_factory=dict)
    provider: str = "heuristic"

    @property
    def formatted(self) -> str:
        return "\n".join(f"{turn.speaker}: {turn.text}" for turn in self.turns if turn.text)

    @property
    def customer_text(self) -> str:
        return " ".join(turn.text for turn in self.turns if turn.speaker == "Customer").strip()

    @property
    def agent_text(self) -> str:
        return " ".join(turn.text for turn in self.turns if turn.speaker == "Agent").strip()


def _normalize_role(label: str) -> str:
    compact = label.strip().lower().replace("_", " ")
    if compact == "customer":
        return "Customer"
    if compact == "agent":
        return "Agent"
    if compact in {"speaker a", "speaker a"}:
        return "Customer"
    if compact in {"speaker b", "speaker b"}:
        return "Agent"
    return label.strip().title()


def _overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _guess_speaker_from_text(text: str, index: int, fallback: str | None = None) -> str:
    lower = text.lower()
    customer_score = sum(1 for term in CUSTOMER_TERMS if term in lower)
    agent_score = sum(1 for term in AGENT_TERMS if term in lower)
    if "?" in text and any(term in lower for term in ["your name", "your job", "you need", "can i help", "what about", "use it for", "brand preference"]):
        agent_score += 2
    if any(term in lower for term in ["i can suggest", "we currently", "we have", "i will share", "i'll share", "both are good", "available"]):
        agent_score += 2
    if any(term in lower for term in ["i want", "i need", "not really", "not sure", "i'll think", "i will think", "get back to you"]):
        customer_score += 2
    if customer_score > agent_score:
        return "Customer"
    if agent_score > customer_score:
        return "Agent"
    if fallback in {"Agent", "Customer"}:
        return fallback
    return "Customer" if index % 2 == 0 else "Agent"


def _sentence_parts(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_SPLIT_RX.split(text.strip()) if part.strip()]


def _split_turn_by_sentence(
    turn: TranscriptTurn,
    start_index: int,
    *,
    preserve_speaker: bool = False,
) -> list[TranscriptTurn]:
    parts = _sentence_parts(turn.text)
    resolved_speaker = (
        _guess_speaker_from_text(turn.text, start_index, turn.speaker)
        if preserve_speaker
        else None
    )
    if len(parts) <= 1:
        guessed = resolved_speaker or _guess_speaker_from_text(turn.text, start_index, turn.speaker)
        return [
            TranscriptTurn(
                speaker=guessed,
                raw_speaker=turn.raw_speaker,
                text=turn.text,
                start=turn.start,
                end=turn.end,
            )
        ]

    duration = None
    if turn.start is not None and turn.end is not None and turn.end > turn.start:
        duration = turn.end - turn.start
    total_chars = max(1, sum(len(part) for part in parts))
    cursor = turn.start

    split_turns: list[TranscriptTurn] = []
    for offset, part in enumerate(parts):
        part_start = cursor
        part_end = None
        if duration is not None and cursor is not None:
            part_duration = duration * (len(part) / total_chars)
            part_end = min(turn.end, cursor + part_duration) if turn.end is not None else cursor + part_duration
            cursor = part_end

        split_turns.append(
            TranscriptTurn(
                speaker=resolved_speaker
                or _guess_speaker_from_text(part, start_index + offset, turn.speaker),
                raw_speaker=turn.raw_speaker,
                text=part,
                start=part_start,
                end=part_end,
            )
        )

    return split_turns


def _refine_turn_roles(turns: list[TranscriptTurn], *, preserve_speakers: bool = False) -> list[TranscriptTurn]:
    refined: list[TranscriptTurn] = []
    sentence_index = 0
    for turn in turns:
        split_turns = _split_turn_by_sentence(turn, sentence_index, preserve_speaker=preserve_speakers)
        refined.extend(split_turns)
        sentence_index += len(split_turns)
    return _merge_turns(refined)


def _merge_turns(turns: list[TranscriptTurn]) -> list[TranscriptTurn]:
    merged: list[TranscriptTurn] = []
    for turn in turns:
        if not turn.text.strip():
            continue
        if merged and merged[-1].speaker == turn.speaker:
            merged[-1].text = f"{merged[-1].text} {turn.text}".strip()
            merged[-1].end = turn.end if turn.end is not None else merged[-1].end
        else:
            merged.append(turn)
    return merged


def _ffmpeg_executable() -> str:
    for ffmpeg_dir in REPO_ROOT.glob("ffmpeg-*"):
        candidate = ffmpeg_dir / "bin" / "ffmpeg.exe"
        if candidate.exists():
            return str(candidate)
    return "ffmpeg"


def _load_audio_mono(audio_path: Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav_path = Path(tmp.name)

    command = [
        _ffmpeg_executable(),
        "-y",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        str(wav_path),
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with wave.open(str(wav_path), "rb") as handle:
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    finally:
        wav_path.unlink(missing_ok=True)

    return samples, rate


def _segment_samples(samples: np.ndarray, sample_rate: int, start: float, end: float) -> np.ndarray:
    start_index = max(0, int(start * sample_rate))
    end_index = min(len(samples), int(max(end, start + 0.2) * sample_rate))
    return samples[start_index:end_index]


def _frame_audio(samples: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if len(samples) < frame_size:
        padded = np.pad(samples, (0, frame_size - len(samples)))
        return padded.reshape(1, frame_size)
    frame_count = 1 + (len(samples) - frame_size) // hop_size
    shape = (frame_count, frame_size)
    strides = (samples.strides[0] * hop_size, samples.strides[0])
    return np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides).copy()


def _acoustic_features(samples: np.ndarray, sample_rate: int) -> list[float]:
    if len(samples) == 0:
        return [0.0] * 10

    frame_size = int(sample_rate * 0.025)
    hop_size = int(sample_rate * 0.010)
    frames = _frame_audio(samples, frame_size, hop_size)
    window = np.hanning(frame_size).astype(np.float32)
    windowed = frames * window

    rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-9)
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
    spectrum = np.abs(np.fft.rfft(windowed, axis=1)) + 1e-9
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    spectral_sum = np.sum(spectrum, axis=1)
    centroid = np.sum(spectrum * freqs, axis=1) / spectral_sum
    bandwidth = np.sqrt(np.sum(spectrum * np.square(freqs - centroid[:, None]), axis=1) / spectral_sum)
    peak_freq = freqs[np.argmax(spectrum, axis=1)]

    return [
        float(np.mean(rms)),
        float(np.std(rms)),
        float(np.percentile(rms, 90)),
        float(np.mean(zcr)),
        float(np.std(zcr)),
        float(np.mean(centroid)),
        float(np.std(centroid)),
        float(np.mean(bandwidth)),
        float(np.std(bandwidth)),
        float(np.mean(peak_freq)),
    ]


def _role_map_from_cluster_text(cluster_texts: dict[int, list[str]]) -> dict[int, str]:
    scores: dict[int, int] = {}
    for cluster_id, texts in cluster_texts.items():
        joined = " ".join(texts).lower()
        customer_score = sum(1 for term in CUSTOMER_TERMS if term in joined)
        agent_score = sum(1 for term in AGENT_TERMS if term in joined)
        scores[cluster_id] = customer_score - agent_score

    if not scores:
        return {}

    if len(scores) == 1:
        only_cluster = next(iter(scores))
        return {only_cluster: "Customer" if scores[only_cluster] >= 0 else "Agent"}

    customer_cluster = max(scores, key=lambda cluster_id: scores[cluster_id])
    return {cluster_id: ("Customer" if cluster_id == customer_cluster else "Agent") for cluster_id in scores}


def _heuristic_audio_diarization(whisper_segments: list[dict[str, Any]], provider: str = "heuristic") -> DiarizationResult:
    turns = [
        TranscriptTurn(
            speaker=_guess_speaker_from_text(str(segment.get("text", "")), index),
            raw_speaker=f"SPEAKER_{index % 2}",
            text=str(segment.get("text", "")).strip(),
            start=float(segment.get("start", 0.0) or 0.0),
            end=float(segment.get("end", 0.0) or 0.0),
        )
        for index, segment in enumerate(whisper_segments)
    ]
    return DiarizationResult(turns=_refine_turn_roles(turns), speaker_map={"SPEAKER_0": "Customer", "SPEAKER_1": "Agent"}, provider=provider)


def _free_local_diarization(audio_path: Path, whisper_segments: list[dict[str, Any]]) -> DiarizationResult | None:
    usable_segments = [
        segment
        for segment in whisper_segments
        if str(segment.get("text", "")).strip()
        and float(segment.get("end", 0.0) or 0.0) > float(segment.get("start", 0.0) or 0.0)
    ]
    if len(usable_segments) < 2:
        return None

    try:
        samples, sample_rate = _load_audio_mono(audio_path)
        feature_rows = [
            _acoustic_features(
                _segment_samples(
                    samples,
                    sample_rate,
                    float(segment.get("start", 0.0) or 0.0),
                    float(segment.get("end", 0.0) or 0.0),
                ),
                sample_rate,
            )
            for segment in usable_segments
        ]
        scaled = StandardScaler().fit_transform(np.asarray(feature_rows, dtype=np.float32))
        labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(scaled)
    except Exception as exc:
        logger.warning("Free local diarization failed, falling back to text heuristics: %s", exc)
        return None

    cluster_texts: dict[int, list[str]] = {}
    for label, segment in zip(labels, usable_segments):
        cluster_texts.setdefault(int(label), []).append(str(segment.get("text", "")).strip())

    role_map = _role_map_from_cluster_text(cluster_texts)
    turns = [
        TranscriptTurn(
            speaker=role_map.get(int(label), _guess_speaker_from_text(str(segment.get("text", "")), index)),
            raw_speaker=f"SPEAKER_{int(label)}",
            text=str(segment.get("text", "")).strip(),
            start=float(segment.get("start", 0.0) or 0.0),
            end=float(segment.get("end", 0.0) or 0.0),
        )
        for index, (label, segment) in enumerate(zip(labels, usable_segments))
    ]
    speaker_map = {f"SPEAKER_{cluster_id}": role for cluster_id, role in role_map.items()}
    return DiarizationResult(
        turns=_refine_turn_roles(turns, preserve_speakers=True),
        speaker_map=speaker_map,
        provider="free-local-kmeans+stable-roles",
    )


@lru_cache(maxsize=1)
def _load_pyannote_pipeline():
    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("PYANNOTE_AUTH_TOKEN")
    if not token:
        logger.info("Pyannote disabled because no Hugging Face token is configured.")
        return None

    try:
        from pyannote.audio import Pipeline

        pipeline_name = os.getenv("PYANNOTE_PIPELINE", "pyannote/speaker-diarization-community-1")
        try:
            return Pipeline.from_pretrained(pipeline_name, token=token)
        except TypeError:
            return Pipeline.from_pretrained(pipeline_name, use_auth_token=token)
    except Exception as exc:
        logger.warning("Could not load pyannote diarization pipeline: %s", exc)
        return None


def diarize_audio_segments(audio_path: Path, whisper_segments: list[dict[str, Any]]) -> DiarizationResult:
    if not whisper_segments:
        return DiarizationResult(turns=[], provider="empty")

    backend = os.getenv("DIARIZATION_BACKEND", "free-local").strip().lower()
    if backend in {"free", "free-local", "local", "kmeans"}:
        result = _free_local_diarization(audio_path, whisper_segments)
        if result is not None:
            return result
        return _heuristic_audio_diarization(whisper_segments, provider="heuristic-free-local-fallback")

    pipeline = _load_pyannote_pipeline()
    if pipeline is None:
        result = _free_local_diarization(audio_path, whisper_segments)
        if result is not None:
            return result
        return _heuristic_audio_diarization(whisper_segments, provider="heuristic-free-local-fallback")

    try:
        diarization = pipeline(str(audio_path), num_speakers=2)
    except Exception as exc:
        logger.warning("Pyannote diarization failed, falling back to heuristic speakers: %s", exc)
        result = _free_local_diarization(audio_path, whisper_segments)
        if result is not None:
            return result
        return _heuristic_audio_diarization(whisper_segments, provider="heuristic-pyannote-fallback")

    speaker_intervals: list[tuple[float, float, str]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_intervals.append((float(turn.start), float(turn.end), str(speaker)))

    raw_turns: list[TranscriptTurn] = []
    raw_speaker_order: list[str] = []
    for index, segment in enumerate(whisper_segments):
        start = float(segment.get("start", 0.0) or 0.0)
        end = float(segment.get("end", start) or start)
        text = str(segment.get("text", "")).strip()
        overlaps: dict[str, float] = {}
        for interval_start, interval_end, speaker in speaker_intervals:
            overlap = _overlap_seconds(start, end, interval_start, interval_end)
            if overlap > 0:
                overlaps[speaker] = overlaps.get(speaker, 0.0) + overlap
        raw_speaker = max(overlaps, key=overlaps.get) if overlaps else f"SPEAKER_{index % 2}"
        if raw_speaker not in raw_speaker_order:
            raw_speaker_order.append(raw_speaker)
        raw_turns.append(TranscriptTurn(speaker=raw_speaker, raw_speaker=raw_speaker, text=text, start=start, end=end))

    speaker_scores: dict[str, dict[str, int]] = {speaker: {"customer": 0, "agent": 0} for speaker in raw_speaker_order}
    for turn in raw_turns:
        lower = turn.text.lower()
        speaker_scores.setdefault(turn.speaker, {"customer": 0, "agent": 0})
        speaker_scores[turn.speaker]["customer"] += sum(1 for term in CUSTOMER_TERMS if term in lower)
        speaker_scores[turn.speaker]["agent"] += sum(1 for term in AGENT_TERMS if term in lower)

    speaker_map: dict[str, str] = {}
    if raw_speaker_order:
        customer_raw = max(raw_speaker_order, key=lambda s: (speaker_scores[s]["customer"] - speaker_scores[s]["agent"], -raw_speaker_order.index(s)))
        speaker_map[customer_raw] = "Customer"
        for raw in raw_speaker_order:
            speaker_map.setdefault(raw, "Agent")

    turns = [
        TranscriptTurn(
            speaker=speaker_map.get(turn.speaker, _guess_speaker_from_text(turn.text, index)),
            raw_speaker=turn.raw_speaker,
            text=turn.text,
            start=turn.start,
            end=turn.end,
        )
        for index, turn in enumerate(raw_turns)
    ]
    return DiarizationResult(
        turns=_refine_turn_roles(turns, preserve_speakers=True),
        speaker_map=speaker_map,
        provider="pyannote.audio+stable-roles",
    )


def diarize_text(text: str) -> DiarizationResult:
    matches = list(SPEAKER_LABEL_RX.finditer(text))
    if matches:
        turns: list[TranscriptTurn] = []
        for index, match in enumerate(matches):
            next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            turn_text = text[match.end() : next_start].strip()
            if turn_text:
                turns.append(TranscriptTurn(speaker=_normalize_role(match.group(1)), raw_speaker=match.group(1), text=turn_text))
        return DiarizationResult(turns=_merge_turns(turns), provider="explicit-labels")

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    if not sentences:
        sentences = [text.strip()] if text.strip() else []
    turns = [TranscriptTurn(speaker=_guess_speaker_from_text(sentence, index), text=sentence) for index, sentence in enumerate(sentences)]
    return DiarizationResult(turns=_merge_turns(turns), speaker_map={"SPEAKER_0": "Customer", "SPEAKER_1": "Agent"}, provider="heuristic-text")
