from __future__ import annotations

import csv
import json
import os
import re
import shutil
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import WebSocket, WebSocketDisconnect, FastAPI, File, HTTPException, UploadFile
from src.api.worker import background_worker, queue, JOBS
import uuid
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.aspect_sentiment.audio import WhisperTranscriber
from src.aspect_sentiment.diarization import DiarizationResult, diarize_audio_segments, diarize_text
from src.aspect_sentiment.engine import AspectSentimentEngine
from src.aspect_sentiment.follow_up_alerts import (
    detect_follow_up_alerts,
    init_follow_up_db,
    list_follow_up_alerts,
    save_follow_up_alerts,
    update_follow_up_status,
)
from src.aspect_sentiment.llama_extraction import derive_features, get_sentiment, process_text, rule_based_features, summarize_conversation
from src.aspect_sentiment.privacy import PrivacyResult, extract_and_redact_pii
from src.aspect_sentiment.schemas import AnalysisResponse, PipelineStage
from src.aspect_sentiment.probability_fusion import fuse_probabilities

REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPT_CSV_PATHS = [
    REPO_ROOT / "data" / "raw" / "transcripts.csv",
]
MODEL_DIR = REPO_ROOT / "models"
CONVERSION_MODEL_PATH = MODEL_DIR / "sales_conversion_model.pkl"
MODEL_FEATURES_PATH = MODEL_DIR / "sales_conversion_features.pkl"
MODEL_METRICS_PATH = MODEL_DIR / "sales_conversion_metrics.json"
TRANSCRIPT_CSV_FIELDS = [
    "file_name",
    "text",
    "language",
    "duration_s",
    "timestamp",
    "products",
    "brands",
    "budget",
    "features",
    "intent",
    "decision_stage",
    "use_case",
    "objections",
    "sentiment",
    "confidence_score",
    "hesitation_score",
    "delay_flag",
    "conversion_label",
    "conversion_probability",
    "conversion_prediction",
    "model_accuracy",
    "model_precision",
    "model_recall",
    "model_f1",
    "xgboost_base_probability",
    "intent_score",
    "behavioral_score",
    "emotion_score",
    "engagement_score",
    "extraction_provider",
    "pii_redaction_count",
    "raw_features_json",
]
TRANSCRIPT_CSV_LOCK = threading.Lock()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file(REPO_ROOT / ".env.local")
load_env_file(REPO_ROOT / ".env")


def append_transcript_csv(
    *,
    source_name: str,
    text: str,
    result: dict[str, Any] | None = None,
    language: str | None = None,
    duration_s: float | None = None,
) -> None:
    cleaned_text = " ".join(text.split())
    if not cleaned_text:
        return

    row = {
        "file_name": source_name,
        "text": cleaned_text,
        "language": language or "",
        "duration_s": duration_s if duration_s is not None else "",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **csv_feature_columns(result or {}),
    }

    with TRANSCRIPT_CSV_LOCK:
        for csv_path in TRANSCRIPT_CSV_PATHS:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            ensure_transcript_csv_schema(csv_path)
            needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
            with csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=TRANSCRIPT_CSV_FIELDS)
                if needs_header:
                    writer.writeheader()
                writer.writerow(row)


def ensure_transcript_csv_schema(csv_path: Path) -> None:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames == TRANSCRIPT_CSV_FIELDS:
            return
        rows = list(reader)

    upgraded_rows = [{field: row.get(field, "") for field in TRANSCRIPT_CSV_FIELDS} for row in rows]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRANSCRIPT_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(upgraded_rows)


def _csv_join(values: list[str]) -> str:
    unique = []
    for value in values:
        clean = str(value or "").strip()
        if clean and clean.lower() not in {item.lower() for item in unique}:
            unique.append(clean)
    return ", ".join(unique)


def csv_feature_columns(result: dict[str, Any]) -> dict[str, Any]:
    raw_features = result.get("rawFeatures")
    if not isinstance(raw_features, list):
        raw_features = []

    by_label: dict[str, list[str]] = {}
    for feature in raw_features:
        if not isinstance(feature, dict):
            continue
        label = str(feature.get("label") or "FEATURE").upper()
        value = str(feature.get("name") or feature.get("value") or "").strip()
        if value:
            by_label.setdefault(label, []).append(value)

    pipeline_features = result.get("pipelineFeatures") if isinstance(result.get("pipelineFeatures"), dict) else {}
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    conversion = result.get("conversionScore") if isinstance(result.get("conversionScore"), dict) else {}
    prediction = result.get("prediction") if isinstance(result.get("prediction"), dict) else {}
    debug_metrics = prediction.get("debug_metrics") if isinstance(prediction.get("debug_metrics"), dict) else {}
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    model_metrics = load_model_metrics()

    return {
        "products": _csv_join(by_label.get("PRODUCT", [])),
        "brands": _csv_join(by_label.get("BRAND", [])),
        "budget": _csv_join(by_label.get("BUDGET", [])),
        "features": _csv_join(by_label.get("FEATURE", [])),
        "intent": _csv_join(by_label.get("INTENT", [])),
        "decision_stage": _csv_join(by_label.get("DECISION_STAGE", [])),
        "use_case": _csv_join(by_label.get("USE_CASE", [])),
        "objections": _csv_join(by_label.get("OBJECTION", []) + by_label.get("OBJECTION_TYPE", [])),
        "sentiment": summary.get("dominant", ""),
        "confidence_score": pipeline_features.get("confidence_score", ""),
        "hesitation_score": pipeline_features.get("hesitation_score", ""),
        "delay_flag": pipeline_features.get("delay_flag", ""),
        "conversion_label": conversion.get("label", ""),
        "conversion_probability": conversion.get("probability", ""),
        "conversion_prediction": prediction.get("prediction", ""),
        "model_accuracy": model_metrics.get("accuracy", ""),
        "model_precision": model_metrics.get("precision", ""),
        "model_recall": model_metrics.get("recall", ""),
        "model_f1": model_metrics.get("f1", ""),
        "xgboost_base_probability": debug_metrics.get("xgboost_base", ""),
        "intent_score": debug_metrics.get("intent_score", ""),
        "behavioral_score": debug_metrics.get("behavioral_score_scaled", ""),
        "emotion_score": debug_metrics.get("emotion_score", ""),
        "engagement_score": debug_metrics.get("engagement_score", ""),
        "extraction_provider": metadata.get("extractionProvider", pipeline_features.get("extraction_provider", "")),
        "pii_redaction_count": metadata.get("piiRedactionCount", pipeline_features.get("privacy_redaction_count", "")),
        "raw_features_json": json.dumps(raw_features, ensure_ascii=True),
    }


def load_model_metrics() -> dict[str, Any]:
    if not MODEL_METRICS_PATH.exists():
        return {}
    try:
        payload = json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def bundled_ffmpeg_path() -> str | None:
    for ffmpeg_dir in REPO_ROOT.glob("ffmpeg-*"):
        candidate = ffmpeg_dir / "bin" / "ffmpeg.exe"
        if candidate.exists():
            return str(candidate)
    return shutil.which("ffmpeg")


# Add bundled ffmpeg to PATH for child processes (e.g. Whisper)
_ffmpeg_bin = bundled_ffmpeg_path()
if _ffmpeg_bin:
    _ffmpeg_dir = str(Path(_ffmpeg_bin).parent)
    if _ffmpeg_dir not in os.environ["PATH"]:
        os.environ["PATH"] = _ffmpeg_dir + os.path.pathsep + os.environ["PATH"]


def privacy_safe_csv_text(result: dict[str, Any], fallback_text: str) -> str:
    turns = result.get("diarizedTranscript")
    if isinstance(turns, list) and turns:
        safe_turns: list[str] = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            speaker = str(turn.get("speaker") or "Speaker")
            turn_text = str(turn.get("text") or "").strip()
            if not turn_text:
                continue
            if speaker == "Customer":
                turn_text = extract_and_redact_pii(turn_text).cleaned_text
            safe_turns.append(f"{speaker}: {turn_text}")
        if safe_turns:
            return " ".join(safe_turns)

    return str(result.get("customerBehavioralTranscript") or result.get("transcript") or fallback_text)


app = FastAPI(title="AI Audio Analysis API", version="1.0.0")

DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]
allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", ",".join(DEFAULT_ALLOWED_ORIGINS)).split(",")
    if origin.strip()
]


@app.on_event("startup")
async def startup_event():
    init_follow_up_db()
    asyncio.create_task(background_worker(transcriber, diarize_audio_segments, analyze_text_payload))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from src.aspect_sentiment.groq_audio import GroqCloudTranscriber

engine = AspectSentimentEngine()
if os.getenv("USE_GROQ_WHISPER", "true").lower() == "true":
    transcriber = GroqCloudTranscriber()
else:
    transcriber = WhisperTranscriber()
conversion_model = joblib.load(CONVERSION_MODEL_PATH)
model_features = joblib.load(MODEL_FEATURES_PATH)


class TextAnalysisRequest(BaseModel):
    text: str
    sourceName: str = "typed-conversation"


class FollowUpStatusRequest(BaseModel):
    status: str


def to_dict(response: AnalysisResponse) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return response.dict()


def completed_stage(id: str, title: str, detail: str) -> PipelineStage:
    return PipelineStage(id=id, title=title, status="completed", detail=detail)


def budget_value(value: str) -> int:
    digits = "".join(ch for ch in value if ch.isdigit())
    return int(digits) if digits else 0


MONEY_CONTEXT_RX = re.compile(
    r"\b(?P<context>budget|salary|earning|income|pay|price|cost)\b"
    r"[^.?!]{0,80}?"
    r"(?P<currency>rs\.?|inr|₹)?\s*"
    r"(?<![A-Za-z])(?P<amount>[0-9][0-9,]*(?:\.\d+)?)"
    r"\s*(?P<suffix>k|lakh|lakhs)?"
    r"(?![\d,])"
    r"(?!\s*(?:gb|tb|ram|ssd|inch)\b)",
    re.IGNORECASE,
)
BUDGET_AMOUNT_RX = re.compile(
    r"\b(?:my\s+budget(?:\s+is)?|budget\s+(?:is|of)|"
    r"(?:hoping|want|need|trying)\s+to\s+(?:stay|keep\s+it)\s+(?:at|around|under|within)|"
    r"(?:stay|keep\s+it)\s+(?:at|around|under|within))"
    r"[^.?!]{0,30}?"
    r"(?P<currency>rs\.?|inr|₹)?\s*"
    r"(?<![A-Za-z])(?P<amount>[0-9][0-9,]*(?:\.\d+)?)"
    r"\s*(?P<suffix>k|lakh|lakhs)?"
    r"(?![\d,])"
    r"(?!\s*(?:gb|tb|ram|ssd|inch)\b)",
    re.IGNORECASE,
)
SELF_INTRO_RX = re.compile(r"\b(?i:my name is|i am|i'm|this is)\s+(?P<name>[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,2})(?=\b|[.,!?])")
GREETING_RX = re.compile(r"\b(?:hi|hello|good morning|good afternoon)\s+(?P<name>[A-Z][A-Za-z]+)\b", re.IGNORECASE)
OCCUPATION_RX = re.compile(r"\b(?:i am|i'm|working as|work as)\s+(?:an?\s+)?(?P<job>teacher|student|engineer|doctor|developer|manager|salesperson|consultant|designer)\b", re.IGNORECASE)
PRODUCT_NAME_TERMS = {
    "iphone",
    "samsung",
    "galaxy",
    "ultra",
    "pro",
    "max",
    "plus",
    "s25",
    "s24",
    "laptop",
    "phone",
    "mobile",
    "tv",
    "ac",
    "refrigerator",
    "washing",
    "machine",
}
NON_PERSON_NAME_TERMS = {
    "interested",
    "looking",
    "earning",
    "teacher",
    "student",
    "calling",
    "sure",
    "here",
    "product",
    "service",
    "proposal",
    "quotation",
    "pricing",
}


def normalize_money(amount: str, suffix: str | None = None) -> str:
    value = float(amount.replace(",", ""))
    suffix = (suffix or "").lower()
    if suffix == "k":
        value *= 1000
    elif suffix in {"lakh", "lakhs"}:
        value *= 100000
    return str(int(value)) if value.is_integer() else str(value)


def _valid_money_amount(amount: str, suffix: str | None, currency: str | None) -> bool:
    try:
        value = float(normalize_money(amount, suffix))
    except ValueError:
        return False
    return value >= 100 or bool(suffix) or bool(currency)


def _valid_person_name(value: str) -> bool:
    clean = value.strip().strip(".,!?")
    if not clean or any(ch.isdigit() for ch in clean):
        return False
    parts = [part.lower() for part in clean.split()]
    if any(part in NON_PERSON_NAME_TERMS or part in PRODUCT_NAME_TERMS for part in parts):
        return False
    return all(part[:1].isalpha() for part in parts)


def _append_entity(entities: list[dict[str, Any]], entity_type: str, value: str, source: str) -> None:
    clean = value.strip().strip(".,!?")
    if not clean:
        return
    if entity_type in {"customer_name", "agent_name"} and not _valid_person_name(clean):
        return
    opposite_type = "agent_name" if entity_type == "customer_name" else "customer_name"
    if entity_type in {"customer_name", "agent_name"}:
        if any(item["type"] == opposite_type and item["value"].lower() == clean.lower() for item in entities):
            return
    key = (entity_type, clean.lower())
    if key not in {(item["type"], item["value"].lower()) for item in entities}:
        entities.append({"type": entity_type, "value": clean, "source": source, "start": None, "end": None})


def local_structured_entities(text: str, diarization: DiarizationResult) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []

    for match in BUDGET_AMOUNT_RX.finditer(text):
        if _valid_money_amount(
            match.group("amount"),
            match.group("suffix"),
            match.group("currency"),
        ):
            _append_entity(
                entities,
                "budget",
                normalize_money(match.group("amount"), match.group("suffix")),
                "local-regex",
            )

    for match in MONEY_CONTEXT_RX.finditer(text):
        context = match.group("context").lower()
        if not _valid_money_amount(
            match.group("amount"),
            match.group("suffix"),
            match.group("currency"),
        ):
            continue
        amount = normalize_money(match.group("amount"), match.group("suffix"))
        entity_type = "budget" if context == "budget" else "product_price" if context in {"price", "cost"} else "income"
        _append_entity(entities, entity_type, amount, "local-regex")
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        lower = sentence.lower()
        if any(term in lower for term in ["earning", "income", "salary", "per month"]):
            for match in re.finditer(r"(?:rs\.?|inr|₹)?\s*([0-9][0-9,]*(?:\.\d+)?)\s*(k|lakh|lakhs)?", sentence, re.IGNORECASE):
                _append_entity(entities, "income", normalize_money(match.group(1), match.group(2)), "local-regex")

    for turn in diarization.turns:
        if turn.speaker == "Customer":
            for match in SELF_INTRO_RX.finditer(turn.text):
                candidate = match.group("name")
                if candidate.lower() not in {"looking", "earning", "teacher", "interested", "calling", "sure", "here"}:
                    _append_entity(entities, "customer_name", candidate, "speaker-regex")
            for match in GREETING_RX.finditer(turn.text):
                candidate = match.group("name")
                if candidate.lower() not in {"sir", "madam", "maam", "ma'am", "there"}:
                    _append_entity(entities, "agent_name", candidate, "speaker-regex")
            for match in OCCUPATION_RX.finditer(turn.text):
                _append_entity(entities, "job_title", match.group("job"), "speaker-regex")
        elif turn.speaker == "Agent":
            for match in SELF_INTRO_RX.finditer(turn.text):
                candidate = match.group("name")
                if candidate.lower() not in {"calling", "sure", "here", "just"}:
                    _append_entity(entities, "agent_name", candidate, "speaker-regex")
            for match in GREETING_RX.finditer(turn.text):
                candidate = match.group("name")
                if candidate.lower() not in {"sir", "madam", "maam", "ma'am", "there"}:
                    _append_entity(entities, "customer_name", candidate, "speaker-regex")

    return entities


def build_conversion_row(extraction: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    row: dict[str, Any] = {
        "budget": 0,
        "sentiment_score": extraction.get("sentiment_score", 0),
        "confidence_score": extraction.get("confidence_score", 0),
        "hesitation_score": extraction.get("hesitation_score", 0),
        "delay_flag": extraction.get("delay_flag", 0),
        "feature_count": extraction.get("feature_count", 0),
        "brand_count": extraction.get("brand_count", 0),
        "interaction_length": extraction.get("interaction_length", 0),
    }

    from src.aspect_sentiment.mapping_engine import process_extractions
    
    # Semantic Mapping Layer applies taxonomy normalization
    normalized = process_extractions(extraction.get("raw_features", []))
    for k, v in normalized.to_xgboost_dict().items():
        row[k] = v

    explanation_row = pd.DataFrame([row])
    model_row = explanation_row.copy()
    for column in model_features:
        if column not in model_row.columns:
            model_row[column] = 0

    return model_row[model_features], explanation_row


def explain_prediction(row: pd.DataFrame) -> list[str]:
    def value(column: str, default: int | float = 0) -> int | float:
        if column not in row.columns:
            return default
        return row[column].values[0]

    reasons: list[str] = []
    if value("confidence_score") > 0.6:
        reasons.append("Customer shows buying intent")
    if value("hesitation_score") >= 2:
        reasons.append("Customer is hesitant")
    if value("delay_flag") == 1:
        reasons.append("Customer postponed decision")
    if value("sentiment_score") > 0.3:
        reasons.append("Positive sentiment")
    if not reasons:
        reasons.append("Limited buying signals detected")
    return reasons


def predict_with_trained_model(extraction: dict[str, Any], text: str, agent_text: str = "") -> dict[str, Any]:
    model_row, explanation_row = build_conversion_row(extraction)
    xgboost_prob = float(conversion_model.predict_proba(model_row)[0][1])
    
    raw_features = extraction.get("raw_features", [])
    sentiment_score = float(extraction.get("sentiment_score", 0))
    
    return fuse_probabilities(
        xgboost_prob=xgboost_prob,
        transcript=text,
        raw_features=raw_features,
        sentiment_score=sentiment_score,
        agent_transcript=agent_text,
    )


def fallback_extraction(text: str) -> dict[str, Any]:
    features = rule_based_features(text)
    sentiment = get_sentiment(text)
    derived = derive_features(text, features)
    
    return {
        "raw_features": features,
        "sentiment_score": sentiment,
        **derived,
        "extraction_provider": "local-fallback",
    }


def pii_payload(privacy: PrivacyResult, extra_entities: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    grouped: dict[str, list[str]] = {}
    unique_entities: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for entity in privacy.entities:
        key = (entity.type, str(entity.value).lower().strip())
        if key not in seen:
            seen.add(key)
            unique_entities.append({
                "type": entity.type,
                "value": entity.value,
                "source": entity.source,
                "start": entity.start,
                "end": entity.end,
            })
            grouped.setdefault(entity.type, []).append(entity.value)

    for entity in extra_entities or []:
        key = (entity["type"], str(entity["value"]).lower().strip())
        if key not in seen:
            seen.add(key)
            unique_entities.append({
                "type": entity["type"],
                "value": entity["value"],
                "source": entity.get("source", "local"),
                "start": entity.get("start"),
                "end": entity.get("end"),
            })
            grouped.setdefault(entity["type"], []).append(entity["value"])

    return {
        "entities": unique_entities,
        "grouped": grouped,
        "redactionCount": privacy.redaction_count,
        "provider": privacy.provider,
    }


def transcript_payload(diarization: DiarizationResult) -> list[dict[str, Any]]:
    return [
        {
            "speaker": turn.speaker,
            "rawSpeaker": turn.raw_speaker,
            "text": turn.text,
            "start": turn.start,
            "end": turn.end,
        }
        for turn in diarization.turns
    ]


def summarize_customer_behavior(customer_text: str, extraction: dict[str, Any], conversation_summary: dict[str, Any] = None) -> dict[str, Any]:
    if conversation_summary is None:
        conversation_summary = {}
        
    return {
        "focus": "customer-only",
        "intentSignals": conversation_summary.get("intentScore", 0),
        "hesitationScore": conversation_summary.get("hesitationScore", 0),
        "urgencySignals": conversation_summary.get("urgencyScore", 0),
        "objectionSignals": len([f for f in extraction.get("raw_features", []) if f["label"] == "OBJECTION"]),
        "wordCount": len(customer_text.split()),
        "privacySafe": True,
    }


async def pipeline_response(
    text: str,
    source_name: str,
    source_type: str,
    started: float,
    diarization: DiarizationResult | None = None,
    transcription_confidence: float | None = None,
    whisper_model: str | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    diarized = diarization or diarize_text(text)
    customer_text = diarized.customer_text or text
    agent_text = diarized.agent_text
    privacy = extract_and_redact_pii(customer_text)
    local_entities = local_structured_entities(text, diarized)
    llama_text = privacy.cleaned_text

    extraction = await process_text(llama_text)
    if extraction is None:
        extraction = fallback_extraction(llama_text)
    extraction["privacy_redaction_count"] = privacy.redaction_count
    extraction["analysis_scope"] = "customer_only"
    conversation_summary = await summarize_conversation(
        transcript=" ".join(text.split()),
        customer_text=llama_text,
        agent_text=agent_text,
    )

    prediction = predict_with_trained_model(extraction, llama_text, agent_text)
    raw_features = extraction.get("raw_features", [])
    sentiment_score = float(extraction.get("sentiment_score", 0))
    audio_quality = None
    if transcription_confidence is not None:
        if transcription_confidence >= 0.82:
            quality_label = "Good"
        elif transcription_confidence >= 0.65:
            quality_label = "Fair"
        else:
            quality_label = "Poor"
        audio_quality = {
            "label": quality_label,
            "confidence": transcription_confidence,
            "language": language,
            "whisperModel": whisper_model,
        }
    
    def get_sentiment_label(score: float, txt: str) -> str:
        txt_lower = txt.lower()
        if any(w in txt_lower for w in ["confusing", "too many options", "don't know"]):
            return "Confused"
        if any(w in txt_lower for w in ["love", "amazing", "exactly what i need", "perfect"]):
            return "Emotionally Engaged"
        if any(w in txt_lower for w in ["maybe", "not sure", "thinking", "think about it"]):
            return "Hesitant"
        if score > 0.6: return "Very Positive"
        if score > 0.2: return "Positive"
        if score > 0.05: return "Mildly Positive"
        if score < -0.2: return "Negative"
        return "Neutral"

    dominant = get_sentiment_label(sentiment_score, llama_text)
    privacy_info = pii_payload(privacy, local_entities)
    detected_follow_ups = await detect_follow_up_alerts(
        customer_text=llama_text,
        diarization=diarized,
        privacy_payload=privacy_info,
    )
    follow_up_alerts = save_follow_up_alerts(
        detected_follow_ups,
        source_name=source_name,
        source_type=source_type,
    )

    products = [
        {
            "name": feature.get("name", ""),
            "entityType": feature.get("label", "FEATURE"),
            "sentiment": dominant,
            "score": sentiment_score,
            "confidence": extraction.get("confidence_score", 0.5),
            "mentions": 1,
            "context": feature.get("label", "Feature"),
        }
        for feature in raw_features
    ]

    return {
        "transcript": text,
        "diarizedTranscript": transcript_payload(diarized),
        "customerTranscript": customer_text,
        "customerBehavioralTranscript": llama_text,
        "agentTranscript": agent_text,
        "privacy": privacy_info,
        "followUpAlerts": follow_up_alerts,
        "customerBehaviorSummary": summarize_customer_behavior(llama_text, extraction, conversation_summary),
        "conversationSummary": conversation_summary,
        "normalizedText": " ".join(text.split()),
        "rawFeatures": raw_features,
        "pipelineFeatures": extraction,
        "products": products,
        "summary": {
            "positive": 100 if dominant == "positive" else 0,
            "negative": 100 if dominant == "negative" else 0,
            "neutral": 100 if dominant == "neutral" else 0,
            "dominant": dominant,
            "averageScore": sentiment_score,
            "totalProducts": len(raw_features),
        },
        "conversionScore": {
            "probability": prediction["probability"],
            "label": prediction["label"],
            "confidence": round(abs(prediction["probability"] - 0.5) * 2, 2),
            "features": extraction,
            "model": CONVERSION_MODEL_PATH.name,
        },
        "audioQuality": audio_quality,
        "prediction": prediction,
        "metadata": {
            "sourceType": source_type,
            "sourceName": source_name,
            "processingMs": int((time.perf_counter() - started) * 1000),
            "extractionProvider": extraction.get("extraction_provider", "llama"),
            "modelFeatures": len(model_features),
            "diarizationProvider": diarized.provider,
            "analysisScope": "customer_only_privacy_safe",
            "piiRedactionCount": privacy.redaction_count,
            "transcriptionConfidence": transcription_confidence,
            "whisperModel": whisper_model,
            "language": language,
        },
        "pipeline": [
            completed_stage("diarization", "Speaker diarization", f"Transcript separated with {diarized.provider}").dict(),
            completed_stage("privacy", "Local PII extraction", f"{privacy.redaction_count} sensitive item(s) redacted before LLaMA").dict(),
            completed_stage("llama", "Customer-only LLaMA extraction", "Structured sales features extracted from cleaned customer speech").dict(),
            completed_stage("follow-up-alerts", "Follow-up alert detection", f"{len(follow_up_alerts)} alert(s) saved").dict(),
            completed_stage("model", "Customer-weighted conversion model", "Hybrid conversion model executed").dict(),
        ],
    }


async def analyze_text_payload(
    text: str,
    *,
    source_name: str,
    source_type: str,
    language: str | None = None,
    transcription_confidence: float | None = None,
    whisper_model: str | None = None,
    pipeline: list[PipelineStage] | None = None,
    diarization: DiarizationResult | None = None,
    start_time: float | None = None,
) -> dict[str, Any]:
    if not text.strip():
        raise HTTPException(status_code=400, detail="Conversation text is required.")

    started = start_time or time.perf_counter()
    return await pipeline_response(
        text,
        source_name,
        source_type,
        started,
        diarization=diarization,
        transcription_confidence=transcription_confidence,
        whisper_model=whisper_model,
        language=language,
    )


@app.get("/health")
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "llamaConfigured": bool(engine.llama_api_key),
        "leadModelLoaded": bool(engine.lead_model),
        "whisperModel": transcriber.model_size,
        "diarizationBackend": os.getenv("DIARIZATION_BACKEND", "free-local"),
    }


@app.get("/api/readiness")
def readiness() -> dict[str, Any]:
    checks = {
        "conversionModel": CONVERSION_MODEL_PATH.exists(),
        "modelFeatures": MODEL_FEATURES_PATH.exists(),
        "modelMetrics": MODEL_METRICS_PATH.exists(),
        "leadModel": (REPO_ROOT / "data" / "processed" / "lead_scoring_model.joblib").exists(),
        "ffmpeg": bundled_ffmpeg_path() is not None,
        "llamaConfigured": bool(engine.llama_api_key),
        "allowedOriginsConfigured": bool(allowed_origins),
    }
    return {
        "status": "ready" if all(checks.values()) else "degraded",
        "checks": checks,
        "whisperModel": transcriber.model_size,
        "whisperDevice": transcriber.device,
        "diarizationBackend": os.getenv("DIARIZATION_BACKEND", "free-local"),
        "modelMetrics": load_model_metrics(),
    }


@app.get("/api/follow-up-alerts")
def get_follow_up_alerts(
    priority: str | None = None,
    status: str | None = None,
    customer_name: str | None = None,
) -> dict[str, Any]:
    return {
        "alerts": list_follow_up_alerts(
            priority=priority,
            status=status,
            customer_name=customer_name,
        )
    }


@app.patch("/api/follow-up-alerts/{alert_id}")
def patch_follow_up_alert(alert_id: str, request: FollowUpStatusRequest) -> dict[str, Any]:
    try:
        alert = update_follow_up_status(alert_id, request.status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not alert:
        raise HTTPException(status_code=404, detail="Follow-up alert not found")
    return {"alert": alert}


@app.post("/api/analyze")
async def analyze_text(request: TextAnalysisRequest) -> dict[str, Any]:
    result = await analyze_text_payload(
        request.text,
        source_name=request.sourceName,
        source_type="text",
    )
    append_transcript_csv(
        source_name=request.sourceName,
        text=privacy_safe_csv_text(result, request.text),
        result=result,
        language=result.get("metadata", {}).get("language"),
    )
    return result


@app.post("/api/upload")
async def upload_audio(audio: UploadFile = File(...)) -> dict[str, Any]:
    if not audio.filename:
        raise HTTPException(status_code=400, detail="Audio filename is required.")

    suffix = Path(audio.filename).suffix or ".audio"
    job_id = str(uuid.uuid4())
    started = time.perf_counter()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(await audio.read())

    JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "filename": audio.filename,
        "started_at": started,
        "result": None
    }
    
    await queue.put({
        "job_id": job_id,
        "tmp_path": str(tmp_path),
        "filename": audio.filename
    })

    return {"job_id": job_id, "status": "pending"}

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str) -> dict[str, Any]:
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]



@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    buffer = b""
    try:
        while True:
            data = await websocket.receive_bytes()
            buffer += data
            # In a real production system, you would pass the chunk to a streaming transcriber
            # For this prototype, we simulate accumulation and return a ping
            await websocket.send_json({"status": "receiving", "bytes_received": len(buffer)})
    except WebSocketDisconnect:
        print("Client disconnected from audio stream.")
