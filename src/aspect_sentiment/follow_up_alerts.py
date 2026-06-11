from __future__ import annotations

import json
import re
import csv
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.aspect_sentiment.diarization import DiarizationResult
from src.aspect_sentiment.llama_extraction import call_llama


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = REPO_ROOT / "data" / "processed" / "follow_up_alerts.csv"
FOLLOW_UP_DB_LOCK = threading.Lock()

VALID_PRIORITIES = {"High", "Medium", "Low"}
VALID_STATUSES = {"Pending", "Completed"}
SENTENCE_SPLIT_RX = re.compile(r"(?<=[.!?])\s+")
FOLLOW_UP_CSV_FIELDS = [
    "id",
    "follow_up_required",
    "customer_name",
    "company_name",
    "action_needed",
    "priority",
    "reason",
    "source_text",
    "created_date",
    "status",
    "source_name",
    "source_type",
]


@dataclass(slots=True)
class FollowUpAlert:
    follow_up_required: bool
    customer_name: str
    company_name: str
    action_needed: str
    priority: str
    reason: str
    source_text: str
    status: str = "Pending"
    id: str | None = None
    source_name: str = ""
    source_type: str = ""
    created_date: str | None = None

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "follow_up_required": self.follow_up_required,
            "customer_name": self.customer_name,
            "company_name": self.company_name,
            "action_needed": self.action_needed,
            "priority": self.priority,
            "reason": self.reason,
            "source_text": self.source_text,
            "status": self.status,
            "created_date": self.created_date,
            "source_name": self.source_name,
            "source_type": self.source_type,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def init_follow_up_db(csv_path: Path = DEFAULT_CSV_PATH) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=FOLLOW_UP_CSV_FIELDS).writeheader()
        return

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames == FOLLOW_UP_CSV_FIELDS:
            return
        rows = list(reader)

    upgraded_rows = [{field: row.get(field, "") for field in FOLLOW_UP_CSV_FIELDS} for row in rows]
    _write_alert_rows(upgraded_rows, csv_path)


def _row_to_alert(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id", ""),
        "follow_up_required": str(row.get("follow_up_required", "")).lower() in {"true", "1", "yes"},
        "customer_name": row.get("customer_name", ""),
        "company_name": row.get("company_name", ""),
        "action_needed": row.get("action_needed", ""),
        "priority": row.get("priority", "Low"),
        "reason": row.get("reason", ""),
        "source_text": row.get("source_text", ""),
        "created_date": row.get("created_date", ""),
        "status": row.get("status", "Pending"),
        "source_name": row.get("source_name", ""),
        "source_type": row.get("source_type", ""),
    }


def _read_alert_rows(csv_path: Path = DEFAULT_CSV_PATH) -> list[dict[str, str]]:
    init_follow_up_db(csv_path)
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        return [
            {field: row.get(field, "") for field in FOLLOW_UP_CSV_FIELDS}
            for row in csv.DictReader(handle)
        ]


def _write_alert_rows(rows: list[dict[str, Any]], csv_path: Path = DEFAULT_CSV_PATH) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FOLLOW_UP_CSV_FIELDS)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in FOLLOW_UP_CSV_FIELDS} for row in rows])


def list_follow_up_alerts(
    *,
    priority: str | None = None,
    status: str | None = None,
    customer_name: str | None = None,
    csv_path: Path = DEFAULT_CSV_PATH,
) -> list[dict[str, Any]]:
    rows = [_row_to_alert(row) for row in _read_alert_rows(csv_path)]
    if priority and priority in VALID_PRIORITIES:
        rows = [row for row in rows if row["priority"] == priority]
    if status and status in VALID_STATUSES:
        rows = [row for row in rows if row["status"] == status]
    if customer_name:
        customer_filter = customer_name.lower()
        rows = [row for row in rows if customer_filter in row["customer_name"].lower()]

    rows.sort(key=lambda row: row["created_date"], reverse=True)
    rows.sort(key=lambda row: 0 if row["status"] == "Pending" else 1)
    return rows


def update_follow_up_status(alert_id: str, status: str, csv_path: Path = DEFAULT_CSV_PATH) -> dict[str, Any] | None:
    if status not in VALID_STATUSES:
        raise ValueError("Invalid follow-up status")

    with FOLLOW_UP_DB_LOCK:
        rows = _read_alert_rows(csv_path)
        updated_row: dict[str, str] | None = None
        for row in rows:
            if row.get("id") == alert_id:
                row["status"] = status
                updated_row = row
                break
        if updated_row:
            _write_alert_rows(rows, csv_path)
    return _row_to_alert(updated_row) if updated_row else None


def save_follow_up_alerts(
    alerts: list[FollowUpAlert],
    *,
    source_name: str,
    source_type: str,
    csv_path: Path = DEFAULT_CSV_PATH,
) -> list[dict[str, Any]]:
    if not alerts:
        return []

    created: list[FollowUpAlert] = []
    created_at = utc_now()
    with FOLLOW_UP_DB_LOCK:
        rows = _read_alert_rows(csv_path)
        for alert in alerts:
            if not alert.follow_up_required or not alert.action_needed.strip():
                continue
            alert.id = alert.id or str(uuid.uuid4())
            alert.source_name = source_name
            alert.source_type = source_type
            alert.created_date = alert.created_date or created_at
            rows.append(
                {
                    "id": alert.id,
                    "follow_up_required": "true",
                    "customer_name": alert.customer_name,
                    "company_name": alert.company_name,
                    "action_needed": alert.action_needed,
                    "priority": alert.priority,
                    "reason": alert.reason,
                    "source_text": alert.source_text,
                    "created_date": alert.created_date,
                    "status": alert.status,
                    "source_name": alert.source_name,
                    "source_type": alert.source_type,
                }
            )
            created.append(alert)
        _write_alert_rows(rows, csv_path)

    return [alert.to_api_dict() for alert in created]


def _customer_turns_text(diarization: DiarizationResult, fallback_text: str) -> str:
    turns = [turn.text.strip() for turn in diarization.turns if turn.speaker == "Customer" and turn.text.strip()]
    return "\n".join(turns) if turns else fallback_text


def _entity_value(grouped: dict[str, list[str]], *keys: str) -> str:
    for key in keys:
        values = grouped.get(key) or []
        if values:
            return str(values[0])
    return ""


def _normalize_priority(value: Any, source_text: str) -> str:
    priority = str(value or "").strip().title()
    if priority in VALID_PRIORITIES:
        return priority
    source_lower = source_text.lower()
    if any(term in source_lower for term in ("urgent", "today", "tomorrow", "as soon", "asap")):
        return "High"
    if any(term in source_lower for term in ("next week", "later", "some time", "review", "discuss internally")):
        return "Medium"
    return "Low"


def _clean_alert_item(item: dict[str, Any], defaults: dict[str, str]) -> FollowUpAlert | None:
    source_text = str(item.get("source_text") or item.get("source_statement") or "").strip()
    action_needed = str(item.get("action_needed") or item.get("action") or "").strip()
    reason = str(item.get("reason") or "").strip()
    if not action_needed or not source_text:
        return None

    return FollowUpAlert(
        follow_up_required=bool(item.get("follow_up_required", True)),
        customer_name=str(item.get("customer_name") or defaults.get("customer_name") or "").strip(),
        company_name=str(item.get("company_name") or defaults.get("company_name") or "").strip(),
        action_needed=action_needed,
        priority=_normalize_priority(item.get("priority"), source_text),
        reason=reason or "Customer indicated a future sales action or response is needed.",
        source_text=source_text,
        status="Pending",
    )


def _dedupe_alerts(alerts: list[FollowUpAlert]) -> list[FollowUpAlert]:
    seen: set[tuple[str, str]] = set()
    deduped: list[FollowUpAlert] = []
    for alert in alerts:
        key = (alert.action_needed.lower(), alert.source_text.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(alert)
    return deduped


def build_follow_up_prompt(customer_text: str, defaults: dict[str, str]) -> str:
    return f"""
Detect customer follow-up intent in this sales conversation.

Return strict JSON:
{{
  "alerts": [
    {{
      "follow_up_required": true,
      "customer_name": "",
      "company_name": "",
      "action_needed": "",
      "priority": "High|Medium|Low",
      "reason": "",
      "source_text": "",
      "status": "Pending"
    }}
  ]
}}

Rules:
- Detect explicit and implicit requests for future action, response, callback, meeting, proposal, quotation, demo, document sharing, clarification, or customer engagement.
- Do not rely on keywords alone; infer the customer's actual intent from context.
- Create one alert per distinct follow-up action.
- Use only customer statements as source_text.
- If no future action is required, return {{"alerts": []}}.
- Do not invent names. Use these known defaults only when appropriate:
  customer_name: {defaults.get("customer_name", "")}
  company_name: {defaults.get("company_name", "")}

Customer statements:
{customer_text}
"""


async def detect_follow_up_alerts(
    *,
    customer_text: str,
    diarization: DiarizationResult,
    privacy_payload: dict[str, Any],
) -> list[FollowUpAlert]:
    source_text = _customer_turns_text(diarization, customer_text)
    grouped = privacy_payload.get("grouped") if isinstance(privacy_payload.get("grouped"), dict) else {}
    defaults = {
        "customer_name": _entity_value(grouped, "customer_name", "person", "name"),
        "company_name": _entity_value(grouped, "company_name", "organization", "org"),
    }

    alerts: list[FollowUpAlert] = []
    try:
        response = await call_llama(
            messages=[
                {"role": "system", "content": "You detect sales CRM follow-up tasks and return only strict JSON."},
                {"role": "user", "content": build_follow_up_prompt(source_text, defaults)},
            ]
        )
        content = response["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        for item in parsed.get("alerts", []):
            if isinstance(item, dict):
                alert = _clean_alert_item(item, defaults)
                if alert:
                    alerts.append(alert)
    except Exception as exc:
        print("Follow-up alert LLaMA detection failed:", exc)
        alerts.extend(_fallback_follow_up_alerts(source_text, defaults))

    return _dedupe_alerts(alerts)


def _fallback_follow_up_alerts(customer_text: str, defaults: dict[str, str]) -> list[FollowUpAlert]:
    patterns = [
        (r"\b(call|callback|ring)\b.*\b(me|us)\b", "Call the customer back", "Customer requested a future call."),
        (r"\b(send|share|email|forward)\b.*\b(proposal|quotation|quote|details|document|information|info)\b", "Share requested sales information", "Customer requested materials or details."),
        (r"\b(schedule|arrange|book)\b.*\b(demo|meeting|call)\b", "Schedule the requested engagement", "Customer asked to arrange a future engagement."),
        (r"\b(review|discuss|think about|consider)\b.*\b(get back|later|internally|team|management)\b", "Follow up after customer review", "Customer needs time before responding."),
        (r"\b(know more|more information|more details|clarify|explain)\b", "Provide clarification or additional information", "Customer asked for more clarity."),
    ]
    alerts: list[FollowUpAlert] = []
    sentences = [part.strip() for part in SENTENCE_SPLIT_RX.split(customer_text) if part.strip()]
    for sentence in sentences:
        sentence_lower = sentence.lower()
        for pattern, action, reason in patterns:
            if re.search(pattern, sentence_lower):
                alerts.append(
                    FollowUpAlert(
                        follow_up_required=True,
                        customer_name=defaults.get("customer_name", ""),
                        company_name=defaults.get("company_name", ""),
                        action_needed=action,
                        priority=_normalize_priority(None, sentence),
                        reason=reason,
                        source_text=sentence,
                    )
                )
                break
    return alerts
