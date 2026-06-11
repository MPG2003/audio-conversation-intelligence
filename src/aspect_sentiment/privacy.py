from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable

import spacy

EMAIL_RX = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RX = re.compile(r"(?:(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3,5}\)?[\s.-]?)?\d{3,5}[\s.-]?\d{4,6})")
CUSTOMER_NAME_RX = re.compile(
    r"\b(?i:my name is|this is|i am|i'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})(?=\b|[.,!?])",
)
ADDRESS_RX = re.compile(
    r"\b(?:i live (?:at|in)|my address is|address is)\s+([A-Za-z0-9][A-Za-z0-9,.\- /]{2,80})",
    re.IGNORECASE,
)
CUSTOMER_PHONE_RX = re.compile(
    r"\b(?:my\s+)?(?:phone|mobile|contact|number)(?:\s+number)?\s*(?:is|:)?\s*((?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3,5}\)?[\s.-]?)?\d{3,5}[\s.-]?\d{4,6})",
    re.IGNORECASE,
)
COMPANY_RX = re.compile(r"\b(?:company is|work(?:ing)? at|from company)\s+([A-Z][A-Za-z0-9&.\- ]{2,40})", re.IGNORECASE)
JOB_RX = re.compile(r"\b(?:profession is|working as|work as|job title is)\s+(?:an?\s+)?([A-Za-z][A-Za-z /-]{2,40})", re.IGNORECASE)
GENDER_RX = re.compile(r"\b(?:i am|i'm|gender is)\s+(male|female|non[- ]?binary|man|woman)\b", re.IGNORECASE)
NON_NAME_TERMS = {
    "apple",
    "calling",
    "dell",
    "earning",
    "galaxy",
    "hp",
    "here",
    "interested",
    "iphone",
    "lenovo",
    "looking",
    "samsung",
    "sure",
    "teacher",
    "student",
    "engineer",
    "doctor",
    "developer",
    "manager",
}


@dataclass(slots=True)
class PiiEntity:
    type: str
    value: str
    start: int
    end: int
    source: str


@dataclass(slots=True)
class PrivacyResult:
    cleaned_text: str
    entities: list[PiiEntity] = field(default_factory=list)
    redaction_count: int = 0
    provider: str = "regex+spacy"


@lru_cache(maxsize=1)
def load_privacy_nlp():
    for model_name in ("en_core_web_sm", "en_core_web_md"):
        try:
            return spacy.load(model_name)
        except OSError:
            continue
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def _add_regex_entities(entities: list[PiiEntity], text: str, pattern: re.Pattern[str], entity_type: str) -> None:
    for match in pattern.finditer(text):
        value = match.group(1).strip() if match.lastindex else match.group(0).strip()
        start = match.start(1) if match.lastindex else match.start()
        end = match.end(1) if match.lastindex else match.end()
        if value:
            entities.append(PiiEntity(type=entity_type, value=value, start=start, end=end, source="regex"))


def _dedupe_entities(entities: Iterable[PiiEntity]) -> list[PiiEntity]:
    ordered = sorted(entities, key=lambda item: (item.start, -(item.end - item.start)))
    deduped: list[PiiEntity] = []
    occupied: list[tuple[int, int]] = []
    for entity in ordered:
        if any(not (entity.end <= start or entity.start >= end) for start, end in occupied):
            continue
        deduped.append(entity)
        occupied.append((entity.start, entity.end))
    return deduped


def _valid_name(value: str) -> bool:
    words = [word.lower() for word in re.findall(r"[A-Za-z]+", value)]
    return bool(words) and len(words) <= 3 and not any(word in NON_NAME_TERMS for word in words)


def extract_and_redact_pii(text: str) -> PrivacyResult:
    entities: list[PiiEntity] = []
    _add_regex_entities(entities, text, EMAIL_RX, "email")
    _add_regex_entities(entities, text, CUSTOMER_NAME_RX, "customer_name")
    _add_regex_entities(entities, text, CUSTOMER_PHONE_RX, "customer_number")
    _add_regex_entities(entities, text, PHONE_RX, "phone")
    _add_regex_entities(entities, text, COMPANY_RX, "company")
    _add_regex_entities(entities, text, JOB_RX, "job_title")
    _add_regex_entities(entities, text, GENDER_RX, "gender")
    _add_regex_entities(entities, text, ADDRESS_RX, "address_location")

    nlp = load_privacy_nlp()
    doc = nlp(text)
    for ent in doc.ents:
        context_start = max(0, ent.start_char - 30)
        preceding = text[context_start : ent.start_char]
        context = text[context_start : ent.end_char].lower()
        if ent.label_ == "PERSON" and _valid_name(ent.text) and re.search(
            r"\b(?:my name is|this is|i am|i'm)\s+$", preceding, re.IGNORECASE
        ):
            entities.append(PiiEntity("customer_name", ent.text, ent.start_char, ent.end_char, "spacy"))
        elif ent.label_ in {"GPE", "LOC", "FAC"} and any(
            marker in context for marker in ("i live at", "i live in", "my address is", "address is")
        ):
            entities.append(PiiEntity("address_location", ent.text, ent.start_char, ent.end_char, "spacy"))

    entities = [
        entity
        for entity in _dedupe_entities(entities)
        if entity.type != "customer_name" or _valid_name(entity.value)
    ]
    cleaned = text
    for entity in sorted(entities, key=lambda item: item.start, reverse=True):
        cleaned = f"{cleaned[:entity.start]}[{entity.type.upper()}_REDACTED]{cleaned[entity.end:]}"

    return PrivacyResult(cleaned_text=cleaned, entities=entities, redaction_count=len(entities))
