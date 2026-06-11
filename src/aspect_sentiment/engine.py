from __future__ import annotations

import json
import logging
import math
import os
import re
import httpx
import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from statistics import fmean
from typing import Iterable

import joblib
import pandas as pd
import spacy
from spacy.tokens import Doc, Span
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .schemas import (
    AnalysisMetadata,
    AnalysisResponse,
    ConversionScore,
    HighlightRange,
    PipelineStage,
    ProductSentiment,
    SentimentCounts,
    SentimentSummary,
)

logger = logging.getLogger(__name__)

SPACE_RX = re.compile(r"\s+")
EDGE_PUNCT_RX = re.compile(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$")
ALPHA_RX = re.compile(r"[A-Za-z]")
DIGIT_RX = re.compile(r"\d")
BUDGET_RX = re.compile(r"(?i)^(?:rs\.?|inr|₹|\$|€|£)?\s*\d[\d,]*(?:\.\d+)?\s*(?:[kKmMlL]|lakh|lakhs|cr|crore|crores|bucks|ks)?$")
SENTENCE_SPLIT_RX = re.compile(r"(?<=[.!?])\s+")

GENERIC_ASPECT_TERMS = {
    "app",
    "apps",
    "brand",
    "brands",
    "call",
    "conversation",
    "experience",
    "feature",
    "features",
    "issue",
    "issues",
    "item",
    "items",
    "people",
    "person",
    "product",
    "products",
    "service",
    "services",
    "something",
    "stuff",
    "team",
    "thing",
    "things",
    "today",
}
CLAUSE_BREAKERS = {"but", "however", "though", "although", "yet", "while"}
VALID_ENTITY_LABELS = {"PRODUCT", "BRAND", "BUDGET", "FEATURE", "ISSUE", "INTENT", "URGENCY", "DECISION_STAGE"}
GENERIC_FEATURES = {"good", "best", "nice", "great", "unknown"}
INVALID_ENTITY_TERMS = {
    "bonus",
    "cashback",
    "consideration",
    "deal",
    "deals",
    "decision",
    "discount",
    "discounts",
    "daily use",
    "emi",
    "emi option",
    "emi options",
    "family",
    "festive",
    "festive offer",
    "medium",
    "no",
    "offer",
    "offers",
    "photos",
    "strong",
    "weak",
    "yes",
    "maybe",
    "none",
    "na",
}
BRANDS = {
    "apple",
    "asus",
    "blue star",
    "bosch",
    "daikin",
    "dell",
    "godrej",
    "haier",
    "hp",
    "ifb",
    "lenovo",
    "lg",
    "mi",
    "oneplus",
    "oppo",
    "panasonic",
    "samsung",
    "sony",
    "vivo",
    "whirlpool",
    "xiaomi",
}
PRODUCTS = {
    "ac",
    "air conditioner",
    "camera",
    "dishwasher",
    "earbuds",
    "headphones",
    "laptop",
    "microwave",
    "mobile",
    "phone",
    "refrigerator",
    "smartphone",
    "smartwatch",
    "tablet",
    "tv",
    "vacuum cleaner",
    "washing machine",
    "desktop",
}
PREFERENCES = {"budget", "durability", "energy efficiency", "performance", "reliability"}
LEAD_CATEGORICAL_COLS = [
    "product",
    "preference",
    "intent_strength",
    "decision_stage",
    "sentiment",
    "hesitation",
    "follow_up_needed",
    "offer_given",
    "emi_option",
    "product_suggested",
]
LEAD_NUMERIC_COLS = ["budget", "brand_count", "use_case_count", "word_count", "sentence_count"]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEAD_MODEL_PATH = REPO_ROOT / "data" / "processed" / "lead_scoring_model.joblib"

FEATURE_READER_SYSTEM_PROMPT = "You are an expert reader for an AI sales CRM. Extract structured sales data. Return STRICT JSON."


@dataclass(slots=True)
class AspectMention:
    key: str
    name: str
    context: str
    start_char: int
    end_char: int
    text: str
    label: str = "ASPECT"


@dataclass(slots=True)
class ExtractionResult:
    mentions: list[AspectMention]
    provider: str


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@lru_cache(maxsize=1)
def load_nlp():
    for model_name in ("en_core_web_sm", "en_core_web_md"):
        try:
            logger.info("Loading spaCy model '%s'", model_name)
            return spacy.load(model_name)
        except OSError:
            continue

    logger.warning("No pretrained spaCy English model found. Falling back to a blank English pipeline.")
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


@lru_cache(maxsize=1)
def load_vader() -> SentimentIntensityAnalyzer:
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(
        {
            "crisp": 2.1,
            "drain": -2.4,
            "drains": -2.6,
            "lag": -2.2,
            "laggy": -2.6,
            "overheat": -2.7,
            "overheats": -2.8,
            "premium": 1.8,
            "responsive": 2.1,
            "sharp": 2.0,
            "sluggish": -2.4,
            "smooth": 2.2,
            "stable": 1.7,
        }
    )
    return analyzer


@lru_cache(maxsize=1)
def load_lead_model() -> dict | None:
    model_path = Path(os.getenv("LEAD_SCORING_MODEL_PATH", str(DEFAULT_LEAD_MODEL_PATH)))
    if not model_path.exists():
        logger.warning("Lead scoring model not found at %s", model_path)
        return None
    return joblib.load(model_path)


class AspectSentimentEngine:
    def __init__(self) -> None:
        self.nlp = load_nlp()
        self.analyzer = load_vader()
        self.spacy_model_name = getattr(self.nlp, "meta", {}).get("name") or "blank-en"
        self.llama_api_key = os.getenv("LLAMA_API_KEY") or os.getenv("GROQ_API_KEY")
        raw_url = os.getenv("LLAMA_API_URL", "https://api.groq.com/openai/v1/chat/completions")
        self.llama_api_url = raw_url if raw_url.endswith("/chat/completions") else f"{raw_url.rstrip('/')}/chat/completions"
        self.llama_model = os.getenv("LLAMA_MODEL", "llama-3.3-70b-versatile")
        self.lead_model = load_lead_model()

    @staticmethod
    def normalize_text(text: str) -> str:
        return SPACE_RX.sub(" ", text).strip()

    def parse(self, text: str) -> Doc:
        return self.nlp(self.normalize_text(text))

    @staticmethod
    def _strip_edges(text: str) -> str:
        return EDGE_PUNCT_RX.sub("", text).strip()

    @staticmethod
    def _valid_phrase(text: str) -> bool:
        return bool(text and ALPHA_RX.search(text))

    @staticmethod
    def _valid_entity_text(text: str) -> bool:
        return bool(text and (ALPHA_RX.search(text) or DIGIT_RX.search(text)))

    @staticmethod
    def _compact_entity_name(text: str) -> str:
        return SPACE_RX.sub(" ", text).strip().lower()

    @staticmethod
    def _is_budget_value(text: str) -> bool:
        normalized = text.strip().lower().replace(" ", "")
        return bool(BUDGET_RX.match(normalized))

    def _is_relevant(self, text: str) -> bool:
        text_lower = text.lower()
        keywords = {"buy", "price", "budget", "product", "purchase", "model", "cost", "review", "issue", "battery", "display"}
        if any(word in text_lower for word in keywords):
            return True
        if any(p in text_lower for p in PRODUCTS):
            return True
        if any(b in text_lower for b in BRANDS):
            return True
        return False

    def _classify_known_entity(self, name: str) -> str | None:
        normalized = self._compact_entity_name(name)
        if normalized in BRANDS:
            return "BRAND"
        if normalized in PRODUCTS:
            return "PRODUCT"
        if self._is_budget_value(normalized):
            return "BUDGET"
        return None

    def _is_valid_extracted_entity(self, name: str, label: str) -> bool:
        normalized = self._compact_entity_name(name)
        if not self._valid_entity_text(normalized):
            return False
        if len(normalized) < 2:
            return False
        if label == "FEATURE" and normalized in GENERIC_FEATURES:
            return False
        if normalized in GENERIC_ASPECT_TERMS or normalized in INVALID_ENTITY_TERMS:
            return False
        if any(term in normalized.split() for term in INVALID_ENTITY_TERMS):
            return False
        if label not in VALID_ENTITY_LABELS:
            return False
        if label == "BUDGET":
            return self._is_budget_value(normalized)
        if label == "BRAND" and not any(b in normalized for b in BRANDS):
            return False
        if label == "PRODUCT" and not any(p in normalized for p in PRODUCTS):
            return False
        return len(normalized.split()) <= 3

    @staticmethod
    def _binary_signal(value: str) -> int:
        return 1 if value.lower().strip() in {"yes", "true", "1"} else 0

    @staticmethod
    def _budget_number(value: str) -> float:
        normalized = value.lower().replace(",", "").replace("rs", "").replace("inr", "").replace("₹", "").strip()
        multiplier = 1.0
        if normalized.endswith("k"):
            multiplier = 1000.0
            normalized = normalized[:-1]
        elif normalized.endswith("lakh"):
            multiplier = 100000.0
            normalized = normalized[:-4]
        elif normalized.endswith("lakhs"):
            multiplier = 100000.0
            normalized = normalized[:-5]
        try:
            return float(normalized.strip()) * multiplier
        except ValueError:
            return 0.0

    @staticmethod
    def _normalize_model_category(value: str) -> str:
        return value.lower().strip().replace(" ", "_")

    def _lead_features_from_mentions(
        self,
        mentions: list[AspectMention],
        products: list[ProductSentiment],
        word_count: int,
        sentence_count: int,
    ) -> dict[str, int | float | str]:
        by_label: dict[str, list[str]] = defaultdict(list)
        for mention in mentions:
            by_label[mention.label].append(self._compact_entity_name(mention.name))

        product = by_label.get("PRODUCT", ["unknown"])[0]
        if product not in PRODUCTS:
            product = "unknown"

        preference = by_label.get("PREFERENCE", ["unknown"])[0]
        if preference not in PREFERENCES:
            preference = "unknown"

        budget = 0.0
        if by_label.get("BUDGET"):
            budget = self._budget_number(by_label["BUDGET"][0])

        sentiment = "neutral"
        if products:
            avg_score = fmean(product_item.score for product_item in products)
            sentiment = self._label_for_score(avg_score)

        return {
            "id": 0,
            "product": product,
            "budget": budget,
            "brand_count": len(set(by_label.get("BRAND", []))),
            "use_case_count": len(set(by_label.get("USE_CASE", []))),
            "preference": preference,
            "intent_strength": by_label.get("INTENT", ["medium"])[0],
            "decision_stage": by_label.get("DECISION_STAGE", ["consideration"])[0],
            "sentiment": sentiment,
            "hesitation": self._binary_signal(by_label.get("HESITATION", ["no"])[0]),
            "follow_up_needed": self._binary_signal(by_label.get("FOLLOW_UP", ["no"])[0]),
            "offer_given": self._binary_signal(by_label.get("OFFER", ["no"])[0]),
            "emi_option": self._binary_signal(by_label.get("EMI", ["no"])[0]),
            "product_suggested": self._binary_signal(by_label.get("PRODUCT_SUGGESTED", ["no"])[0]),
            "word_count": word_count,
            "sentence_count": sentence_count,
        }

    def predict_conversion(
        self,
        mentions: list[AspectMention],
        products: list[ProductSentiment],
        word_count: int,
        sentence_count: int,
    ) -> ConversionScore | None:
        if not self.lead_model:
            return None

        features = self._lead_features_from_mentions(mentions, products, word_count, sentence_count)
        payload = self.lead_model
        feature_columns = payload["feature_columns"]
        model = payload["model"]
        scaler = payload.get("scaler")

        frame = pd.DataFrame([features])
        frame["product"] = frame["product"].map(self._normalize_model_category)
        frame["preference"] = frame["preference"].map(self._normalize_model_category)

        encoded = pd.get_dummies(frame, columns=[col for col in LEAD_CATEGORICAL_COLS if col in frame], drop_first=False)
        for column in feature_columns:
            if column not in encoded:
                encoded[column] = 0
        encoded = encoded[feature_columns]

        if scaler:
            numeric_columns = [column for column in LEAD_NUMERIC_COLS if column in encoded]
            encoded[numeric_columns] = scaler.transform(encoded[numeric_columns])

        probability = float(model.predict_proba(encoded)[0][1])
        label = "hot" if probability >= 0.7 else "warm" if probability >= 0.4 else "cold"
        confidence = round(abs(probability - 0.5) * 2, 2)
        return ConversionScore(
            probability=round(probability, 3),
            label=label,
            confidence=confidence,
            features=features,
            model=str(Path(os.getenv("LEAD_SCORING_MODEL_PATH", str(DEFAULT_LEAD_MODEL_PATH))).name),
        )

    def _aspect_name_from_span(self, span: Span) -> tuple[str, str] | None:
        if not span.text.strip():
            return None

        display_parts: list[str] = []
        key_parts: list[str] = []

        for token in span:
            if token.is_space or token.is_punct:
                continue
            if token.pos_ in {"DET", "PRON"}:
                continue
            if token.is_stop and token.pos_ not in {"NOUN", "PROPN"}:
                continue
            if token.pos_ in {"NOUN", "PROPN"} or token.dep_ == "compound" or not token.pos_:
                display_parts.append(token.text)
                lemma = token.lemma_.lower() if token.lemma_ not in {"", "-PRON-"} else token.text.lower()
                key_parts.append(lemma)

        if not display_parts:
            fallback = [
                token.text
                for token in span
                if token.is_alpha and not token.is_stop and token.pos_ not in {"DET", "PRON"}
            ]
            if fallback:
                display_parts = fallback[-2:]
                key_parts = [part.lower() for part in display_parts]

        name = self._strip_edges(" ".join(display_parts)).lower()
        key = self._strip_edges(" ".join(key_parts)).lower()
        if not self._valid_phrase(name):
            return None
        if key in GENERIC_ASPECT_TERMS or name in GENERIC_ASPECT_TERMS:
            return None
        if len(name.split()) > 4:
            return None

        return key, name

    @staticmethod
    def _trim_span(doc: Doc, start: int, end: int) -> Span:
        while start < end and doc[start].is_space:
            start += 1
        while end > start and doc[end - 1].is_space:
            end -= 1
        return doc[start:end]

    def _context_window(self, span: Span) -> str:
        sent = span.sent
        start = sent.start
        end = sent.end

        for index in range(span.start - 1, sent.start - 1, -1):
            token = span.doc[index]
            if token.text in {",", ";", ":"} or token.lower_ in CLAUSE_BREAKERS:
                start = index + 1
                break
            if token.lower_ in {"and", "or"}:
                lookback_start = max(sent.start, index - 4)
                if any(span.doc[left].pos_ in {"VERB", "AUX", "ADJ", "ADV"} for left in range(lookback_start, index)):
                    start = index + 1
                    break

        saw_predicate = False
        for index in range(span.end, sent.end):
            token = span.doc[index]
            if token.text in {",", ";", ":"} or token.lower_ in CLAUSE_BREAKERS:
                end = index
                break
            if token.pos_ in {"VERB", "AUX", "ADJ", "ADV"}:
                saw_predicate = True
            if token.lower_ in {"and", "or"} and saw_predicate:
                next_index = index + 1
                while next_index < sent.end and span.doc[next_index].is_space:
                    next_index += 1
                if next_index < sent.end:
                    next_token = span.doc[next_index]
                    if next_token.pos_ in {"DET", "NOUN", "PROPN", "PRON"} or next_token.lower_ in {"the", "this", "that"}:
                        end = index
                        break

        candidate = self._trim_span(span.doc, start, end)
        context = self.normalize_text(candidate.text)

        if len(context.split()) < 3:
            local_start = max(sent.start, span.start - 6)
            local_end = min(sent.end, span.end + 6)
            context = self.normalize_text(self._trim_span(span.doc, local_start, local_end).text)

        return context or self.normalize_text(sent.text)

    @staticmethod
    def _dedupe_mentions(mentions: Iterable[AspectMention]) -> list[AspectMention]:
        seen: set[tuple[str, int, int]] = set()
        deduped: list[AspectMention] = []
        for mention in mentions:
            signature = (mention.key, mention.start_char, mention.end_char)
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(mention)
        return deduped

    def _build_feature_prompt(self, text: str) -> str:
        return (
            f"{FEATURE_READER_SYSTEM_PROMPT}\n"
            'Format: {"features":[{"name":"value","label":"PRODUCT|BRAND|BUDGET|FEATURE|ISSUE|INTENT|URGENCY|DECISION_STAGE","context":"short evidence"}]}\n\n'
            "Rules:\n"
            "- Only extract relevant product/sales info.\n"
            "- If a field is missing, skip it.\n"
            "- Do NOT guess or hallucinate.\n"
            "- Do NOT output generic words: yes, no, maybe, medium, consideration, decision, offer, discount, emi, none.\n"
            "- If not product-related -> return empty JSON: {\"features\": []}\n"
            "- Keep names short, ideally 1-3 words.\n\n"
            "Definitions:\n"
            "- INTENT: buying / exploring\n"
            "- URGENCY: immediate / later\n"
            "- DECISION_STAGE: early / mid / final\n\n"
            "Example:\n"
            'Input: "I want a Samsung TV under 50000, planning to buy this week"\n'
            "Output:\n"
            '{"features": ['
            '{"name":"tv", "label":"PRODUCT", "context":"Samsung TV"}, '
            '{"name":"samsung", "label":"BRAND", "context":"Samsung TV"}, '
            '{"name":"50000", "label":"BUDGET", "context":"under 50000"}, '
            '{"name":"buying", "label":"INTENT", "context":"planning to buy"}, '
            '{"name":"immediate", "label":"URGENCY", "context":"this week"}, '
            '{"name":"final", "label":"DECISION_STAGE", "context":"planning to buy"}'
            ']}\n\n'
            f"Input:\n{text}\n"
            "Output:\n"
        )

    @staticmethod
    def _sentence_candidates(text: str) -> list[str]:
        parts = [part.strip() for part in SENTENCE_SPLIT_RX.split(text) if part.strip()]
        return parts or [text.strip()]

    def _find_context_in_text(self, text: str, context: str, feature_name: str) -> str:
        normalized_text = self.normalize_text(text)
        normalized_context = self.normalize_text(context)
        if normalized_context and normalized_context.lower() in normalized_text.lower():
            return normalized_context

        candidates = self._sentence_candidates(normalized_text)
        feature_lower = feature_name.lower()
        for candidate in candidates:
            if feature_lower in candidate.lower():
                return candidate
        return normalized_context or normalized_text

    def _locate_span(self, text: str, feature_name: str, context: str) -> tuple[int, int, str]:
        normalized_text = self.normalize_text(text)
        feature_lower = feature_name.lower().strip()
        if not feature_lower:
            return 0, 0, normalized_text

        start = normalized_text.lower().find(feature_lower)
        if start != -1:
            return start, start + len(feature_name), self._find_context_in_text(normalized_text, context, feature_name)

        words = feature_lower.split()
        if words:
            fallback = words[-1]
            start = normalized_text.lower().find(fallback)
            if start != -1:
                return start, start + len(fallback), self._find_context_in_text(normalized_text, context, feature_name)

        return 0, len(feature_name), self._find_context_in_text(normalized_text, context, feature_name)

    def _extract_mentions_from_feature_items(self, text: str, items: list[dict[str, object]]) -> list[AspectMention]:
        mentions: list[AspectMention] = []
        for item in items:
            name = self._compact_entity_name(str(item.get("name", "")))
            label = self.normalize_text(str(item.get("label", "FEATURE"))).upper()
            if label not in VALID_ENTITY_LABELS:
                continue

            if self._is_budget_value(name) and label != "BUDGET":
                label = "BUDGET"

            known_label = self._classify_known_entity(name)
            if known_label:
                label = known_label

            if not self._is_valid_extracted_entity(name, label):
                continue

            key = f"{label}:{self._strip_edges(name).lower()}"
            start_char, end_char, resolved_context = self._locate_span(text, name, str(item.get("context", "")))
            mentions.append(
                AspectMention(
                    key=key,
                    name=name,
                    context=resolved_context,
                    start_char=start_char,
                    end_char=end_char,
                    text=text[start_char:end_char] if end_char > start_char else name,
                    label=label,
                )
            )

        return self._dedupe_mentions(mentions)

    async def _extract_mentions_with_llama_api(self, text: str) -> list[AspectMention]:
        if not self.llama_api_key:
            raise RuntimeError("Set LLAMA_API_KEY or GROQ_API_KEY before starting the backend.")

        if not self._is_relevant(text):
            return []

        payload = {
            "model": self.llama_model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": FEATURE_READER_SYSTEM_PROMPT},
                {"role": "user", "content": self._build_feature_prompt(text)},
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.llama_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "aspect-sentiment-client/1.0",
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.llama_api_url, json=payload, headers=headers, timeout=90.0
                )
                response.raise_for_status()
                data = response.json()
                
                message = data.get("choices", [{}])[0].get("message", {})
                model_response = str(message.get("content", "")).strip()
                if not model_response:
                    return []
                    
                parsed = json.loads(model_response)
                features = parsed.get("features", [])
                
                items = [item for item in features if isinstance(item, dict)]
                return self._extract_mentions_from_feature_items(text, items)
                
            except httpx.HTTPStatusError as exc:
                logger.error("Groq API HTTP error %s: %s", exc.response.status_code, exc.response.text)
            except httpx.RequestError as exc:
                logger.error("Groq API connection error: %s", exc)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.error("Error parsing Groq response: %s", exc)

        return []



    def extract_mentions(self, doc: Doc) -> list[AspectMention]:
        mentions: list[AspectMention] = []
        covered_tokens: set[int] = set()
        has_dependencies = doc.has_annotation("DEP")

        if has_dependencies:
            for chunk in doc.noun_chunks:
                aspect = self._aspect_name_from_span(chunk)
                if not aspect:
                    continue
                key, name = aspect
                mentions.append(
                    AspectMention(
                        key=key,
                        name=name,
                        context=self._context_window(chunk),
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                        text=chunk.text,
                    )
                )
                covered_tokens.update(range(chunk.start, chunk.end))

        for token in doc:
            if token.i in covered_tokens:
                continue
            if token.is_space or token.is_punct or token.is_stop:
                continue

            if doc.has_annotation("POS"):
                if token.pos_ not in {"NOUN", "PROPN"}:
                    continue
            elif not token.is_alpha:
                continue

            span = doc[token.i : token.i + 1]
            aspect = self._aspect_name_from_span(span)
            if not aspect:
                continue
            key, name = aspect
            mentions.append(
                AspectMention(
                    key=key,
                    name=name,
                    context=self._context_window(span),
                    start_char=token.idx,
                    end_char=token.idx + len(token.text),
                    text=token.text,
                )
            )

        return self._dedupe_mentions(mentions)

    async def extract_mentions_with_provider(self, text: str) -> ExtractionResult:
        normalized = self.normalize_text(text)
        mentions = await self._extract_mentions_with_llama_api(normalized)
        return ExtractionResult(mentions=mentions, provider=f"llama:{self.llama_model}")

    @staticmethod
    def _label_for_score(score: float) -> str:
        if score > 0.05:
            return "positive"
        if score < -0.05:
            return "negative"
        return "neutral"

    @staticmethod
    def _confidence(scores: list[float], mention_count: int) -> float:
        if not scores:
            return 0.35
        strength = fmean(abs(score) for score in scores)
        mention_bonus = min(0.16, max(0, mention_count - 1) * 0.04)
        return round(min(0.99, 0.42 + strength * 0.46 + mention_bonus), 2)

    def score_products(self, mentions: list[AspectMention]) -> list[ProductSentiment]:
        grouped: dict[str, list[AspectMention]] = defaultdict(list)
        for mention in mentions:
            grouped[mention.key].append(mention)

        products: list[ProductSentiment] = []
        for group in grouped.values():
            contexts = list(dict.fromkeys(mention.context for mention in group if mention.context))
            scores = [self.analyzer.polarity_scores(context)["compound"] for context in contexts] or [0.0]
            score = round(float(fmean(scores)), 3)
            label = self._label_for_score(score)

            products.append(
                ProductSentiment(
                    name=group[0].name,
                    entityType=group[0].label,
                    sentiment=label,
                    score=score,
                    confidence=self._confidence(scores, len(group)),
                    mentions=len(group),
                    context=contexts[0] if contexts else "",
                    contexts=contexts,
                    highlights=[
                        HighlightRange(
                            product=group[0].name,
                            text=mention.text,
                            start=mention.start_char,
                            end=mention.end_char,
                            label=mention.label,
                        )
                        for mention in group
                    ],
                )
            )

        return sorted(products, key=lambda item: (-item.mentions, -abs(item.score), item.name))

    @staticmethod
    def _percentages(counts: Counter[str], total: int) -> dict[str, int]:
        if total == 0:
            return {"positive": 0, "negative": 0, "neutral": 0}

        raw = {label: counts.get(label, 0) * 100 / total for label in ("positive", "negative", "neutral")}
        floors = {label: math.floor(value) for label, value in raw.items()}
        remaining = 100 - sum(floors.values())
        order = sorted(raw, key=lambda label: raw[label] - floors[label], reverse=True)
        for index in range(remaining):
            floors[order[index % len(order)]] += 1
        return {label: int(value) for label, value in floors.items()}

    def summarize(self, products: list[ProductSentiment]) -> SentimentSummary:
        counts = Counter(product.sentiment for product in products)
        total = len(products)
        percentages = self._percentages(counts, total)
        average_score = round(float(fmean(product.score for product in products)), 3) if products else 0.0

        top_count = max(counts.values(), default=0)
        leaders = [label for label, count in counts.items() if count == top_count and count > 0]
        dominant = leaders[0] if len(leaders) == 1 else "balanced" if leaders else "neutral"

        return SentimentSummary(
            positive=percentages["positive"],
            negative=percentages["negative"],
            neutral=percentages["neutral"],
            counts=SentimentCounts(
                positive=counts.get("positive", 0),
                negative=counts.get("negative", 0),
                neutral=counts.get("neutral", 0),
            ),
            dominant=dominant,
            averageScore=average_score,
            totalProducts=total,
        )

    @staticmethod
    def extraction_quality(mentions: list[AspectMention], products: list[ProductSentiment]) -> dict[str, int | float | str]:
        label_counts = Counter(mention.label for mention in mentions)
        return {
            "mentionCount": len(mentions),
            "uniqueEntityCount": len(products),
            "labelCount": len(label_counts),
            "avgConfidence": round(float(fmean(product.confidence for product in products)), 2) if products else 0.0,
            **{f"{label.lower()}Count": count for label, count in sorted(label_counts.items())},
        }

    async def analyze_text(
        self,
        text: str,
        *,
        source_name: str,
        source_type: str,
        language: str | None,
        transcription_confidence: float | None,
        whisper_model: str | None,
        pipeline: list[PipelineStage],
        processing_ms: int,
    ) -> AnalysisResponse:
        normalized = self.normalize_text(text)
        doc = self.parse(normalized)
        extraction_result = await self.extract_mentions_with_provider(normalized)
        mentions = extraction_result.mentions
        products = self.score_products(mentions)
        highlights = [highlight for product in products for highlight in product.highlights]
        sentence_count = sum(1 for _ in doc.sents) if normalized else 0
        word_count = sum(1 for token in doc if not token.is_space and not token.is_punct)
        conversion_score = self.predict_conversion(mentions, products, word_count, sentence_count)

        return AnalysisResponse(
            transcript=normalized,
            normalizedText=normalized,
            products=products,
            highlights=highlights,
            summary=self.summarize(products),
            conversionScore=conversion_score,
            pipeline=pipeline,
            metadata=AnalysisMetadata(
                sourceType=source_type,
                sourceName=source_name,
                language=language,
                processingMs=processing_ms,
                extractionProvider=extraction_result.provider,
                transcriptionConfidence=transcription_confidence,
                whisperModel=whisper_model,
                wordCount=word_count,
                sentenceCount=sentence_count,
                extractionQuality=self.extraction_quality(mentions, products),
                createdAt=_utc_timestamp(),
            ),
        )
