import json
import os
import httpx
import asyncio
from pathlib import Path

from dotenv import load_dotenv
import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ================================
# CONFIG
# ================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env.local")

MODEL = os.getenv("LLAMA_MODEL", "llama-3.3-70b-versatile")
BASE_URL = os.getenv("LLAMA_API_URL", "https://api.groq.com/openai/v1")
BASE_URL = BASE_URL.removesuffix("/chat/completions").rstrip("/")
CHAT_COMPLETIONS_URL = f"{BASE_URL}/chat/completions"

sentiment_analyzer = SentimentIntensityAnalyzer()

BRAND_TERMS = [
    "samsung",
    "iphone",
    "apple",
    "oneplus",
    "vivo",
    "oppo",
    "xiaomi",
    "redmi",
    "realme",
    "dell",
    "hp",
    "lenovo",
    "acer",
    "asus",
    "lg",
    "sony",
]
PRODUCT_TERMS = ["phone", "mobile", "cell", "cell phone", "laptop", "tv", "ac", "refrigerator"]
FEATURE_TERMS = [
    "camera",
    "battery",
    "display",
    "storage",
    "ram",
    "processor",
    "performance",
    "charging",
    "video",
]
BUDGET_RX = re.compile(
    r"\b(?:budget(?:\s+is)?|under|within|around|price|cost)\s*(?:is|of|:)?\s*(?:rs\.?|inr|₹)?\s*([0-9][0-9,]*(?:\.\d+)?)\s*(k|lakh|lakhs)?\b",
    re.IGNORECASE,
)
MONEY_RX = re.compile(r"\b(?:rs\.?|inr|₹)?\s*([0-9][0-9,]*(?:\.\d+)?)\s*(k|lakh|lakhs)?\b", re.IGNORECASE)
ENUM_LABELS = {
    "INTENT",
    "URGENCY",
    "DECISION_STAGE",
    "URGENCY_LEVEL",
    "PRICE_SENSITIVITY",
    "BRAND_LOYALTY",
    "FOLLOW_UP_PROBABILITY",
    "EMOTIONAL_CONFIDENCE",
}
ENUM_EVIDENCE = {
    "INTENT": ("buy", "purchase", "want", "need", "interested", "looking for"),
    "URGENCY": ("today", "tomorrow", "week", "month", "urgent", "immediately", "later"),
    "URGENCY_LEVEL": ("today", "tomorrow", "week", "month", "urgent", "immediately", "later"),
    "DECISION_STAGE": ("consider", "compare", "thinking", "decide", "ready", "later", "follow up"),
    "PRICE_SENSITIVITY": ("price", "budget", "cost", "expensive", "cheap", "discount"),
    "BRAND_LOYALTY": ("brand", "prefer", "always use", "loyal"),
    "FOLLOW_UP_PROBABILITY": ("follow up", "call back", "get back", "later", "contact"),
    "EMOTIONAL_CONFIDENCE": ("sure", "definitely", "maybe", "not sure", "confident"),
}


def _normalize_money(amount: str, suffix: str | None = None) -> str:
    value = float(amount.replace(",", ""))
    suffix = (suffix or "").lower()
    if suffix == "k":
        value *= 1000
    elif suffix in {"lakh", "lakhs"}:
        value *= 100000
    return str(int(value)) if value.is_integer() else str(value)


def _append_unique(features, value, label):
    clean_value = str(value).strip()
    if not clean_value:
        return
    key = (clean_value.lower(), label)
    existing = {
        (str(f.get("value", f.get("name", ""))).strip().lower(), str(f.get("label", "")))
        for f in features
    }
    if key not in existing:
        features.append({"value": clean_value, "name": clean_value, "label": label})


def rule_based_features(text):
    """Local safety net for sales facts LLaMA often misses in noisy transcripts."""
    text_lower = text.lower()
    features = []

    for term in PRODUCT_TERMS:
        if re.search(rf"\b{re.escape(term)}s?\b", text_lower):
            canonical = "phone" if term in {"mobile", "cell", "cell phone"} else term
            _append_unique(features, canonical, "PRODUCT")

    for term in BRAND_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", text_lower):
            _append_unique(features, "iPhone" if term == "iphone" else term.title(), "BRAND")

    for term in FEATURE_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", text_lower):
            _append_unique(features, term, "FEATURE")

    for match in BUDGET_RX.finditer(text):
        _append_unique(features, _normalize_money(match.group(1), match.group(2)), "BUDGET")

    if any(term in text_lower for term in ["looking for", "i want", "i need", "interested"]):
        _append_unique(features, "Interested", "INTENT")
    if any(term in text_lower for term in ["consider", "thinking", "not sure", "maybe"]):
        _append_unique(features, "Considering", "DECISION_STAGE")
    if any(term in text_lower for term in ["teacher", "student", "business", "job", "profession"]):
        _append_unique(features, "Work/personal use", "USE_CASE")
    if any(term in text_lower for term in ["expensive", "costly", "discount", "cheap", "low budget", "price sensitive"]):
        _append_unique(features, "High", "PRICE_SENSITIVITY")
    elif any(term in text_lower for term in ["premium", "best model", "top model"]):
        _append_unique(features, "Low", "PRICE_SENSITIVITY")

    return features


async def call_llama(messages):
    api_key = os.getenv("LLAMA_API_KEY")
    if not api_key:
        raise ValueError(
            f"Set LLAMA_API_KEY in your environment or in {PROJECT_ROOT / '.env.local'}"
        )

    payload = {
        "model": MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": messages
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "aspect-sentiment-client/1.0",
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(CHAT_COMPLETIONS_URL, json=payload, headers=headers, timeout=90.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            raise RuntimeError(f"Groq API HTTP {exc.response.status_code}: {detail}") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Groq API connection error: {exc}") from exc



# ================================
# PROMPT
# ================================

SYSTEM_PROMPT = "You are an expert AI that extracts structured sales data."
SUMMARY_SYSTEM_PROMPT = "You are an expert sales call summarizer. Return concise STRICT JSON only."

def build_prompt(text):
    return f"""
Extract structured sales features.

Return JSON:
{{
  "features": [
    {{
      "value": "Exact semantic value (e.g., 'RTX 4050', '85000', 'Considering')",
      "label": "PRODUCT|BRAND|BUDGET|FEATURE|INTENT|URGENCY|DECISION_STAGE|USE_CASE|OBJECTION|URGENCY_LEVEL|OBJECTION_TYPE|PRICE_SENSITIVITY|BRAND_LOYALTY|FOLLOW_UP_PROBABILITY|EMOTIONAL_CONFIDENCE"
    }}
  ]
}}

CRITICAL RULES:
- Extract literal, semantic values, NOT schema names. (e.g., Extract "RTX 4050", NOT "graphics card". Extract "85000", NOT "Budget").
- For BUDGET, extract ONLY the actual numerical amount (e.g., "85000").
- For DECISION_STAGE, strictly use one of: Awareness, Exploring, Evaluating, Considering, Comparing Alternatives, Budget Discussion, Negotiation, Near Purchase, Purchase Delayed, Follow-up Required, Ready to Purchase, Converted, Dropped.
- For INTENT, strictly use one of: Low Interest, Curious, Warm Lead, Interested, High Interest, Strong Buying Intent, Comparison Shopper, Price Sensitive, Hesitant Buyer, Ready to Purchase.
- For OBJECTION, extract explicit reasons for hesitation (e.g., "too expensive", "no EMI").
- For other customer signals like URGENCY_LEVEL, OBJECTION_TYPE, PRICE_SENSITIVITY, BRAND_LOYALTY, FOLLOW_UP_PROBABILITY, EMOTIONAL_CONFIDENCE, provide concise descriptive values.
- If not relevant → return empty list.

Input:
{text}

Output:
"""


def build_summary_prompt(transcript: str, customer_text: str = "", agent_text: str = ""):
    return f"""
Summarize this sales call or customer conversation for a CRM dashboard.

Return JSON:
{{
  "overview": "2 sentence plain-English summary of what happened",
  "customerNeed": "main customer requirement or interest",
  "keyPoints": ["3 to 5 important facts, preferences, objections, offers, or decisions"],
  "outcome": "current call outcome or decision status",
  "nextAction": "best next step for the sales agent",
  "confidence": 0.0
}}

Rules:
- Do not invent facts.
- Keep each field concise.
- Use customer-safe language and avoid exposing private personal details.
- confidence must be a number between 0 and 1.
- If there is not enough context, say so clearly in the values.

Full transcript:
{transcript}

Customer-only transcript:
{customer_text}

Agent-only transcript:
{agent_text}

Output:
"""

# ================================
# PRE-CHECK (Layer 1: Quick Intent Detection)
# ================================

def is_relevant(text):
    text = text.lower().strip()
    
    # Layer 3 filters: Catch empty, garbage, or pure greetings early
    words = text.split()
    if len(words) < 3:
        return False  # Too short
        
    alnum_ratio = sum(c.isalnum() for c in text) / max(len(text), 1)
    if alnum_ratio < 0.5:
        return False  # Garbage input
        
    pure_greetings = ["hi", "hello", "hey", "how are you", "good morning", "good afternoon", "testing"]
    if any(text == g for g in pure_greetings):
        return False  # Just a greeting

    # Layer 1: Force LLaMA for meaningful conversations using Weighted Scoring
    score = 0
    
    # 1. Product conversations
    products = ["laptop", "tv", "phone", "ac", "refrigerator", "samsung", "apple", "lg", "sony", "device", "machine", "software", "service"]
    if any(p in text for p in products): score += 3
    
    # 2. Budget discussions & Numbers
    budget = ["price", "cost", "budget", "how much", "expensive", "cheap", "offer", "discount", "deal"]
    if any(b in text for b in budget): score += 2
    if any(char.isdigit() for char in text): score += 1
    
    # 3. Buying intent
    buying_intent = ["buy", "purchase", "want", "need", "looking", "interested", "recommend", "suggest", "options"]
    if any(i in text for i in buying_intent): score += 3
    
    # 4. Comparisons
    comparisons = ["better", "compare", "difference", "vs", "versus", "which one"]
    if any(c in text for c in comparisons): score += 2
    
    # 5. EMI / Payment
    payments = ["emi", "installment", "finance", "loan", "card", "cash", "payment"]
    if any(p in text for p in payments): score += 2
    
    # 6. Hesitation
    hesitation = ["not sure", "thinking", "maybe", "later", "wait", "consider"]
    if any(h in text for h in hesitation): score += 1
    
    # 7. Contextual depth
    if len(words) > 15: score += 2
    if len(words) > 30: score += 2
        
    # Threshold: If score >= 3, it's a serious conversation requiring LLaMA 3
    return score >= 3

# ================================
# POST-PROCESSING
# ================================

def fix_labels(features):
    fixed = []

    for f in features:
        raw_val = f.get("value", f.get("name", ""))
        name = str(raw_val).lower()
        label = f.get("label", "")

        if any(x in name for x in ["day", "week", "month", "tomorrow", "today"]):
            if "warranty" not in name and "subscription" not in name:
                label = "URGENCY"

        if name in ["coding", "gaming", "office work", "daily use", "programming", "editing"]:
            label = "USE_CASE"

        # LLaMA is usually accurate, so we only override to BUDGET if it clearly contains
        # currency terms or if the LLaMA label is completely wrong.
        currency_markers = ['$', '₹', 'rs', 'rupees', 'budget', 'price', 'cost', 'under']
        if any(c in name for c in currency_markers):
            if not any(x in name for x in ["rtx", "ryzen", "intel", "amd", "gb", "tb"]):
                label = "BUDGET"
            
        # Fix specific common tech components that might get mislabeled
        if any(x in name for x in ["rtx", "ryzen", "intel", "amd", "geforce", "nvidia", "ram", "gb", "tb", "ssd", "warranty"]):
            if label not in ["PRODUCT", "BRAND"]:
                label = "FEATURE"

        f["label"] = label
        f["name"] = raw_val
        f["value"] = raw_val
        fixed.append(f)

    return fixed


def _feature_is_grounded(feature, text):
    value = str(feature.get("value", feature.get("name", ""))).strip()
    label = str(feature.get("label", "")).upper()
    if not value or not label:
        return False
    text_lower = text.lower()
    value_lower = value.lower()
    if label in ENUM_LABELS:
        return any(marker in text_lower for marker in ENUM_EVIDENCE.get(label, ()))
    if value_lower in text_lower:
        return True

    if label == "BUDGET":
        normalized_value = re.sub(r"\D", "", value)
        for match in BUDGET_RX.finditer(text):
            normalized_budget = re.sub(r"\D", "", _normalize_money(match.group(1), match.group(2)))
            if normalized_value == normalized_budget:
                return True
        return False

    meaningful_words = [
        word
        for word in re.findall(r"[a-z0-9]+", value_lower)
        if len(word) >= 3 and word not in {"the", "and", "with", "for"}
    ]
    return bool(meaningful_words) and all(
        re.search(rf"\b{re.escape(word)}\b", text_lower) for word in meaningful_words
    )


def merge_rule_features(llama_features, text):
    features = [
        feature
        for feature in fix_labels(llama_features or [])
        if _feature_is_grounded(feature, text)
    ]
    for feature in rule_based_features(text):
        _append_unique(features, feature.get("value", ""), feature.get("label", "FEATURE"))
    return features

# ================================
# SENTIMENT
# ================================

def get_sentiment(text):
    return sentiment_analyzer.polarity_scores(text)["compound"]

# ================================
# FEATURE ENGINEERING
# ================================

def derive_features(text, features):
    text_lower = text.lower()

    # ======================
    # CONFIDENCE (IMPROVED)
    # ======================
    if any(w in text_lower for w in ["definitely", "sure", "will buy", "confirm"]):
        confidence = 0.9
    elif any(w in text_lower for w in ["probably"]):
        confidence = 0.7
    elif any(w in text_lower for w in ["maybe", "not sure", "thinking"]):
        confidence = 0.3
    else:
        confidence = 0.5

    # ======================
    # HESITATION (STRONG FIX)
    # ======================
    hesitation_words = ["maybe", "thinking", "not sure", "later", "wait", "consider"]
    hesitation = sum(1 for w in hesitation_words if w in text_lower)

    # 🔥 EXTRA: Delay signal (VERY IMPORTANT)
    delay_flag = 0
    if any(w in text_lower for w in ["later", "get back", "think about", "not now"]):
        delay_flag = 1
        hesitation += 2   # increase hesitation strongly

    # ======================
    # COUNTS
    # ======================
    brands = [f for f in features if f["label"] == "BRAND"]
    feats = [f for f in features if f["label"] == "FEATURE"]

    # ======================
    # INTERACTION LENGTH
    # ======================
    length = len(text.split())
    if length < 50:
        interaction = 1
    elif length < 120:
        interaction = 2
    else:
        interaction = 3

    # ======================
  
    # ======================
    # Penalize confidence if hesitation high
    if hesitation >= 2:
        confidence = max(0.2, confidence - 0.3)

    return {
        "confidence_score": confidence,
        "hesitation_score": hesitation,
        "delay_flag": delay_flag,   # 🔥 NEW FEATURE
        "brand_count": len(brands),
        "feature_count": len(feats),
        "interaction_length": interaction
    }

# ================================
# MAIN PIPELINE
# ================================

async def process_text(text):

    if not is_relevant(text):
        print("⚠️ Out-of-context input")
        return None

    try:
        response = await call_llama(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(text)}
            ]
        )

        content = response["choices"][0]["message"]["content"]
        data = json.loads(content)

        features = merge_rule_features(data.get("features", []), text)

        sentiment = get_sentiment(text)
        derived = derive_features(text, features)

        return {
            "raw_features": features,
            "sentiment_score": sentiment,
            **derived
        }

    except Exception as e:
        print("API Error:", e)
        return None


def _fallback_summary(transcript: str, customer_text: str = "", agent_text: str = ""):
    source = " ".join((customer_text or transcript).split())
    lower = source.lower()

    need = "Customer requirement is not clear from the conversation."
    for product in PRODUCT_TERMS:
        if re.search(rf"\b{re.escape(product)}s?\b", lower):
            need = f"Customer is interested in a {product}."
            break

    budget = None
    for match in BUDGET_RX.finditer(source):
        budget = _normalize_money(match.group(1), match.group(2))
        break

    key_points = []
    if need:
        key_points.append(need)
    if budget:
        key_points.append(f"Budget mentioned: {budget}.")
    if any(term in lower for term in ["not sure", "maybe", "thinking", "later", "get back"]):
        key_points.append("Customer showed hesitation or delayed the decision.")
    if any(term in (agent_text or transcript).lower() for term in ["offer", "discount", "emi"]):
        key_points.append("Agent discussed an offer, discount, or EMI option.")

    if not key_points:
        key_points.append("Conversation has limited sales detail.")

    outcome = "Follow-up required" if any(term in lower for term in ["later", "get back", "think about"]) else "Conversation analyzed"
    next_action = "Follow up with the customer and address the main requirement or objection."

    return {
        "overview": source[:220] + ("..." if len(source) > 220 else "") if source else "No conversation text was available to summarize.",
        "customerNeed": need,
        "keyPoints": key_points[:5],
        "outcome": outcome,
        "nextAction": next_action,
        "confidence": 0.45,
        "provider": "local-fallback",
    }


async def summarize_conversation(transcript: str, customer_text: str = "", agent_text: str = ""):
    clean_transcript = " ".join(transcript.split())
    if not clean_transcript:
        return _fallback_summary(clean_transcript, customer_text, agent_text)

    try:
        response = await call_llama(
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": build_summary_prompt(clean_transcript, customer_text, agent_text)},
            ]
        )
        content = response["choices"][0]["message"]["content"]
        data = json.loads(content)

        key_points = data.get("keyPoints", [])
        if not isinstance(key_points, list):
            key_points = [str(key_points)]

        confidence = data.get("confidence", 0.7)
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.7

        return {
            "overview": str(data.get("overview") or "").strip() or "Summary was generated, but no overview was returned.",
            "customerNeed": str(data.get("customerNeed") or "").strip() or "Customer requirement is not clear from the conversation.",
            "keyPoints": [str(item).strip() for item in key_points if str(item).strip()][:5],
            "outcome": str(data.get("outcome") or "").strip() or "Outcome not clearly stated.",
            "nextAction": str(data.get("nextAction") or "").strip() or "Follow up with the customer.",
            "confidence": confidence,
            "provider": f"llama:{MODEL}",
        }
    except Exception as e:
        print("Summary API Error:", e)
        return _fallback_summary(clean_transcript, customer_text, agent_text)

# ================================
# TEST
# ================================

if __name__ == "__main__":
    text = "Start. Hi, I want a laptop under 60,000. Sure. What will you mainly use it for? Programming and basic gaming. Do you have any brand performance like Dell HP or Lenovo? Not really. Just something reliable. Oh, okay. I can suggest a Dell Inspired on and a Lenovo idea part in that range. Both are good for programming. Okay, but I'm not sure if I should buy now. We currently have an offer and the EMA options available. I think about it and get back to you later. Okay, sure. I'll share the details with you. Thank you"

    result = asyncio.run(process_text(text))

    print("\nFinal Output:")
    print(json.dumps(result, indent=2))
