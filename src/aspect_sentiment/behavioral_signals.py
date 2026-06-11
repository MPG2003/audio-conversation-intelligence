import json
import re
from pathlib import Path

# Load weights
BASE_DIR = Path(__file__).resolve().parent
with open(BASE_DIR / "signal_weights.json", "r") as f:
    SIGNAL_WEIGHTS = json.load(f)

POSITIVE_SIGNALS = SIGNAL_WEIGHTS.get("positive_signals", {})
NEGATIVE_SIGNALS = SIGNAL_WEIGHTS.get("negative_signals", {})

def _check_keyword_signal(text_lower: str, keywords: list[str]) -> bool:
    for keyword in keywords:
        escaped = re.escape(keyword.lower())
        if re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text_lower):
            return True
    return False

def detect_signals(transcript: str, raw_features: list) -> dict:
    """
    Analyzes the conversation to detect specific behavioral signals.
    Returns the detected signals and a normalized score.
    """
    text_lower = transcript.lower()
    detected_positive = []
    detected_negative = []
    
    score = 0
    
    # Positive Signal Detection
    if _check_keyword_signal(text_lower, ["emi", "installment", "finance", "loan"]):
        detected_positive.append("asked_for_emi")
        score += POSITIVE_SIGNALS.get("asked_for_emi", 0)
        
    if _check_keyword_signal(text_lower, ["call me", "get back", "reach out", "follow up"]):
        detected_positive.append("asked_for_followup")
        score += POSITIVE_SIGNALS.get("asked_for_followup", 0)
        
    if _check_keyword_signal(text_lower, ["whatsapp", "whats app"]):
        detected_positive.append("asked_for_whatsapp")
        score += POSITIVE_SIGNALS.get("asked_for_whatsapp", 0)
        
    if _check_keyword_signal(text_lower, ["reserve", "book", "hold it", "keep it"]):
        detected_positive.append("asked_for_reservation")
        score += POSITIVE_SIGNALS.get("asked_for_reservation", 0)
        
    if _check_keyword_signal(text_lower, ["warranty", "guarantee"]):
        detected_positive.append("asked_for_warranty")
        score += POSITIVE_SIGNALS.get("asked_for_warranty", 0)
        
    if _check_keyword_signal(text_lower, ["cashback", "discount", "offer"]):
        detected_positive.append("asked_for_cashback")
        score += POSITIVE_SIGNALS.get("asked_for_cashback", 0)
        
    if _check_keyword_signal(text_lower, ["stock", "available", "in store"]):
        detected_positive.append("asked_for_stock_availability")
        score += POSITIVE_SIGNALS.get("asked_for_stock_availability", 0)
        
    if _check_keyword_signal(text_lower, ["specs", "ram", "processor", "battery", "storage"]):
        detected_positive.append("asked_for_specifications")
        score += POSITIVE_SIGNALS.get("asked_for_specifications", 0)
        
    if _check_keyword_signal(text_lower, ["difference", "better", "compare", "vs"]):
        detected_positive.append("detailed_product_comparison")
        score += POSITIVE_SIGNALS.get("detailed_product_comparison", 0)
        
    if _check_keyword_signal(text_lower, ["love", "amazing", "exactly what i need", "perfect"]):
        detected_positive.append("emotional_leaning")
        score += POSITIVE_SIGNALS.get("emotional_leaning", 0)
        
    if _check_keyword_signal(text_lower, ["pay", "card", "cash", "upi", "payment"]):
        detected_positive.append("asked_for_payment_details")
        score += POSITIVE_SIGNALS.get("asked_for_payment_details", 0)
        
    # Negative Signal Detection
    if _check_keyword_signal(text_lower, ["maybe", "not sure", "thinking", "think about it"]):
        detected_negative.append("hesitation")
        score += NEGATIVE_SIGNALS.get("hesitation", 0)
        
    if _check_keyword_signal(text_lower, ["later", "next month", "wait"]):
        detected_negative.append("delayed_decision")
        score += NEGATIVE_SIGNALS.get("delayed_decision", 0)
        
    if _check_keyword_signal(text_lower, ["ask my", "check with", "parents", "wife", "husband"]):
        detected_negative.append("wants_external_approval")
        score += NEGATIVE_SIGNALS.get("wants_external_approval", 0)
        
    if _check_keyword_signal(text_lower, ["too expensive", "costly", "high price", "budget"]):
        detected_negative.append("price_concern")
        score += NEGATIVE_SIGNALS.get("price_concern", 0)
        
    if _check_keyword_signal(text_lower, ["confusing", "too many options", "don't know which"]):
        detected_negative.append("comparison_confusion")
        score += NEGATIVE_SIGNALS.get("comparison_confusion", 0)

    # LLaMA feature incorporation
    for f in raw_features:
        label = f.get("label", "")
        name = str(f.get("value", f.get("name", ""))).lower()
        
        if label == "OBJECTION":
            if "expensive" in name or "price" in name:
                detected_negative.append("price_concern_explicit")
                score += NEGATIVE_SIGNALS.get("price_concern", 0)
            elif "think" in name or "sure" in name:
                detected_negative.append("hesitation_explicit")
                score += NEGATIVE_SIGNALS.get("hesitation", 0)
                
    # Normalize score (e.g. max 100 points -> 1.0)
    MAX_POINTS = 100
    normalized_score = max(-1.0, min(1.0, score / MAX_POINTS))
    
    return {
        "raw_score": score,
        "normalized_score": normalized_score,
        "detected_positive": list(set(detected_positive)),
        "detected_negative": list(set(detected_negative))
    }
