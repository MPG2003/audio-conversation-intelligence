def calculate_intent_score(raw_features: list) -> float:
    """
    Transforms LLaMA qualitative INTENT/DECISION tags into quantitative score.
    """
    intent_mapping = {
        "strong buying intent": 1.0,
        "ready to purchase": 1.0,
        "high interest": 0.8,
        "warm lead": 0.6,
        "interested": 0.5,
        "comparison shopper": 0.4,
        "curious": 0.3,
        "price sensitive": 0.3,
        "hesitant buyer": 0.2,
        "low interest": 0.1
    }
    
    decision_mapping = {
        "converted": 1.0,
        "ready to purchase": 1.0,
        "near purchase": 0.9,
        "negotiation": 0.8,
        "evaluating": 0.6,
        "comparing alternatives": 0.5,
        "considering": 0.5,
        "budget discussion": 0.4,
        "exploring": 0.3,
        "awareness": 0.2,
        "follow-up required": 0.4,
        "purchase delayed": 0.1,
        "dropped": 0.0
    }
    
    intent_val = 0.5 # default
    
    for f in raw_features:
        label = f.get("label", "")
        name = str(f.get("value", f.get("name", ""))).lower()
        
        if label == "INTENT":
            intent_val = max(intent_val, intent_mapping.get(name, 0.5))
        elif label == "DECISION_STAGE":
            intent_val = max(intent_val, decision_mapping.get(name, 0.5))
            
    return intent_val

def calculate_emotion_score(sentiment_score: float, raw_features: list) -> float:
    """
    Combines VADER sentiment (-1 to 1) with LLaMA emotional tags to output a 0 to 1 score.
    """
    # Normalize VADER to 0-1
    base_emotion = (sentiment_score + 1) / 2
    
    emotion_val = base_emotion
    for f in raw_features:
        if f.get("label") == "EMOTIONAL_CONFIDENCE":
            val = str(f.get("value", "")).lower()
            if "high" in val or "strong" in val or "positive" in val:
                emotion_val = min(1.0, emotion_val + 0.2)
            elif "low" in val or "weak" in val or "negative" in val:
                emotion_val = max(0.0, emotion_val - 0.2)
                
    return emotion_val
