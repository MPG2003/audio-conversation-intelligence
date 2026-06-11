import json
from pathlib import Path
from src.aspect_sentiment.behavioral_signals import detect_signals
from src.aspect_sentiment.conversation_context_engine import analyze_context
from src.aspect_sentiment.lead_scoring_engine import calculate_intent_score, calculate_emotion_score

BASE_DIR = Path(__file__).resolve().parent
with open(BASE_DIR / "scoring_rules.json", "r") as f:
    RULES = json.load(f)

WEIGHTS = RULES.get("fusion_weights", {})

def fuse_probabilities(
    xgboost_prob: float,
    transcript: str,
    raw_features: list,
    sentiment_score: float,
    agent_transcript: str = "",
) -> dict:
    
    # 1. Gather all sub-scores
    behavioral_data = detect_signals(transcript, raw_features)
    context_data = analyze_context(transcript, raw_features)
    
    b_score = behavioral_data["normalized_score"] # -1 to 1
    # Scale b_score from [-1, 1] to [0, 1] for fusion calculation
    b_score_scaled = (b_score + 1) / 2
    
    intent_score = calculate_intent_score(raw_features) # 0 to 1
    emotion_score = calculate_emotion_score(sentiment_score, raw_features) # 0 to 1
    engagement_score = context_data["engagement_score"] # 0 to 1
    customer_word_count = len(transcript.split())
    agent_word_count = len(agent_transcript.split())
    customer_share = customer_word_count / max(customer_word_count + agent_word_count, 1)
    customer_weight = max(0.75, min(1.0, customer_share + 0.2))
    
    # 2. Hybrid Calculation
    # final = (xgb*w_xgb) + (b*w_b) + (intent*w_i) + (emotion*w_e) + (engagement*w_eng)
    final_probability = (
        xgboost_prob * WEIGHTS.get("xgboost_prediction", 0.4) * customer_weight +
        b_score_scaled * WEIGHTS.get("behavioral_score", 0.2) +
        intent_score * WEIGHTS.get("intent_score", 0.2) +
        emotion_score * WEIGHTS.get("emotion_score", 0.1) +
        engagement_score * WEIGHTS.get("engagement_score", 0.1)
    )
    
    # Safety bounds
    final_probability = max(0.0, min(1.0, final_probability))
    
    # 3. Label Generation
    if final_probability >= 0.7:
        label = "hot"
    elif final_probability >= 0.4:
        label = "warm"
    else:
        label = "cold"
        
    # 4. Explainable Reasoning
    reasons = []
    
    # Positive reasons
    if xgboost_prob > 0.6:
        reasons.append("Customer-only historical signals predict high conversion")
    if intent_score > 0.7:
        reasons.append("Strong buying intent detected")
    if engagement_score > 0.7:
        reasons.append("Deep technical/conversational engagement")
    if emotion_score > 0.7:
        reasons.append("Positive emotional leaning")
        
    # Inject behavioral specific reasons
    for pos_sig in behavioral_data["detected_positive"]:
        formatted = pos_sig.replace("_", " ").capitalize()
        reasons.append(formatted)
        
    # Negative reasons (hesitation shouldn't necessarily crash the score, just noted)
    for neg_sig in behavioral_data["detected_negative"]:
        formatted = neg_sig.replace("_", " ").capitalize()
        reasons.append(f"Noted: {formatted}")
        
    if not reasons:
        reasons.append("Standard conversational flow detected")
        
    # Dedup and limit to 5 top reasons
    reasons = list(dict.fromkeys(reasons))[:5]
    
    return {
        "prediction": 1 if final_probability >= 0.5 else 0, # compatibility with original response
        "probability": round(final_probability, 3),
        "label": label,
        "reasons": reasons,
        "debug_metrics": {
            "xgboost_base": round(xgboost_prob, 3),
            "behavioral_score_scaled": round(b_score_scaled, 3),
            "intent_score": round(intent_score, 3),
            "emotion_score": round(emotion_score, 3),
            "engagement_score": round(engagement_score, 3)
            ,"customer_weight": round(customer_weight, 3)
        }
    }
