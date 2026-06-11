from src.aspect_sentiment.semantic_mapper import mapper
from src.aspect_sentiment.normalized_features import NormalizedOutput
import re

def clean_budget(value: str) -> int:
    """Safely extract integer budget from string."""
    digits = re.sub(r'[^\d]', '', str(value))
    return int(digits) if digits else 0


def normalize_decision_stage(value: str) -> str:
    name = value.replace(" ", "_").lower()
    if name in {"converted", "ready_to_purchase", "near_purchase", "negotiation", "decision", "final"}:
        return "decision"
    if name in {"evaluating", "comparing_alternatives", "considering", "budget_discussion", "mid"}:
        return "mid"
    if name in {"purchase_delayed", "follow_up_required", "late"}:
        return "late"
    if name in {"exploring", "awareness", "curious", "early"}:
        return "early"
    return "mid"


def normalize_urgency(value: str) -> str:
    name = value.replace(" ", "_").lower()
    if any(term in name for term in ["today", "now", "immediate", "urgent", "high"]):
        return "immediate"
    if any(term in name for term in ["tomorrow", "week", "soon", "medium"]):
        return "soon"
    if any(term in name for term in ["month", "later", "low", "flexible"]):
        return "flexible"
    if any(term in name for term in ["no_rush", "browsing", "exploring"]):
        return "no_rush"
    return "flexible"


def normalize_price_sensitivity(value: str) -> str:
    name = value.lower()
    if any(term in name for term in ["high", "very", "expensive", "cheap", "discount", "low budget", "price sensitive"]):
        return "high"
    if any(term in name for term in ["low", "premium", "not important"]):
        return "low"
    return "medium"

def process_extractions(raw_features: list[dict]) -> NormalizedOutput:
    """
    Takes the raw features from LLaMA 3 and converts them into the 
    normalized XGBoost categorical structure.
    """
    output = NormalizedOutput()
    
    for f in raw_features:
        name = str(f.get("name", "")).strip().lower()
        label = str(f.get("label", "")).upper()
        
        if not name:
            continue
            
        # 1. Budget extraction
        if label == "BUDGET" or any(c in name for c in ['$', '₹', 'rs', 'rupees']):
            # Prevent GPU names like RTX 3050 from becoming a budget
            gpu = mapper.extract_gpu(name)
            if gpu:
                # It's a GPU, not a budget!
                output.use_cases.append("gaming") # Infer gaming
                continue
                
            val = clean_budget(name)
            if val > 0:
                output.budget = val
            continue
            
        # 2. Urgency
        if label == "URGENCY" or label == "URGENCY_LEVEL":
            output.urgency = normalize_urgency(name)
            continue
            
        # 2.5 Decision Stage
        if label == "DECISION_STAGE":
            output.decision_stage = normalize_decision_stage(name)
            continue

        if label in {"PRICE_SENSITIVITY", "OBJECTION_TYPE"}:
            output.price_sensitivity = normalize_price_sensitivity(name)
            continue
            
        # 3. Semantic Mapping for Product, Brand, Use Case
        mapped_tuples = mapper.map_term(name)
        
        # If mapping engine found nothing, use raw label as fallback
        if not mapped_tuples:
            if label == "PRODUCT": output.products.append(name.replace(" ", "_"))
            elif label == "BRAND": output.brands.append(name.replace(" ", "_"))
            elif label == "USE_CASE": output.use_cases.append(name.replace(" ", "_"))
            
        for val, mapped_label in mapped_tuples:
            if mapped_label == "PRODUCT" and val not in output.products:
                output.products.append(val)
            elif mapped_label == "BRAND" and val not in output.brands:
                output.brands.append(val)
            elif mapped_label == "USE_CASE" and val not in output.use_cases:
                output.use_cases.append(val)
                
    return output
