from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class NormalizedFeature:
    name: str
    label: str
    original_value: str

@dataclass
class NormalizedOutput:
    products: List[str] = field(default_factory=list)
    brands: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)
    budget: int = 0
    urgency: Optional[str] = None
    decision_stage: Optional[str] = None
    price_sensitivity: Optional[str] = None
    
    def to_xgboost_dict(self) -> dict:
        result = {"budget": self.budget}
        for p in self.products:
            result[f"product_{p}"] = 1
        for b in self.brands:
            result[f"brand_{b}"] = 1
        for u in self.use_cases:
            result[f"use_case_{u}"] = 1
            
        if self.urgency:
            result[f"urgency_{self.urgency}"] = 1
            
        if self.decision_stage:
            result[f"decision_stage_{self.decision_stage}"] = 1

        if self.price_sensitivity:
            result[f"price_sensitivity_{self.price_sensitivity}"] = 1
            
        return result
