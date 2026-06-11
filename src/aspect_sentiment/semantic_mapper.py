import json
from pathlib import Path
import re

TAXONOMY_PATH = Path(__file__).parent / "taxonomy.json"

class SemanticMapper:
    def __init__(self):
        self.taxonomy = {}
        self._load_taxonomy()

    def _load_taxonomy(self):
        try:
            with open(TAXONOMY_PATH, 'r', encoding='utf-8') as f:
                self.taxonomy = json.load(f)
        except Exception as e:
            print(f"Error loading taxonomy: {e}")
            self.taxonomy = {"products": {}, "brands": {}, "use_cases": {}}

    def extract_gpu(self, text: str) -> str:
        """Extract GPU models separately so they aren't confused with budgets."""
        text = text.lower()
        gpu_patterns = [r'rtx\s*\d{4}', r'gtx\s*\d{3,4}', r'rx\s*\d{4}', r'radeon', r'geforce']
        for pattern in gpu_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return ""

    def map_term(self, term: str) -> list[tuple[str, str]]:
        """Maps a raw term to a list of normalized (value, label) tuples."""
        term_lower = term.lower()
        results = []

        # Check Products
        for category, keywords in self.taxonomy.get("products", {}).items():
            if category in term_lower or any(kw in term_lower for kw in keywords):
                results.append((category, "PRODUCT"))

        # Check Brands
        for brand, aliases in self.taxonomy.get("brands", {}).items():
            if brand in term_lower or any(alias in term_lower for alias in aliases):
                results.append((brand, "BRAND"))

        # Check Use Cases
        for use_case, keywords in self.taxonomy.get("use_cases", {}).items():
            if use_case in term_lower or any(kw in term_lower for kw in keywords):
                results.append((use_case, "USE_CASE"))

        # Eliminate duplicates
        return list(set(results))

mapper = SemanticMapper()
