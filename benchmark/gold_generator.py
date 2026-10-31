
from .role_extractor import has_select_star
from .types import GoldLabel, SpiderExample, Violation

def generate_gold_label(example: SpiderExample, violations: list[Violation]) -> GoldLabel:
    if has_select_star(example.sql) or violations:
        return GoldLabel(type="REFUSE", sql=None)
    return GoldLabel(type="SQL", sql=example.query)
