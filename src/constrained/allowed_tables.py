
from __future__ import annotations

def allowed_tables(db_schema: dict) -> set[str]:

    return set(db_schema.get("table_names_original", []))
