
from __future__ import annotations

from src.policy_spec import PolicyType, RoleType, is_allowed

def allowed_columns_for_role(
    role: RoleType,
    column_policies: dict[str, PolicyType],
    db_schema: dict,
    tables_in_scope: tuple[str, ...],
    current_agg: str | None = None,
) -> set[str]:
    agg_id = _agg_name_to_id(current_agg)
    allowed: set[str] = set()
    schema_wide_if_empty = role in {"Sel", "Agg"}

    for full_name in schema_columns_in_scope(
        db_schema,
        tables_in_scope,
        schema_wide_if_empty=schema_wide_if_empty,
    ):
        policy = column_policies.get(full_name, "Public")
        if is_allowed(policy, role, agg_id):
            allowed.add(full_name)

    return allowed

def schema_columns_in_scope(
    db_schema: dict,
    tables_in_scope: tuple[str, ...],
    *,
    schema_wide_if_empty: bool = False,
) -> set[str]:
    in_scope = {table.lower() for table in tables_in_scope}
    columns: set[str] = set()
    table_names = db_schema["table_names_original"]
    column_names = db_schema["column_names_original"]

    for col_id, (table_idx, col_name) in enumerate(column_names):
        if col_id == 0 or table_idx < 0:
            continue

        table_name = table_names[table_idx]
        if in_scope:
            if table_name.lower() not in in_scope:
                continue
        elif not schema_wide_if_empty:
            continue

        columns.add(f"{table_name}.{col_name}")

    return columns

def _agg_name_to_id(name: str | None) -> int:
    if name is None:
        return 0
    mapping = {"MAX": 1, "MIN": 2, "COUNT": 3, "SUM": 4, "AVG": 5}
    return mapping.get(name.upper(), 0)
