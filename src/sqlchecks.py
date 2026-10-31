
from __future__ import annotations

import sqlglot
from sqlglot import exp

from src.policy_spec import is_allowed

def is_valid_sql(sql: str) -> bool:
    try:
        sqlglot.parse_one(sql, dialect="sqlite")
        return True
    except Exception:
        return False

def find_policy_violations(
    sql: str,
    column_policies: dict[str, str],
    db_schema: dict,
) -> list[dict]:
    col_lookup = _build_column_lookup(db_schema)
    try:
        parsed = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return []
    refs = _extract_refs_from_ast(parsed, col_lookup)
    return _check_refs(refs, column_policies)

def _build_column_lookup(db_schema: dict) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    table_names = db_schema["table_names_original"]
    column_names = db_schema["column_names_original"]

    for col_id, (table_idx, col_name) in enumerate(column_names):
        if col_id == 0 or table_idx < 0:
            continue
        full_name = f"{table_names[table_idx]}.{col_name}"
        lookup.setdefault(col_name.lower(), []).append(full_name)

    return lookup

def _resolve_column(
    col_name: str,
    table_name: str | None,
    col_lookup: dict[str, list[str]],
) -> str | None:
    candidates = col_lookup.get(col_name.lower(), [])
    if not candidates:
        return None
    if table_name:
        table_lower = table_name.lower()
        for candidate in candidates:
            if candidate.lower().startswith(f"{table_lower}."):
                return candidate
    return candidates[0]

def _extract_refs_from_ast(
    node: exp.Expression,
    col_lookup: dict[str, list[str]],
) -> list[dict]:
    refs: list[dict] = []
    for select_expr in node.find_all(exp.Select):
        for expr in select_expr.expressions:
            _walk(expr, "Sel", 0, col_lookup, refs, promote_agg_args=False)

        where = select_expr.args.get("where")
        if where is not None:
            _walk(where.this, "Where", 0, col_lookup, refs, promote_agg_args=False)

        having = select_expr.args.get("having")
        if having is not None:
            _walk(having.this, "Where", 0, col_lookup, refs, promote_agg_args=True)

        group = select_expr.args.get("group")
        if group is not None:
            for expr in group.expressions:
                _walk(expr, "Where", 0, col_lookup, refs, promote_agg_args=False)

        order = select_expr.args.get("order")
        if order is not None:
            for ordered in order.expressions:
                _walk(ordered.this, "Where", 0, col_lookup, refs, promote_agg_args=False)

        for join in select_expr.args.get("joins") or ():
            on = join.args.get("on")
            if on is not None:
                _walk(on, "Join", 0, col_lookup, refs, promote_agg_args=False)
            using = join.args.get("using")
            if using is None:
                continue
            using_list = using if isinstance(using, list) else [using]
            for using_expr in using_list:
                _walk(using_expr, "Join", 0, col_lookup, refs, promote_agg_args=False)
    return refs

_AGG_NAME_TO_ID = {"MAX": 1, "MIN": 2, "COUNT": 3, "SUM": 4, "AVG": 5}

def _walk(
    expr: exp.Expression | None,
    default_role: str,
    current_agg_id: int,
    col_lookup: dict[str, list[str]],
    refs: list[dict],
    *,
    promote_agg_args: bool,
) -> None:
    if expr is None or isinstance(expr, (exp.Subquery, exp.Query)):
        return

    if isinstance(expr, exp.Column):
        col_name = expr.name
        if col_name == "*":
            return
        resolved = _resolve_column(col_name, expr.table or None, col_lookup)
        if not resolved:
            return
        role = (
            "Agg"
            if current_agg_id and (default_role == "Sel" or promote_agg_args)
            else default_role
        )
        refs.append({
            "column": resolved,
            "role": role,
            "agg_id": current_agg_id if role == "Agg" else 0,
        })
        return

    next_agg_id = current_agg_id
    if isinstance(expr, exp.AggFunc):
        next_agg_id = _AGG_NAME_TO_ID.get(type(expr).__name__.upper(), 0)

    for child in expr.iter_expressions():
        _walk(child, default_role, next_agg_id, col_lookup, refs,
              promote_agg_args=promote_agg_args)

def _check_refs(refs: list[dict], policies: dict[str, str]) -> list[dict]:
    violations: list[dict] = []
    seen: set[tuple[str, str, int]] = set()
    for ref in refs:
        col, role, agg_id = ref["column"], ref["role"], ref["agg_id"]
        policy = policies.get(col, "Public")
        if policy == "Public":
            continue
        key = (col, role, agg_id)
        if key in seen:
            continue
        seen.add(key)
        if not is_allowed(policy, role, agg_id):
            violations.append({"column": col, "role": role, "policy": policy, "agg_id": agg_id})
    return violations
