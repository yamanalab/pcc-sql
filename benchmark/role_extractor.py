
from typing import Any

from .types import ColumnRef, RoleType, TableSchema

def extract_roles(sql: dict[str, Any], schema: TableSchema) -> list[ColumnRef]:
    refs: list[ColumnRef] = []

    # SELECT clause
    refs.extend(_extract_from_select(sql["select"], schema))

    # FROM clause JOIN conditions
    if sql["from"]["conds"]:
        refs.extend(_extract_from_conds(sql["from"]["conds"], schema, "Join"))

    # WHERE clause
    if sql["where"]:
        refs.extend(_extract_from_conds(sql["where"], schema, "Where"))

    # Subqueries in FROM
    for table_unit in sql["from"]["table_units"]:
        if table_unit[0] == "sql":
            refs.extend(extract_roles(table_unit[1], schema))

    # INTERSECT/UNION/EXCEPT
    for op in ["intersect", "union", "except"]:
        if sql.get(op):
            refs.extend(extract_roles(sql[op], schema))

    return refs

def _extract_from_select(select: list, schema: TableSchema) -> list[ColumnRef]:
    # select = [isDistinct, [[agg_id, val_unit], ...]]
    _, val_units = select
    refs: list[ColumnRef] = []

    for item in val_units:
        agg_id, val_unit = item
        refs.extend(_extract_from_val_unit(val_unit, schema, agg_id))

    return refs

def _extract_from_val_unit(
    val_unit: list, schema: TableSchema, outer_agg_id: int
) -> list[ColumnRef]:
    # val_unit = [unit_op, col_unit1, col_unit2]
    _, col_unit1, col_unit2 = val_unit
    refs: list[ColumnRef] = []

    for col_unit in [col_unit1, col_unit2]:
        if col_unit is None:
            continue

        agg_id, col_id, _ = col_unit

        # Skip col_id=0 (*) - COUNT(*) is always allowed
        if col_id == 0:
            continue

        # Effective agg_id: prefer col_unit's own, fallback to outer
        effective_agg = agg_id if agg_id != 0 else outer_agg_id

        # Determine role
        if effective_agg != 0:
            role: RoleType = "Agg"
        else:
            role = "Sel"

        table, column = _resolve_col_id(schema, col_id)
        if table:  # Skip if unresolved
            refs.append(ColumnRef(table, column, role, effective_agg))

    return refs

def _extract_from_conds(
    conds: list, schema: TableSchema, role: RoleType
) -> list[ColumnRef]:
    refs: list[ColumnRef] = []

    for item in conds:
        # Skip "and" / "or" connectors
        if isinstance(item, str):
            continue

        # Condition format: [not_op, op_id, val_unit, val1, val2]
        _, _, val_unit, val1, val2 = item

        # Extract columns from val_unit (left side of comparison)
        refs.extend(_extract_cols_from_val_unit(val_unit, schema, role))

        # Handle val1, val2 (right side of comparison)
        for val in [val1, val2]:
            if val is None:
                continue

            # Subquery
            if isinstance(val, dict):
                refs.extend(extract_roles(val, schema))
            # Column reference: [agg_id, col_id, isDistinct]
            elif isinstance(val, (list, tuple)) and len(val) == 3:
                agg_id, col_id, _ = val
                if col_id != 0:  # Skip "*"
                    table, column = _resolve_col_id(schema, col_id)
                    if table:
                        refs.append(ColumnRef(table, column, role, agg_id))

    return refs

def _extract_cols_from_val_unit(
    val_unit: list, schema: TableSchema, role: RoleType
) -> list[ColumnRef]:
    # val_unit = [unit_op, col_unit1, col_unit2]
    _, col_unit1, col_unit2 = val_unit
    refs: list[ColumnRef] = []

    for col_unit in [col_unit1, col_unit2]:
        if col_unit is None:
            continue

        agg_id, col_id, _ = col_unit

        if col_id == 0:  # Skip "*"
            continue

        table, column = _resolve_col_id(schema, col_id)
        if table:
            refs.append(ColumnRef(table, column, role, agg_id))

    return refs

def _resolve_col_id(schema: TableSchema, col_id: int) -> tuple[str, str]:
    if col_id == 0:
        return ("", "*")
    if col_id >= len(schema.column_names):
        return ("", "")  # Invalid col_id

    table_idx, col_name = schema.column_names[col_id]
    if table_idx < 0:
        return ("", col_name)

    return (schema.table_names[table_idx], col_name)

def has_select_star(sql: dict[str, Any]) -> bool:
    _, val_units = sql["select"]

    for item in val_units:
        agg_id, val_unit = item
        _, col_unit1, _ = val_unit

        if col_unit1 is not None:
            inner_agg_id, col_id, _ = col_unit1
            # SELECT * (not COUNT(*))
            if col_id == 0 and inner_agg_id == 0 and agg_id == 0:
                return True

    # Check subqueries and set operations
    for table_unit in sql["from"]["table_units"]:
        if table_unit[0] == "sql":
            if has_select_star(table_unit[1]):
                return True

    for op in ["intersect", "union", "except"]:
        if sql.get(op):
            if has_select_star(sql[op]):
                return True

    return False
