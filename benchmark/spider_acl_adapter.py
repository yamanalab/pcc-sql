from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path


_GRANT_WITH_COLUMNS_RE = re.compile(
    r"""
    GRANT \s+ SELECT \s*
    \(
        (?P<columns>[^)]*)
    \)
    \s+ ON \s+
    (?:(?P<schema>"[^"]+"|[A-Za-z0-9_]+)\.)?
    (?P<table>"[^"]+"|[A-Za-z0-9_]+)
    \s+ TO \s+
    (?P<role>"[^"]+"|[A-Za-z0-9_]+)
    \s* ;
    """,
    re.IGNORECASE | re.VERBOSE,
)
_GRANT_TABLE_RE = re.compile(
    r"""
    GRANT \s+ SELECT
    \s+ ON \s+
    (?:(?P<schema>"[^"]+"|[A-Za-z0-9_]+)\.)?
    (?P<table>"[^"]+"|[A-Za-z0-9_]+)
    \s+ TO \s+
    (?P<role>"[^"]+"|[A-Za-z0-9_]+)
    \s* ;
    """,
    re.IGNORECASE | re.VERBOSE,
)
_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?P<table>[^(\s;]+)\s*\((?P<body>.*?)\);",
    re.IGNORECASE | re.DOTALL,
)
_TYPE_SUFFIXES = [
    "timestamp without time zone",
    "timestamp with time zone",
    "double precision",
    "character varying",
    "varying character",
    "without time zone",
    "with time zone",
    "smallint",
    "bigint",
    "integer",
    "numeric",
    "decimal",
    "boolean",
    "varchar",
    "double",
    "float",
    "real",
    "date",
    "time",
    "text",
    "int",
]


def load_rcr_jsonl_from_zip(zip_path: str | Path, member_name: str) -> list[dict]:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member_name) as f:
            return [json.loads(line) for line in f.read().decode("utf-8").splitlines()]


def parse_schema_ddl_to_tables_json(db_id: str, schema_ddl: str) -> dict:
    table_names_original: list[str] = []
    column_names_original: list[tuple[int, str]] = [(-1, "*")]
    column_names: list[tuple[int, str]] = [(-1, "*")]
    column_types: list[str] = []

    for match in _CREATE_TABLE_RE.finditer(schema_ddl):
        table_name = _unquote_identifier(match.group("table"))
        table_idx = len(table_names_original)
        table_names_original.append(table_name)

        for raw_column in _split_column_defs(match.group("body")):
            col_name, col_type = _parse_column_def(raw_column)
            if col_name is None:
                continue
            column_names_original.append((table_idx, col_name))
            column_names.append((table_idx, col_name.lower()))
            column_types.append(_sql_type_to_spider(col_type))

    return {
        "db_id": db_id,
        "table_names_original": table_names_original,
        "table_names": [name.lower() for name in table_names_original],
        "column_names_original": column_names_original,
        "column_names": column_names,
        "column_types": column_types,
        "primary_keys": [],
        "foreign_keys": [],
    }


def compile_rcr_column_policies(
    access_policy_sql: str,
    db_schema: dict,
    user: str,
) -> dict[str, str]:
    table_names = db_schema["table_names_original"]
    column_policies: dict[str, str] = {}
    columns_by_table: dict[str, list[str]] = {}

    for col_id, (table_idx, col_name) in enumerate(db_schema["column_names_original"]):
        if col_id == 0 or table_idx < 0:
            continue
        table_name = table_names[table_idx]
        columns_by_table.setdefault(table_name.lower(), []).append(col_name)
        column_policies[f"{table_name}.{col_name}"] = "Hidden"

    if user.lower().startswith(f'{db_schema["db_id"].lower()}_'):
        target_role = user
    else:
        target_role = f'{db_schema["db_id"]}_{user}'
    target_role_lower = target_role.lower()

    for match in _GRANT_WITH_COLUMNS_RE.finditer(access_policy_sql):
        if _unquote_identifier(match.group("role")).lower() != target_role_lower:
            continue
        table_name = _unquote_identifier(match.group("table"))
        for col_name in _split_grant_columns(match.group("columns")):
            full_name = _resolve_column_full_name(table_name, col_name, db_schema)
            if full_name is not None:
                column_policies[full_name] = "Public"

    for match in _GRANT_TABLE_RE.finditer(access_policy_sql):
        if match.groupdict().get("columns") is not None:
            continue
        if _unquote_identifier(match.group("role")).lower() != target_role_lower:
            continue
        table_name = _unquote_identifier(match.group("table"))
        for col_name in columns_by_table.get(table_name.lower(), []):
            column_policies[f"{table_name}.{col_name}"] = "Public"

    return column_policies


def convert_rcr_record(record: dict, idx: int, db_schema: dict) -> dict:
    ground_truth_label = record.get("ground_truth_label", "") or ""
    if ground_truth_label.lstrip().upper().startswith("ERROR:"):
        gold_label = {"type": "REFUSE", "sql": None}
    else:
        gold_label = {"type": "SQL", "sql": record["query"]}

    user = record.get("user", "")
    return {
        "id": f"spider_acl_{idx:06d}",
        "db_id": record["db_id"],
        "question": record["question"],
        "original_sql": record["query"],
        "column_policies": compile_rcr_column_policies(
            record["access_policy_sql"],
            db_schema,
            user,
        ),
        "violations_original": [],
        "gold_label": gold_label,
        "rcr_user": user,
        "rcr_ground_truth_label": ground_truth_label,
        "rcr_access_policy_sql": record["access_policy_sql"],
    }


def build_rcr_spider_eval_artifacts(records: list[dict]) -> tuple[list[dict], list[dict]]:
    schemas_by_db: dict[str, dict] = {}
    eval_records: list[dict] = []

    for idx, record in enumerate(records):
        db_id = record["db_id"]
        db_schema = schemas_by_db.get(db_id)
        if db_schema is None:
            db_schema = parse_schema_ddl_to_tables_json(db_id, record["schema_ddl"])
            schemas_by_db[db_id] = db_schema
        eval_records.append(convert_rcr_record(record, idx, db_schema))

    tables_json = [schemas_by_db[db_id] for db_id in sorted(schemas_by_db)]
    return eval_records, tables_json


def _split_grant_columns(text: str) -> list[str]:
    return [_unquote_identifier(part.strip()) for part in text.split(",") if part.strip()]


def _resolve_column_full_name(table_name: str, col_name: str, db_schema: dict) -> str | None:
    table_names = db_schema["table_names_original"]
    for table_idx, original_table_name in enumerate(table_names):
        if original_table_name.lower() != table_name.lower():
            continue
        for col_id, (col_table_idx, original_col_name) in enumerate(
            db_schema["column_names_original"]
        ):
            if col_id == 0 or col_table_idx != table_idx:
                continue
            if original_col_name.lower() == col_name.lower():
                return f"{original_table_name}.{original_col_name}"
    return None


def _sql_type_to_spider(sql_type: str | None) -> str:
    if not sql_type:
        return "text"
    kind_upper = str(sql_type).upper()
    numeric = ("INT", "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC", "BIGINT", "SMALLINT", "REAL")
    if any(token in kind_upper for token in numeric):
        return "number"
    if any(token in kind_upper for token in ("DATE", "TIME")):
        return "time"
    if "BOOL" in kind_upper:
        return "boolean"
    return "text"


def _unquote_identifier(name: str) -> str:
    stripped = name.strip()
    if stripped.startswith('"') and stripped.endswith('"'):
        return stripped[1:-1]
    return stripped


def _split_column_defs(body: str) -> list[str]:
    return [part.strip() for part in body.split(",") if part.strip()]


def _parse_column_def(raw_column: str) -> tuple[str | None, str | None]:
    stripped = raw_column.strip()
    lowered = stripped.lower()

    for type_suffix in _TYPE_SUFFIXES:
        if lowered.endswith(type_suffix):
            col_name = stripped[: -len(type_suffix)].strip()
            if not col_name:
                return None, None
            return _unquote_identifier(col_name), type_suffix

    parts = stripped.rsplit(" ", 1)
    if len(parts) != 2:
        return None, None
    return _unquote_identifier(parts[0]), parts[1]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert the Spider-ACL release (Klisura et al., 2026) into the eval format",
    )
    parser.add_argument("--input-zip", required=True)
    parser.add_argument("--member-name", default="access_control_dataset.jsonl")
    parser.add_argument("--output-dir", default="data_spider_acl")
    args = parser.parse_args()

    records = load_rcr_jsonl_from_zip(args.input_zip, args.member_name)
    eval_records, tables_json = build_rcr_spider_eval_artifacts(records)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "dev.json", "w", encoding="utf-8") as f:
        json.dump(eval_records, f, ensure_ascii=False, indent=2)
    with open(output_dir / "tables.json", "w", encoding="utf-8") as f:
        json.dump(tables_json, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(eval_records)} records, {len(tables_json)} schemas to {output_dir}")


if __name__ == "__main__":
    main()
