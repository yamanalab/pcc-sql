
import json
from pathlib import Path

from .types import SpiderExample, TableSchema

def load_schemas(tables_path: str | Path) -> dict[str, TableSchema]:
    with open(tables_path) as f:
        data = json.load(f)

    schemas: dict[str, TableSchema] = {}
    for db in data:
        schema = TableSchema(
            db_id=db["db_id"],
            table_names=db["table_names_original"],
            column_names=[(col[0], col[1]) for col in db["column_names_original"]],
            column_types=db["column_types"],
            primary_keys=db["primary_keys"],
            foreign_keys=[(fk[0], fk[1]) for fk in db["foreign_keys"]],
        )
        schemas[db["db_id"]] = schema

    return schemas

def load_examples(json_path: str | Path) -> list[SpiderExample]:
    with open(json_path) as f:
        data = json.load(f)

    examples = []
    for item in data:
        example = SpiderExample(
            db_id=item["db_id"],
            question=item["question"],
            query=item["query"],
            sql=item["sql"],
        )
        examples.append(example)

    return examples

def get_column_by_id(schema: TableSchema, col_id: int) -> tuple[str, str]:
    if col_id == 0:
        return ("*", "*")
    table_idx, col_name = schema.column_names[col_id]
    if table_idx < 0:
        return ("", col_name)
    return (schema.table_names[table_idx], col_name)

def get_tables_with_column(schema: TableSchema, column_name: str) -> list[str]:
    tables = []
    col_lower = column_name.lower()
    for table_idx, col in schema.column_names[1:]:  # Skip col_id=0 (*)
        if col.lower() == col_lower and table_idx >= 0:
            tables.append(schema.table_names[table_idx])
    return tables

def get_columns_for_table(schema: TableSchema, table_name: str) -> list[tuple[int, str]]:
    result = []
    table_lower = table_name.lower()
    for i, (table_idx, col) in enumerate(schema.column_names):
        if table_idx >= 0 and schema.table_names[table_idx].lower() == table_lower:
            result.append((i, col))
    return result

def is_primary_key(schema: TableSchema, col_id: int) -> bool:
    return col_id in schema.primary_keys
