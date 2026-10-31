
from dataclasses import dataclass, field
from typing import Any, Literal

PolicyType = Literal["Public", "ConditionOnly", "AggregateOnly", "Hidden"]
RoleType = Literal["Sel", "Join", "Where", "Agg"]
GoldLabelType = Literal["SQL", "REFUSE"]

@dataclass
class ColumnRef:

    table: str
    column: str
    role: RoleType
    agg_id: int  # 0=none, 1=max, 2=min, 3=count, 4=sum, 5=avg

@dataclass
class Violation:

    column: str  # "table.column" format
    role: RoleType
    policy: PolicyType
    agg_id: int

@dataclass
class GoldLabel:

    type: GoldLabelType
    sql: str | None = None

@dataclass
class TableSchema:

    db_id: str
    table_names: list[str]
    column_names: list[tuple[int, str]]
    column_types: list[str]
    primary_keys: list[int]
    foreign_keys: list[tuple[int, int]]

    def resolve_column(self, col_id: int) -> str:
        if col_id == 0:
            return "*"
        table_idx, col_name = self.column_names[col_id]
        if table_idx < 0:
            return col_name
        return f"{self.table_names[table_idx]}.{col_name}"

@dataclass
class SpiderExample:

    db_id: str
    question: str
    query: str
    sql: dict[str, Any]

@dataclass
class ProcessedExample:

    id: str
    db_id: str
    question: str
    original_sql: str
    column_policies: dict[str, PolicyType]
    violations_original: list[Violation]
    gold_label: GoldLabel
