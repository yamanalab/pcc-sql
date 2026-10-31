
import json
from pathlib import Path

from .types import GoldLabel, PolicyType, Violation

def write_dataset(records: list[dict], output_path: Path, split: str) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{split}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(records)} records to {output_file}")
    _print_statistics(records, split)

def _print_statistics(records: list[dict], split: str) -> None:
    total = len(records)
    if total == 0:
        print(f"\n=== Output Statistics ({split}) ===\n  No records.")
        return
    with_violations = sum(1 for r in records if r["violations_original"])
    gold_sql = sum(1 for r in records if r["gold_label"]["type"] == "SQL")
    print(f"\n=== Output Statistics ({split}) ===")
    print(f"  Total records     : {total}")
    print(f"  With violations   : {with_violations} ({with_violations / total * 100:.1f}%)")
    print(f"  Gold SQL          : {gold_sql} ({gold_sql / total * 100:.1f}%)")
    print(f"  Gold REFUSE       : {total - gold_sql} ({(total - gold_sql) / total * 100:.1f}%)")

def format_record(
    record_id: str,
    db_id: str,
    question: str,
    original_sql: str,
    column_policies: dict[str, PolicyType],
    violations_original: list[Violation],
    gold_label: GoldLabel,
) -> dict:
    return {
        "id": record_id,
        "db_id": db_id,
        "question": question,
        "original_sql": original_sql,
        "column_policies": column_policies,
        "violations_original": [
            {"column": v.column, "role": v.role, "policy": v.policy, "agg_id": v.agg_id}
            for v in violations_original
        ],
        "gold_label": {"type": gold_label.type, "sql": gold_label.sql},
    }
