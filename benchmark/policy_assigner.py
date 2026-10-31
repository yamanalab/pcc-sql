
import json
import random
import re
from pathlib import Path

from .types import PolicyType, TableSchema

# Stage 1, deterministic Hidden (PII).
_HIDDEN_PATTERNS = [
    re.compile(r"email", re.I),
    re.compile(r"phone", re.I),
    re.compile(r"address", re.I),
    re.compile(r"gender", re.I),
    re.compile(r"nationality", re.I),
    re.compile(r"birth", re.I),
    re.compile(r"ssn", re.I),
    re.compile(r"password", re.I),
    re.compile(r"^sex$", re.I),
    re.compile(r"^weight$", re.I),
    re.compile(r"^height$", re.I),
    re.compile(r"^age$", re.I),
]

# Stage 1, name regex (Public by default, Hidden under `p`).
_NAME_PATTERNS = [
    re.compile(r"^name$", re.I),
    re.compile(r"_name$", re.I),
    re.compile(r"^name_", re.I),
]


def _matches_any(name: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(p.search(name) for p in patterns)


def assign_policy(
    column_name: str,
    role_counts: dict[str, int],
    rng: random.Random,
    p: float,
    p_relax: float = 0.0,
) -> PolicyType:
    if _matches_any(column_name, _HIDDEN_PATTERNS):
        return "Public" if rng.random() < p_relax else "Hidden"
    if _matches_any(column_name, _NAME_PATTERNS):
        return "Hidden" if rng.random() < p else "Public"

    join_count = role_counts.get("Join", 0)
    agg_count = role_counts.get("Agg", 0)
    other_count = role_counts.get("Sel", 0) + role_counts.get("Where", 0)

    if join_count > 0:
        if rng.random() < p_relax:
            return "Public"
        return "Hidden" if rng.random() < p else "ConditionOnly"
    if agg_count > 0 and other_count == 0:
        if rng.random() < p_relax:
            return "Public"
        return "Hidden" if rng.random() < p else "AggregateOnly"
    return "Public"


def assign_policies_for_db(
    schema: TableSchema,
    db_usage: dict[str, dict[str, int]],
    rng: random.Random,
    p: float,
    p_relax: float = 0.0,
) -> dict[str, PolicyType]:
    policies: dict[str, PolicyType] = {}
    for col_id in range(1, len(schema.column_names)):
        table_idx, col_name = schema.column_names[col_id]
        if table_idx < 0:
            continue
        table_name = schema.table_names[table_idx]
        col_full = f"{table_name}.{col_name}"
        role_counts = db_usage.get(col_full, {})
        policies[col_full] = assign_policy(col_name, role_counts, rng, p, p_relax)
    return policies


def collect_role_usage(
    examples: list,
    schemas: dict[str, TableSchema],
) -> dict[str, dict[str, dict[str, int]]]:
    from collections import defaultdict

    from .role_extractor import extract_roles

    usage: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    for ex in examples:
        if ex.db_id not in schemas:
            continue
        try:
            refs = extract_roles(ex.sql, schemas[ex.db_id])
        except Exception:
            continue
        for ref in refs:
            col_full = f"{ref.table}.{ref.column}"
            usage[ex.db_id][col_full][ref.role] += 1
    return {
        db: {col: dict(roles) for col, roles in cols.items()}
        for db, cols in usage.items()
    }


def generate_all_policies(
    schemas: dict[str, TableSchema],
    output_dir: str | Path,
    examples: list,
    seed: int = 42,
    p: float = 0.0,
    p_relax: float = 0.0,
) -> dict[str, dict[str, PolicyType]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    usage = collect_role_usage(examples, schemas)
    col_count = sum(len(cols) for cols in usage.values())
    print(f"  Collected role usage from {len(examples)} examples ({col_count} columns observed)")

    all_policies: dict[str, dict[str, PolicyType]] = {}
    for db_id in sorted(schemas):
        rng = random.Random(f"{seed}_{db_id}")
        policies = assign_policies_for_db(
            schemas[db_id], usage.get(db_id, {}), rng, p, p_relax
        )
        all_policies[db_id] = policies
        with open(output_dir / f"{db_id}.json", "w") as f:
            json.dump(
                {"db_id": db_id, "seed": seed, "p": p, "p_relax": p_relax, "policies": policies},
                f, indent=2,
            )
    return all_policies


def print_policy_stats(
    schemas: dict[str, TableSchema],
    policies: dict[str, dict[str, PolicyType]],
) -> None:
    stats = {"Public": 0, "ConditionOnly": 0, "AggregateOnly": 0, "Hidden": 0}
    for db_policies in policies.values():
        for policy in db_policies.values():
            stats[policy] += 1
    total = sum(stats.values())
    print("\nPolicy Distribution:")
    for policy, count in stats.items():
        pct = (count / total * 100) if total else 0.0
        print(f"  {policy:10s}: {count:4d} ({pct:5.1f}%)")
