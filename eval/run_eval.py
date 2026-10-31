
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from tqdm import tqdm

from src.sqlchecks import find_policy_violations, is_valid_sql

from .baselines import DirectProvider, RetryProvider, VerifierProvider, create_base_provider
from .prompt import build_prompt, load_schemas_raw
from .scorer import parse_model_output, print_report, score

def load_dataset(data_dir: Path) -> list[dict]:
    with open(data_dir / "dev.json") as f:
        return json.load(f)

def run_sql(sql: str, db_path: str, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout

    def _abort() -> int:
        return 1 if time.monotonic() > deadline else 0

    with sqlite3.connect(db_path) as conn:
        conn.set_progress_handler(_abort, 10000)
        try:
            conn.cursor().execute(sql).fetchall()
            return True
        except sqlite3.Error:
            return False

def process_record(record, schemas, provider, schema_dir: Path):
    db_schema = schemas.get(record["db_id"])
    if db_schema is None:
        return None
    prompt = build_prompt(record["question"], db_schema, record["column_policies"])
    provider.set_context(record["column_policies"], db_schema)

    started = time.perf_counter()
    try:
        raw_output = provider.generate(prompt)
        prediction = parse_model_output(raw_output)
    except Exception as e:
        raw_output = f"ERROR: {e}"
        prediction = {"type": "REFUSE", "sql": None}
    latency = time.perf_counter() - started

    result = {
        "gold_label": record["gold_label"],
        "prediction": prediction,
        "raw_output": raw_output,
        "latency_sec": latency,
    }
    usage = provider.get_last_usage_stats()
    if usage is not None:
        result["usage"] = usage

    if prediction["type"] == "SQL" and prediction["sql"]:
        pred_sql = prediction["sql"]
        result["syntax_valid"] = is_valid_sql(pred_sql)
        result["pred_violations"] = find_policy_violations(
            pred_sql, record["column_policies"], db_schema,
        )
        db_path = schema_dir / "database" / record["db_id"] / f"{record['db_id']}.sqlite"
        result["exec_success"] = run_sql(pred_sql, str(db_path))
    return result

METHODS = ("pcc-sql", "direct", "retry", "verifier")

def create_provider(method: str, model: str, retries: int):
    if method == "pcc-sql":
        from src.constrained.policy_state_provider import PolicyStateLocalProvider
        return PolicyStateLocalProvider(model)
    base = create_base_provider(model)
    if method == "direct":
        return DirectProvider(base)
    if method == "retry":
        return RetryProvider(base, max_retries=retries)
    return VerifierProvider(base)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PCC-SQL and the baselines on Spider-CU / Spider-ACL",
    )
    parser.add_argument("--method", choices=METHODS, default="pcc-sql")
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="HuggingFace model ID, or a claude-* model ID (direct/retry/verifier only)",
    )
    parser.add_argument("--retries", type=int, default=3, help="Max retries for --method retry")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--schema-dir", default="spider_data/spider_data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    schema_dir = Path(args.schema_dir)
    if not (schema_dir / "database").is_dir():
        raise SystemExit(
            f"Spider SQLite DBs not found at {schema_dir / 'database'}. "
            "Run `bash setup_spider.sh` first."
        )
    dataset = load_dataset(data_dir)
    schemas = load_schemas_raw(schema_dir / "tables.json")
    if args.limit:
        dataset = dataset[: args.limit]

    print(f"{args.method}  model={args.model}  N={len(dataset)}")
    provider = create_provider(args.method, args.model, args.retries)

    results = []
    for record in tqdm(dataset, desc="Evaluating"):
        r = process_record(record, schemas, provider, schema_dir)
        if r is not None:
            results.append(r)

    print_report(score(results), args.method, args.model)

if __name__ == "__main__":
    main()
