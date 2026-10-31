
import argparse
from pathlib import Path

from tqdm import tqdm

from .gold_generator import generate_gold_label
from .output_writer import format_record, write_dataset
from .policy_assigner import generate_all_policies, print_policy_stats
from .qa_checker import print_qa_report, run_qa_check, save_qa_report
from .role_extractor import extract_roles
from .spider_loader import load_examples, load_schemas
from .violation_checker import check_violations


def _process_split(examples, schemas, all_policies, split):
    records = []
    for idx, ex in enumerate(tqdm(examples, desc=f"spider:{split}")):
        schema = schemas[ex.db_id]
        policies = all_policies.get(ex.db_id, {})
        refs = extract_roles(ex.sql, schema)
        violations = check_violations(refs, policies)
        gold_label = generate_gold_label(ex, violations)
        records.append(format_record(
            record_id=f"{split}_{idx:05d}",
            db_id=ex.db_id, question=ex.question, original_sql=ex.query,
            column_policies=policies, violations_original=violations,
            gold_label=gold_label,
        ))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Spider-CU benchmark")
    parser.add_argument("--spider-path", default="spider_data/spider_data")
    parser.add_argument("--output-path", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p", type=float, default=0.0,
                        help="Hidden injection probability")
    parser.add_argument("--p-relax", type=float, default=0.0,
                        help="Relaxation probability (Hidden / X-Only -> Public)")
    args = parser.parse_args()

    spider_path = Path(args.spider_path)
    output_path = Path(args.output_path)
    print(f"Spider-CU builder  seed={args.seed}  p={args.p}  p_relax={args.p_relax}")

    print("\n[spider] Loading schemas...")
    schemas = load_schemas(spider_path / "tables.json")
    print(f"  {len(schemas)} database schemas")

    train_path = spider_path / "train_spider.json"
    train_examples = load_examples(train_path) if train_path.exists() else []
    dev_examples = load_examples(spider_path / "dev.json")
    usage_examples = train_examples + dev_examples

    print(f"\n[spider] Assigning policies (seed={args.seed}, p={args.p}, p_relax={args.p_relax})...")
    all_policies = generate_all_policies(
        schemas, output_path / "policies", seed=args.seed, p=args.p,
        examples=usage_examples, p_relax=args.p_relax,
    )
    print_policy_stats(schemas, all_policies)

    splits = [("dev", dev_examples)]
    if train_examples:
        splits.insert(0, ("train", train_examples))
    for split, examples in splits:
        print(f"\n[spider] Processing {split}...")
        write_dataset(_process_split(examples, schemas, all_policies, split),
                      output_path, split)

    print("\n[spider] QA checks...")
    reports = [run_qa_check(output_path, split) for split, _ in splits]
    for r in reports:
        print_qa_report(r)
    save_qa_report(reports, output_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
