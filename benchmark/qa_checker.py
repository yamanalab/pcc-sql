
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class QAReport:
    split: str
    total_records: int
    violation_rate: float
    refuse_rate: float
    role_distribution: dict[str, int] = field(default_factory=dict)
    policy_distribution: dict[str, int] = field(default_factory=dict)
    db_refuse_rates: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

def run_qa_check(data_path: Path, split: str) -> QAReport:
    with open(data_path / f"{split}.json") as f:
        data = json.load(f)
    if not data:
        return QAReport(split=split, total_records=0, violation_rate=0, refuse_rate=0,
                        warnings=["No data found"])

    warnings: list[str] = []

    with_violations = sum(1 for r in data if r["violations_original"])
    violation_rate = with_violations / len(data)
    refuse_count = sum(1 for r in data if r["gold_label"]["type"] == "REFUSE")
    refuse_rate = refuse_count / len(data)

    # v2: violation_rate == refuse_rate (no rewrite path), so warn only on extremes
    if violation_rate < 0.05:
        warnings.append(f"Violation rate unusually low: {violation_rate:.1%}")
    if refuse_rate < 0.05:
        warnings.append(f"REFUSE rate unusually low: {refuse_rate:.1%}")

    db_refuse: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "refuse": 0})
    for r in data:
        db_refuse[r["db_id"]]["total"] += 1
        if r["gold_label"]["type"] == "REFUSE":
            db_refuse[r["db_id"]]["refuse"] += 1
    db_refuse_rates = {
        db: d["refuse"] / d["total"] if d["total"] else 0.0 for db, d in db_refuse.items()
    }
    rates = list(db_refuse_rates.values())
    if len(rates) > 1 and statistics.stdev(rates) > 0.3:
        warnings.append(f"High DB REFUSE rate variance: stdev={statistics.stdev(rates):.2f}")

    role_counter: Counter[str] = Counter()
    policy_counter: Counter[str] = Counter()
    for r in data:
        for v in r["violations_original"]:
            role_counter[v["role"]] += 1
            policy_counter[v["policy"]] += 1

    total_violations = sum(role_counter.values())
    if total_violations and role_counter.get("Join", 0) / total_violations < 0.02:
        warnings.append(
            f"Join violations very rare: "
            f"{role_counter.get('Join', 0)}/{total_violations}"
        )

    return QAReport(
        split=split,
        total_records=len(data),
        violation_rate=violation_rate,
        refuse_rate=refuse_rate,
        role_distribution=dict(role_counter),
        policy_distribution=dict(policy_counter),
        db_refuse_rates=db_refuse_rates,
        warnings=warnings,
    )

def print_qa_report(report: QAReport) -> None:
    print(f"\n=== QA Report: {report.split} ===")
    print(f"  Total records  : {report.total_records}")
    print(f"  Violation rate : {report.violation_rate:.1%}")
    print(f"  REFUSE rate    : {report.refuse_rate:.1%}")
    if report.role_distribution:
        print("  Role distribution:")
        for role, count in sorted(report.role_distribution.items()):
            print(f"    {role}: {count}")
    if report.policy_distribution:
        print("  Policy distribution:")
        for policy, count in sorted(report.policy_distribution.items()):
            print(f"    {policy}: {count}")
    if report.warnings:
        print("  WARNINGS:")
        for w in report.warnings:
            print(f"    ! {w}")

def save_qa_report(reports: list[QAReport], output_path: Path) -> None:
    data = [
        {
            "split": r.split,
            "total_records": r.total_records,
            "violation_rate": r.violation_rate,
            "refuse_rate": r.refuse_rate,
            "role_distribution": r.role_distribution,
            "policy_distribution": r.policy_distribution,
            "warnings": r.warnings,
        }
        for r in reports
    ]
    with open(output_path / "qa_report.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved QA report to {output_path / 'qa_report.json'}")
