
from src.policy_spec import is_allowed as shared_is_allowed

from .types import ColumnRef, PolicyType, RoleType, Violation

def is_allowed(policy: PolicyType, role: RoleType, agg_id: int) -> bool:
    return shared_is_allowed(policy, role, agg_id)

def check_violations(
    refs: list[ColumnRef], policies: dict[str, PolicyType]
) -> list[Violation]:
    violations: list[Violation] = []

    for ref in refs:
        col_key = f"{ref.table}.{ref.column}"
        policy = policies.get(col_key, "Public")

        if not is_allowed(policy, ref.role, ref.agg_id):
            violations.append(
                Violation(
                    column=col_key,
                    role=ref.role,
                    policy=policy,
                    agg_id=ref.agg_id,
                )
            )

    return violations
