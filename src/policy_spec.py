
from __future__ import annotations

from typing import Literal, TypeAlias

RoleType: TypeAlias = Literal["Sel", "Join", "Where", "Agg"]
PolicyType: TypeAlias = Literal["Public", "ConditionOnly", "AggregateOnly", "Hidden"]

PERMISSION_TABLE: dict[PolicyType, dict[RoleType, bool]] = {
    "Public": {"Sel": True, "Join": True, "Where": True, "Agg": True},
    "ConditionOnly": {"Sel": False, "Join": True, "Where": True, "Agg": False},
    "AggregateOnly": {"Sel": False, "Join": False, "Where": False, "Agg": True},
    "Hidden": {"Sel": False, "Join": False, "Where": False, "Agg": False},
}
AGGREGATIONONLY_ALLOWED_AGGS = {3, 4, 5}  # count=3, sum=4, avg=5

def is_allowed(policy: PolicyType | str, role: RoleType | str, agg_id: int = 0) -> bool:
    allowed = PERMISSION_TABLE.get(policy, {}).get(role, True)
    if allowed and policy == "AggregateOnly" and role == "Agg":
        return agg_id in AGGREGATIONONLY_ALLOWED_AGGS
    return allowed
