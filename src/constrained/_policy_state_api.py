
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .canonical_sql_grammar import GrammarBranch
from .sql_state_parser import PolicySQLCurrentState

if TYPE_CHECKING:
    from ._policy_state_impl import PolicyStateAdmissibilityRuntime

@dataclass(frozen=True)
class PolicyStateDecodingContext:

    generated_ids: tuple[int, ...]
    text: str
    state: PolicySQLCurrentState
    branch: GrammarBranch
    slot_ids: tuple[int, ...]

@dataclass(frozen=True)
class PolicyStateAdmissibilityResult:

    decision: Literal["allow", "force_eos", "force_stop"]
    allowed_token_ids: frozenset[int]
    context: PolicyStateDecodingContext | None = None
    debug_info: dict[str, object] | None = None

    @property
    def should_force_eos(self) -> bool:
        return self.decision in {"force_eos", "force_stop"}

    @property
    def terminated_unsafe(self) -> bool:
        return self.decision == "force_stop"

def build_policy_state_context(
    runtime: "PolicyStateAdmissibilityRuntime",
    *,
    generated_ids: list[int] | tuple[int, ...],
    text: str | None = None,
) -> PolicyStateDecodingContext:
    return runtime.build_context(generated_ids=generated_ids, text=text)

def _forced_admissibility_result(
    runtime: "PolicyStateAdmissibilityRuntime",
    *,
    decision: Literal["force_eos", "force_stop"],
    context: PolicyStateDecodingContext | None = None,
    debug_info: dict[str, object] | None = None,
) -> PolicyStateAdmissibilityResult:
    return PolicyStateAdmissibilityResult(
        decision=decision,
        allowed_token_ids=frozenset(runtime._eos_allowed_token_ids()),
        context=context,
        debug_info=debug_info,
    )

def _context_failure_result(
    runtime: "PolicyStateAdmissibilityRuntime",
    *,
    context: PolicyStateDecodingContext,
    failure_reason: str,
) -> PolicyStateAdmissibilityResult:
    return _forced_admissibility_result(
        runtime,
        decision="force_stop",
        context=context,
        debug_info=runtime.build_debug_info_from_context(
            context,
            failure_reason=failure_reason,
        ),
    )

def compute_admissible_next_tokens(
    runtime: "PolicyStateAdmissibilityRuntime",
    *,
    generated_ids: list[int] | tuple[int, ...],
    text: str | None = None,
) -> PolicyStateAdmissibilityResult:
    if text is None:
        text = runtime.tokenizer.decode(generated_ids, skip_special_tokens=True)

    if text.rstrip().endswith(";"):
        debug_info = runtime.validate_terminal_output(
            text,
            terminated_by_eos=False,
        )
        if debug_info is not None:
            return _forced_admissibility_result(
                runtime,
                decision="force_stop",
                debug_info={**debug_info, "decision_stage": "mask"},
            )
        return _forced_admissibility_result(runtime, decision="force_eos")

    context = build_policy_state_context(
        runtime,
        generated_ids=generated_ids,
        text=text,
    )
    state = context.state

    if state.unsupported:
        return _context_failure_result(
            runtime,
            context=context,
            failure_reason="unsupported_subset",
        )

    if runtime._finalized_scope_select_list_has_out_of_scope_reference(
        text,
        state,
        final_output=False,
    ):
        return _context_failure_result(
            runtime,
            context=context,
            failure_reason="out_of_scope_select_reference",
        )

    allowed_token_ids = runtime._collect_allowed_token_ids(
        generated_ids=list(context.generated_ids),
        text=text,
        state=context.state,
        branch=context.branch,
        slot_ids=list(context.slot_ids),
    )
    if not allowed_token_ids:
        return _context_failure_result(
            runtime,
            context=context,
            failure_reason="no_valid_continuation",
        )

    return PolicyStateAdmissibilityResult(
        decision="allow",
        allowed_token_ids=frozenset(allowed_token_ids),
        context=context,
    )

__all__ = [
    "PolicyStateAdmissibilityResult",
    "PolicyStateDecodingContext",
    "build_policy_state_context",
    "compute_admissible_next_tokens",
]
