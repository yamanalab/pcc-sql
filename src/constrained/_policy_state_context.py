
from __future__ import annotations

from ._policy_state_api import PolicyStateDecodingContext
from .canonical_sql_grammar import GrammarBranch
from .sql_state_parser import PolicySQLCurrentState

__all__ = ["_PolicyStateContextMixin"]

class _PolicyStateContextMixin:

    def build_context(
        self,
        *,
        generated_ids: list[int] | tuple[int, ...],
        text: str | None = None,
    ) -> PolicyStateDecodingContext:
        generated_ids_tuple = tuple(generated_ids)
        if text is None:
            text = self.tokenizer.decode(generated_ids_tuple, skip_special_tokens=True)

        state, branch, slot_ids = self._snapshot_for_text(text)
        return PolicyStateDecodingContext(
            generated_ids=generated_ids_tuple,
            text=text,
            state=state,
            branch=branch,
            slot_ids=slot_ids,
        )

    def _snapshot_for_text(
        self,
        text: str,
    ) -> tuple[PolicySQLCurrentState, GrammarBranch, tuple[int, ...]]:
        state = self.state_parser.get_current_state(text)
        branch = self.grammar.allowed_branches(text, state)
        slot_ids = tuple(self._slot_token_ids(text, state))
        return state, branch, slot_ids

    def build_debug_info_from_context(
        self,
        context: PolicyStateDecodingContext,
        *,
        failure_reason: str,
        **extra: object,
    ) -> dict[str, object]:
        return self._build_debug_info(
            failure_reason=failure_reason,
            text=context.text,
            state=context.state,
            branch=context.branch,
            slot_ids=list(context.slot_ids),
            **extra,
        )

    def _build_debug_info(
        self,
        *,
        failure_reason: str,
        text: str,
        state,
        branch,
        slot_ids: list[int],
        **extra: object,
    ) -> dict[str, object]:
        return {
            "decision_stage": "mask",
            "failure_reason": failure_reason,
            "unsupported_reason": state.unsupported_reason,
            "prefix": text,
            "slot_prefix": self.tokenizer.decode(slot_ids, skip_special_tokens=True),
            "clause": state.clause,
            "role": state.role,
            "predicate_context": state.predicate_context,
            "tables_in_scope": list(state.tables_in_scope),
            "current_agg": state.current_agg,
            "identifier_mode": state.identifier_mode,
            "table_mode": state.table_mode,
            "query_depth": state.query_depth,
            "in_string_literal": state.in_string_literal,
            "branch": {
                "structural_literals": list(branch.structural_literals),
                "allow_column_slot": branch.allow_column_slot,
                "allow_table_slot": branch.allow_table_slot,
                "allow_where_value_slot": branch.allow_where_value_slot,
                "allow_limit_value_slot": branch.allow_limit_value_slot,
                "allow_eos": branch.allow_eos,
            },
            **extra,
        }
