
from __future__ import annotations

__all__ = ["_PolicyStateTerminalValidationMixin"]

class _PolicyStateTerminalValidationMixin:

    def validate_terminal_output(
        self,
        output_text: str,
        *,
        terminated_by_eos: bool,
    ) -> dict[str, object] | None:
        normalized_output = self._normalize_terminal_output(output_text)
        context = self.build_context(generated_ids=(), text=normalized_output)
        slot_ids_list = list(context.slot_ids)
        failure_reason = self._terminal_failure_details(
            normalized_output,
            terminated_by_eos=terminated_by_eos,
            state=context.state,
            branch=context.branch,
            slot_ids=slot_ids_list,
        )
        if failure_reason is None:
            return None

        debug_info = self.build_debug_info_from_context(
            context,
            failure_reason=failure_reason,
        )
        debug_info["decision_stage"] = "decode"
        debug_info["terminated_by_eos"] = terminated_by_eos
        return debug_info

    @staticmethod
    def _normalize_terminal_output(output_text: str) -> str:
        normalized_output = output_text.rstrip()
        if normalized_output.endswith(";"):
            normalized_output = normalized_output[:-1].rstrip()
        return normalized_output

    def _terminal_failure_details(
        self,
        normalized_output: str,
        *,
        terminated_by_eos: bool,
        state,
        branch,
        slot_ids: list[int],
    ) -> str | None:
        can_exit_slot = self._can_exit_current_slot(normalized_output, state, branch, slot_ids)
        failure_reason = self._terminal_structural_failure(
            normalized_output,
            terminated_by_eos=terminated_by_eos,
            state=state,
            slot_ids=slot_ids,
        )
        if failure_reason is not None:
            return failure_reason

        if branch.allow_eos and can_exit_slot:
            return None

        return "eos_leakage" if terminated_by_eos else "nonterminal_decode_stop"

    def _terminal_structural_failure(
        self,
        normalized_output: str,
        *,
        terminated_by_eos: bool,
        state,
        slot_ids: list[int],
    ) -> str | None:
        if state.unsupported:
            return "unsupported_subset"

        if self._finalized_scope_select_list_has_out_of_scope_reference(
            normalized_output,
            state,
            final_output=True,
        ):
            return "out_of_scope_select_reference"

        if state.in_string_literal:
            return "unfinished_string_literal"
        if state.query_depth > 0 or self.grammar.balanced_parens(normalized_output) > 0:
            return "unfinished_subquery"
        if not terminated_by_eos:
            slot_failure = self._current_schema_slot_terminal_failure(
                normalized_output,
                state,
                slot_ids,
            )
            if slot_failure is not None:
                return slot_failure
        if self.grammar.at_predicate_rhs(normalized_output, state):
            return "unfinished_where_rhs"
        if (
            self._limit_value_slot_text(normalized_output).strip()
            and not self.grammar.at_completed_limit_value(normalized_output)
        ):
            return "unfinished_limit_value"
        if state.current_agg is not None:
            if state.current_agg == "COUNT":
                if not self.grammar.at_completed_count_arg(normalized_output):
                    return "unfinished_aggregate"
            elif not self.grammar.at_completed_unary_agg_arg(normalized_output):
                return "unfinished_aggregate"

        return None
