
from __future__ import annotations

import re
from dataclasses import dataclass

from .grammar_constants import SELECT_MODIFIER_ALTERNATION

__all__ = ["_PolicyStateSlotHelperMixin"]

@dataclass(frozen=True)
class _SchemaSlotStatus:

    incomplete: bool = False
    has_continuation: bool = False

class _PolicyStateSlotHelperMixin:

    def _slot_started(self, text: str, state) -> bool:
        return bool(self._slot_text(text, state).strip())

    def _table_slot_status(
        self,
        text: str,
        state,
        slot_ids: list[int],
        *,
        allowed_table_names: set[str] | frozenset[str] | None = None,
    ) -> _SchemaSlotStatus:
        if not state.table_mode or not self._slot_started(text, state):
            return _SchemaSlotStatus()

        tables = allowed_table_names if allowed_table_names is not None else self._all_table_names
        incomplete = not self.table_trie.is_complete_allowed(slot_ids, tables)
        if not incomplete:
            return _SchemaSlotStatus()

        return _SchemaSlotStatus(
            incomplete=True,
            has_continuation=bool(self.table_trie.get_allowed_token_ids(slot_ids, tables)),
        )

    def _table_slot_complete(
        self,
        slot_ids: list[int],
        *,
        allowed_table_names: set[str] | frozenset[str] | None = None,
    ) -> bool:
        tables = allowed_table_names if allowed_table_names is not None else self._all_table_names
        return self.table_trie.is_complete_allowed(slot_ids, tables)

    def _table_slot_token_ids(
        self,
        slot_ids: list[int],
        *,
        allowed_table_names: set[str] | frozenset[str] | None = None,
    ) -> set[int]:
        tables = allowed_table_names if allowed_table_names is not None else self._all_table_names
        if self._table_slot_complete(slot_ids, allowed_table_names=tables):
            return set()
        return self.table_trie.get_allowed_token_ids(slot_ids, tables)

    def _table_slot_incomplete(
        self,
        text: str,
        state,
        slot_ids: list[int],
        *,
        allowed_table_names: set[str] | frozenset[str] | None = None,
    ) -> bool:
        return self._table_slot_status(
            text,
            state,
            slot_ids,
            allowed_table_names=allowed_table_names,
        ).incomplete

    def _table_slot_has_continuation(
        self,
        text: str,
        state,
        slot_ids: list[int],
        *,
        allowed_table_names: set[str] | frozenset[str] | None = None,
    ) -> bool:
        return self._table_slot_status(
            text,
            state,
            slot_ids,
            allowed_table_names=allowed_table_names,
        ).has_continuation

    def _column_slot_complete(self, state, slot_ids: list[int]) -> bool:
        if state.role == "Agg" and state.current_agg == "COUNT":
            if self._count_star_token_ids(slot_ids):
                return False
            slot_text = self.tokenizer.decode(slot_ids, skip_special_tokens=True).lstrip()
            if slot_text == "*":
                return True

            allowed_columns = self._allowed_columns_for_current_slot(state)
            if not allowed_columns:
                return False
            return slot_text in self._count_distinct_variants(allowed_columns)

        allowed_columns = self._allowed_columns_for_current_slot(state)
        if not allowed_columns:
            return False

        return self.allowed_trie.is_complete_allowed(slot_ids, allowed_columns)

    def _column_slot_has_continuation(self, text: str, state, slot_ids: list[int]) -> bool:
        if state.role == "Agg" and state.current_agg == "COUNT":
            return bool(self._count_arg_token_ids(text, state, slot_ids))

        allowed_columns = self._allowed_columns_for_current_slot(state)
        if not allowed_columns:
            return False

        return bool(self.allowed_trie.get_allowed_token_ids(slot_ids, allowed_columns))

    def _identifier_slot_status(
        self,
        text: str,
        state,
        slot_ids: list[int],
    ) -> _SchemaSlotStatus:
        if not state.identifier_mode or not self._slot_started(text, state):
            return _SchemaSlotStatus()

        incomplete = not self._column_slot_complete(state, slot_ids)
        if not incomplete:
            return _SchemaSlotStatus()

        return _SchemaSlotStatus(
            incomplete=True,
            has_continuation=self._column_slot_has_continuation(text, state, slot_ids)
            or bool(self._textual_column_continuation_token_ids(text, state)),
        )

    def _identifier_slot_incomplete(self, text: str, state, slot_ids: list[int]) -> bool:
        return self._identifier_slot_status(text, state, slot_ids).incomplete

    def _identifier_slot_has_any_continuation(
        self,
        text: str,
        state,
        slot_ids: list[int],
    ) -> bool:
        return self._identifier_slot_status(text, state, slot_ids).has_continuation

    def _current_schema_slot_status(
        self,
        text: str,
        state,
        slot_ids: list[int],
        *,
        allowed_table_names: set[str] | frozenset[str] | None = None,
    ) -> _SchemaSlotStatus:
        if state.table_mode:
            return self._table_slot_status(
                text,
                state,
                slot_ids,
                allowed_table_names=allowed_table_names,
            )
        if state.identifier_mode:
            return self._identifier_slot_status(text, state, slot_ids)
        return _SchemaSlotStatus()

    def _current_schema_slot_incomplete(
        self,
        text: str,
        state,
        slot_ids: list[int],
        *,
        allowed_table_names: set[str] | frozenset[str] | None = None,
    ) -> bool:
        return self._current_schema_slot_status(
            text,
            state,
            slot_ids,
            allowed_table_names=allowed_table_names,
        ).incomplete

    def _current_schema_slot_has_continuation(
        self,
        text: str,
        state,
        slot_ids: list[int],
        *,
        allowed_table_names: set[str] | frozenset[str] | None = None,
    ) -> bool:
        return self._current_schema_slot_status(
            text,
            state,
            slot_ids,
            allowed_table_names=allowed_table_names,
        ).has_continuation

    def _current_schema_slot_terminal_failure(
        self,
        text: str,
        state,
        slot_ids: list[int],
    ) -> str | None:
        if not self._current_schema_slot_incomplete(text, state, slot_ids):
            return None
        if state.table_mode:
            return "partial_table_reference"
        if state.identifier_mode:
            return "partial_identifier"
        return None

    def _slot_completion_literals(self, text: str, state) -> tuple[str, ...]:
        if state.role == "Agg":
            return (")",)

        if state.role == "Sel" and state.clause == "SELECT":
            return self.grammar.select_exit_literals(text)

        if state.role == "Where":
            return self.grammar.operator_literals(
                text,
                ("=", "!=", "<>", "<", ">", "<=", ">="),
            )

        if state.role == "Join":
            fragment = self.grammar.current_clause_fragment(text, ("ON", "AND"))
            if fragment and "=" in fragment:
                return self.grammar.join_predicate_exit_literals(
                    text,
                    allow_close=state.query_depth > 0,
                )
            return self.grammar.operator_literals(text, ("=",))

        if state.clause == "GROUP_BY":
            return self.grammar.group_by_exit_literals(text)

        if state.clause == "ORDER_BY":
            return self.grammar.order_by_exit_literals(text)

        return ()

    def _column_slot_allows_eos(self, text: str, state) -> bool:
        if state.clause in {"GROUP_BY", "ORDER_BY"}:
            return state.query_depth == 0

        if state.role == "Join":
            fragment = self.grammar.current_clause_fragment(text, ("ON", "AND"))
            return bool(fragment and "=" in fragment and state.query_depth == 0)

        return False

    def _slot_token_ids(self, text: str, state) -> list[int]:
        slot_text = self._slot_text(text, state)
        return self._encode_identifier_slot(slot_text)

    def _encode_identifier_slot(self, slot_text: str) -> list[int]:
        normalized = self._normalize_slot_prefix(slot_text)
        if not normalized:
            return []
        return self.tokenizer.encode(normalized, add_special_tokens=False)

    @staticmethod
    def _normalize_slot_prefix(slot_text: str) -> str:
        if not slot_text:
            return ""
        stripped = slot_text.lstrip()
        if not stripped:
            return ""
        if stripped != slot_text:
            return f" {stripped}"
        return slot_text

    def _slot_text(self, text: str, state) -> str:
        if state.table_mode:
            match = re.search(
                r"(?:\bFROM\b|\bJOIN\b)(\s*(?:[A-Za-z_][A-Za-z0-9_]*)?)$",
                text,
                flags=re.IGNORECASE,
            )
            return match.group(1) if match else ""

        if state.clause == "GROUP_BY":
            match = re.search(
                r"(?:\bGROUP\s+BY\b|,)(\s*(?:[A-Za-z_][A-Za-z0-9_\.]*)?)$",
                text,
                flags=re.IGNORECASE,
            )
            return match.group(1) if match else ""

        if state.clause == "ORDER_BY":
            match = re.search(
                r"(?:\bORDER\s+BY\b|,)(\s*(?:[A-Za-z_][A-Za-z0-9_\.]*)?)$",
                text,
                flags=re.IGNORECASE,
            )
            return match.group(1) if match else ""

        if not state.identifier_mode or state.role is None:
            return ""

        if state.role == "Sel":
            # The optional modifier group skips past `SELECT DISTINCT ` so
            # that the "DISTINCT" keyword is not mistaken for a partial
            # column name. The modifier list comes from
            # `canonical_sql_grammar.SELECT_MODIFIERS`.
            match = re.search(
                rf"(?:\bSELECT\b(?:\s+(?:{SELECT_MODIFIER_ALTERNATION})\b)?|,)"
                r"(\s*(?:[A-Za-z_][A-Za-z0-9_\.]*)?)$",
                text,
                flags=re.IGNORECASE,
            )
            return match.group(1) if match else ""

        if state.role == "Where":
            match = re.search(
                r"(?:\bWHERE\b|\bHAVING\b|\bAND\b|\bOR\b)"
                r"(\s*(?:[A-Za-z_][A-Za-z0-9_\.]*)?)$",
                text,
                flags=re.IGNORECASE,
            )
            return match.group(1) if match else ""

        if state.role == "Join":
            match = re.search(
                r"(?:\bON\b|\bAND\b|=)(\s*(?:[A-Za-z_][A-Za-z0-9_\.]*)?)$",
                text,
                flags=re.IGNORECASE,
            )
            return match.group(1) if match else ""

        if state.role == "Agg":
            if state.current_agg == "COUNT":
                match = re.search(r"COUNT\s*\(([^()]*)$", text, flags=re.IGNORECASE)
                return match.group(1) if match else ""

            match = re.search(
                r"(?:AVG|SUM|MAX|MIN)\s*\((\s*(?:[A-Za-z_][A-Za-z0-9_\.]*)?)$",
                text,
                flags=re.IGNORECASE,
            )
            return match.group(1) if match else ""

        return ""

    def _where_value_slot_text(self, text: str, state=None) -> str:
        separators = ("WHERE", "AND", "OR")
        if state is not None and state.predicate_context == "HAVING":
            separators = ("HAVING", "AND", "OR")

        fragment = self.grammar.current_clause_fragment(text, separators)
        if not fragment:
            return ""

        match = re.search(r"(?:!=|<>|<=|>=|=|<|>)(.*)$", fragment)
        return match.group(1) if match else ""

    def _limit_value_slot_text(self, text: str) -> str:
        match = re.search(r"\bLIMIT(\s*-?\d*(?:\.\d*)?)$", text, flags=re.IGNORECASE)
        return match.group(1) if match else ""
