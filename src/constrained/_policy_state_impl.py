
from __future__ import annotations

import re
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._policy_state_context import _PolicyStateContextMixin
from ._identifier_variants import normalize_identifier_reference
from ._policy_state_scope import _PolicyStateSelectScopeMixin
from ._policy_state_slots import _PolicyStateSlotHelperMixin
from ._policy_state_terminal import _PolicyStateTerminalValidationMixin
from ._policy_state_tokens import _PolicyStateTokenHelperMixin
from .allowed_columns import allowed_columns_for_role
from .allowed_tables import allowed_tables
from .allowed_trie import AllowedContinuationTrie
from .canonical_sql_grammar import (
    _IDENTIFIER_REF,
    CanonicalSQLGrammar,
)
from .literal_trie import LiteralContinuationTrie
from .sql_state_parser import PolicySQLStateParser
from .table_trie import TableContinuationTrie

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

__all__ = ["PolicyStateAdmissibilityRuntime"]

@dataclass(frozen=True)
class _RuntimeBranchAssembly:

    structural_literals: frozenset[str]
    allow_eos: bool
    can_exit_slot: bool
    force_where_value_slot: bool
    allowed_tables: frozenset[str] | None = None

class PolicyStateAdmissibilityRuntime(
    _PolicyStateTerminalValidationMixin,
    _PolicyStateContextMixin,
    _PolicyStateSelectScopeMixin,
    _PolicyStateTokenHelperMixin,
    _PolicyStateSlotHelperMixin,
):

    _SHARED_DECODED_TOKEN_CACHE: "weakref.WeakKeyDictionary[object, dict[int, str]]" = (
        weakref.WeakKeyDictionary()
    )
    _SHARED_TOKEN_MATCH_CACHE: "weakref.WeakKeyDictionary[object, dict[str, set[int]]]" = (
        weakref.WeakKeyDictionary()
    )

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        allowed_trie: AllowedContinuationTrie,
        table_trie: TableContinuationTrie,
        column_policies: dict[str, str],
        db_schema: dict,
    ):
        self.tokenizer = tokenizer
        self.allowed_trie = allowed_trie
        self.table_trie = table_trie
        self.column_policies = column_policies
        self.db_schema = db_schema
        self._all_table_names = allowed_tables(self.db_schema)
        self.state_parser = PolicySQLStateParser(db_schema)
        self.grammar = CanonicalSQLGrammar()
        self.literal_trie = LiteralContinuationTrie(tokenizer, self.grammar.all_literals())
        self._unique_bare_columns = self._compute_unique_bare_columns(db_schema)
        self._single_char_token_cache: dict[str, set[int]] = {}
        self._decoded_token_cache = self._SHARED_DECODED_TOKEN_CACHE.setdefault(tokenizer, {})
        self._token_match_cache = self._SHARED_TOKEN_MATCH_CACHE.setdefault(tokenizer, {})

    def _eos_allowed_token_ids(self) -> set[int]:
        eos_id = self.tokenizer.eos_token_id
        return {eos_id} if eos_id is not None else set()

    def _collect_allowed_token_ids(
        self,
        generated_ids: list[int],
        text: str,
        state,
        branch,
        slot_ids: list[int],
    ) -> set[int]:
        allowed: set[int] = set()
        restricted = False
        assembly = self._assemble_runtime_branch(text, state, branch, slot_ids)

        if assembly.allow_eos and assembly.can_exit_slot:
            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None:
                allowed.add(eos_id)
            allowed |= self.literal_trie.get_allowed_token_ids(generated_ids, {";"})
            restricted = True

        if assembly.structural_literals and assembly.can_exit_slot:
            allowed |= self.literal_trie.get_allowed_token_ids(
                generated_ids,
                assembly.structural_literals,
            )
            restricted = True

        if assembly.allowed_tables is not None:
            allowed |= self._table_slot_token_ids(
                slot_ids,
                allowed_table_names=assembly.allowed_tables,
            )
            restricted = True

        if branch.allow_column_slot and state.identifier_mode:
            allowed |= self._column_slot_token_ids(text, state, slot_ids)
            allowed |= self._aggregate_entry_token_ids(text, state)
            restricted = True

        if state.identifier_mode:
            keyword_token_ids = self._identifier_slot_keyword_continuation_token_ids(
                generated_ids,
                text,
                state,
            )
            if keyword_token_ids:
                allowed |= keyword_token_ids
                restricted = True

        if branch.allow_where_value_slot or assembly.force_where_value_slot:
            allowed |= self._where_value_token_ids(generated_ids, text, state)
            restricted = True

        if branch.allow_limit_value_slot:
            allowed |= self._limit_value_token_ids(text)
            restricted = True

        return allowed if restricted else set()

    def _assemble_runtime_branch(
        self,
        text: str,
        state,
        branch,
        slot_ids: list[int],
    ) -> _RuntimeBranchAssembly:
        scope_controls = self._current_scope_table_controls(text, state)
        structural_literals = set(
            scope_controls.filter_structural_literals(tuple(branch.structural_literals)),
        )
        allow_eos = branch.allow_eos and not scope_controls.blocks_scope_freeze

        if (
            branch.allow_column_slot
            and state.identifier_mode
            and self._slot_started(text, state)
            and self._column_slot_complete(state, slot_ids)
        ):
            structural_literals.update(self._slot_completion_literals(text, state))
            allow_eos = allow_eos or self._column_slot_allows_eos(text, state)

        force_where_value_slot = self._blocked_where_value_identifier_exit(text, state)
        allowed_tables = None
        if branch.allow_table_slot and state.table_mode:
            allowed_tables = frozenset(scope_controls.allowed_table_names(self._all_table_names))

        return _RuntimeBranchAssembly(
            structural_literals=frozenset(structural_literals),
            allow_eos=allow_eos,
            can_exit_slot=self._can_exit_current_slot(
                text,
                state,
                branch,
                slot_ids,
                force_where_value_slot=force_where_value_slot,
            ),
            force_where_value_slot=force_where_value_slot,
            allowed_tables=allowed_tables,
        )

    def _can_exit_current_slot(
        self,
        text: str,
        state,
        branch,
        slot_ids: list[int],
        *,
        force_where_value_slot: bool | None = None,
    ) -> bool:
        slot_text = self._slot_text(text, state)
        if slot_text and any(
            literal.startswith(slot_text) for literal in branch.structural_literals
        ):
            return True

        if force_where_value_slot is None:
            force_where_value_slot = self._blocked_where_value_identifier_exit(text, state)
        if force_where_value_slot:
            return False

        if self._current_schema_slot_incomplete(text, state, slot_ids):
            return False

        return True

    def _columns_for_role(
        self,
        role: str,
        *,
        state=None,
        tables_in_scope: tuple[str, ...] | None = None,
        current_agg: str | None = None,
    ) -> set[str]:
        if tables_in_scope is None:
            tables_in_scope = () if state is None else state.tables_in_scope

        return allowed_columns_for_role(
            role=role,
            column_policies=self.column_policies,
            db_schema=self.db_schema,
            tables_in_scope=tables_in_scope,
            current_agg=current_agg,
        )

    def _allowed_columns_for_current_slot(self, state) -> set[str]:
        if state.role is None and state.clause in {"GROUP_BY", "ORDER_BY"}:
            # Bare GROUP BY / ORDER BY identifiers are not first-class roles in
            # the benchmark, but when they appear inside WHERE subqueries the
            # checker walks them as predicate context. Keep both clauses on the
            # same policy-safe column set as WHERE.
            return self._policy_safe_clause_columns(state)
        if state.role is None:
            return set()

        return self._columns_for_role(
            state.role,
            state=state,
            current_agg=state.current_agg,
        )

    def _identifier_slot_entry_keyword_literals(self, text: str, state) -> frozenset[str]:
        if not state.identifier_mode:
            return frozenset()

        slot_text = self._slot_text(text, state)
        slot_prefix = slot_text.lstrip()
        if not slot_prefix or not text.endswith(slot_text):
            return frozenset()

        slot_entry_text = text[: -len(slot_text)]
        slot_entry_state = self.state_parser.get_current_state(slot_entry_text)
        slot_entry_branch = self.grammar.allowed_branches(slot_entry_text, slot_entry_state)
        if not slot_entry_branch.structural_literals:
            return frozenset()

        slot_prefix_upper = slot_prefix.upper()
        return frozenset(
            literal
            for literal in slot_entry_branch.structural_literals
            if literal.lstrip()
            and literal.lstrip()[0].isalpha()
            and literal.lstrip().upper().startswith(slot_prefix_upper)
        )

    def _identifier_slot_keyword_continuation_token_ids(
        self,
        generated_ids: list[int] | tuple[int, ...],
        text: str,
        state,
    ) -> set[int]:
        keyword_literals = self._identifier_slot_entry_keyword_literals(text, state)
        if not keyword_literals:
            return set()
        return self.literal_trie.get_allowed_token_ids(
            list(generated_ids),
            set(keyword_literals),
        )

    def _reference_allowed_for_role(
        self,
        reference: str,
        *,
        role: str,
        state,
        current_agg: str | None = None,
    ) -> bool:
        allowed_columns = self._columns_for_role(
            role,
            state=state,
            current_agg=current_agg,
        )
        return self._reference_in_allowed_columns(reference, allowed_columns)

    def _policy_safe_clause_columns(self, state) -> set[str]:
        return self._columns_for_role(
            "Where",
            state=state,
            current_agg=state.current_agg,
        )

    def _blocked_where_value_identifier_exit(self, text: str, state) -> bool:
        if state.role != "Where" or state.predicate_context != "WHERE":
            return False

        slot_text = self._where_value_slot_text(text, state)
        stripped = slot_text.strip()
        if not stripped or not re.fullmatch(_IDENTIFIER_REF, stripped, re.IGNORECASE):
            return False

        if self._reference_allowed_for_role(stripped, role="Where", state=state):
            return False

        return self._reference_has_allowed_continuation(
            stripped,
            role="Where",
            state=state,
        )

    def _reference_has_allowed_continuation(
        self,
        reference: str,
        *,
        role: str,
        state,
        current_agg: str | None = None,
    ) -> bool:
        allowed_columns = self._columns_for_role(
            role,
            state=state,
            current_agg=current_agg,
        )
        if not allowed_columns:
            return False

        slot_ids = self.tokenizer.encode(reference.strip(), add_special_tokens=False)
        if not slot_ids:
            return False
        if self.allowed_trie.is_complete_allowed(slot_ids, allowed_columns):
            return False
        if self.allowed_trie.get_allowed_token_ids(slot_ids, allowed_columns):
            return True

        stripped_reference = reference.strip()
        return any(
            variant.lstrip().startswith(stripped_reference)
            and variant.lstrip() != stripped_reference
            for variant in self._allowed_column_variants(allowed_columns)
        )

    def _reference_in_allowed_columns(
        self,
        reference: str,
        allowed_columns: set[str],
    ) -> bool:
        token_ids = self.tokenizer.encode(reference.strip(), add_special_tokens=False)
        if token_ids and self.allowed_trie.is_complete_allowed(token_ids, allowed_columns):
            return True

        normalized_reference = normalize_identifier_reference(reference)
        if not normalized_reference:
            return False

        for full_name in allowed_columns:
            _, _, column_name = full_name.partition(".")
            if normalized_reference == full_name.lower():
                return True
            if (
                column_name.lower() in self._unique_bare_columns
                and normalized_reference == column_name.lower()
            ):
                return True

        return False

    @staticmethod
    def _identifier_case_variants(identifier: str) -> tuple[str, ...]:
        lowered = identifier.lower()
        if lowered == identifier:
            return (identifier,)
        return (identifier, lowered)
