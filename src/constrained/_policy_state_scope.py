
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ._identifier_variants import normalize_identifier_reference
from .canonical_sql_grammar import _IDENTIFIER_REF
from .grammar_constants import SELECT_MODIFIER_ALTERNATION

__all__ = ["_PolicyStateSelectScopeMixin"]

# Pre-compiled regex derived from `canonical_sql_grammar.SELECT_MODIFIERS`.
# A leading SELECT modifier (e.g. DISTINCT) belongs to the SELECT itself,
# not to the first projection expression; the scope analyzer must drop it
# before classifying the expression.
_SELECT_MODIFIER_PREFIX = re.compile(
    rf"^\s*(?:{SELECT_MODIFIER_ALTERNATION})\s+",
    re.IGNORECASE,
)

def _strip_select_modifier_prefix(expression: str) -> str:
    return _SELECT_MODIFIER_PREFIX.sub("", expression, count=1)

@dataclass(frozen=True)
class _ScopeTableControls:

    missing_tables: frozenset[str] | None = None
    current_scope_has_tables: bool = False

    @property
    def blocks_scope_freeze(self) -> bool:
        return self.missing_tables is not None and self.current_scope_has_tables

    def allowed_table_names(
        self,
        all_table_names: set[str] | frozenset[str],
    ) -> set[str]:
        if self.missing_tables is None:
            return set(all_table_names)
        return set(all_table_names) & set(self.missing_tables)

    def filter_structural_literals(self, structural_literals: tuple[str, ...]) -> tuple[str, ...]:
        if not self.blocks_scope_freeze:
            return structural_literals
        return tuple(
            literal for literal in structural_literals if literal.strip().upper().startswith("JOIN")
        )

class _PolicyStateSelectScopeMixin:

    def _finalized_scope_select_list_has_out_of_scope_reference(
        self,
        text: str,
        state,
        *,
        final_output: bool,
    ) -> bool:
        current_depth = state.query_depth

        for scope in self._select_scopes(text):
            if scope.from_start is None or not scope.tables:
                continue

            if scope.closed:
                finalized = True
            elif scope.depth < current_depth:
                finalized = True
            elif scope.depth == current_depth:
                finalized = final_output or self._current_scope_select_references_frozen(state)
            else:
                finalized = False

            if not finalized:
                continue

            select_list = text[scope.select_end : scope.from_start]
            for expr in self._split_top_level_csv(select_list):
                if not self._select_expr_allowed_for_scope(expr, tuple(scope.tables)):
                    return True

        return False

    @staticmethod
    def _current_scope_select_references_frozen(state) -> bool:
        if state.table_mode or state.clause in {None, "SELECT", "FROM", "JOIN", "ON"}:
            return False
        return state.role != "Join"

    def _select_expr_allowed_for_scope(
        self,
        expression: str,
        current_scope_tables: tuple[str, ...],
    ) -> bool:
        expr = _strip_select_modifier_prefix(expression).strip()
        if not expr:
            return False
        if re.fullmatch(r"1", expr):
            return True
        if re.fullmatch(r"COUNT\s*\(\s*\*\s*\)", expr, re.IGNORECASE):
            return True
        count_distinct_ref = self.grammar.count_distinct_expr_reference(expr)
        if count_distinct_ref is not None:
            allowed_columns = self._columns_for_role(
                "Agg",
                tables_in_scope=current_scope_tables,
                current_agg="COUNT",
            )
            return self._reference_in_allowed_columns(
                count_distinct_ref,
                allowed_columns,
            )

        agg_match = re.fullmatch(
            rf"(AVG|SUM|MAX|MIN)\s*\(\s*({_IDENTIFIER_REF})\s*\)",
            expr,
            re.IGNORECASE,
        )
        if agg_match:
            allowed_columns = self._columns_for_role(
                "Agg",
                tables_in_scope=current_scope_tables,
                current_agg=agg_match.group(1).upper(),
            )
            return self._reference_in_allowed_columns(agg_match.group(2), allowed_columns)

        allowed_columns = self._columns_for_role(
            "Sel",
            tables_in_scope=current_scope_tables,
        )
        return self._reference_in_allowed_columns(expr, allowed_columns)

    def _current_select_scope(self, text: str, state) -> "_SelectScope | None":
        current_scope: _SelectScope | None = None
        for scope in self._select_scopes(text):
            if scope.closed or scope.depth != state.query_depth:
                continue
            if current_scope is None or scope.select_end >= current_scope.select_end:
                current_scope = scope
        return current_scope

    def _missing_select_tables_for_scope(
        self,
        text: str,
        scope: "_SelectScope | None",
    ) -> set[str] | None:
        if scope is None:
            return None

        select_end = scope.from_start if scope.from_start is not None else len(text)
        exprs = self._split_top_level_csv(text[scope.select_end : select_end])
        if not exprs:
            return None

        scope_tables = tuple(scope.tables)
        have_scope_tables = bool(scope_tables)
        missing: set[str] = set()

        for raw_expr in exprs:
            expr = _strip_select_modifier_prefix(raw_expr)
            stripped_expr = expr.strip()
            if re.fullmatch(r"1", stripped_expr):
                continue
            if re.fullmatch(r"COUNT\s*\(\s*\*\s*\)", stripped_expr, re.IGNORECASE):
                continue

            if have_scope_tables and self._select_expr_allowed_for_scope(expr, scope_tables):
                continue

            candidates = self._select_expr_candidate_tables(expr)
            if not candidates:
                return set()

            unresolved = candidates - set(scope_tables)
            if not unresolved and have_scope_tables:
                return set()
            missing |= unresolved or candidates

        return missing or None

    def _current_scope_table_controls(self, text: str, state) -> _ScopeTableControls:
        scope = self._current_select_scope(text, state)
        missing = self._missing_select_tables_for_scope(text, scope)
        return _ScopeTableControls(
            missing_tables=None if missing is None else frozenset(missing),
            current_scope_has_tables=bool(scope and scope.tables),
        )

    def _current_scope_missing_select_tables(self, text: str, state) -> set[str] | None:
        scope = self._current_select_scope(text, state)
        return self._missing_select_tables_for_scope(text, scope)

    def _select_expr_candidate_tables(self, expression: str) -> set[str]:
        expr = expression.strip()
        if not expr:
            return set()
        if re.fullmatch(r"1", expr):
            return set()
        if re.fullmatch(r"COUNT\s*\(\s*\*\s*\)", expr, re.IGNORECASE):
            return set()
        count_distinct_ref = self.grammar.count_distinct_expr_reference(expr)
        if count_distinct_ref is not None:
            return self._candidate_tables_for_reference(
                count_distinct_ref,
                role="Agg",
                current_agg="COUNT",
            )

        agg_match = re.fullmatch(
            rf"(AVG|SUM|MAX|MIN)\s*\(\s*({_IDENTIFIER_REF})\s*\)",
            expr,
            re.IGNORECASE,
        )
        if agg_match:
            return self._candidate_tables_for_reference(
                agg_match.group(2),
                role="Agg",
                current_agg=agg_match.group(1).upper(),
            )

        if re.fullmatch(_IDENTIFIER_REF, expr, re.IGNORECASE):
            return self._candidate_tables_for_reference(expr, role="Sel")

        return set()

    def _candidate_tables_for_reference(
        self,
        reference: str,
        *,
        role: str,
        current_agg: str | None = None,
    ) -> set[str]:
        normalized_reference = normalize_identifier_reference(reference)
        if not normalized_reference:
            return set()

        allowed_columns = self._columns_for_role(
            role,
            tables_in_scope=(),
            current_agg=current_agg,
        )

        candidates: set[str] = set()
        for full_name in allowed_columns:
            table_name, _, column_name = full_name.partition(".")
            if normalized_reference == full_name.lower():
                candidates.add(table_name)
            elif (
                "." not in normalized_reference
                and column_name.lower() in self._unique_bare_columns
                and normalized_reference == column_name.lower()
            ):
                candidates.add(table_name)
        return candidates

    def _allowed_tables_for_current_slot(self, text: str, state) -> set[str]:
        controls = self._current_scope_table_controls(text, state)
        return controls.allowed_table_names(self._all_table_names)

    def _filter_scope_freeze_exits(
        self,
        text: str,
        state,
        *,
        structural_literals: tuple[str, ...],
    ) -> tuple[str, ...]:
        controls = self._current_scope_table_controls(text, state)
        return controls.filter_structural_literals(structural_literals)

    def _select_scopes(self, text: str) -> list["_SelectScope"]:
        depth = 0
        in_string = False
        scopes: list[_SelectScope] = []
        completed: list[_SelectScope] = []
        i = 0

        while i < len(text):
            ch = text[i]
            if in_string:
                if ch == "'" and i + 1 < len(text) and text[i + 1] == "'":
                    i += 2
                    continue
                if ch == "'":
                    in_string = False
                i += 1
                continue

            if ch == "'":
                in_string = True
                i += 1
                continue

            if ch == "(":
                depth += 1
                i += 1
                continue

            if ch == ")":
                depth = max(depth - 1, 0)
                while scopes and scopes[-1].depth > depth:
                    scope = scopes.pop()
                    scope.closed = True
                    completed.append(scope)
                i += 1
                continue

            if ch.isalpha() or ch == "_":
                start = i
                i += 1
                while i < len(text) and (text[i].isalnum() or text[i] == "_"):
                    i += 1
                token = text[start:i]
                upper = token.upper()

                if upper == "SELECT":
                    scopes.append(_SelectScope(depth=depth, select_end=i))
                    continue

                current = scopes[-1] if scopes and scopes[-1].depth == depth else None
                if current is None:
                    continue

                if upper == "FROM":
                    if current.from_start is None:
                        current.from_start = start
                    current.awaiting_table = True
                    continue

                if upper == "JOIN":
                    current.awaiting_table = True
                    continue

                if upper == "AS":
                    continue

                if current.awaiting_table:
                    canonical = self.state_parser.canonical_table_name(token)
                    current.awaiting_table = False
                    if canonical and canonical not in current.seen:
                        current.seen.add(canonical)
                        current.tables.append(canonical)
                continue

            i += 1

        return completed + scopes

    @staticmethod
    def _split_top_level_csv(fragment: str) -> list[str]:
        parts: list[str] = []
        depth = 0
        in_string = False
        start = 0
        i = 0

        while i < len(fragment):
            ch = fragment[i]
            if in_string:
                if ch == "'" and i + 1 < len(fragment) and fragment[i + 1] == "'":
                    i += 2
                    continue
                if ch == "'":
                    in_string = False
                i += 1
                continue

            if ch == "'":
                in_string = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(depth - 1, 0)
            elif ch == "," and depth == 0:
                parts.append(fragment[start:i].strip())
                start = i + 1
            i += 1

        tail = fragment[start:].strip()
        if tail:
            parts.append(tail)
        return parts

@dataclass
class _SelectScope:

    depth: int
    select_end: int
    from_start: int | None = None
    tables: list[str] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)
    awaiting_table: bool = False
    closed: bool = False
