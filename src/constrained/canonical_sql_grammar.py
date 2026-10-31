
from __future__ import annotations

import re
from dataclasses import dataclass

from .sql_state_parser import PolicySQLCurrentState

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_QUOTED_IDENT = r"`[^`]+`"
_IDENT_PART = rf"(?:{_IDENT}|{_QUOTED_IDENT})"
_QUALIFIED_IDENT = rf"{_IDENT_PART}\.{_IDENT_PART}"
_IDENTIFIER_REF = rf"(?:{_QUALIFIED_IDENT}|{_IDENT_PART})"
_NUMBER = r"-?\d+(?:\.\d+)?"
_STRING = r"'(?:''|[^'])*'"
_BOOLEAN_OR_NULL = r"(?:TRUE|FALSE|NULL)"
_SIMPLE_VALUE = rf"(?:{_QUALIFIED_IDENT}|{_NUMBER}|{_STRING}|{_BOOLEAN_OR_NULL})"
_COMPARISON_OP = r"(?:=|!=|<>|<=|>=|<|>)"

# Re-exported for callers that pull constants from `canonical_sql_grammar`.
from .grammar_constants import (  # noqa: E402,F401
    CLAUSE_KEYWORD_CONTINUATIONS,
    RESERVED_IDENTIFIER_NAMES,
    SELECT_ENTRY_KEYWORDS,
    SELECT_MODIFIERS,
    SELECT_MODIFIER_ALTERNATION,
)
_JOIN_PREDICATE = re.compile(
    rf"{_QUALIFIED_IDENT}\s*=\s*{_QUALIFIED_IDENT}\s*$",
    re.IGNORECASE,
)
_JOIN_LHS = re.compile(rf"{_QUALIFIED_IDENT}\s*$", re.IGNORECASE)

@dataclass(frozen=True)
class GrammarBranch:

    structural_literals: tuple[str, ...] = ()
    allow_column_slot: bool = False
    allow_table_slot: bool = False
    allow_where_value_slot: bool = False
    allow_limit_value_slot: bool = False
    allow_eos: bool = False

class CanonicalSQLGrammar:

    def allowed_branches(self, text: str, state: PolicySQLCurrentState) -> GrammarBranch:
        stripped = text.strip()
        leading = text.lstrip()

        if not stripped:
            return GrammarBranch(structural_literals=("SELECT ",))

        if (
            " " not in leading
            and 0 < len(leading) < len("SELECT")
            and self._starts_with_partial_keyword(leading.upper(), "SELECT")
        ):
            return GrammarBranch(structural_literals=("SELECT ",))

        if self._at_subquery_select_start(text):
            return GrammarBranch(structural_literals=("SELECT ",))

        if self._at_scalar_subquery_select_start(text):
            return GrammarBranch(structural_literals=("SELECT ",))

        if (
            state.clause == "SELECT"
            and state.role == "Sel"
            and self._at_completed_select_expr(text)
        ):
            return GrammarBranch(
                structural_literals=self.select_exit_literals(text),
                allow_column_slot=state.identifier_mode,
            )

        if self._at_select_entry(state):
            literals = [
                self.literal(text, "AVG("),
                self.literal(text, "MAX("),
                self.literal(text, "MIN("),
                self.literal(text, "SUM("),
                self.literal(text, "COUNT(*)"),
            ]
            if not self._select_projection_has_modifier(text):
                for modifier in SELECT_MODIFIERS:
                    literals.append(self.literal(text, f"{modifier} "))
            if state.query_depth > 0:
                literals.append(self.literal(text, "1"))
            return GrammarBranch(
                structural_literals=tuple(literals),
                allow_column_slot=True,
            )

        if state.clause == "FROM" and self._at_completed_from_table(text):
            return GrammarBranch(
                structural_literals=self._from_exit_literals(text, state.query_depth > 0),
                allow_table_slot=state.table_mode,
                allow_eos=state.query_depth == 0,
            )

        if state.clause == "JOIN" and self._at_completed_join_table(text):
            return GrammarBranch(
                structural_literals=(self._keyword_literal(text, "ON"),),
                allow_table_slot=state.table_mode,
            )

        if self._at_table_entry(state):
            return GrammarBranch(allow_table_slot=True)

        partial_branch = self._continuing_structural_literal_branch(text, state)
        if partial_branch is not None:
            return partial_branch

        if self._at_completed_join_predicate(text):
            return GrammarBranch(
                structural_literals=self.join_predicate_exit_literals(
                    text,
                    allow_close=state.query_depth > 0,
                ),
                allow_column_slot=state.identifier_mode,
                allow_eos=state.query_depth == 0,
            )

        if self._at_completed_join_lhs(text):
            return GrammarBranch(
                structural_literals=self.operator_literals(text, ("=",)),
                allow_column_slot=state.identifier_mode,
            )

        if state.role == "Join":
            return GrammarBranch(allow_column_slot=state.identifier_mode)

        if (
            state.predicate_context in {"WHERE", "HAVING"}
            and self._at_completed_predicate(text, state)
        ):
            return GrammarBranch(
                structural_literals=self._predicate_exit_literals(
                    text,
                    state,
                    allow_close=state.query_depth > 0,
                ),
                # A numeric predicate value can be extended with more digits;
                # keep the value slot open alongside structural exits.
                allow_where_value_slot=self._predicate_value_is_open_numeric(text, state),
                allow_eos=state.query_depth == 0,
            )

        if state.predicate_context in {"WHERE", "HAVING"} and self._at_completed_predicate_lhs(
            text,
            state,
        ):
            return GrammarBranch(
                structural_literals=self.operator_literals(
                    text,
                    ("=", "!=", "<>", "<", ">", "<=", ">="),
                ),
                allow_column_slot=state.identifier_mode,
            )

        if self.at_predicate_rhs(text, state):
            return GrammarBranch(allow_where_value_slot=True)

        if state.predicate_context == "WHERE" and state.role == "Where":
            structural_literals = ()
            if state.query_depth == 0:
                structural_literals = self._predicate_entry_literals(text)
            return GrammarBranch(
                structural_literals=structural_literals,
                allow_column_slot=state.identifier_mode,
            )

        if state.predicate_context == "HAVING" and state.role == "Where":
            return GrammarBranch(structural_literals=self._having_entry_literals(text))

        if state.role == "Agg":
            if state.predicate_context == "HAVING" and state.current_agg == "COUNT":
                return GrammarBranch(structural_literals=("*)",))
            if state.current_agg == "COUNT":
                if self.at_completed_count_arg(text):
                    return GrammarBranch(
                        structural_literals=(")",),
                        allow_column_slot=state.identifier_mode,
                    )
                return GrammarBranch(
                    structural_literals=("*)",),
                    allow_column_slot=state.identifier_mode,
                )
            if self.at_completed_unary_agg_arg(text):
                return GrammarBranch(
                    structural_literals=(")",),
                    allow_column_slot=state.identifier_mode,
                )
            return GrammarBranch(allow_column_slot=state.identifier_mode)

        if state.clause == "GROUP_BY" and self._at_completed_group_by_expr(text):
            return GrammarBranch(
                structural_literals=self.group_by_exit_literals(text),
                allow_column_slot=state.identifier_mode,
                allow_eos=state.query_depth == 0,
            )

        if self._at_group_by_entry(state):
            return GrammarBranch(allow_column_slot=state.identifier_mode)

        if state.clause == "ORDER_BY" and self._at_completed_order_by_direction(text):
            return GrammarBranch(
                structural_literals=self._order_by_direction_exit_literals(text),
                allow_eos=state.query_depth == 0,
            )

        if state.clause == "ORDER_BY" and self._at_completed_order_by_expr(text):
            return GrammarBranch(
                structural_literals=self.order_by_exit_literals(text),
                allow_column_slot=state.identifier_mode,
                allow_eos=state.query_depth == 0,
            )

        if self._at_order_by_entry(state):
            return GrammarBranch(allow_column_slot=state.identifier_mode)

        if self.at_completed_unary_agg_arg(text):
            return GrammarBranch(structural_literals=(")",))

        if self.at_completed_count_arg(text):
            return GrammarBranch(structural_literals=(")",))

        if state.clause == "LIMIT" and self.at_completed_limit_value(text):
            return GrammarBranch(
                structural_literals=(")",) if state.query_depth > 0 else (),
                # `LIMIT 1` can extend to `LIMIT 10`, so keep the value slot open.
                allow_limit_value_slot=True,
                allow_eos=state.query_depth == 0,
            )

        if self._at_limit_entry(state):
            return GrammarBranch(allow_limit_value_slot=True)

        return GrammarBranch()

    def all_literals(self) -> set[str]:
        literals = {
            "SELECT ",
            "AVG(",
            " AVG(",
            "MAX(",
            " MAX(",
            "MIN(",
            " MIN(",
            "SUM(",
            " SUM(",
            "COUNT(*)",
            " COUNT(*)",
            "1",
            " 1",
            ", ",
            "FROM ",
            " FROM ",
            "JOIN ",
            " JOIN ",
            "WHERE ",
            " WHERE ",
            "GROUP BY ",
            " GROUP BY ",
            "HAVING ",
            " HAVING ",
            "ORDER BY ",
            " ORDER BY ",
            "LIMIT ",
            " LIMIT ",
            "ON ",
            " ON ",
            "AND ",
            " AND ",
            "OR ",
            " OR ",
            "ASC",
            " ASC",
            "DESC",
            " DESC",
            "EXISTS (",
            " EXISTS (",
            "NOT EXISTS (",
            " NOT EXISTS (",
            "(",
            " (",
            "TRUE",
            " TRUE",
            "FALSE",
            " FALSE",
            "NULL",
            " NULL",
            "=",
            "= ",
            " = ",
            "!=",
            "!= ",
            " != ",
            "<>",
            "<> ",
            " <> ",
            "<",
            "< ",
            " < ",
            ">",
            "> ",
            " > ",
            "<=",
            "<= ",
            " <= ",
            ">=",
            ">= ",
            " >= ",
            "*)",
            ")",
            ";",
        }
        # SELECT modifiers (e.g. DISTINCT) — registered in both
        # leading-space and no-leading-space variants because the literal
        # _literal helper picks one or the other based on the preceding text.
        for modifier in SELECT_MODIFIERS:
            literals.add(f"{modifier} ")
            literals.add(f" {modifier} ")
        return literals

    def _at_select_entry(self, state: PolicySQLCurrentState) -> bool:
        return state.role == "Sel" and state.clause == "SELECT"

    def _at_table_entry(self, state: PolicySQLCurrentState) -> bool:
        return state.table_mode

    def _at_group_by_entry(self, state: PolicySQLCurrentState) -> bool:
        return state.clause == "GROUP_BY" and state.identifier_mode

    def _at_order_by_entry(self, state: PolicySQLCurrentState) -> bool:
        return state.clause == "ORDER_BY" and state.identifier_mode

    def _at_limit_entry(self, state: PolicySQLCurrentState) -> bool:
        return state.clause == "LIMIT"

    def _at_subquery_select_start(self, text: str) -> bool:
        stripped = text.rstrip()
        upper = stripped.upper()
        return upper.endswith("EXISTS (") or upper.endswith("NOT EXISTS (")

    def _at_scalar_subquery_select_start(self, text: str) -> bool:
        trimmed = text.rstrip()
        if not trimmed.endswith("("):
            return False
        return bool(re.search(rf"{_COMPARISON_OP}\s*\($", trimmed, re.IGNORECASE))

    def _at_completed_select_expr(self, text: str) -> bool:
        trimmed = text.rstrip()
        if re.search(r"\bCOUNT\s*\(\s*\*\s*\)\s*$", trimmed, re.IGNORECASE):
            return True
        count_distinct_ref = self.count_distinct_expr_reference(trimmed)
        if count_distinct_ref is not None:
            return self._identifier_reference_allowed(count_distinct_ref)
        if re.search(r"(?:^|[\s,(])1\s*$", trimmed):
            return True

        agg_match = re.search(
            rf"\b(?:AVG|SUM|MAX|MIN)\s*\(\s*({_IDENTIFIER_REF})\s*\)\s*$",
            trimmed,
            re.IGNORECASE,
        )
        if agg_match:
            return self._identifier_reference_allowed(agg_match.group(1))

        ref_match = re.search(rf"({_IDENTIFIER_REF})\s*$", trimmed, re.IGNORECASE)
        if not ref_match:
            return False
        candidate = ref_match.group(1).strip()
        if self._identifier_could_extend_to_keyword(candidate, SELECT_ENTRY_KEYWORDS):
            return False
        return self._identifier_reference_allowed(candidate)

    def _at_completed_from_table(self, text: str) -> bool:
        trimmed = text.rstrip()
        return bool(re.search(rf"\bFROM\s+{_IDENT}\s*$", trimmed, re.IGNORECASE))

    def _at_completed_join_table(self, text: str) -> bool:
        trimmed = text.rstrip()
        return bool(re.search(rf"\bJOIN\s+{_IDENT}\s*$", trimmed, re.IGNORECASE))

    def _at_completed_join_lhs(self, text: str) -> bool:
        fragment = self.current_clause_fragment(text, ("ON", "AND"))
        if fragment is None:
            return False
        upper = fragment.upper()
        if upper.startswith("NOT EXISTS") or upper.startswith("EXISTS"):
            return False
        if re.search(_COMPARISON_OP, fragment):
            return False
        return bool(_JOIN_LHS.fullmatch(fragment.rstrip()))

    def _at_completed_join_predicate(self, text: str) -> bool:
        fragment = self.current_clause_fragment(text, ("ON", "AND"))
        if fragment is None:
            return False
        return bool(_JOIN_PREDICATE.fullmatch(fragment.rstrip()))

    def _at_completed_where_lhs(self, text: str) -> bool:
        fragment = self.current_clause_fragment(text, ("WHERE", "AND", "OR"))
        if fragment is None:
            return False
        upper = fragment.upper()
        if upper.startswith("NOT EXISTS") or upper.startswith("EXISTS"):
            return False
        if re.search(_COMPARISON_OP, fragment):
            return False
        if self._identifier_could_extend_to_keyword(
            fragment.rstrip(),
            CLAUSE_KEYWORD_CONTINUATIONS[("WHERE", "Where")],
        ):
            return False
        return self._identifier_reference_allowed(fragment.rstrip())

    def _at_completed_where_predicate(self, text: str) -> bool:
        fragment = self.current_clause_fragment(text, ("WHERE", "AND", "OR"))
        if fragment is None:
            return False

        stripped = fragment.rstrip()
        upper = stripped.upper()
        if upper.startswith("NOT EXISTS") or upper.startswith("EXISTS"):
            return stripped.endswith(")") and self.balanced_parens(fragment) == 0
        scalar_subquery_match = re.search(
            rf"({_IDENTIFIER_REF})\s*{_COMPARISON_OP}\s*\(\s*SELECT\b",
            stripped,
            re.IGNORECASE,
        )
        if scalar_subquery_match:
            return (
                self._identifier_reference_allowed(scalar_subquery_match.group(1))
                and stripped.endswith(")")
                and self.balanced_parens(fragment) == 0
            )
        match = re.search(
            rf"({_IDENTIFIER_REF})\s*{_COMPARISON_OP}\s*({_SIMPLE_VALUE})\s*$",
            stripped,
            re.IGNORECASE,
        )
        return bool(match and self._identifier_reference_allowed(match.group(1)))

    def _at_completed_having_lhs(self, text: str) -> bool:
        fragment = self.current_clause_fragment(text, ("HAVING", "AND", "OR"))
        if fragment is None:
            return False
        stripped = fragment.rstrip()
        if re.search(_COMPARISON_OP, stripped):
            return False
        if re.fullmatch(r"COUNT\s*\(\s*\*\s*\)", stripped, re.IGNORECASE):
            return True
        match = re.fullmatch(
            rf"(AVG|SUM)\s*\(\s*({_IDENTIFIER_REF})\s*\)",
            stripped,
            re.IGNORECASE,
        )
        return bool(match and self._identifier_reference_allowed(match.group(2)))

    def _at_completed_having_predicate(self, text: str) -> bool:
        fragment = self.current_clause_fragment(text, ("HAVING", "AND", "OR"))
        if fragment is None:
            return False
        return bool(
            re.fullmatch(
                rf"(COUNT\s*\(\s*\*\s*\)|(?:AVG|SUM)\s*\(\s*({_IDENTIFIER_REF})\s*\))"
                rf"\s*{_COMPARISON_OP}\s*({_NUMBER})",
                fragment.rstrip(),
                re.IGNORECASE,
            )
        )

    def _at_where_rhs(self, text: str) -> bool:
        fragment = self.current_clause_fragment(text, ("WHERE", "AND", "OR"))
        if fragment is None:
            return False
        stripped = fragment.rstrip()
        if self._has_top_level_clause_boundary(stripped):
            return False
        if self._at_completed_where_predicate(text):
            return False
        return bool(re.search(_COMPARISON_OP, stripped))

    def _at_having_rhs(self, text: str) -> bool:
        fragment = self.current_clause_fragment(text, ("HAVING", "AND", "OR"))
        if fragment is None:
            return False
        stripped = fragment.rstrip()
        if self._has_top_level_clause_boundary(stripped):
            return False
        if self._at_completed_having_predicate(text):
            return False
        if not re.search(
            rf"^(COUNT\s*\(\s*\*\s*\)|(?:AVG|SUM)\s*\(\s*{_IDENTIFIER_REF}\s*\))\s*{_COMPARISON_OP}",
            stripped,
            re.IGNORECASE,
        ):
            return False
        return bool(re.search(_COMPARISON_OP, stripped))

    def at_completed_unary_agg_arg(self, text: str) -> bool:
        trimmed = text.rstrip()
        match = re.search(
            rf"\b(?:AVG|SUM|MAX|MIN)\s*\(\s*({_IDENTIFIER_REF})\s*$",
            trimmed,
            re.IGNORECASE,
        )
        return bool(match and self._identifier_reference_allowed(match.group(1)))

    def _at_completed_count_star(self, text: str) -> bool:
        trimmed = text.rstrip()
        return bool(re.search(r"\bCOUNT\s*\(\s*\*\s*$", trimmed, re.IGNORECASE))

    def _count_distinct_arg_reference(self, text: str) -> str | None:
        match = re.search(
            rf"\bCOUNT\s*\(\s*DISTINCT\s+({_IDENTIFIER_REF})\s*$",
            text.rstrip(),
            re.IGNORECASE,
        )
        return match.group(1) if match else None

    def count_distinct_expr_reference(self, text: str) -> str | None:
        match = re.search(
            rf"\bCOUNT\s*\(\s*DISTINCT\s+({_IDENTIFIER_REF})\s*\)\s*$",
            text.rstrip(),
            re.IGNORECASE,
        )
        return match.group(1) if match else None

    def at_completed_count_arg(self, text: str) -> bool:
        if self._at_completed_count_star(text):
            return True
        count_distinct_ref = self._count_distinct_arg_reference(text)
        return bool(
            count_distinct_ref and self._identifier_reference_allowed(count_distinct_ref)
        )

    def _at_completed_group_by_expr(self, text: str) -> bool:
        trimmed = text.rstrip()
        match = re.search(
            rf"(?:GROUP\s+BY|,)\s*({_IDENTIFIER_REF})\s*$",
            trimmed,
            re.IGNORECASE,
        )
        return bool(match and self._identifier_reference_allowed(match.group(1)))

    def _at_completed_order_by_expr(self, text: str) -> bool:
        trimmed = text.rstrip()
        match = re.search(
            rf"(?:ORDER\s+BY|,)\s*({_IDENTIFIER_REF})\s*$",
            trimmed,
            re.IGNORECASE,
        )
        return bool(match and self._identifier_reference_allowed(match.group(1)))

    def _at_completed_order_by_direction(self, text: str) -> bool:
        trimmed = text.rstrip()
        match = re.search(
            rf"(?:ORDER\s+BY|,)\s*({_IDENTIFIER_REF})\s+(?:ASC|DESC)\s*$",
            trimmed,
            re.IGNORECASE,
        )
        return bool(match and self._identifier_reference_allowed(match.group(1)))

    def at_completed_limit_value(self, text: str) -> bool:
        trimmed = text.rstrip()
        return bool(re.search(r"\bLIMIT\s+-?\d+\s*$", trimmed, re.IGNORECASE))

    def _select_projection_has_modifier(self, text: str) -> bool:
        match = re.search(r"\bSELECT\s+(.*)$", text, re.IGNORECASE | re.DOTALL)
        if not match:
            return False
        tail = match.group(1)
        clause_end = re.search(
            r"\b(FROM|WHERE|GROUP\s+BY|ORDER\s+BY|LIMIT|HAVING)\b",
            tail,
            re.IGNORECASE,
        )
        if clause_end:
            tail = tail[: clause_end.start()]
        return bool(
            re.search(rf"\b(?:{SELECT_MODIFIER_ALTERNATION})\b", tail, re.IGNORECASE)
        )

    def _where_predicate_value_is_open_numeric(self, text: str) -> bool:
        fragment = self.current_clause_fragment(text, ("WHERE", "AND", "OR"))
        if fragment is None:
            return False
        match = re.search(
            rf"{_COMPARISON_OP}\s*(-?\d+(?:\.\d+)?)\s*$",
            fragment,
        )
        return bool(match)

    def _having_predicate_value_is_open_numeric(self, text: str) -> bool:
        fragment = self.current_clause_fragment(text, ("HAVING", "AND", "OR"))
        if fragment is None:
            return False
        match = re.search(
            rf"{_COMPARISON_OP}\s*(-?\d+(?:\.\d+)?)\s*$",
            fragment,
        )
        return bool(match)

    def _predicate_value_is_open_numeric(
        self,
        text: str,
        state: PolicySQLCurrentState,
    ) -> bool:
        if state.predicate_context == "HAVING":
            return self._having_predicate_value_is_open_numeric(text)
        if state.predicate_context == "WHERE":
            return self._where_predicate_value_is_open_numeric(text)
        return False

    def current_clause_fragment(
        self,
        text: str,
        separators: tuple[str, ...],
    ) -> str | None:
        target_depth = self.balanced_parens(text)
        separator_set = {token.upper() for token in separators}
        last_end: int | None = None
        depth = 0
        in_string = False
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
                i += 1
                continue

            if ch.isalpha() and depth == target_depth:
                start = i
                i += 1
                while i < len(text) and text[i].isalpha():
                    i += 1
                token = text[start:i].upper()
                if token in separator_set:
                    last_end = i
                continue

            i += 1

        if last_end is None:
            return None
        return text[last_end:].lstrip()

    def _has_top_level_clause_boundary(self, fragment: str) -> bool:
        depth = 0
        in_string = False
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
                i += 1
                continue

            if ch == "(":
                depth += 1
                i += 1
                continue

            if ch == ")":
                depth = max(depth - 1, 0)
                i += 1
                continue

            if depth == 0 and ch.isalpha():
                start = i
                i += 1
                while i < len(fragment) and fragment[i].isalpha():
                    i += 1
                token = fragment[start:i].upper()
                if start == 0:
                    continue
                if token in {"HAVING", "LIMIT", "JOIN", "WHERE"}:
                    return True
                if token in {"GROUP", "ORDER"}:
                    j = i
                    while j < len(fragment) and fragment[j].isspace():
                        j += 1
                    k = j
                    while k < len(fragment) and fragment[k].isalpha():
                        k += 1
                    if fragment[j:k].upper() == "BY":
                        return True
                continue

            i += 1

        return False

    def _predicate_entry_literals(self, text: str) -> tuple[str, ...]:
        return (
            self._keyword_literal(text, "EXISTS", trailing=" ("),
            self._keyword_literal(text, "NOT EXISTS", trailing=" ("),
        )

    def _having_entry_literals(self, text: str) -> tuple[str, ...]:
        return (
            self.literal(text, "AVG("),
            self.literal(text, "SUM("),
            self.literal(text, "COUNT(*)"),
        )

    def _at_completed_predicate_lhs(
        self,
        text: str,
        state: PolicySQLCurrentState,
    ) -> bool:
        if state.predicate_context == "WHERE":
            return self._at_completed_where_lhs(text)
        if state.predicate_context == "HAVING":
            return self._at_completed_having_lhs(text)
        return False

    def _at_completed_predicate(
        self,
        text: str,
        state: PolicySQLCurrentState,
    ) -> bool:
        if state.predicate_context == "WHERE":
            return self._at_completed_where_predicate(text)
        if state.predicate_context == "HAVING":
            return self._at_completed_having_predicate(text)
        return False

    def at_predicate_rhs(
        self,
        text: str,
        state: PolicySQLCurrentState,
    ) -> bool:
        if state.predicate_context == "WHERE":
            return self._at_where_rhs(text)
        if state.predicate_context == "HAVING":
            return self._at_having_rhs(text)
        return False

    def _continuing_structural_literal_branch(
        self,
        text: str,
        state: PolicySQLCurrentState,
    ) -> GrammarBranch | None:
        if state.table_mode:
            return None

        if state.role == "Where" and re.search(
            r"(WHERE|HAVING|AND|OR)\s*$",
            text,
            re.IGNORECASE,
        ):
            return None

        if state.role == "Join" and re.search(r"(ON|AND|=)\s*$", text, re.IGNORECASE):
            return None

        if state.clause in {"GROUP_BY", "ORDER_BY"} and state.identifier_mode:
            return None

        if state.clause == "LIMIT":
            return None

        candidates: list[str] = []
        boundary_specs = (
            (self._at_subquery_select_start, lambda prefix: ("SELECT ",)),
            (self._at_scalar_subquery_select_start, lambda prefix: ("SELECT ",)),
            (self._at_completed_select_expr, self.select_exit_literals),
            (
                self._at_completed_from_table,
                lambda prefix: self._from_exit_literals(prefix, state.query_depth > 0),
            ),
            (
                self._at_completed_join_predicate,
                lambda prefix: self.join_predicate_exit_literals(
                    prefix,
                    allow_close=state.query_depth > 0,
                ),
            ),
            (
                self._at_completed_where_predicate,
                lambda prefix: self._where_predicate_exit_literals(
                    prefix,
                    allow_close=state.query_depth > 0,
                ),
            ),
            (self._at_completed_group_by_expr, self.group_by_exit_literals),
            (self._at_completed_order_by_expr, self.order_by_exit_literals),
            (self._at_completed_order_by_direction, self._order_by_direction_exit_literals),
        )

        for checker, builder in boundary_specs:
            candidates.extend(self._continuing_literals_from_boundary(text, checker, builder))

        if not candidates:
            return None

        unique: list[str] = []
        seen: set[str] = set()
        for literal in candidates:
            if literal not in seen:
                seen.add(literal)
                unique.append(literal)
        return GrammarBranch(structural_literals=tuple(unique))

    def _continuing_literals_from_boundary(
        self,
        text: str,
        checker,
        builder,
    ) -> list[str]:
        matches: list[str] = []
        max_len = max((len(lit) for lit in self.all_literals()), default=0)
        max_suffix = min(len(text), max_len - 1)
        text_lower = text.lower()

        for suffix_len in range(1, max_suffix + 1):
            prefix_text = text[:-suffix_len]
            if not checker(prefix_text):
                continue

            suffix = text_lower[-suffix_len:]
            for literal in builder(prefix_text):
                literal_lower = literal.lower()
                if suffix_len < len(literal) and literal_lower.startswith(suffix):
                    matches.append(literal)

        return matches

    def select_exit_literals(self, text: str) -> tuple[str, ...]:
        return (", ", self._keyword_literal(text, "FROM"))

    def _from_exit_literals(self, text: str, allow_close: bool) -> tuple[str, ...]:
        literals = [
            self._keyword_literal(text, "JOIN"),
            self._keyword_literal(text, "WHERE"),
            self._keyword_literal(text, "GROUP BY"),
            self._keyword_literal(text, "ORDER BY"),
            self._keyword_literal(text, "LIMIT"),
        ]
        if allow_close:
            literals.append(")")
        return tuple(literals)

    def join_predicate_exit_literals(
        self,
        text: str,
        allow_close: bool,
    ) -> tuple[str, ...]:
        literals = [
            self._keyword_literal(text, "AND"),
            self._keyword_literal(text, "JOIN"),
            self._keyword_literal(text, "WHERE"),
            self._keyword_literal(text, "GROUP BY"),
            self._keyword_literal(text, "ORDER BY"),
            self._keyword_literal(text, "LIMIT"),
        ]
        if allow_close:
            literals.append(")")
        return tuple(literals)

    def _where_predicate_exit_literals(
        self,
        text: str,
        allow_close: bool,
    ) -> tuple[str, ...]:
        literals = [
            self._keyword_literal(text, "AND"),
            self._keyword_literal(text, "OR"),
            self._keyword_literal(text, "GROUP BY"),
            self._keyword_literal(text, "ORDER BY"),
            self._keyword_literal(text, "LIMIT"),
        ]
        if allow_close:
            literals.append(")")
        return tuple(literals)

    def _having_predicate_exit_literals(
        self,
        text: str,
        allow_close: bool,
    ) -> tuple[str, ...]:
        literals = [
            self._keyword_literal(text, "AND"),
            self._keyword_literal(text, "OR"),
            self._keyword_literal(text, "ORDER BY"),
            self._keyword_literal(text, "LIMIT"),
        ]
        if allow_close:
            literals.append(")")
        return tuple(literals)

    def _predicate_exit_literals(
        self,
        text: str,
        state: PolicySQLCurrentState,
        *,
        allow_close: bool,
    ) -> tuple[str, ...]:
        if state.predicate_context == "HAVING":
            return self._having_predicate_exit_literals(text, allow_close=allow_close)
        return self._where_predicate_exit_literals(text, allow_close=allow_close)

    def group_by_exit_literals(self, text: str) -> tuple[str, ...]:
        return (
            ", ",
            self._keyword_literal(text, "HAVING"),
            self._keyword_literal(text, "ORDER BY"),
            self._keyword_literal(text, "LIMIT"),
        )

    def order_by_exit_literals(self, text: str) -> tuple[str, ...]:
        return (
            ", ",
            self.literal(text, "ASC"),
            self.literal(text, "DESC"),
            self._keyword_literal(text, "LIMIT"),
        )

    def _order_by_direction_exit_literals(self, text: str) -> tuple[str, ...]:
        return (
            ", ",
            self._keyword_literal(text, "LIMIT"),
        )

    def operator_literals(
        self,
        text: str,
        operators: tuple[str, ...],
    ) -> tuple[str, ...]:
        return tuple(self._operator_literal(text, operator) for operator in operators)

    def _keyword_literal(
        self,
        text: str,
        keyword: str,
        trailing: str = " ",
    ) -> str:
        needs_leading = bool(text) and not text.endswith((" ", "\n", "\t", "("))
        prefix = " " if needs_leading else ""
        return f"{prefix}{keyword}{trailing}"

    def literal(self, text: str, literal: str) -> str:
        needs_leading = bool(text) and not text.endswith((" ", "\n", "\t", "("))
        prefix = " " if needs_leading else ""
        return f"{prefix}{literal}"

    def _operator_literal(self, text: str, operator: str) -> str:
        if text.endswith((" ", "\n", "\t")):
            return f"{operator} "
        return f" {operator} "

    @staticmethod
    def _identifier_reference_allowed(identifier: str) -> bool:
        stripped = identifier.strip()
        if not stripped:
            return False
        if "." in stripped:
            return bool(re.fullmatch(_QUALIFIED_IDENT, stripped, re.IGNORECASE))
        if re.fullmatch(_QUOTED_IDENT, stripped):
            return True
        return bool(re.fullmatch(_IDENT, stripped, re.IGNORECASE)) and (
            stripped.upper() not in RESERVED_IDENTIFIER_NAMES
        )

    @staticmethod
    def balanced_parens(text: str) -> int:
        depth = 0
        in_string = False
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
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            i += 1
        return depth

    @staticmethod
    def _starts_with_partial_keyword(text: str, keyword: str) -> bool:
        if len(text) > len(keyword):
            return False
        return keyword.startswith(text)

    @staticmethod
    def _identifier_could_extend_to_keyword(
        candidate: str,
        keywords: frozenset[str],
    ) -> bool:
        candidate_upper = candidate.strip().upper()
        if not candidate_upper or "." in candidate_upper:
            return False
        return any(
            keyword.startswith(candidate_upper) and candidate_upper != keyword
            for keyword in keywords
        )
