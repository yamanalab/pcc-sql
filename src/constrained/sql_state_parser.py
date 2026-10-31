
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from src.policy_spec import RoleType

from .clause_tracker import (
    _find_last_clause,
    _in_string_literal,
    _paren_depth_at_end,
    keyword_based_role,
)
from .grammar_constants import (
    ALIAS_FORBIDDEN_KEYWORDS,
    IDENTIFIER_TAIL_NON_OPENERS,
    SELECT_MODIFIER_ALTERNATION,
)

_CLAUSE_PATTERN = re.compile(
    r"\b(SELECT|FROM|WHERE|HAVING|GROUP\s+BY|ORDER\s+BY|LIMIT|ON|JOIN)\b",
    re.IGNORECASE,
)
_TABLE_PATTERN = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
_UNSUPPORTED_PATTERN = re.compile(
    r"\b(UNION|INTERSECT|EXCEPT)\b",
    re.IGNORECASE,
)
_ALIAS_PATTERN = re.compile(
    r"\b(?:FROM|JOIN)\s+[A-Za-z_][A-Za-z0-9_]*\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)\b",
    re.IGNORECASE,
)
_IDENTIFIER_SUFFIX = re.compile(r"[A-Za-z_][A-Za-z0-9_\.]*$")
# Order matters: NOT IN must be checked before IN (the IN pattern would
# also match the "IN" inside "NOT IN", but reporting NOT IN is more useful).
_UNSUPPORTED_FRAGMENT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bNOT\s+IN\s*\(", re.IGNORECASE), "NOT IN"),
    (re.compile(r"\bIN\s*\(", re.IGNORECASE), "IN"),
    (re.compile(r"\bANY\s*\(", re.IGNORECASE), "ANY"),
    (re.compile(r"\bALL\s*\(", re.IGNORECASE), "ALL"),
    (re.compile(r"\bSELECT\s+\*", re.IGNORECASE), "SELECT *"),
)

# Identifier-slot opener patterns indexed by role or clause. Each entry is
# a (after_ws, no_ws) pair: `after_ws` matches when the text ends with a
# whitespace right after the separator, `no_ws` matches when the cursor
# sits at the boundary without trailing whitespace.
_ROLE_IDENT_OPENERS: dict[str, tuple[re.Pattern[str], re.Pattern[str]]] = {
    "Sel": (
        re.compile(rf"(SELECT|,|{SELECT_MODIFIER_ALTERNATION})\s+$", re.IGNORECASE),
        re.compile(rf"(SELECT|,|{SELECT_MODIFIER_ALTERNATION})\s*$", re.IGNORECASE),
    ),
    "Where": (
        re.compile(r"(WHERE|HAVING|AND|OR)\s+$", re.IGNORECASE),
        re.compile(r"(WHERE|HAVING|AND|OR)\s*$", re.IGNORECASE),
    ),
    "Join": (
        re.compile(r"(ON|AND|=)\s+$", re.IGNORECASE),
        re.compile(r"(ON|AND|=)\s*$", re.IGNORECASE),
    ),
    "Agg": (
        # `DISTINCT\s+` requires a trailing space before the slot starts.
        re.compile(r"(?:AVG|SUM|MAX|MIN)\s*\(\s*$|COUNT\s*\(\s*(?:DISTINCT\s+)?$", re.IGNORECASE),
        # `DISTINCT\s*` allows the cursor to sit right after `DISTINCT`.
        re.compile(r"(?:AVG|SUM|MAX|MIN)\s*\(\s*$|COUNT\s*\(\s*(?:DISTINCT\s*)?$", re.IGNORECASE),
    ),
}
_CLAUSE_IDENT_OPENERS: dict[str, tuple[re.Pattern[str], re.Pattern[str]]] = {
    "GROUP_BY": (
        re.compile(r"(GROUP\s+BY|,)\s+$", re.IGNORECASE),
        re.compile(r"(GROUP\s+BY|,)\s*[A-Za-z_][A-Za-z0-9_\.]*$", re.IGNORECASE),
    ),
    "ORDER_BY": (
        re.compile(r"(ORDER\s+BY|,)\s+$", re.IGNORECASE),
        re.compile(r"(ORDER\s+BY|,)\s*[A-Za-z_][A-Za-z0-9_\.]*$", re.IGNORECASE),
    ),
}
# Cursor at the end of a clause keyword without trailing whitespace,
# e.g. "GROUP BY" with the cursor right after the "Y".
_CLAUSE_KEYWORD_END_PATTERNS: dict[str, re.Pattern[str]] = {
    "GROUP_BY": re.compile(r"GROUP\s+BY\s*$", re.IGNORECASE),
    "ORDER_BY": re.compile(r"ORDER\s+BY\s*$", re.IGNORECASE),
}
PredicateContext = Literal["WHERE", "HAVING", "ON"]

@dataclass(frozen=True)
class PolicySQLCurrentState:

    clause: str | None
    role: RoleType | None
    predicate_context: PredicateContext | None
    tables_in_scope: tuple[str, ...]
    current_agg: str | None
    identifier_mode: bool
    table_mode: bool
    query_depth: int
    in_string_literal: bool
    unsupported: bool
    unsupported_reason: str | None = None

class PolicySQLStateParser:

    def __init__(self, db_schema: dict):
        self._table_name_map = {
            table.lower(): table
            for table in db_schema.get("table_names_original", [])
        }

    def canonical_table_name(self, name: str) -> str | None:
        return self._table_name_map.get(name.lower())

    def get_current_state(self, text: str) -> PolicySQLCurrentState:
        in_string_literal = _in_string_literal(text)
        scope_state = self._scan_query_scopes(text)
        clause = self._last_clause(text)
        current_agg = self._current_aggregate(text)
        tracked_role = keyword_based_role(text)
        predicate_context = self._predicate_context(text, tracked_role)
        role = self._role(tracked_role, predicate_context, current_agg)
        unsupported_reason = self._unsupported_reason(
            text,
            scope_state,
            current_agg=current_agg,
            predicate_context=predicate_context,
        )
        tables_in_scope = scope_state.tables_in_scope
        identifier_mode = False if in_string_literal else self._identifier_mode(text, role, clause)
        table_mode = False if in_string_literal else self._table_mode(text)

        return PolicySQLCurrentState(
            clause=clause,
            role=role,
            predicate_context=predicate_context,
            tables_in_scope=tables_in_scope,
            current_agg=current_agg,
            identifier_mode=identifier_mode,
            table_mode=table_mode,
            query_depth=scope_state.query_depth,
            in_string_literal=in_string_literal,
            unsupported=unsupported_reason is not None,
            unsupported_reason=unsupported_reason,
        )

    def _role(
        self,
        tracked_role: RoleType | None,
        predicate_context: PredicateContext | None,
        current_agg: str | None,
    ) -> RoleType | None:
        if predicate_context == "HAVING" and current_agg is not None:
            return "Agg"
        return tracked_role

    def _predicate_context(
        self,
        text: str,
        role: RoleType | None,
    ) -> PredicateContext | None:
        if role not in {"Agg", "Join", "Where"}:
            return None

        target_depth = _paren_depth_at_end(text)
        for depth in range(target_depth, -1, -1):
            clause = _find_last_clause(text, depth)
            if clause is None:
                continue

            clause_upper = clause.upper().strip()
            if role == "Join" and clause_upper == "ON":
                return "ON"
            if role in {"Agg", "Where"} and clause_upper in {"WHERE", "HAVING"}:
                return clause_upper

        return None

    def _unsupported_reason(
        self,
        text: str,
        scope_state: "_ScopeState",
        *,
        current_agg: str | None,
        predicate_context: PredicateContext | None,
    ) -> str | None:
        if scope_state.query_depth > 1:
            return "DEEP_SUBQUERY"

        unsupported = _UNSUPPORTED_PATTERN.search(text)
        if unsupported:
            return unsupported.group(1).upper()

        if predicate_context == "HAVING":
            if not scope_state.current_scope_has_group_by:
                return "HAVING"
            if current_agg not in {None, "COUNT", "AVG", "SUM"}:
                return "HAVING"

        # v1 avoids aliases to keep scope tracking simple. Detect "FROM t x"
        # or "FROM t AS x" once the alias token is fully written.
        for alias in _ALIAS_PATTERN.finditer(text):
            alias_name = alias.group(1)
            if alias_name.upper() in ALIAS_FORBIDDEN_KEYWORDS:
                continue
            if alias_name.lower() not in self._table_name_map:
                return "ALIAS"

        for pattern, reason in _UNSUPPORTED_FRAGMENT_PATTERNS:
            if pattern.search(text):
                return reason

        return None

    def _last_clause(self, text: str) -> str | None:
        matches = list(_CLAUSE_PATTERN.finditer(text))
        if not matches:
            return None
        return matches[-1].group(1).upper().replace(" ", "_")

    def _current_aggregate(self, text: str) -> str | None:
        depth = 0
        i = len(text) - 1

        while i >= 0:
            ch = text[i]
            if ch == ")":
                depth += 1
            elif ch == "(":
                if depth > 0:
                    depth -= 1
                else:
                    prefix = text[:i].rstrip().upper()
                    for agg in ("COUNT", "AVG", "SUM", "MAX", "MIN"):
                        if prefix.endswith(agg):
                            start = len(prefix) - len(agg)
                            if start == 0 or not prefix[start - 1].isalnum():
                                return agg
                    return None
            i -= 1

        return None

    def _identifier_mode(
        self,
        text: str,
        role: RoleType | None,
        clause: str | None,
    ) -> bool:
        upper = text.upper()
        role_opener = _ROLE_IDENT_OPENERS.get(role) if role is not None else None
        clause_opener = _CLAUSE_IDENT_OPENERS.get(clause) if clause is not None else None

        if text.endswith((" ", "\n", "\t")):
            if role_opener is not None:
                return bool(role_opener[0].search(upper))
            if clause_opener is not None:
                return bool(clause_opener[0].search(upper))
            return False

        keyword_end = _CLAUSE_KEYWORD_END_PATTERNS.get(clause) if clause is not None else None
        if keyword_end is not None and keyword_end.search(upper):
            return True

        if clause_opener is not None:
            return bool(clause_opener[1].search(text))

        if role is None:
            return False

        # An identifier-shaped suffix means we are still inside the current
        # slot, unless it is itself a keyword.
        match = _IDENTIFIER_SUFFIX.search(text)
        if match and match.group(0).upper() not in IDENTIFIER_TAIL_NON_OPENERS:
            return True

        if role_opener is not None:
            return bool(role_opener[1].search(upper))
        return False

    def _table_mode(self, text: str) -> bool:
        upper = text.upper()

        if re.search(r"(FROM|JOIN)\s*$", upper):
            return True

        if text.endswith((" ", "\n", "\t")):
            return bool(re.search(r"(FROM|JOIN)\s+$", upper))

        match = _IDENTIFIER_SUFFIX.search(text)
        if match:
            prefix = upper[:match.start()]
            if re.search(r"(FROM|JOIN)\s+$", prefix):
                return True

        return False

    def _scan_query_scopes(self, text: str) -> "_ScopeState":
        depth = 0
        in_string = False
        scopes: list[_Scope] = []
        i = 0
        n = len(text)

        while i < n:
            ch = text[i]
            if in_string:
                if ch == "'" and i + 1 < n and text[i + 1] == "'":
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
                    scopes.pop()
                i += 1
                continue

            if ch.isalpha() or ch == "_":
                start = i
                i += 1
                while i < n and (text[i].isalnum() or text[i] == "_"):
                    i += 1
                token = text[start:i]
                upper = token.upper()

                if upper == "SELECT":
                    scopes.append(_Scope(depth=depth))
                    continue

                if upper == "GROUP":
                    j = i
                    while j < n and text[j].isspace():
                        j += 1
                    by_start = j
                    while j < n and text[j].isalpha():
                        j += 1
                    if by_start < j and text[by_start:j].upper() == "BY" and scopes:
                        scopes[-1].has_group_by = True
                        i = j
                        continue

                if upper in {"FROM", "JOIN"}:
                    if scopes:
                        scopes[-1].awaiting_table = True
                    continue

                if upper == "AS":
                    continue

                if scopes and scopes[-1].awaiting_table:
                    canonical = self._table_name_map.get(token.lower())
                    scopes[-1].awaiting_table = False
                    if canonical and canonical not in scopes[-1].seen:
                        scopes[-1].seen.add(canonical)
                        scopes[-1].tables.append(canonical)
                    continue

                continue

            i += 1

        tables_in_scope: list[str] = []
        seen: set[str] = set()
        for scope in scopes:
            for table in scope.tables:
                if table not in seen:
                    seen.add(table)
                    tables_in_scope.append(table)

        return _ScopeState(
            tables_in_scope=tuple(tables_in_scope),
            current_scope_has_group_by=bool(scopes and scopes[-1].has_group_by),
            query_depth=max(len(scopes) - 1, 0),
        )

@dataclass
class _Scope:
    depth: int
    tables: list[str] = None
    seen: set[str] = None
    has_group_by: bool = False
    awaiting_table: bool = False

    def __post_init__(self) -> None:
        if self.tables is None:
            self.tables = []
        if self.seen is None:
            self.seen = set()

@dataclass(frozen=True)
class _ScopeState:
    tables_in_scope: tuple[str, ...]
    current_scope_has_group_by: bool
    query_depth: int
