
from __future__ import annotations

SELECT_MODIFIERS: tuple[str, ...] = ("DISTINCT",)
SELECT_MODIFIER_ALTERNATION: str = "|".join(SELECT_MODIFIERS)

SELECT_ENTRY_KEYWORDS: frozenset[str] = frozenset(
    (*SELECT_MODIFIERS, "AVG", "SUM", "MAX", "MIN", "COUNT")
)

CLAUSE_KEYWORD_CONTINUATIONS: dict[tuple[str, str | None], frozenset[str]] = {
    ("SELECT", "Sel"): SELECT_ENTRY_KEYWORDS,
    ("SELECT", "Agg"): frozenset({"DISTINCT"}),
    ("WHERE", "Where"): frozenset({"EXISTS", "NOT"}),
    ("HAVING", "Where"): frozenset({"AVG", "SUM", "COUNT"}),
}

ALIAS_FORBIDDEN_KEYWORDS: frozenset[str] = frozenset({
    "SELECT", "FROM", "WHERE", "HAVING", "GROUP", "ORDER", "BY", "LIMIT",
    "JOIN", "ON", "AND", "OR", "EXISTS", "NOT", "IN", "ASC", "DESC",
})

RESERVED_IDENTIFIER_NAMES: frozenset[str] = frozenset({
    "AND", "ASC", "AVG", "BY", "COUNT", "DESC", "DISTINCT", "EXISTS",
    "FALSE", "FROM", "GROUP", "JOIN", "LIMIT", "MAX", "MIN", "NOT",
    "NULL", "ON", "OR", "ORDER", "SELECT", "SUM", "TRUE", "WHERE",
})

IDENTIFIER_TAIL_NON_OPENERS: frozenset[str] = frozenset({
    "SELECT", "FROM", "WHERE", "AND", "ON", "JOIN",
})
