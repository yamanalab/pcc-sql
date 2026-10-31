
from __future__ import annotations

import re

_AGG_FUNCTIONS = frozenset({"COUNT", "AVG", "SUM", "MAX", "MIN"})

_KW_PATTERN = re.compile(
    r"\b(SELECT|FROM|WHERE|HAVING|LIMIT"
    r"|GROUP\s+BY|ORDER\s+BY"
    r"|(?:LEFT\s+|RIGHT\s+|INNER\s+|OUTER\s+|CROSS\s+)?JOIN"
    r"|ON|UNION|INTERSECT|EXCEPT)\b",
    re.IGNORECASE,
)

_AGG_PATTERN = re.compile(
    r"\b(COUNT|AVG|SUM|MAX|MIN)\s*\(", re.IGNORECASE
)

def keyword_based_role(text: str) -> str | None:
    target_depth = _paren_depth_at_end(text)
    # `SELECT COUNT(` puts the cursor at depth 1 but the SELECT keyword sits
    # at depth 0; walk depths from current down to 0.
    last_clause = None
    for d in range(target_depth, -1, -1):
        last_clause = _find_last_clause(text, d)
        if last_clause is not None:
            break

    if last_clause is None:
        return None

    clause_upper = last_clause.upper().strip()
    if clause_upper == "SELECT":
        return "Agg" if _cursor_in_aggregate(text, target_depth) else "Sel"
    if clause_upper in ("WHERE", "HAVING"):
        return "Where"
    if clause_upper == "ON":
        return "Join"
    return None

def _scan_end_state(text: str) -> tuple[int, bool]:
    depth = 0
    in_string = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            if ch == "'" and i + 1 < n and text[i + 1] == "'":
                i += 2  # escaped quote
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
    return max(depth, 0), in_string

def _paren_depth_at_end(text: str) -> int:
    depth, _ = _scan_end_state(text)
    return depth

def _in_string_literal(text: str) -> bool:
    _, in_string = _scan_end_state(text)
    return in_string

def _find_last_clause(text: str, target_depth: int) -> str | None:
    depth = 0
    in_string = False
    last_kw: str | None = None
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
            depth -= 1
            i += 1
            continue

        if depth == target_depth and ch.isalpha():
            m = _KW_PATTERN.match(text, i)
            if m:
                kw = m.group(0)
                if i == 0 or not text[i - 1].isalnum():
                    last_kw = kw
                i = m.end()
                continue

        i += 1

    return last_kw

def _cursor_in_aggregate(text: str, target_depth: int) -> bool:
    # Scan from end of text backwards for an unmatched "AGG(".
    depth = 0
    in_string = False
    n = len(text)

    i = n - 1
    while i >= 0:
        ch = text[i]

        if in_string:
            if ch == "'" and i > 0 and text[i - 1] == "'":
                i -= 2
                continue
            if ch == "'":
                in_string = False
            i -= 1
            continue
        if ch == "'":
            in_string = True
            i -= 1
            continue

        if ch == ")":
            depth += 1
        elif ch == "(":
            if depth > 0:
                depth -= 1
            else:
                before = text[:i].rstrip()
                before_upper = before.upper()
                for agg in _AGG_FUNCTIONS:
                    if before_upper.endswith(agg):
                        prefix_end = len(before) - len(agg)
                        if prefix_end == 0 or not before[prefix_end - 1].isalnum():
                            return True
                break
        i -= 1

    return False
