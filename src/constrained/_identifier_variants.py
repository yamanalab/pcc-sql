
from __future__ import annotations

import re

_PLAIN_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

def identifier_case_variants(identifier: str) -> tuple[str, ...]:
    lowered = identifier.lower()
    if lowered == identifier:
        return (identifier,)
    return (identifier, lowered)

def identifier_sql_variants(identifier: str) -> tuple[str, ...]:
    variants = list(identifier_case_variants(identifier))
    if _PLAIN_IDENTIFIER_RE.fullmatch(identifier):
        return tuple(variants)

    for case_variant in identifier_case_variants(identifier):
        variants.append(f"`{case_variant}`")

    unique: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        if variant not in seen:
            seen.add(variant)
            unique.append(variant)
    return tuple(unique)

def column_reference_variants(
    table_name: str,
    column_name: str,
    *,
    allow_bare: bool = False,
    prefixes: tuple[str, ...] = ("",),
) -> list[str]:
    variants: list[str] = []
    table_variants = identifier_sql_variants(table_name)
    column_variants = identifier_sql_variants(column_name)

    for prefix in prefixes:
        for table_variant in table_variants:
            for column_variant in column_variants:
                variants.append(f"{prefix}{table_variant}.{column_variant}")

    if allow_bare:
        for prefix in prefixes:
            for column_variant in column_variants:
                variants.append(f"{prefix}{column_variant}")

    unique: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        if variant not in seen:
            seen.add(variant)
            unique.append(variant)
    return unique

def unquote_identifier_component(identifier: str) -> str:
    stripped = identifier.strip()
    if len(stripped) >= 2 and stripped[0] == "`" and stripped[-1] == "`":
        return stripped[1:-1]
    return stripped

def normalize_identifier_reference(reference: str) -> str:
    parts = [unquote_identifier_component(part) for part in reference.strip().split(".")]
    return ".".join(parts).lower()
