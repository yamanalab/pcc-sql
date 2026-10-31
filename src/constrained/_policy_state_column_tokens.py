
from __future__ import annotations

import re

from ._identifier_variants import column_reference_variants
from ._policy_state_token_matching import _PolicyStateTokenMatchingMixin

_IDENTIFIER_CONTINUATION_TOKEN_RE = re.compile(r"[A-Za-z0-9_`\. ()]+")
_AGGREGATE_ENTRY_TOKEN_RE = re.compile(r"\([A-Za-z0-9_`\. ()]+")
_COUNT_DISTINCT_CONTINUATION_TOKEN_RE = re.compile(r"[A-Za-z0-9_`\. ()]+")

__all__ = ["_PolicyStateColumnTokenMixin"]

class _PolicyStateColumnTokenMixin(_PolicyStateTokenMatchingMixin):

    def _column_slot_token_ids(self, text: str, state, slot_ids: list[int]) -> set[int]:
        if state.role == "Agg" and state.current_agg == "COUNT":
            return self._count_arg_token_ids(text, state, slot_ids)

        allowed_columns = self._allowed_columns_for_current_slot(state)
        if not allowed_columns:
            return set()

        allowed = self.allowed_trie.get_allowed_token_ids(slot_ids, allowed_columns)
        if allowed:
            return allowed
        return self._textual_column_continuation_token_ids(text, state)

    def _count_star_token_ids(self, slot_ids: list[int]) -> set[int]:
        star_ids = self.tokenizer.encode("*", add_special_tokens=False)
        if not star_ids:
            return set()

        if len(slot_ids) > len(star_ids):
            return set()

        prefix = star_ids[: len(slot_ids)]
        if slot_ids != prefix:
            return set()

        if len(slot_ids) == len(star_ids):
            return set()

        return {star_ids[len(slot_ids)]}

    def _count_arg_token_ids(self, text: str, state, slot_ids: list[int]) -> set[int]:
        return self._count_star_token_ids(slot_ids) | self._count_distinct_token_ids(
            text,
            state,
        )

    def _count_distinct_token_ids(self, text: str, state) -> set[int]:
        allowed_columns = self._allowed_columns_for_current_slot(state)
        if not allowed_columns:
            return set()

        slot_text = self._slot_text(text, state).lstrip()
        variants = self._count_distinct_variants(
            allowed_columns,
            allow_close=True,
        )
        if not variants:
            return set()

        candidate_token_ids = self._token_ids_extending_variants(
            cache_key="count_distinct_continuation_candidates",
            predicate=lambda token_text: bool(
                _COUNT_DISTINCT_CONTINUATION_TOKEN_RE.fullmatch(token_text)
            ),
            variants=variants,
            current_prefix=slot_text,
        )
        return self._filter_completion_safe_tokens(
            candidate_token_ids,
            slot_text=slot_text,
            variants=variants,
        )

    def _textual_column_continuation_token_ids(self, text: str, state) -> set[int]:
        slot_text = self._slot_text(text, state).lstrip()
        if not slot_text:
            return set()

        allowed_columns = self._allowed_columns_for_current_slot(state)
        variants = self._allowed_column_variants(allowed_columns)
        stripped_variants = {variant.lstrip() for variant in variants}
        candidate_token_ids = self._token_ids_extending_variants(
            cache_key="identifier_continuation_candidates",
            predicate=lambda token_text: bool(
                _IDENTIFIER_CONTINUATION_TOKEN_RE.fullmatch(token_text)
            ),
            variants=stripped_variants,
            current_prefix=slot_text,
        )
        return self._filter_completion_safe_tokens(
            candidate_token_ids,
            slot_text=slot_text,
            variants=stripped_variants,
        )

    def _filter_completion_safe_tokens(
        self,
        candidate_token_ids: set[int],
        *,
        slot_text: str,
        variants: set[str],
    ) -> set[int]:
        if not candidate_token_ids or not variants:
            return set()

        current_prefix = slot_text.lstrip()
        plen = len(current_prefix)
        # Pre-strip the slot prefix from each variant once. This means we only
        # need to compare each candidate token's text against the variant
        # suffixes — avoiding repeated prefix matching against the same slot
        # for every (token, variant) pair.
        suffix_variants: list[str] = []
        for variant in variants:
            stripped = variant.lstrip()
            if stripped.startswith(current_prefix):
                suffix_variants.append(stripped[plen:])
        if not suffix_variants:
            return set()
        first_chars = {s[0] for s in suffix_variants if s}

        safe: set[int] = set()
        for token_id in candidate_token_ids:
            token_text = self._decoded_token_text(token_id)
            if not token_text:
                # Empty token text trivially keeps current_prefix unchanged.
                safe.add(token_id)
                continue
            # Cheap reject most tokens by their first char before the full
            # startswith check.
            if token_text[0] not in first_chars:
                continue
            if any(s.startswith(token_text) for s in suffix_variants):
                safe.add(token_id)
        return safe

    def _aggregate_entry_token_ids(self, text: str, state) -> set[int]:
        slot_text = self._slot_text(text, state).strip()
        agg = slot_text.upper()
        if agg == "COUNT":
            return self._count_star_entry_token_ids() | self._count_distinct_entry_token_ids(state)
        if agg not in {"AVG", "SUM", "MAX", "MIN"}:
            return set()

        allowed_columns = self._columns_for_role(
            "Agg",
            state=state,
            current_agg=agg,
        )
        variants = self._allowed_column_variants(allowed_columns, prefix="(")
        return self._token_ids_extending_variants(
            cache_key="aggregate_entry_candidates",
            predicate=lambda token_text: bool(_AGGREGATE_ENTRY_TOKEN_RE.fullmatch(token_text)),
            variants=variants,
        )

    def _count_star_entry_token_ids(self) -> set[int]:
        return self._token_ids_matching(
            "count_star_entry",
            lambda token_text: bool(token_text) and "(*)".startswith(token_text),
        )

    def _count_distinct_entry_token_ids(self, state) -> set[int]:
        allowed_columns = self._columns_for_role(
            "Agg",
            state=state,
            current_agg="COUNT",
        )
        variants = self._count_distinct_variants(
            allowed_columns,
            prefix="(",
            allow_close=True,
        )
        return self._token_ids_extending_variants(
            cache_key="count_distinct_entry_candidates",
            predicate=lambda token_text: bool(
                token_text.startswith("(")
                and _COUNT_DISTINCT_CONTINUATION_TOKEN_RE.fullmatch(token_text[1:])
            ),
            variants=variants,
        )

    def _count_distinct_variants(
        self,
        allowed_columns: set[str],
        *,
        prefix: str = "",
        allow_close: bool = False,
    ) -> set[str]:
        variants = self._allowed_column_variants(
            allowed_columns,
            prefix=f"{prefix}DISTINCT ",
        )
        if not allow_close:
            return variants
        return variants | {f"{variant})" for variant in variants}

    def _allowed_column_variants(
        self,
        allowed_columns: set[str],
        *,
        prefix: str = "",
    ) -> set[str]:
        variants: set[str] = set()
        for full_name in allowed_columns:
            table_name, _, column_name = full_name.partition(".")
            variants.update(
                column_reference_variants(
                    table_name,
                    column_name,
                    allow_bare=column_name.lower() in self._unique_bare_columns,
                    prefixes=(prefix,),
                )
            )
        return variants

    @staticmethod
    def _compute_unique_bare_columns(db_schema: dict) -> set[str]:
        counts: dict[str, int] = {}
        for col_id, (table_idx, col_name) in enumerate(db_schema["column_names_original"]):
            if col_id == 0 or table_idx < 0:
                continue
            normalized = col_name.lower()
            counts[normalized] = counts.get(normalized, 0) + 1
        return {col_name for col_name, count in counts.items() if count == 1}
