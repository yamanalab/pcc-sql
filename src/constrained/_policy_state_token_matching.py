
from __future__ import annotations

import re
from collections.abc import Callable, Iterable

__all__ = ["_PolicyStateTokenMatchingMixin"]

class _PolicyStateTokenMatchingMixin:

    def _single_char_token_ids(self, chars: str) -> set[int]:
        allowed: set[int] = set()
        for ch in chars:
            cached = self._single_char_token_cache.get(ch)
            if cached is None:
                token_ids = self.tokenizer.encode(ch, add_special_tokens=False)
                cached = {token_ids[0]} if len(token_ids) == 1 else set()
                self._single_char_token_cache[ch] = cached
            allowed |= cached
        return allowed

    def _whitespace_token_ids(self) -> set[int]:
        return self._token_ids_matching(
            "whitespace",
            lambda text: bool(text) and text.isspace(),
        )

    def _regex_token_ids(
        self,
        cache_key: str,
        pattern: str | re.Pattern[str],
    ) -> set[int]:
        compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
        return self._token_ids_matching(
            cache_key,
            lambda text: bool(compiled.fullmatch(text)),
        )

    def _token_ids_extending_variants(
        self,
        *,
        cache_key: str,
        predicate: Callable[[str], bool],
        variants: Iterable[str],
        current_prefix: str = "",
    ) -> set[int]:
        variant_set = set(variants)
        if not variant_set:
            return set()

        # Pre-strip `current_prefix` from variants once, so per-token check
        # collapses to a small set of plain startswith comparisons.
        plen = len(current_prefix)
        suffix_variants: list[str] = []
        for variant in variant_set:
            if variant.startswith(current_prefix):
                suffix_variants.append(variant[plen:])
        if not suffix_variants:
            return set()
        first_chars = {s[0] for s in suffix_variants if s}

        candidate_token_ids = self._token_ids_matching(cache_key, predicate)
        out: set[int] = set()
        for token_id in candidate_token_ids:
            token_text = self._decoded_token_text(token_id)
            if not token_text:
                out.add(token_id)
                continue
            if token_text[0] not in first_chars:
                continue
            if any(s.startswith(token_text) for s in suffix_variants):
                out.add(token_id)
        return out

    def _token_ids_matching(
        self,
        cache_key: str,
        predicate: Callable[[str], bool],
    ) -> set[int]:
        cached = self._token_match_cache.get(cache_key)
        if cached is not None:
            return cached

        matched: set[int] = set()
        for token_id in self._all_token_ids():
            text = self._decoded_token_text(token_id)
            if text and predicate(text):
                matched.add(token_id)

        self._token_match_cache[cache_key] = matched
        return matched

    def _decoded_token_text(self, token_id: int) -> str:
        cached = self._decoded_token_cache.get(token_id)
        if cached is None:
            cached = self.tokenizer.decode([token_id], skip_special_tokens=True)
            self._decoded_token_cache[token_id] = cached
        return cached

    def _all_token_ids(self) -> range | list[int]:
        try:
            return range(len(self.tokenizer))
        except TypeError:
            pass

        for attr in ("_id_to_piece", "_id_to_char", "_id_to_word"):
            mapping = getattr(self.tokenizer, attr, None)
            if mapping is not None:
                return list(mapping)

        return []
