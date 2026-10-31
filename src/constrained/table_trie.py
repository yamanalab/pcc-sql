
from __future__ import annotations

from typing import TYPE_CHECKING

from ._continuation_trie_base import ContinuationTrieBase

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

__all__ = ["TableContinuationTrie"]

class TableContinuationTrie(ContinuationTrieBase[str]):

    def __init__(self, tokenizer: "PreTrainedTokenizerBase", db_schema: dict):
        super().__init__(tokenizer)
        self._build(db_schema)

    def _build(self, db_schema: dict) -> None:
        for table_name in db_schema["table_names_original"]:
            for variant in self._table_variants(table_name):
                token_ids = self.tokenizer.encode(variant, add_special_tokens=False)
                if token_ids:
                    self._insert(token_ids, table_name)
        self._compute_reachable(self.root)

    def _table_variants(self, table_name: str) -> list[str]:
        variants: list[str] = []
        for prefix in ("", " ", "(", ", "):
            for table_variant in self._identifier_case_variants(table_name):
                variants.append(f"{prefix}{table_variant}")
        unique: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            if variant not in seen:
                seen.add(variant)
                unique.append(variant)
        return unique

    @staticmethod
    def _identifier_case_variants(identifier: str) -> tuple[str, ...]:
        lowered = identifier.lower()
        if lowered == identifier:
            return (identifier,)
        return (identifier, lowered)
