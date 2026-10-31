
from __future__ import annotations

from typing import TYPE_CHECKING

from ._continuation_trie_base import ContinuationTrieBase
from ._identifier_variants import column_reference_variants, identifier_case_variants

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

__all__ = ["AllowedContinuationTrie"]

class AllowedContinuationTrie(ContinuationTrieBase[str]):

    def __init__(self, tokenizer: "PreTrainedTokenizerBase", db_schema: dict):
        super().__init__(tokenizer)
        self._build(db_schema)

    def _build(self, db_schema: dict) -> None:
        table_names = db_schema["table_names_original"]
        column_names = db_schema["column_names_original"]
        column_counts: dict[str, int] = {}

        for col_id, (table_idx, col_name) in enumerate(column_names):
            if col_id == 0 or table_idx < 0:
                continue
            column_counts[col_name.lower()] = column_counts.get(col_name.lower(), 0) + 1

        for col_id, (table_idx, col_name) in enumerate(column_names):
            if col_id == 0 or table_idx < 0:
                continue

            table_name = table_names[table_idx]
            full_name = f"{table_name}.{col_name}"
            for variant in self._column_variants(
                table_name,
                col_name,
                allow_bare=column_counts.get(col_name.lower(), 0) == 1,
            ):
                token_ids = self.tokenizer.encode(variant, add_special_tokens=False)
                if token_ids:
                    self._insert(token_ids, full_name)

        self._compute_reachable(self.root)

    def _column_variants(
        self,
        table_name: str,
        col_name: str,
        *,
        allow_bare: bool = False,
    ) -> list[str]:
        return column_reference_variants(
            table_name,
            col_name,
            allow_bare=allow_bare,
            prefixes=("", " ", "(", ", "),
        )

    @staticmethod
    def _identifier_case_variants(identifier: str) -> tuple[str, ...]:
        return identifier_case_variants(identifier)
