
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from ._continuation_trie_base import ContinuationTrieBase, ContinuationTrieNode

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

__all__ = ["LiteralContinuationTrie"]

class LiteralContinuationTrie(ContinuationTrieBase[str]):

    def __init__(self, tokenizer: "PreTrainedTokenizerBase", literals: Iterable[str]):
        super().__init__(tokenizer)
        self._build(literals)

    def _build(self, literals: Iterable[str]) -> None:
        for literal in literals:
            for variant in self._literal_variants(literal):
                token_ids = self.tokenizer.encode(variant, add_special_tokens=False)
                if token_ids:
                    self._insert(token_ids, literal)
        self._compute_reachable(self.root)

    @staticmethod
    def _literal_variants(literal: str) -> list[str]:
        variants = [literal]
        lower = literal.lower()
        if lower != literal:
            variants.append(lower)
        return variants

    def get_allowed_token_ids(
        self,
        recent_ids: list[int],
        allowed_literals: set[str] | frozenset[str],
    ) -> set[int]:
        # Override: structural literals lack an explicit slot boundary, so
        # try every suffix start in the recent token window — that lets
        # `"SELECT col F"` still match `"FROM "` when only `"F"` is emitted.
        if not allowed_literals:
            return set()

        if self._max_depth <= 0:
            active_nodes: list[ContinuationTrieNode[str]] = [self.root]
        else:
            window = recent_ids[-self._max_depth:]
            # Track all currently-alive walks (each pointing to a trie node) in
            # parallel: extend them by one token per window step instead of
            # restarting from root for every suffix.
            active_walks: list[ContinuationTrieNode[str]] = []
            for tid in window:
                next_walks: list[ContinuationTrieNode[str]] = []
                for node in active_walks:
                    child = node.children.get(tid)
                    if child is not None:
                        next_walks.append(child)
                root_child = self.root.children.get(tid)
                if root_child is not None:
                    next_walks.append(root_child)
                active_walks = next_walks
            # Dedup walks that landed on the same trie node.
            active_nodes = [self.root]
            seen: set[int] = set()
            for node in active_walks:
                node_id = id(node)
                if node_id not in seen:
                    seen.add(node_id)
                    active_nodes.append(node)

        allowed: set[int] = set()
        for node in active_nodes:
            for tid, child in node.children.items():
                if tid in allowed:
                    continue
                if not child.reachable.isdisjoint(allowed_literals):
                    allowed.add(tid)
        return allowed
