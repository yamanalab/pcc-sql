
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

__all__ = ["ContinuationTrieBase", "ContinuationTrieNode"]

T = TypeVar("T")

@dataclass
class ContinuationTrieNode(Generic[T]):
    children: dict[int, "ContinuationTrieNode[T]"] = field(default_factory=dict)
    completes: set[T] = field(default_factory=set)
    reachable: set[T] = field(default_factory=set)

class ContinuationTrieBase(Generic[T]):

    def __init__(self, tokenizer: "PreTrainedTokenizerBase"):
        self.tokenizer = tokenizer
        self.root: ContinuationTrieNode[T] = ContinuationTrieNode()
        self._max_depth = 0

    def _insert(self, token_ids: list[int], item: T) -> None:
        node = self.root
        for tid in token_ids:
            if tid not in node.children:
                node.children[tid] = ContinuationTrieNode()
            node = node.children[tid]
        node.completes.add(item)
        self._max_depth = max(self._max_depth, len(token_ids))

    def _compute_reachable(self, node: ContinuationTrieNode[T]) -> set[T]:
        node.reachable = set(node.completes)
        for child in node.children.values():
            node.reachable |= self._compute_reachable(child)
        return node.reachable

    def get_allowed_token_ids(
        self,
        slot_token_ids: list[int],
        allowed: set[T] | frozenset[T],
    ) -> set[int]:
        if not allowed:
            return set()

        node = self.root
        if self._max_depth > 0:
            window_start = -self._max_depth
            for tid in slot_token_ids[window_start:]:
                child = node.children.get(tid)
                if child is None:
                    return set()
                node = child

        # `isdisjoint` avoids allocating an intermediate intersection set per child.
        return {
            tid
            for tid, child in node.children.items()
            if not child.reachable.isdisjoint(allowed)
        }

    def is_complete_allowed(
        self,
        slot_token_ids: list[int],
        allowed: set[T] | frozenset[T],
    ) -> bool:
        if not allowed:
            return False

        node = self.root
        if self._max_depth > 0:
            for tid in slot_token_ids[-self._max_depth:]:
                child = node.children.get(tid)
                if child is None:
                    return False
                node = child

        return not node.completes.isdisjoint(allowed)
