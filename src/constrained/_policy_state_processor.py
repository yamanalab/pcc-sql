
from __future__ import annotations

from typing import TYPE_CHECKING

from ._policy_state_api import compute_admissible_next_tokens
from ._policy_state_impl import PolicyStateAdmissibilityRuntime
from .allowed_trie import AllowedContinuationTrie
from .canonical_sql_grammar import CanonicalSQLGrammar
from .sql_state_parser import PolicySQLStateParser
from .table_trie import TableContinuationTrie

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

class PolicyStateLogitsProcessor:

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        allowed_trie: AllowedContinuationTrie,
        table_trie: TableContinuationTrie,
        column_policies: dict[str, str],
        db_schema: dict,
        prompt_length: int,
    ):
        self.prompt_length = prompt_length
        self._runtime = PolicyStateAdmissibilityRuntime(
            tokenizer=tokenizer,
            allowed_trie=allowed_trie,
            table_trie=table_trie,
            column_policies=column_policies,
            db_schema=db_schema,
        )
        self.terminated_unsafe = False
        self._debug_info: dict[str, object] | None = None

    @property
    def runtime(self) -> PolicyStateAdmissibilityRuntime:
        return self._runtime

    @property
    def state_parser(self) -> PolicySQLStateParser:
        return self._runtime.state_parser

    @property
    def grammar(self) -> CanonicalSQLGrammar:
        return self._runtime.grammar

    def __call__(self, input_ids, scores):
        import torch

        generated_ids = input_ids[0, self.prompt_length :].tolist()
        text = self._runtime.tokenizer.decode(generated_ids, skip_special_tokens=True)
        result = compute_admissible_next_tokens(
            self._runtime,
            generated_ids=generated_ids,
            text=text,
        )
        self.terminated_unsafe = result.terminated_unsafe
        self._debug_info = result.debug_info

        if result.should_force_eos:
            return self._force_eos(scores)

        valid_ids = [
            tid for tid in result.allowed_token_ids if 0 <= tid < scores.shape[-1]
        ]
        if not valid_ids:
            self.terminated_unsafe = True
            return self._force_eos(scores)

        allowed = torch.tensor(sorted(valid_ids), dtype=torch.long, device=scores.device)
        blocked = torch.ones_like(scores, dtype=torch.bool)
        blocked[0, allowed] = False
        scores[blocked] = float("-inf")
        return scores

    def validate_terminal_output(
        self,
        output_text: str,
        *,
        terminated_by_eos: bool,
    ) -> dict[str, object] | None:
        return self._runtime.validate_terminal_output(
            output_text,
            terminated_by_eos=terminated_by_eos,
        )

    def get_debug_info(self) -> dict[str, object] | None:
        return self._debug_info

    def _force_eos(self, scores):
        eos_id = self._runtime.tokenizer.eos_token_id
        if eos_id is None:
            return scores
        scores[0, :] = float("-inf")
        scores[0, eos_id] = 0.0
        return scores

__all__ = ["PolicyStateLogitsProcessor"]
