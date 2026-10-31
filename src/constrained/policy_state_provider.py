
from __future__ import annotations

from .allowed_trie import AllowedContinuationTrie
from .policy_state_mask import PolicyStateLogitsProcessor
from .table_trie import TableContinuationTrie

class PolicyStateLocalProvider:

    def __init__(self, model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"):
        self.model = model
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model_obj = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        self._column_policies: dict[str, str] | None = None
        self._db_schema: dict | None = None
        self._allowed_trie: AllowedContinuationTrie | None = None
        self._table_trie: TableContinuationTrie | None = None
        self._last_debug_info: dict[str, object] | None = None
        self._last_usage_stats: dict[str, object] | None = None

    def set_context(
        self,
        column_policies: dict[str, str],
        db_schema: dict,
    ) -> None:
        self._column_policies = column_policies
        self._db_schema = db_schema
        self._allowed_trie = AllowedContinuationTrie(self.tokenizer, db_schema)
        self._table_trie = TableContinuationTrie(self.tokenizer, db_schema)

    def _decode_with_policy_state(
        self,
        prompt: str,
    ) -> tuple[str, bool, PolicyStateLogitsProcessor]:
        import torch

        if (
            self._column_policies is None
            or self._db_schema is None
            or self._allowed_trie is None
            or self._table_trie is None
        ):
            raise RuntimeError("Must call set_context() before generate()")

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model_obj.device)
        prompt_length = inputs["input_ids"].shape[1]

        processor = PolicyStateLogitsProcessor(
            tokenizer=self.tokenizer,
            allowed_trie=self._allowed_trie,
            table_trie=self._table_trie,
            column_policies=self._column_policies,
            db_schema=self._db_schema,
            prompt_length=prompt_length,
        )

        with torch.no_grad():
            outputs = self.model_obj.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                logits_processor=[processor],
            )

        generated_ids = outputs[0][prompt_length:].tolist()
        eos_id = self.tokenizer.eos_token_id
        terminated_by_eos = bool(
            generated_ids and eos_id is not None and generated_ids[-1] == eos_id
        )
        if terminated_by_eos:
            generated_ids = generated_ids[:-1]
        self._last_usage_stats = {
            "provider_calls": 1,
            "input_tokens": int(prompt_length),
            "output_tokens": len(generated_ids),
            "total_tokens": int(prompt_length) + len(generated_ids),
            "token_stats_available": True,
        }
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return output_text, terminated_by_eos, processor

    def _handle_mask_or_empty_output(
        self,
        output_text: str,
        terminated_by_eos: bool,
        processor: PolicyStateLogitsProcessor,
    ) -> str | None:
        if processor.terminated_unsafe:
            self._last_debug_info = {
                "status": "refuse",
                "output_text": output_text,
                **(processor.get_debug_info() or {}),
            }
            return "REFUSE"

        if not output_text:
            self._last_debug_info = {
                "status": "refuse",
                "decision_stage": "decode",
                "failure_reason": "empty_output",
                "output_text": output_text,
            }
            return "REFUSE"

        debug_info = processor.validate_terminal_output(
            output_text,
            terminated_by_eos=terminated_by_eos,
        )
        if debug_info is not None:
            self._last_debug_info = {
                "status": "refuse",
                "output_text": output_text,
                **debug_info,
            }
            return "REFUSE"

        return None

    def generate(self, prompt: str) -> str:
        self._last_debug_info = None
        self._last_usage_stats = None
        output_text, terminated_by_eos, processor = self._decode_with_policy_state(prompt)
        early_result = self._handle_mask_or_empty_output(
            output_text,
            terminated_by_eos,
            processor,
        )
        if early_result is not None:
            return early_result
        self._last_debug_info = {
            "status": "accepted",
            "decision_stage": "decode",
            "failure_reason": None,
            "output_text": output_text,
        }
        return output_text

    def get_last_debug_info(self) -> dict[str, object] | None:
        return self._last_debug_info

    def get_last_usage_stats(self) -> dict[str, object] | None:
        return self._last_usage_stats
