
from __future__ import annotations

import re
import string

from ._policy_state_token_matching import _PolicyStateTokenMatchingMixin

_SIMPLE_STRING_LITERAL_CHARS = (
    string.ascii_letters + string.digits + " _-./%:,()&+@#$!?="
)
_SIMPLE_STRING_LITERAL_CHAR_SET = set(_SIMPLE_STRING_LITERAL_CHARS)
_WHERE_VALUE_KEYWORDS = frozenset({"TRUE", "FALSE", "NULL"})
# Initial literals also allow "(" — the literal trie consumes it; the
# subquery body itself is dispatched by grammar.allowed_branches.
_WHERE_VALUE_INITIAL_LITERALS = _WHERE_VALUE_KEYWORDS | frozenset({"("})

__all__ = ["_PolicyStateLiteralTokenMixin"]

def _numeric_prefix_kind(
    stripped_slot_text: str,
    *,
    allow_decimal: bool,
    allow_negative: bool,
) -> str | None:
    if stripped_slot_text == "":
        return "empty"

    sign = r"-?" if allow_negative else ""
    if allow_negative and stripped_slot_text == "-":
        return "sign_only"
    if re.fullmatch(rf"{sign}\d+", stripped_slot_text):
        return "integer"
    if allow_decimal and re.fullmatch(rf"{sign}\d+\.", stripped_slot_text):
        return "decimal_point"
    if allow_decimal and re.fullmatch(rf"{sign}\d+\.\d*", stripped_slot_text):
        return "decimal_digits"
    return None

def _simple_string_body(
    text: str,
    *,
    allow_leading_space: bool,
    require_opening_quote: bool,
) -> str | None:
    candidate = text.lstrip() if allow_leading_space else text
    if not candidate:
        return None

    if require_opening_quote:
        if not candidate.startswith("'"):
            return None
        body = candidate[1:]
    else:
        body = candidate

    if "'" in body[:-1]:
        return None
    if body.endswith("'"):
        body = body[:-1]
    return body

class _PolicyStateLiteralTokenMixin(_PolicyStateTokenMatchingMixin):

    def _where_value_token_ids(
        self,
        generated_ids: list[int],
        text: str,
        state,
    ) -> set[int]:
        slot_text = self._where_value_slot_text(text, state)
        stripped = slot_text.lstrip()
        is_having = state.predicate_context == "HAVING"

        if not stripped:
            if is_having:
                return self._whitespace_token_ids() | self._numeric_token_ids(
                    "",
                    allow_decimal=False,
                    allow_negative=True,
                )
            return (
                self._whitespace_token_ids()
                | self._single_char_token_ids("'")
                | self._numeric_token_ids("", allow_decimal=False, allow_negative=True)
                | self._string_literal_start_token_ids()
                | self._literal_token_ids(
                    generated_ids,
                    text,
                    _WHERE_VALUE_INITIAL_LITERALS,
                )
                | self._where_value_column_token_ids(state, slot_text)
            )

        if self._is_numeric_prefix(stripped):
            return self._numeric_token_ids(stripped, allow_decimal=True, allow_negative=True)

        # HAVING beyond an in-progress number has no other valid continuation.
        if is_having:
            return set()

        if stripped.startswith("'"):
            allowed = self._string_literal_token_ids(stripped)
            suffixes = self._where_value_string_literal_exit_suffixes(state)
            if suffixes:
                allowed |= self._string_literal_close_with_suffix_token_ids(suffixes)
            return allowed

        if stripped.startswith("("):
            # Scalar / EXISTS subquery: grammar.allowed_branches dispatches
            # the inner SELECT scope, so this path adds nothing once "(" is in.
            return set()

        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\.]*", stripped, re.IGNORECASE):
            return self._where_value_column_token_ids(
                state,
                slot_text,
            ) | self._where_value_keyword_token_ids(
                slot_text,
                _WHERE_VALUE_KEYWORDS,
            )

        return self._literal_token_ids(
            generated_ids,
            text,
            _WHERE_VALUE_KEYWORDS,
        )

    def _where_value_column_token_ids(self, state, slot_text: str) -> set[int]:
        allowed_columns = self._columns_for_role("Where", state=state)
        if not allowed_columns:
            return set()

        slot_ids = self._encode_identifier_slot(slot_text)
        if self.allowed_trie.is_complete_allowed(slot_ids, allowed_columns):
            return set()

        return self.allowed_trie.get_allowed_token_ids(slot_ids, allowed_columns)

    def _where_value_keyword_token_ids(
        self,
        slot_text: str,
        keywords: set[str] | frozenset[str],
    ) -> set[int]:
        slot_ids = self._encode_identifier_slot(slot_text)
        if not slot_ids:
            return set()
        return self.literal_trie.get_allowed_token_ids(slot_ids, keywords)

    def _limit_value_token_ids(self, text: str) -> set[int]:
        slot_text = self._limit_value_slot_text(text)
        if not slot_text.strip():
            return self._whitespace_token_ids() | self._numeric_token_ids(
                "",
                allow_decimal=False,
                allow_negative=True,
            )
        return self._numeric_token_ids(
            slot_text.lstrip(),
            allow_decimal=False,
            allow_negative=True,
        )

    def _numeric_token_ids(
        self,
        stripped_slot_text: str,
        *,
        allow_decimal: bool,
        allow_negative: bool,
    ) -> set[int]:
        prefix_kind = _numeric_prefix_kind(
            stripped_slot_text,
            allow_decimal=allow_decimal,
            allow_negative=allow_negative,
        )
        if prefix_kind == "empty":
            chars = string.digits + ("-" if allow_negative else "")
            return self._single_char_token_ids(chars) | self._numeric_token_start_ids(
                allow_negative=allow_negative
            )

        if prefix_kind == "sign_only":
            return self._single_char_token_ids(string.digits) | self._numeric_digit_token_ids()

        if prefix_kind == "integer":
            allowed = self._single_char_token_ids(string.digits) | self._numeric_digit_token_ids()
            if allow_decimal and "." not in stripped_slot_text:
                allowed |= self._single_char_token_ids(".") | self._numeric_decimal_token_ids()
            return allowed

        if prefix_kind in {"decimal_point", "decimal_digits"}:
            return self._single_char_token_ids(string.digits) | self._numeric_digit_token_ids()

        return set()

    def _string_literal_token_ids(self, stripped_slot_text: str) -> set[int]:
        if self._is_closed_string_literal(stripped_slot_text):
            return set()

        return (
            self._single_char_token_ids(_SIMPLE_STRING_LITERAL_CHARS + "'")
            | self._string_literal_continuation_token_ids()
        )

    def _string_literal_start_token_ids(self) -> set[int]:
        return self._token_ids_matching(
            "string_literal_start",
            lambda text: self._is_string_literal_token_text(
                text,
                safe_chars=_SIMPLE_STRING_LITERAL_CHAR_SET,
                allow_leading_space=True,
            ),
        )

    def _string_literal_continuation_token_ids(self) -> set[int]:
        return self._token_ids_matching(
            "string_literal_continuation",
            lambda text: self._is_string_literal_token_text(
                text,
                safe_chars=_SIMPLE_STRING_LITERAL_CHAR_SET,
                allow_leading_space=False,
                require_opening_quote=False,
            ),
        )

    def _string_literal_close_with_suffix_token_ids(
        self,
        suffixes: tuple[str, ...],
    ) -> set[int]:
        suffix_key = "|".join(suffixes)
        return self._token_ids_matching(
            f"string_literal_suffix::{suffix_key}",
            lambda text: self._is_string_literal_suffix_token_text(
                text,
                safe_chars=_SIMPLE_STRING_LITERAL_CHAR_SET,
                suffixes=suffixes,
            ),
        )

    def _numeric_token_start_ids(self, *, allow_negative: bool) -> set[int]:
        cache_key = f"numeric_start::{allow_negative}"
        return self._token_ids_matching(
            cache_key,
            lambda text: self._is_numeric_token_text(
                text,
                allow_negative=allow_negative,
                allow_decimal=True,
                allow_leading_space=True,
            ),
        )

    def _numeric_digit_token_ids(self) -> set[int]:
        return self._regex_token_ids("numeric_digits", r"\d+")

    def _numeric_decimal_token_ids(self) -> set[int]:
        return self._regex_token_ids("numeric_decimal", r"\.\d*|\d+\.\d*")

    def _literal_token_ids(
        self,
        generated_ids: list[int],
        text: str,
        keywords: set[str] | frozenset[str],
    ) -> set[int]:
        return self.literal_trie.get_allowed_token_ids(
            generated_ids,
            {self.grammar.literal(text, keyword) for keyword in keywords},
        )

    @staticmethod
    def _where_value_string_literal_exit_suffixes(state) -> tuple[str, ...]:
        suffixes = [")"] if state.query_depth > 0 else []
        if state.query_depth == 0:
            suffixes.append(";")
        return tuple(suffixes)

    @staticmethod
    def _is_string_literal_token_text(
        text: str,
        *,
        safe_chars: set[str],
        allow_leading_space: bool,
        require_opening_quote: bool = True,
    ) -> bool:
        body = _simple_string_body(
            text,
            allow_leading_space=allow_leading_space,
            require_opening_quote=require_opening_quote,
        )
        return body is not None and all(ch in safe_chars for ch in body)

    @staticmethod
    def _is_string_literal_suffix_token_text(
        text: str,
        *,
        safe_chars: set[str],
        suffixes: tuple[str, ...],
    ) -> bool:
        for suffix in suffixes:
            if not text.endswith(suffix):
                continue
            body = text[: -len(suffix)]
            if not body.endswith("'"):
                continue
            inner = body[:-1]
            if "'" in inner:
                continue
            if all(ch in safe_chars for ch in inner):
                return True
        return False

    @staticmethod
    def _is_numeric_token_text(
        text: str,
        *,
        allow_negative: bool,
        allow_decimal: bool,
        allow_leading_space: bool,
    ) -> bool:
        candidate = text.lstrip() if allow_leading_space else text
        return (
            _numeric_prefix_kind(
                candidate,
                allow_decimal=allow_decimal,
                allow_negative=allow_negative,
            )
            is not None
        )

    @staticmethod
    def _is_numeric_prefix(stripped_slot_text: str) -> bool:
        return (
            _numeric_prefix_kind(
                stripped_slot_text,
                allow_decimal=True,
                allow_negative=True,
            )
            is not None
        )

    @staticmethod
    def _is_closed_string_literal(stripped_slot_text: str) -> bool:
        return bool(re.fullmatch(r"'[^']*'", stripped_slot_text, re.DOTALL))
