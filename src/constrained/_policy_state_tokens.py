
from __future__ import annotations

from ._policy_state_column_tokens import _PolicyStateColumnTokenMixin
from ._policy_state_literal_tokens import _PolicyStateLiteralTokenMixin

__all__ = ["_PolicyStateTokenHelperMixin"]

class _PolicyStateTokenHelperMixin(
    _PolicyStateColumnTokenMixin,
    _PolicyStateLiteralTokenMixin,
):
    """Compose the internal token-helper mixins for policy-state decoding."""
