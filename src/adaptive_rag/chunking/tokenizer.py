"""
chunking.tokenizer
------------------
Token counting utility using tiktoken with graceful offline heuristic fallback.
"""

from typing import Any

_TIKTOKEN_ENCODER = None


def count_tokens(text: str) -> int:
    """Return token count for a text string using cl100k_base or length heuristic."""
    global _TIKTOKEN_ENCODER
    if not text:
        return 0

    if _TIKTOKEN_ENCODER is None:
        try:
            import tiktoken
            _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TIKTOKEN_ENCODER = False

    if _TIKTOKEN_ENCODER is not False:
        try:
            return len(_TIKTOKEN_ENCODER.encode(text, disallowed_special=()))
        except Exception:
            pass

    # Fallback heuristic: 1 token ~= 4 characters for English text
    return max(1, len(text) // 4)
