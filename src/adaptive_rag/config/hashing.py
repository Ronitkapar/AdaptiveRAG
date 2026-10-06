"""
config.hashing
--------------
Deterministic hashing and canonical JSON serialization utilities.
Guarantees byte-stable representations and content-addressed cache keys.
"""

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(data: Any) -> str:
    """Serialize data into a deterministic, byte-stable JSON string.

    Uses sorted keys, no whitespace around separators, and utf-8 encoding.
    """
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def compute_sha256(text: str) -> str:
    """Compute hex SHA-256 digest of a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_file_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_config_hash(data: Any, prefix_len: int = 16) -> str:
    """Compute a truncated deterministic hash for a configuration dictionary or object.

    Handles Pydantic models by using model_dump(mode='json').
    """
    if hasattr(data, "model_dump"):
        data = data.model_dump(mode="json")
    json_bytes = canonical_json(data).encode("utf-8")
    full_hex = hashlib.sha256(json_bytes).hexdigest()
    return full_hex[:prefix_len]
