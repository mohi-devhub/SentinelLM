"""API key generation and hashing.

Keys are high-entropy random tokens, not user passwords, so a fast
cryptographic hash (SHA-256) is sufficient — there's no need for bcrypt/
argon2's deliberately-slow properties, which exist to slow down guessing
low-entropy secrets.
"""

from __future__ import annotations

import hashlib
import secrets

_KEY_PREFIX = "sk-sentinel-"
_PREFIX_DISPLAY_LEN = len(_KEY_PREFIX) + 8  # enough to distinguish keys at a glance


def generate_api_key() -> str:
    """Return a new random plaintext API key. Never persisted — only its hash is."""
    return f"{_KEY_PREFIX}{secrets.token_urlsafe(24)}"


def hash_key(plaintext_key: str) -> str:
    """Return the SHA-256 hex digest of a plaintext API key."""
    return hashlib.sha256(plaintext_key.encode()).hexdigest()


def key_prefix(plaintext_key: str) -> str:
    """Return a short prefix of a plaintext key, for display/lookup only."""
    return plaintext_key[:_PREFIX_DISPLAY_LEN]
