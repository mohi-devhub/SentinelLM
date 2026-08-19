"""Unit tests for sentinel.tenancy.keys — generation, hashing, prefixing."""

from __future__ import annotations

from sentinel.tenancy.keys import generate_api_key, hash_key, key_prefix


def test_generate_api_key_has_expected_prefix():
    key = generate_api_key()
    assert key.startswith("sk-sentinel-")


def test_generate_api_key_is_random():
    assert generate_api_key() != generate_api_key()


def test_hash_key_is_deterministic():
    key = generate_api_key()
    assert hash_key(key) == hash_key(key)


def test_hash_key_differs_for_different_keys():
    assert hash_key("key-a") != hash_key("key-b")


def test_hash_key_is_a_sha256_hex_digest():
    digest = hash_key("some-key")
    assert len(digest) == 64
    int(digest, 16)  # raises ValueError if not valid hex


def test_hash_key_never_contains_the_plaintext():
    key = "sk-sentinel-super-secret-value"
    assert key not in hash_key(key)


def test_key_prefix_is_a_short_leading_substring():
    key = "sk-sentinel-abcdefghijklmnop"
    prefix = key_prefix(key)
    assert key.startswith(prefix)
    assert len(prefix) < len(key)
