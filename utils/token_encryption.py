"""
utils/token_encryption.py
──────────────────────────
AES-256-GCM helpers for encrypting third-party access tokens (Messenger/
Instagram Page tokens, and any future channel credential) before they're
written to Mongo. GCM is authenticated — a wrong key or a tampered
ciphertext fails decryption loudly instead of returning garbage.

Key format: TOKEN_ENCRYPTION_KEY (config.settings) must be 64 hex
characters (32 raw bytes). Generate one with: openssl rand -hex 32
Losing or rotating this key makes every already-stored token undecryptable.
"""

from __future__ import annotations

import base64
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from config import settings

_NONCE_SIZE = 12  # 96-bit nonce — the size AES-GCM is designed for


class TokenEncryptionError(Exception):
    pass


def _load_key() -> bytes:
    raw = settings.TOKEN_ENCRYPTION_KEY
    if not raw:
        raise TokenEncryptionError(
            "TOKEN_ENCRYPTION_KEY is not set — generate one with `openssl rand -hex 32`"
        )
    try:
        key = bytes.fromhex(raw)
    except ValueError:
        raise TokenEncryptionError(
            "TOKEN_ENCRYPTION_KEY must be a 64-character hex string (32 raw bytes)"
        )
    if len(key) != 32:
        raise TokenEncryptionError(
            f"TOKEN_ENCRYPTION_KEY must decode to 32 bytes for AES-256, got {len(key)}"
        )
    return key


def encrypt_token(plaintext: str) -> str:
    """Encrypts a token for storage. Returns a single base64 string
    (nonce + ciphertext) safe to write directly into a Mongo document."""
    key = _load_key()
    nonce = os.urandom(_NONCE_SIZE)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext.encode("utf-8"), None)
    return base64.b64encode(nonce + ciphertext).decode("ascii")


def decrypt_token(encrypted: str) -> str:
    """Reverses encrypt_token. Raises TokenEncryptionError if the value is
    malformed, was encrypted under a different key, or was tampered with —
    callers should treat that as "this credential is unusable" (prompt a
    reconnect), not retry."""
    key = _load_key()
    try:
        raw = base64.b64decode(encrypted)
    except Exception as exc:
        raise TokenEncryptionError("Encrypted token is not valid base64") from exc

    if len(raw) <= _NONCE_SIZE:
        raise TokenEncryptionError("Encrypted token is too short to contain a nonce")

    nonce, ciphertext = raw[:_NONCE_SIZE], raw[_NONCE_SIZE:]
    try:
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
    except Exception as exc:
        # Deliberately vague — never surface *why* decryption failed (wrong
        # key vs. tampered ciphertext) to a caller or a log line.
        raise TokenEncryptionError("Failed to decrypt token") from exc
    return plaintext.decode("utf-8")
