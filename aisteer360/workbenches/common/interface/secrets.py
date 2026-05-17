"""Symmetric encryption for API keys stored at rest.

Keys are encrypted with Fernet (AES-128-CBC + HMAC-SHA256) before being written to the SQLite
database, and decrypted only when the agent claims a run. The Fernet key itself is resolved from
`AISTEER_SECRET_KEY` when set (recommended for production), otherwise from a persistent file at
`~/.aisteer360/secret.key` that is auto-generated on first use.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

_KEY_ENV = "AISTEER_SECRET_KEY"
_KEY_FILE = Path.home() / ".aisteer360" / "secret.key"

_fernet: Fernet | None = None


def _resolve_fernet_key() -> bytes:
    """Resolve the Fernet key from env var or a persistent file, generating one if needed."""
    env = os.environ.get(_KEY_ENV)
    if env:
        return env.encode("utf-8")

    if _KEY_FILE.exists():
        return _KEY_FILE.read_bytes().strip()

    key = Fernet.generate_key()
    _KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    _KEY_FILE.write_bytes(key)
    try:
        _KEY_FILE.chmod(0o600)
    except OSError:
        logger.debug("Could not chmod %s", _KEY_FILE)
    logger.info("Generated new server secret key at %s", _KEY_FILE)
    return key


def get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        _fernet = Fernet(_resolve_fernet_key())
    return _fernet


def encrypt(plaintext: str) -> str:
    """Encrypt a string, returning a URL-safe base64 token."""
    return get_fernet().encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt(token: str) -> str:
    """Decrypt a Fernet token back to plaintext."""
    return get_fernet().decrypt(token.encode("utf-8")).decode("utf-8")


__all__ = ["encrypt", "decrypt", "get_fernet"]
