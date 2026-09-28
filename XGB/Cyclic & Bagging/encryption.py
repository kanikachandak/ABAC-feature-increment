# encryption.py  –  Lightweight AES-256-GCM encryption for FL model parameters.
#
# Design goals:
#   • Lossless        – AES-GCM is perfectly reversible; model bytes are unchanged
#   • Authenticated   – GCM authentication tag detects tampering / corruption
#   • Lightweight     – AES-GCM is hardware-accelerated on modern CPUs; overhead
#                       for a 50–500 KB XGBoost model is < 1 ms
#   • Zero model impact – encryption wraps the serialised bytes, the loaded model
#                         is byte-for-byte identical to the original
#
# Usage:
#   Set FL_SHARED_SECRET environment variable to any string before running.
#   Server and all clients must use the same value.
#   If unset, a default is used (fine for local experiments; change for production).
#
#   pip install cryptography   (already required by flwr)

import os
import logging
from functools import lru_cache

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidTag

_log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_NONCE_LEN  = 12   # 96-bit GCM nonce (NIST recommended)
_KEY_LEN    = 32   # 256-bit AES key
_DEFAULT_SECRET = "fl-xgb-policy-mining-2024"   # change in production


# ── Key derivation ────────────────────────────────────────────────────────────

def derive_key(shared_secret: str, context: str = "fl-xgb-v1") -> bytes:
    """Derive a 256-bit AES key from a shared secret string using HKDF-SHA256.

    HKDF (RFC 5869) turns an arbitrary-length secret into a cryptographically
    strong fixed-length key.  The context string differentiates keys for
    different purposes if the same secret is reused elsewhere.
    """
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=_KEY_LEN,
        salt=None,
        info=context.encode(),
    )
    return hkdf.derive(shared_secret.encode())


@lru_cache(maxsize=1)
def get_fl_key() -> bytes:
    """Return the FL encryption key (derived once, cached for the process lifetime).

    Reads FL_SHARED_SECRET from the environment.  The key is the same on the
    server and every client as long as they share the same environment variable.
    """
    secret = os.environ.get("FL_SHARED_SECRET", _DEFAULT_SECRET)
    if secret == _DEFAULT_SECRET:
        _log.warning(
            "[Encryption] FL_SHARED_SECRET not set — using default secret. "
            "Set FL_SHARED_SECRET=<your-secret> before running in production."
        )
    key = derive_key(secret)
    _log.info("[Encryption] AES-256-GCM key derived (HKDF-SHA256). "
              "Secret source: %s",
              "FL_SHARED_SECRET env var" if secret != _DEFAULT_SECRET
              else "default (set FL_SHARED_SECRET for production)")
    return key


# ── Core encrypt / decrypt ────────────────────────────────────────────────────

def encrypt_bytes(data: bytes, key: bytes) -> bytes:
    """Encrypt *data* with AES-256-GCM.

    Wire format:  nonce (12 B) | ciphertext | GCM-tag (16 B)

    Properties:
    • Confidentiality – ciphertext is computationally indistinguishable from random
    • Integrity       – the 16-byte authentication tag detects any bit-flip or
                        tampering; decrypt raises InvalidTag instead of returning
                        corrupted data
    • Freshness       – random 96-bit nonce means encrypting the same model twice
                        produces different ciphertexts (no replay detection needed
                        because GCM-tag already covers the nonce)
    """
    nonce = os.urandom(_NONCE_LEN)
    ct    = AESGCM(key).encrypt(nonce, data, None)   # aad=None
    return nonce + ct


def decrypt_bytes(data: bytes, key: bytes) -> bytes:
    """Decrypt AES-256-GCM ciphertext.

    Raises:
        cryptography.exceptions.InvalidTag  – if the data was tampered with or
                                              the wrong key is used.
        ValueError                          – if data is shorter than nonce length.
    """
    if len(data) < _NONCE_LEN + 16:   # nonce + minimum 16-byte GCM tag
        raise ValueError(
            f"Ciphertext too short ({len(data)} B) — expected at least "
            f"{_NONCE_LEN + 16} B (nonce + GCM tag)."
        )
    nonce, ct = data[:_NONCE_LEN], data[_NONCE_LEN:]
    try:
        return AESGCM(key).decrypt(nonce, ct, None)
    except InvalidTag:
        raise InvalidTag(
            "Decryption failed: authentication tag mismatch. "
            "Possible causes: wrong FL_SHARED_SECRET, data corruption, "
            "or tampering detected."
        )


# ── Convenience wrappers for Flower Parameters tensors ────────────────────────

def encrypt_tensors(tensors: list[bytes], key: bytes) -> list[bytes]:
    """Encrypt every tensor in a Parameters.tensors list."""
    return [encrypt_bytes(bytes(t), key) for t in tensors]


def decrypt_tensors(tensors: list[bytes], key: bytes) -> list[bytes]:
    """Decrypt every tensor in a Parameters.tensors list."""
    return [decrypt_bytes(bytes(t), key) for t in tensors]
