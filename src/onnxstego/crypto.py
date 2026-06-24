from __future__ import annotations

import base64
import os
import re
import struct
from collections.abc import Callable
from dataclasses import dataclass

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

MASTER_KEY_SIZE = 32
SESSION_SALT_SIZE = 16
CHACHA20_POLY1305_NONCE_SIZE = 12
CHACHA20_POLY1305_TAG_SIZE = 16
KEY_PREFIX = "onxs1_"

HEADER_VERSION = 1
HEADER_FLAGS = 0
HEADER_STRUCT = struct.Struct(">BBI16s12s")
HEADER_PLAINTEXT_SIZE = HEADER_STRUCT.size
HEADER_ENVELOPE_SIZE = (
    CHACHA20_POLY1305_NONCE_SIZE + HEADER_PLAINTEXT_SIZE + CHACHA20_POLY1305_TAG_SIZE
)
HEADER_AAD = b"onnx-stego:v1:header"
PAYLOAD_AAD_PREFIX = b"onnx-stego:v1:payload:"


class AuthenticationError(Exception):
    """Raised when a hidden message cannot be authenticated."""


@dataclass(frozen=True)
class StaticKeys:
    header_key: bytes
    header_position_key: bytes


@dataclass(frozen=True)
class PayloadKeys:
    payload_key: bytes
    position_key: bytes


@dataclass(frozen=True)
class Header:
    payload_ciphertext_len: int
    salt: bytes
    payload_nonce: bytes
    version: int = HEADER_VERSION
    flags: int = HEADER_FLAGS

    def to_plaintext(self) -> bytes:
        return HEADER_STRUCT.pack(
            self.version,
            self.flags,
            self.payload_ciphertext_len,
            self.salt,
            self.payload_nonce,
        )

    @classmethod
    def from_plaintext(cls, data: bytes) -> "Header":
        if len(data) != HEADER_PLAINTEXT_SIZE:
            raise AuthenticationError("invalid stego header length")
        version, flags, payload_len, salt, payload_nonce = HEADER_STRUCT.unpack(data)
        if version != HEADER_VERSION or flags != HEADER_FLAGS:
            raise AuthenticationError("unsupported stego header")
        if payload_len < CHACHA20_POLY1305_TAG_SIZE:
            raise AuthenticationError("invalid payload length")
        return cls(
            payload_ciphertext_len=payload_len,
            salt=salt,
            payload_nonce=payload_nonce,
            version=version,
            flags=flags,
        )


@dataclass(frozen=True)
class Envelope:
    header: Header
    header_nonce: bytes
    header_ciphertext: bytes
    payload_ciphertext: bytes

    @property
    def header_bytes(self) -> bytes:
        return self.header_nonce + self.header_ciphertext


def generate_master_key() -> bytes:
    return os.urandom(MASTER_KEY_SIZE)


def format_master_key(key: bytes) -> str:
    _require_master_key(key)
    encoded = base64.urlsafe_b64encode(key).decode("ascii").rstrip("=")
    return f"{KEY_PREFIX}{encoded}"


def parse_master_key(value: str) -> bytes:
    text = value.strip()
    if text.startswith(KEY_PREFIX):
        text = text[len(KEY_PREFIX) :]

    if re.fullmatch(r"[0-9a-fA-F]{64}", text):
        key = bytes.fromhex(text)
    else:
        padded = text + "=" * (-len(text) % 4)
        try:
            key = base64.urlsafe_b64decode(padded.encode("ascii"))
        except Exception as exc:  # noqa: BLE001 - normalize parser errors for CLI/API users.
            raise ValueError("master key must be 32 bytes as hex or base64url") from exc

    _require_master_key(key)
    return key


def derive_static_keys(master_key: bytes) -> StaticKeys:
    _require_master_key(master_key)
    return StaticKeys(
        header_key=_hkdf(master_key, b"onnx-stego:v1:static", b"header-aead", 32),
        header_position_key=_hkdf(master_key, b"onnx-stego:v1:static", b"header-positions", 32),
    )


def derive_payload_keys(master_key: bytes, salt: bytes) -> PayloadKeys:
    _require_master_key(master_key)
    if len(salt) != SESSION_SALT_SIZE:
        raise ValueError("payload salt must be 16 bytes")
    return PayloadKeys(
        payload_key=_hkdf(master_key, salt, b"payload-aead", 32),
        position_key=_hkdf(master_key, salt, b"payload-positions", 32),
    )


def encrypt_message(
    master_key: bytes,
    message: bytes,
    *,
    randbytes: Callable[[int], bytes] | None = None,
) -> Envelope:
    _require_master_key(master_key)
    random_bytes = randbytes or os.urandom
    salt = _checked_random(random_bytes, SESSION_SALT_SIZE)
    payload_nonce = _checked_random(random_bytes, CHACHA20_POLY1305_NONCE_SIZE)
    header_nonce = _checked_random(random_bytes, CHACHA20_POLY1305_NONCE_SIZE)

    payload_ciphertext_len = len(message) + CHACHA20_POLY1305_TAG_SIZE
    header = Header(
        payload_ciphertext_len=payload_ciphertext_len,
        salt=salt,
        payload_nonce=payload_nonce,
    )
    header_plaintext = header.to_plaintext()

    payload_keys = derive_payload_keys(master_key, salt)
    payload_aad = PAYLOAD_AAD_PREFIX + header_plaintext
    payload_ciphertext = ChaCha20Poly1305(payload_keys.payload_key).encrypt(
        payload_nonce,
        message,
        payload_aad,
    )

    static_keys = derive_static_keys(master_key)
    header_ciphertext = ChaCha20Poly1305(static_keys.header_key).encrypt(
        header_nonce,
        header_plaintext,
        HEADER_AAD,
    )
    return Envelope(
        header=header,
        header_nonce=header_nonce,
        header_ciphertext=header_ciphertext,
        payload_ciphertext=payload_ciphertext,
    )


def decrypt_header(master_key: bytes, header_bytes: bytes) -> Header:
    _require_master_key(master_key)
    if len(header_bytes) != HEADER_ENVELOPE_SIZE:
        raise AuthenticationError("invalid stego header size")

    header_nonce = header_bytes[:CHACHA20_POLY1305_NONCE_SIZE]
    header_ciphertext = header_bytes[CHACHA20_POLY1305_NONCE_SIZE:]
    static_keys = derive_static_keys(master_key)
    try:
        header_plaintext = ChaCha20Poly1305(static_keys.header_key).decrypt(
            header_nonce,
            header_ciphertext,
            HEADER_AAD,
        )
    except InvalidTag as exc:
        raise AuthenticationError("wrong key or damaged stego header") from exc
    return Header.from_plaintext(header_plaintext)


def decrypt_message(master_key: bytes, header_bytes: bytes, payload_ciphertext: bytes) -> bytes:
    header = decrypt_header(master_key, header_bytes)
    if len(payload_ciphertext) != header.payload_ciphertext_len:
        raise AuthenticationError("payload length does not match authenticated header")

    payload_keys = derive_payload_keys(master_key, header.salt)
    payload_aad = PAYLOAD_AAD_PREFIX + header.to_plaintext()
    try:
        return ChaCha20Poly1305(payload_keys.payload_key).decrypt(
            header.payload_nonce,
            payload_ciphertext,
            payload_aad,
        )
    except InvalidTag as exc:
        raise AuthenticationError("wrong key or damaged stego payload") from exc


def _hkdf(master_key: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        info=info,
    ).derive(master_key)


def _require_master_key(key: bytes) -> None:
    if not isinstance(key, bytes) or len(key) != MASTER_KEY_SIZE:
        raise ValueError("master key must be exactly 32 bytes")


def _checked_random(randbytes: Callable[[int], bytes], size: int) -> bytes:
    value = randbytes(size)
    if len(value) != size:
        raise ValueError("random source returned an unexpected number of bytes")
    return value
