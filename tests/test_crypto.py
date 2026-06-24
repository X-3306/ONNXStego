from __future__ import annotations

import pytest

from onnxstego.crypto import (
    AuthenticationError,
    decrypt_message,
    derive_payload_keys,
    encrypt_message,
    format_master_key,
    generate_master_key,
    parse_master_key,
)


def test_master_key_format_round_trips_base64url_and_hex() -> None:
    key = bytes(range(32))

    formatted = format_master_key(key)

    assert parse_master_key(formatted) == key
    assert parse_master_key(key.hex()) == key
    assert len(generate_master_key()) == 32


def test_encrypt_message_decrypts_with_same_key() -> None:
    key = bytes(range(32))
    envelope = encrypt_message(key, b"tajna wiadomosc")

    recovered = decrypt_message(key, envelope.header_bytes, envelope.payload_ciphertext)

    assert recovered == b"tajna wiadomosc"


def test_encrypt_message_uses_fresh_random_envelopes() -> None:
    key = bytes(range(32))

    first = encrypt_message(key, b"same plaintext")
    second = encrypt_message(key, b"same plaintext")

    assert first.header_bytes != second.header_bytes
    assert first.payload_ciphertext != second.payload_ciphertext
    assert decrypt_message(key, first.header_bytes, first.payload_ciphertext) == b"same plaintext"
    assert decrypt_message(key, second.header_bytes, second.payload_ciphertext) == b"same plaintext"


def test_wrong_key_and_tampering_are_rejected() -> None:
    key = bytes(range(32))
    wrong_key = b"x" * 32
    envelope = encrypt_message(key, b"authenticated")

    with pytest.raises(AuthenticationError):
        decrypt_message(wrong_key, envelope.header_bytes, envelope.payload_ciphertext)

    tampered = bytearray(envelope.payload_ciphertext)
    tampered[0] ^= 0x01

    with pytest.raises(AuthenticationError):
        decrypt_message(key, envelope.header_bytes, bytes(tampered))


def test_payload_position_key_depends_on_session_salt() -> None:
    key = bytes(range(32))

    first = encrypt_message(key, b"payload")
    second = encrypt_message(key, b"payload")

    assert derive_payload_keys(key, first.header.salt).position_key != derive_payload_keys(
        key, second.header.salt
    ).position_key
