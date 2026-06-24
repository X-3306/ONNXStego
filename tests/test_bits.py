from __future__ import annotations

import pytest

from onnxstego.bits import bits_to_bytes, bytes_to_bits, get_lsb, set_lsb


def test_bytes_to_bits_round_trips_in_msb_order() -> None:
    data = b"\x80\x01\xaa"

    bits = bytes_to_bits(data)

    assert bits[:8] == [1, 0, 0, 0, 0, 0, 0, 0]
    assert bits_to_bytes(bits) == data


def test_bits_to_bytes_rejects_non_byte_aligned_input() -> None:
    with pytest.raises(ValueError, match="multiple of 8"):
        bits_to_bytes([1, 0, 1])


def test_set_lsb_changes_only_the_lowest_bit() -> None:
    original = 0b1010_1010

    assert set_lsb(original, 1) == 0b1010_1011
    assert set_lsb(original, 0) == original
    assert get_lsb(0b1111_1110) == 0
    assert get_lsb(0b1111_1111) == 1
