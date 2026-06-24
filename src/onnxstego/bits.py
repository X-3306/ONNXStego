from __future__ import annotations

from collections.abc import Iterable


def bytes_to_bits(data: bytes) -> list[int]:
    """Convert bytes to MSB-first bits."""
    return [(byte >> shift) & 0x01 for byte in data for shift in range(7, -1, -1)]


def bits_to_bytes(bits: Iterable[int]) -> bytes:
    values = list(bits)
    if len(values) % 8 != 0:
        raise ValueError("bit length must be a multiple of 8")

    output = bytearray()
    for index in range(0, len(values), 8):
        byte = 0
        for bit in values[index : index + 8]:
            if bit not in (0, 1):
                raise ValueError("bits must be 0 or 1")
            byte = (byte << 1) | bit
        output.append(byte)
    return bytes(output)


def get_lsb(value: int) -> int:
    return value & 0x01


def set_lsb(value: int, bit: int) -> int:
    if bit not in (0, 1):
        raise ValueError("bit must be 0 or 1")
    return (value & 0xFE) | bit
