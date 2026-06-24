from __future__ import annotations

from collections.abc import Iterable

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


class ChaCha20Rng:
    """Deterministic CSPRNG backed by ChaCha20 and HKDF domain separation."""

    def __init__(self, key: bytes, domain: bytes) -> None:
        if len(key) != 32:
            raise ValueError("position key must be 32 bytes")
        material = HKDF(
            algorithm=hashes.SHA256(),
            length=48,
            salt=b"onnx-stego:v1:position-stream",
            info=domain,
        ).derive(key)
        cipher = Cipher(algorithms.ChaCha20(material[:32], material[32:]), mode=None)
        self._encryptor = cipher.encryptor()
        self._buffer = bytearray()

    def read(self, size: int) -> bytes:
        while len(self._buffer) < size:
            self._buffer.extend(self._encryptor.update(b"\x00" * 4096))
        chunk = bytes(self._buffer[:size])
        del self._buffer[:size]
        return chunk

    def randbelow(self, upper_bound: int) -> int:
        if upper_bound <= 0:
            raise ValueError("upper bound must be positive")

        byte_len = max(1, (upper_bound.bit_length() + 7) // 8)
        range_size = 1 << (8 * byte_len)
        limit = range_size - (range_size % upper_bound)

        while True:
            candidate = int.from_bytes(self.read(byte_len), "big")
            if candidate < limit:
                return candidate % upper_bound


def sample_unique(
    total: int,
    count: int,
    key: bytes,
    domain: bytes,
    *,
    exclude: Iterable[int] | None = None,
) -> list[int]:
    if total < 0 or count < 0:
        raise ValueError("total and count must be non-negative")

    excluded = set(exclude or ())
    if any(position < 0 or position >= total for position in excluded):
        raise ValueError("excluded position outside capacity")

    available = total - len(excluded)
    if count > available:
        raise ValueError("not enough capacity to sample unique positions")

    rng = ChaCha20Rng(key, domain)
    swaps: dict[int, int] = {}
    selected: list[int] = []
    index = 0

    while len(selected) < count:
        if index >= total:
            raise RuntimeError("position sampler exhausted the model")
        chosen_index = index + rng.randbelow(total - index)
        current_value = swaps.get(index, index)
        chosen_value = swaps.get(chosen_index, chosen_index)
        swaps[index] = chosen_value
        swaps[chosen_index] = current_value

        if chosen_value not in excluded:
            selected.append(chosen_value)
        index += 1

    return selected
