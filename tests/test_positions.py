from __future__ import annotations

import pytest

from onnxstego.positions import ChaCha20Rng, sample_unique


def test_sample_unique_is_deterministic_and_has_no_repeats() -> None:
    key = bytes(range(32))

    first = sample_unique(10_000, 500, key, b"domain")
    second = sample_unique(10_000, 500, key, b"domain")

    assert first == second
    assert len(first) == 500
    assert len(set(first)) == 500
    assert all(0 <= position < 10_000 for position in first)


def test_sample_unique_respects_exclusions() -> None:
    key = bytes(range(32))
    excluded = set(sample_unique(1_000, 100, key, b"excluded"))

    selected = sample_unique(1_000, 200, key, b"selected", exclude=excluded)

    assert len(set(selected)) == 200
    assert set(selected).isdisjoint(excluded)


def test_sample_unique_rejects_impossible_request() -> None:
    with pytest.raises(ValueError, match="capacity"):
        sample_unique(10, 11, bytes(range(32)), b"too-many")


def test_randbelow_rejects_invalid_upper_bound() -> None:
    rng = ChaCha20Rng(bytes(range(32)), b"test")

    with pytest.raises(ValueError, match="positive"):
        rng.randbelow(0)
