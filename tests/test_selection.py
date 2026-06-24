from __future__ import annotations

import numpy as np
import onnx
import pytest

from onnxstego.core import CapacityError, embed_message, extract_message
from onnxstego.onnx_io import Float32WeightStore
from onnxstego.selection import NaturalWeightSelector, SelectionMode

from .helpers import make_linear_model


def _make_cover_from_base(base_path, cover_path, *, changed: int = 2048) -> None:
    model = onnx.load(base_path)
    store = Float32WeightStore.from_model(model)
    weights = store.to_float_array()
    for index in range(min(changed, weights.size)):
        weights[index] = np.float32(weights[index] + np.float32(1e-4 + index * 1e-9))
    cursor = 0
    for segment in store.segments:
        count = segment.count
        segment.raw[:] = weights[cursor : cursor + count].astype("<f4").tobytes()
        cursor += count
    store.flush()
    onnx.save(model, cover_path)


def test_natural_selector_returns_only_weights_above_delta_threshold(tmp_path) -> None:
    base_path = tmp_path / "base.onnx"
    cover_path = tmp_path / "cover.onnx"
    make_linear_model(base_path, weights_count=4096)
    _make_cover_from_base(base_path, cover_path, changed=1000)

    selector = NaturalWeightSelector.from_models(
        cover_path,
        base_path,
        min_abs_delta=1e-5,
    )

    candidates = selector.candidate_indices()

    assert len(candidates) == 1000
    assert set(candidates).issubset(set(range(1000)))


def test_natural_selection_round_trips_when_reference_model_is_available(tmp_path) -> None:
    key = bytes(range(32))
    base_path = tmp_path / "base.onnx"
    cover_path = tmp_path / "cover.onnx"
    stego_path = tmp_path / "stego.onnx"
    make_linear_model(base_path, weights_count=4096)
    _make_cover_from_base(base_path, cover_path, changed=4096)

    report = embed_message(
        cover_path,
        stego_path,
        key,
        b"natural selection",
        reference_model_path=base_path,
        selection_mode=SelectionMode.NATURAL,
        natural_min_abs_delta=1e-5,
    )

    assert report.selection_mode == "natural"
    assert report.candidate_weights == 4096
    assert extract_message(
        stego_path,
        key,
        reference_model_path=base_path,
        selection_mode=SelectionMode.NATURAL,
        natural_min_abs_delta=1e-5,
    ) == b"natural selection"


def test_natural_selection_rejects_insufficient_changed_weight_pool(tmp_path) -> None:
    key = bytes(range(32))
    base_path = tmp_path / "base.onnx"
    cover_path = tmp_path / "cover.onnx"
    stego_path = tmp_path / "stego.onnx"
    make_linear_model(base_path, weights_count=4096)
    _make_cover_from_base(base_path, cover_path, changed=128)

    with pytest.raises(CapacityError, match="candidate"):
        embed_message(
            cover_path,
            stego_path,
            key,
            b"too many bits for changed pool",
            reference_model_path=base_path,
            selection_mode=SelectionMode.NATURAL,
            natural_min_abs_delta=1e-5,
        )
