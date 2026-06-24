from __future__ import annotations

import numpy as np
import onnx
import pytest

from onnxstego.core import CapacityError, embed_message, extract_message
from onnxstego.crypto import AuthenticationError, HEADER_ENVELOPE_SIZE, derive_static_keys
from onnxstego.onnx_io import Float32WeightStore
from onnxstego.positions import sample_unique

from .helpers import make_linear_model, run_linear_model


def test_embed_extract_round_trip_and_preserves_onnx_validity(tmp_path) -> None:
    key = bytes(range(32))
    input_model = tmp_path / "input.onnx"
    output_model = tmp_path / "stego.onnx"
    make_linear_model(input_model, weights_count=4096)

    report = embed_message(input_model, output_model, key, b"sekret testowy")

    assert report.required_bits > HEADER_ENVELOPE_SIZE * 8
    assert report.changed_weights <= report.required_bits
    assert report.selection_mode == "uniform"
    assert report.candidate_weights == report.capacity_bits
    assert output_model.exists()
    onnx.checker.check_model(onnx.load(output_model))
    assert extract_message(output_model, key) == b"sekret testowy"


def test_wrong_key_and_tampered_model_fail_to_extract(tmp_path) -> None:
    key = bytes(range(32))
    input_model = tmp_path / "input.onnx"
    output_model = tmp_path / "stego.onnx"
    tampered_model = tmp_path / "tampered.onnx"
    make_linear_model(input_model, weights_count=4096)
    embed_message(input_model, output_model, key, b"do not change")

    with pytest.raises(AuthenticationError):
        extract_message(output_model, b"z" * 32)

    model = onnx.load(output_model)
    store = Float32WeightStore.from_model(model)
    static_keys = derive_static_keys(key)
    header_positions = sample_unique(
        store.total_weights,
        HEADER_ENVELOPE_SIZE * 8,
        static_keys.header_position_key,
        b"header-v1",
    )
    first_bit = store.get_lsb(header_positions[0])
    store.set_lsb(header_positions[0], first_bit ^ 1)
    store.flush()
    onnx.save(model, tampered_model)

    with pytest.raises(AuthenticationError):
        extract_message(tampered_model, key)


def test_embedding_rejects_models_without_enough_float32_capacity(tmp_path) -> None:
    key = bytes(range(32))
    input_model = tmp_path / "tiny.onnx"
    output_model = tmp_path / "tiny-stego.onnx"
    make_linear_model(input_model, weights_count=32)

    with pytest.raises(CapacityError, match="capacity"):
        embed_message(input_model, output_model, key, b"too large for this model")


def test_embedding_float_data_model_round_trips_after_conversion(tmp_path) -> None:
    key = bytes(range(32))
    input_model = tmp_path / "float-data.onnx"
    output_model = tmp_path / "float-data-stego.onnx"
    make_linear_model(input_model, weights_count=4096, use_float_data=True)

    embed_message(input_model, output_model, key, b"float_data path")

    model = onnx.load(output_model)
    assert len(model.graph.initializer[0].raw_data) > 0
    assert len(model.graph.initializer[0].float_data) == 0
    assert extract_message(output_model, key) == b"float_data path"


def test_inference_output_changes_only_slightly_after_embedding(tmp_path) -> None:
    key = bytes(range(32))
    input_model = tmp_path / "input.onnx"
    output_model = tmp_path / "stego.onnx"
    make_linear_model(input_model, weights_count=4096)
    x = np.linspace(-1.0, 1.0, 4096, dtype=np.float32).reshape(1, -1)
    before = run_linear_model(input_model, x)

    embed_message(input_model, output_model, key, b"inference drift check")
    after = run_linear_model(output_model, x)

    assert float(np.max(np.abs(before - after))) < 1e-3
