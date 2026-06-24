from __future__ import annotations

import numpy as np
import onnx

from onnxstego.onnx_io import Float32WeightStore, inspect_capacity

from .helpers import make_linear_model


def test_weight_store_reads_raw_data_as_little_endian_float32(tmp_path) -> None:
    model_path = tmp_path / "raw.onnx"
    original_weights = make_linear_model(model_path, weights_count=128)
    model = onnx.load(model_path)

    store = Float32WeightStore.from_model(model)
    observed = store.to_float_array()

    assert store.total_weights == 129
    np.testing.assert_allclose(observed[:128], original_weights.reshape(-1), rtol=0, atol=0)


def test_weight_store_converts_float_data_to_raw_data(tmp_path) -> None:
    model_path = tmp_path / "float_data.onnx"
    make_linear_model(model_path, weights_count=128, use_float_data=True)
    model = onnx.load(model_path)
    assert len(model.graph.initializer[0].raw_data) == 0
    assert len(model.graph.initializer[0].float_data) == 128

    store = Float32WeightStore.from_model(model)
    store.set_lsb(0, 1)
    store.flush()

    assert len(model.graph.initializer[0].raw_data) == 128 * 4
    assert len(model.graph.initializer[0].float_data) == 0


def test_inspect_capacity_counts_only_float32_initializers(tmp_path) -> None:
    model_path = tmp_path / "model.onnx"
    make_linear_model(model_path, weights_count=256)

    capacity = inspect_capacity(model_path)

    assert capacity.float32_weights == 257
    assert capacity.capacity_bits == 257
    assert capacity.capacity_bytes == 32
