from __future__ import annotations

import onnx

from onnxstego.core import embed_message, extract_message
from onnxstego.empirical import compare_lsb_statistics, compare_weight_arrays
from onnxstego.onnx_io import Float32WeightStore

from .helpers import make_linear_model


def test_empirical_statistics_stay_close_for_sparse_embedding(tmp_path) -> None:
    key = bytes(range(32))
    input_model = tmp_path / "base.onnx"
    output_model = tmp_path / "stego.onnx"
    make_linear_model(input_model, weights_count=100_000)

    base_model = onnx.load(input_model)
    base_store = Float32WeightStore.from_model(base_model)
    base_bits = base_store.lsb_bits()
    base_weights = base_store.to_float_array()

    report = embed_message(input_model, output_model, key, b"krotka wiadomosc")
    assert extract_message(output_model, key) == b"krotka wiadomosc"

    stego_model = onnx.load(output_model)
    stego_store = Float32WeightStore.from_model(stego_model)
    stego_bits = stego_store.lsb_bits()
    stego_weights = stego_store.to_float_array()
    lsb_stats = compare_lsb_statistics(base_bits, stego_bits)
    weight_stats = compare_weight_arrays(base_weights, stego_weights)

    assert report.embedding_density < 0.01
    assert abs(lsb_stats.ones_ratio_delta) < 0.01
    assert lsb_stats.after_chi_square_p_value > 1e-4
    assert weight_stats.changed_count <= report.required_bits
    assert weight_stats.max_abs_delta < 1e-6
