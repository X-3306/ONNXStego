from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402 <---- because sys.path must be set first
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from onnxstego.core import embed_message, extract_message  # noqa: E402
from onnxstego.crypto import AuthenticationError, generate_master_key  # noqa: E402
from onnxstego.empirical import compare_lsb_statistics, compare_weight_arrays  # noqa: E402
from onnxstego.onnx_io import Float32WeightStore  # noqa: E402


def main() -> int:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    base_path = reports_dir / "empirical_base.onnx"
    stego_path = reports_dir / "empirical_stego.onnx"
    inference_base = reports_dir / "inference_base.onnx"
    inference_stego = reports_dir / "inference_stego.onnx"
    report_path = reports_dir / "empirical_report.json"

    key = generate_master_key()
    make_linear_model(base_path, weights_count=100_000, seed=11)
    base_model = onnx.load(base_path)
    base_store = Float32WeightStore.from_model(base_model)
    base_bits = base_store.lsb_bits()
    base_weights = base_store.to_float_array()

    embed_report = embed_message(base_path, stego_path, key, b"empirical safety smoke test")
    recovered = extract_message(stego_path, key)
    wrong_key_rejected = False
    try:
        extract_message(stego_path, b"\x00" * 32)
    except AuthenticationError:
        wrong_key_rejected = True

    stego_model = onnx.load(stego_path)
    onnx.checker.check_model(stego_model)
    stego_store = Float32WeightStore.from_model(stego_model)
    lsb_stats = compare_lsb_statistics(base_bits, stego_store.lsb_bits())
    weight_stats = compare_weight_arrays(base_weights, stego_store.to_float_array())

    make_linear_model(inference_base, weights_count=4096, seed=19)
    x = np.linspace(-1.0, 1.0, 4096, dtype=np.float32).reshape(1, -1)
    before = run_model(inference_base, x)
    embed_message(inference_base, inference_stego, key, b"inference check")
    after = run_model(inference_stego, x)
    max_inference_delta = float(np.max(np.abs(before - after)))

    report = {
        "round_trip_ok": recovered == b"empirical safety smoke test",
        "wrong_key_rejected": wrong_key_rejected,
        "onnx_checker_ok": True,
        "max_inference_delta": max_inference_delta,
        "embed_report": embed_report.to_dict(),
        "lsb_statistics": asdict(lsb_stats),
        "weight_delta_statistics": asdict(weight_stats),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nreport written to {report_path}")
    return 0


def make_linear_model(path: Path, *, weights_count: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.05, size=(weights_count, 1)).astype("<f4")
    bias = np.array([0.125], dtype="<f4")
    w_tensor = numpy_helper.from_array(weights, name="W")
    b_tensor = numpy_helper.from_array(bias, name="B")
    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, weights_count])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["X", "W"], ["Z"]),
            helper.make_node("Add", ["Z", "B"], ["Y"]),
        ],
        "linear",
        [x_info],
        [y_info],
        [w_tensor, b_tensor],
    )
    model = helper.make_model(graph, producer_name="onnx-stego-empirical")
    model.ir_version = 10
    model.opset_import[0].version = 13
    onnx.checker.check_model(model)
    onnx.save(model, path)


def run_model(path: Path, x: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {"X": x})[0]


if __name__ == "__main__":
    raise SystemExit(main())
