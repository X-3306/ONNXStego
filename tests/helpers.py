from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def make_linear_model(
    path: Path,
    *,
    weights_count: int = 4096,
    use_float_data: bool = False,
    seed: int = 7,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.05, size=(weights_count, 1)).astype("<f4")
    bias = np.array([0.125], dtype="<f4")

    if use_float_data:
        w_tensor = helper.make_tensor(
            "W",
            TensorProto.FLOAT,
            weights.shape,
            weights.reshape(-1).astype(float).tolist(),
            raw=False,
        )
        b_tensor = helper.make_tensor(
            "B",
            TensorProto.FLOAT,
            bias.shape,
            bias.astype(float).tolist(),
            raw=False,
        )
    else:
        w_tensor = numpy_helper.from_array(weights, name="W")
        b_tensor = numpy_helper.from_array(bias, name="B")

    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, weights_count])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])
    matmul = helper.make_node("MatMul", ["X", "W"], ["Z"])
    add = helper.make_node("Add", ["Z", "B"], ["Y"])
    graph = helper.make_graph([matmul, add], "linear", [x_info], [y_info], [w_tensor, b_tensor])
    model = helper.make_model(graph, producer_name="onnx-stego-tests")
    model.ir_version = 10
    model.opset_import[0].version = 13
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return weights


def run_linear_model(path: Path, x: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {"X": x.astype(np.float32)})[0]
