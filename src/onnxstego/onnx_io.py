from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto

from onnxstego.bits import get_lsb, set_lsb


class UnsupportedModelError(Exception):
    """Raised when an ONNX model stores float32 data in an unsupported form."""


@dataclass(frozen=True)
class ModelCapacity:
    float32_weights: int
    capacity_bits: int
    capacity_bytes: int
    tensors: int


@dataclass
class _Segment:
    tensor: TensorProto
    start: int
    count: int
    raw: bytearray


class Float32WeightStore:
    def __init__(self, segments: list[_Segment]) -> None:
        self._segments = segments
        self._starts = [segment.start for segment in segments]
        self.total_weights = sum(segment.count for segment in segments)

    @classmethod
    def from_model(cls, model: onnx.ModelProto) -> "Float32WeightStore":
        segments: list[_Segment] = []
        start = 0
        for tensor in model.graph.initializer:
            if tensor.data_type != TensorProto.FLOAT:
                continue
            if tensor.data_location == TensorProto.EXTERNAL:
                raise UnsupportedModelError(
                    f"external float32 initializer is not supported: {tensor.name!r}"
                )
            if tensor.HasField("segment"):
                raise UnsupportedModelError(
                    f"segmented float32 initializer is not supported: {tensor.name!r}"
                )

            count = _tensor_element_count(tensor)
            expected_bytes = count * 4
            if tensor.raw_data:
                if len(tensor.raw_data) < expected_bytes:
                    raise UnsupportedModelError(
                        f"raw_data for {tensor.name!r} is shorter than tensor shape"
                    )
                raw = bytearray(tensor.raw_data[:expected_bytes])
            elif tensor.float_data:
                if len(tensor.float_data) != count:
                    raise UnsupportedModelError(
                        f"float_data for {tensor.name!r} does not match tensor shape"
                    )
                raw = bytearray(np.asarray(tensor.float_data, dtype="<f4").tobytes())
            else:
                raise UnsupportedModelError(
                    f"float32 initializer has no raw_data or float_data: {tensor.name!r}"
                )

            segments.append(_Segment(tensor=tensor, start=start, count=count, raw=raw))
            start += count

        return cls(segments)

    @property
    def segments(self) -> tuple[_Segment, ...]:
        return tuple(self._segments)

    def get_lsb(self, global_index: int) -> int:
        segment, byte_offset = self._locate(global_index)
        return get_lsb(segment.raw[byte_offset])

    def set_lsb(self, global_index: int, bit: int) -> bool:
        segment, byte_offset = self._locate(global_index)
        old_value = segment.raw[byte_offset]
        new_value = set_lsb(old_value, bit)
        segment.raw[byte_offset] = new_value
        return old_value != new_value

    def flush(self) -> None:
        for segment in self._segments:
            segment.tensor.raw_data = bytes(segment.raw)
            del segment.tensor.float_data[:]

    def lsb_bits(self) -> np.ndarray:
        if not self._segments:
            return np.array([], dtype=np.uint8)
        arrays = [
            np.frombuffer(bytes(segment.raw), dtype=np.uint8)[0::4] & np.uint8(1)
            for segment in self._segments
        ]
        return np.concatenate(arrays).astype(np.uint8, copy=False)

    def to_float_array(self) -> np.ndarray:
        if not self._segments:
            return np.array([], dtype="<f4")
        raw = b"".join(bytes(segment.raw) for segment in self._segments)
        return np.frombuffer(raw, dtype="<f4").copy()

    def _locate(self, global_index: int) -> tuple[_Segment, int]:
        if global_index < 0 or global_index >= self.total_weights:
            raise IndexError("weight index outside model capacity")
        segment_index = bisect.bisect_right(self._starts, global_index) - 1
        segment = self._segments[segment_index]
        local_index = global_index - segment.start
        return segment, local_index * 4


def inspect_capacity(model_path: str | Path) -> ModelCapacity:
    model = onnx.load(str(model_path))
    store = Float32WeightStore.from_model(model)
    return ModelCapacity(
        float32_weights=store.total_weights,
        capacity_bits=store.total_weights,
        capacity_bytes=store.total_weights // 8,
        tensors=len(store._segments),
    )


def sanitize_metadata(model: onnx.ModelProto) -> None:
    model.doc_string = ""
    model.producer_name = ""
    model.producer_version = ""
    del model.metadata_props[:]
    model.graph.doc_string = ""


def _tensor_element_count(tensor: TensorProto) -> int:
    if not tensor.dims:
        return 1
    if any(dim < 0 for dim in tensor.dims):
        raise UnsupportedModelError(f"negative dimension in initializer: {tensor.name!r}")
    return math.prod(int(dim) for dim in tensor.dims)
