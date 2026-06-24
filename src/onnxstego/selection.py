from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import numpy as np
import onnx

from onnxstego.onnx_io import Float32WeightStore


class SelectionMode(StrEnum):
    UNIFORM = "uniform"
    NATURAL = "natural"


class NaturalWeightSelector:
    """Select weights that already changed relative to a reference model."""

    def __init__(self, indices: np.ndarray) -> None:
        if indices.dtype != np.int64:
            indices = indices.astype(np.int64, copy=False)
        self._indices = indices

    @classmethod
    def from_models(
        cls,
        cover_model_path: str | Path,
        reference_model_path: str | Path,
        *,
        min_abs_delta: float,
    ) -> "NaturalWeightSelector":
        if min_abs_delta <= 0:
            raise ValueError("natural selection threshold must be positive")

        cover = Float32WeightStore.from_model(onnx.load(str(cover_model_path))).to_float_array()
        reference = Float32WeightStore.from_model(onnx.load(str(reference_model_path))).to_float_array()
        if cover.shape != reference.shape:
            raise ValueError("cover and reference models must expose the same float32 weight shape")

        deltas = np.abs(cover.astype(np.float64) - reference.astype(np.float64))
        indices = np.flatnonzero(deltas >= min_abs_delta).astype(np.int64, copy=False)
        return cls(indices)

    def candidate_indices(self) -> list[int]:
        return self._indices.astype(int).tolist()

    @property
    def count(self) -> int:
        return int(self._indices.size)
