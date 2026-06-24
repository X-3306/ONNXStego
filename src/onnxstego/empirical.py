from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LsbStatistics:
    before_ones_ratio: float
    after_ones_ratio: float
    ones_ratio_delta: float
    before_chi_square: float
    after_chi_square: float
    before_chi_square_p_value: float
    after_chi_square_p_value: float


@dataclass(frozen=True)
class WeightDeltaStatistics:
    changed_count: int
    max_abs_delta: float
    mean_abs_delta: float


def compare_lsb_statistics(before: np.ndarray, after: np.ndarray) -> LsbStatistics:
    if before.shape != after.shape:
        raise ValueError("before and after bit arrays must have the same shape")
    before_ratio = _ones_ratio(before)
    after_ratio = _ones_ratio(after)
    before_chi = _chi_square_against_fair_bits(before)
    after_chi = _chi_square_against_fair_bits(after)
    return LsbStatistics(
        before_ones_ratio=before_ratio,
        after_ones_ratio=after_ratio,
        ones_ratio_delta=after_ratio - before_ratio,
        before_chi_square=before_chi,
        after_chi_square=after_chi,
        before_chi_square_p_value=_chi_square_df1_survival(before_chi),
        after_chi_square_p_value=_chi_square_df1_survival(after_chi),
    )


def compare_weight_arrays(before: np.ndarray, after: np.ndarray) -> WeightDeltaStatistics:
    if before.shape != after.shape:
        raise ValueError("before and after weight arrays must have the same shape")
    deltas = np.abs(after.astype(np.float64) - before.astype(np.float64))
    changed = deltas[deltas > 0]
    return WeightDeltaStatistics(
        changed_count=int(changed.size),
        max_abs_delta=float(changed.max(initial=0.0)),
        mean_abs_delta=float(changed.mean() if changed.size else 0.0),
    )


def _ones_ratio(bits: np.ndarray) -> float:
    if bits.size == 0:
        return 0.0
    return float(np.count_nonzero(bits) / bits.size)


def _chi_square_against_fair_bits(bits: np.ndarray) -> float:
    if bits.size == 0:
        return 0.0
    ones = int(np.count_nonzero(bits))
    zeros = int(bits.size - ones)
    expected = bits.size / 2.0
    return ((zeros - expected) ** 2 / expected) + ((ones - expected) ** 2 / expected)


def _chi_square_df1_survival(chi_square: float) -> float:
    return math.erfc(math.sqrt(max(0.0, chi_square) / 2.0))
