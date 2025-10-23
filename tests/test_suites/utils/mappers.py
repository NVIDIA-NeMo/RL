from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np

from .common import to_numpy


class UnaryMapper(ABC):
    def __call__(self, x: Any) -> np.ndarray:
        arr = to_numpy(x)
        return self.apply(arr)

    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray: ...


class BinaryMapper(ABC):
    @abstractmethod
    def apply(
        self, ref: np.ndarray, exp: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]: ...


class Flatten(UnaryMapper):
    def apply(self, x: np.ndarray) -> np.ndarray:
        return x.ravel()


class AsFloat(UnaryMapper):
    def apply(self, x: np.ndarray) -> np.ndarray:
        return x.astype(float, copy=False)


class Clip(UnaryMapper):
    def __init__(self, min_value: float | None = None, max_value: float | None = None):
        self.min_value = min_value
        self.max_value = max_value

    def apply(self, x: np.ndarray) -> np.ndarray:
        lo = x.min() if self.min_value is None else self.min_value
        hi = x.max() if self.max_value is None else self.max_value
        return np.clip(x, lo, hi)


class Normalize01(UnaryMapper):
    def apply(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(float, copy=False)
        return (
            np.zeros_like(x)
            if x.max() == x.min()
            else (x - x.min()) / (x.max() - x.min())
        )


class BinaryAlign(BinaryMapper):
    """Align two arrays to have the same shape by truncating to minimum size."""

    def apply(self, ref: np.ndarray, exp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Flatten both arrays first
        ref_flat = ref.ravel()
        exp_flat = exp.ravel()

        # Find minimum size
        min_size = min(len(ref_flat), len(exp_flat))

        # Truncate both to minimum size
        return ref_flat[:min_size], exp_flat[:min_size]
