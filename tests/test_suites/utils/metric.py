from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence, Tuple

import numpy as np

from .common import to_numpy
from .mappers import BinaryMapper, UnaryMapper
from .reducers import Reducer
from .types import StatResult


@dataclass
class Metric:
    """A metric is a mappers + reducer."""

    # per-side unary mappers
    preprocess_ref: Sequence[UnaryMapper] = field(default_factory=list)
    preprocess_exp: Sequence[UnaryMapper] = field(default_factory=list)
    # optional joint mappers (alignment, resampling, etc.)
    joint_mappers: Sequence[BinaryMapper] = field(default_factory=list)
    # the reducer that computes statistics
    reducer: Reducer | None = None
    id: str = "Metric"  # TODO(ahmadki): not needed !!

    def _apply_unary(self, x, mappers: Iterable[UnaryMapper]) -> np.ndarray:
        arr = to_numpy(x)
        for m in mappers:
            arr = m(arr)
        return arr

    def _apply_joint(
        self, ref: np.ndarray, exp: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        r, e = ref, exp
        for jm in self.joint_mappers:
            r, e = jm.apply(r, e)
        return r, e

    def run(self, reference, experiment) -> StatResult:
        r = self._apply_unary(reference, self.preprocess_ref)
        e = self._apply_unary(experiment, self.preprocess_exp)
        r, e = self._apply_joint(r, e)
        if self.reducer is None:
            return StatResult({})
        return self.reducer.reduce(r, e)
