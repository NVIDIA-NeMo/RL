from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

import numpy as np


class Tol(Protocol):
    def band(self, baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class SymmetricTol:
    atol: float = 0.0
    rtol: float = 0.0

    def band(self, baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b = self.atol + self.rtol * np.abs(baseline)
        return baseline - b, baseline + b


@dataclass
class AsymmetricTol:
    lower_atol: float = 0.0
    lower_rtol: float = 0.0
    upper_atol: float = 0.0
    upper_rtol: float = 0.0

    def band(self, baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lower_b = self.lower_atol + self.lower_rtol * np.abs(baseline)
        upper_b = self.upper_atol + self.upper_rtol * np.abs(baseline)
        return baseline - lower_b, baseline + upper_b


@dataclass
class SigmaTol:
    k: float = 3.0
    use_mad: bool = False

    def band(self, baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = baseline.astype(float, copy=False)
        mu = float(np.mean(r))
        if self.use_mad:
            med = float(np.median(r))
            mad = float(np.median(np.abs(r - med)))
            sigma = 1.4826 * mad
        else:
            sigma = float(np.std(r, ddof=1))
        lower, upper = mu - self.k * sigma, mu + self.k * sigma
        return np.full_like(baseline, lower), np.full_like(baseline, upper)


@dataclass
class PercentileEnvelope:
    p_low: float = 5.0
    p_high: float = 95.0
    expand: float = 0.0

    def band(self, baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = baseline.astype(float, copy=False).ravel()
        lo = float(np.percentile(r, self.p_low)) - self.expand
        hi = float(np.percentile(r, self.p_high)) + self.expand
        return np.full_like(baseline, lo), np.full_like(baseline, hi)


class PredicateTol:
    def __init__(self, f: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]):
        self.f = f

    def band(self, baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.f(baseline)


class TolChain:
    def __init__(self, parts: Iterable[Tol]):
        self.parts = list(parts)

    def band(self, baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo_all, hi_all = baseline.copy(), baseline.copy()
        for t in self.parts:
            lo, hi = t.band(baseline)
            lo_all = np.minimum(lo_all, lo)
            hi_all = np.maximum(hi_all, hi)
        return lo_all, hi_all
