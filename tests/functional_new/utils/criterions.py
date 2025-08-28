from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from .types import CriterionResult, StatResult


class Criterion(ABC):
    """Abstract base class for all criteria."""

    @abstractmethod
    def judge(self, stat: StatResult) -> CriterionResult:
        """Judge whether the statistical result passes the criterion."""
        ...


class AlwaysPass(Criterion):
    """Criterion that always passes - useful for testing or as a default."""

    def judge(self, stat: StatResult) -> CriterionResult:
        return CriterionResult(True, dict(stat.results))


class ScalarThreshold(Criterion):
    """Gate on a scalar value in stat.results (e.g., 'distance', 'pvalue', 'max_abs_err')."""

    def __init__(self, value_key: str, compare: Callable[[float], bool]):
        self.value_key = value_key
        self.compare = compare

    def judge(self, stat: StatResult) -> CriterionResult:
        v = float(stat.results.get(self.value_key))
        ok = self.compare(v)
        return CriterionResult(ok, {**stat.results, self.value_key: v})


class RangeCriterion(Criterion):
    """Check if a value falls within a specified range."""

    def __init__(
        self, value_key: str, lower: float | None = None, upper: float | None = None
    ):
        self.value_key = value_key
        self.lower = lower
        self.upper = upper

    def judge(self, stat: StatResult) -> CriterionResult:
        v = float(stat.results.get(self.value_key))
        lo_ok = True if self.lower is None else (v >= self.lower)
        hi_ok = True if self.upper is None else (v <= self.upper)
        ok = lo_ok and hi_ok
        return CriterionResult(ok, {**stat.results, self.value_key: v})


class PredicateCriterion(Criterion):
    """Apply a custom predicate function to the StatResult."""

    def __init__(self, predicate: Callable[[StatResult], bool]):
        self.predicate = predicate

    def judge(self, stat: StatResult) -> CriterionResult:
        ok = self.predicate(stat)
        return CriterionResult(ok, dict(stat.results))


class AllOf(Criterion):
    """Criterion that requires ALL sub-criteria to pass."""

    def __init__(self, *criteria: Criterion):
        self.criteria = criteria

    def judge(self, stat: StatResult) -> CriterionResult:
        cur = CriterionResult(True, dict(stat.results))
        for c in self.criteria:
            out = c.judge(StatResult(results=cur.details))
            if not out.passed:
                return out
            cur = out
        return cur


class ExactMatch(Criterion):
    """Criterion that requires exact match (zero residuals)."""

    def __init__(self, residual_key: str = "residual"):
        self.residual_key = residual_key

    def judge(self, stat: StatResult) -> CriterionResult:
        res = stat.results.get(self.residual_key)
        if res is None:
            return CriterionResult(False, dict(stat.results))

        res = np.asarray(res)
        ok = bool(np.all(res == 0))
        det = {**stat.results, "max_abs_residual": float(np.max(np.abs(res)))}
        return CriterionResult(ok, det)


class AnyOf(Criterion):
    """Criterion that requires ANY sub-criterion to pass."""

    def __init__(self, *criteria: Criterion):
        self.criteria = criteria

    def judge(self, stat: StatResult) -> CriterionResult:
        last = None
        for c in self.criteria:
            out = c.judge(stat)
            if out.passed:
                return out
            last = out
        return last if last is not None else CriterionResult(False, dict(stat.results))


def threshold(key: str, max_value: float) -> ScalarThreshold:
    """Create a threshold criterion: value <= max_value"""
    return ScalarThreshold(key, lambda v: v <= max_value)


def min_threshold(key: str, min_value: float) -> ScalarThreshold:
    """Create a minimum threshold criterion: value >= min_value"""
    return ScalarThreshold(key, lambda v: v >= min_value)


def range_check(key: str, min_val: float, max_val: float) -> RangeCriterion:
    """Create a range criterion: min_val <= value <= max_val"""
    return RangeCriterion(key, min_val, max_val)


def exact_match(key: str = "residual") -> ExactMatch:
    """Create a criterion for exact match (zero residuals)"""
    return ExactMatch(key)
