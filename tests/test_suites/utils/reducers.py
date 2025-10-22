from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from .types import StatResult


class Reducer(ABC):
    """Abstract base class for all reducers."""

    @abstractmethod
    def reduce(self, ref: np.ndarray, exp: np.ndarray) -> StatResult:
        """Reduce two arrays to statistical summary."""
        ...


class WassersteinReducer(Reducer):
    """Compute 1D Earth Mover's distance between flattened arrays."""

    def reduce(self, ref: np.ndarray, exp: np.ndarray) -> StatResult:
        r = ref.ravel().astype(float, copy=False)
        e = exp.ravel().astype(float, copy=False)
        d = float(wasserstein_distance(r, e))
        return StatResult({"distance": d})


class KSReducer(Reducer):
    """Two-sample Kolmogorov–Smirnov test statistic and p-value."""

    def __init__(self, alternative: str = "two-sided"):
        self.alternative = alternative

    def reduce(self, ref: np.ndarray, exp: np.ndarray) -> StatResult:
        r = ref.ravel().astype(float, copy=False)
        e = exp.ravel().astype(float, copy=False)
        stat, pvalue = ks_2samp(r, e, alternative=self.alternative, method="auto")
        return StatResult({"D": float(stat), "pvalue": float(pvalue)})


# TODO(ahmadki): missing single array reducer, we don't really need diff part !!
class PercentileDiffReducer(Reducer):
    """Percentile difference reducer."""

    def __init__(self, p: float):
        if not (0.0 <= p <= 100.0):
            raise ValueError("p must be in [0, 100]")
        self.p = p

    def reduce(self, ref: np.ndarray, exp: np.ndarray) -> StatResult:
        r_p = float(np.percentile(ref.astype(float, copy=False), self.p))
        e_p = float(np.percentile(exp.astype(float, copy=False), self.p))
        diff = abs(e_p - r_p)
        return StatResult({"p": self.p, "ref": r_p, "exp": e_p, "diff": diff})


class ResidualsReducer(Reducer):
    """Compute element-wise residuals (exp - ref)."""

    def reduce(self, ref: np.ndarray, exp: np.ndarray) -> StatResult:
        ref_f = ref.astype(float, copy=False)
        exp_f = exp.astype(float, copy=False)

        if ref_f.shape != exp_f.shape:
            raise ValueError(f"Shape mismatch: ref {ref_f.shape} vs exp {exp_f.shape}")

        residual = exp_f - ref_f
        max_abs_residual = float(np.max(np.abs(residual)))
        mean_abs_residual = float(np.mean(np.abs(residual)))

        return StatResult(
            {
                "residual": residual,
                "max_abs_residual": max_abs_residual,
                "mean_abs_residual": mean_abs_residual,
                "residual_shape": residual.shape,
            }
        )


class MSEReducer(Reducer):
    """Mean Squared Error reducer."""

    def reduce(self, ref: np.ndarray, exp: np.ndarray) -> StatResult:
        ref_f = ref.ravel().astype(float, copy=False)
        exp_f = exp.ravel().astype(float, copy=False)

        min_len = min(len(ref_f), len(exp_f))
        ref_f, exp_f = ref_f[:min_len], exp_f[:min_len]

        mse = float(np.mean((ref_f - exp_f) ** 2))
        return StatResult({"mse": mse})


class RMSEReducer(Reducer):
    """Root Mean Squared Error reducer."""

    def reduce(self, ref: np.ndarray, exp: np.ndarray) -> StatResult:
        ref_f = ref.ravel().astype(float, copy=False)
        exp_f = exp.ravel().astype(float, copy=False)

        min_len = min(len(ref_f), len(exp_f))
        ref_f, exp_f = ref_f[:min_len], exp_f[:min_len]

        rmse = float(np.sqrt(np.mean((ref_f - exp_f) ** 2)))
        return StatResult({"rmse": rmse})


class MAPEReducer(Reducer):
    """Mean Absolute Percentage Error reducer."""

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def reduce(self, ref: np.ndarray, exp: np.ndarray) -> StatResult:
        ref_f = ref.ravel().astype(float, copy=False)
        exp_f = exp.ravel().astype(float, copy=False)

        min_len = min(len(ref_f), len(exp_f))
        ref_f, exp_f = ref_f[:min_len], exp_f[:min_len]

        denominator = np.abs(ref_f) + self.epsilon
        mape = float(np.mean(np.abs((ref_f - exp_f) / denominator)) * 100)
        return StatResult({"mape": mape})


class PearsonReducer(Reducer):
    """Pearson correlation coefficient reducer."""

    def reduce(self, ref: np.ndarray, exp: np.ndarray) -> StatResult:
        ref_f = ref.ravel().astype(float, copy=False)
        exp_f = exp.ravel().astype(float, copy=False)

        min_len = min(len(ref_f), len(exp_f))
        ref_f, exp_f = ref_f[:min_len], exp_f[:min_len]

        corr_matrix = np.corrcoef(ref_f, exp_f)
        correlation = (
            float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
        )

        return StatResult({"pearson_correlation": correlation})


class SpearmanReducer(Reducer):
    """Spearman rank correlation reducer."""

    def reduce(self, ref: np.ndarray, exp: np.ndarray) -> StatResult:
        from scipy.stats import spearmanr

        ref_f = ref.ravel().astype(float, copy=False)
        exp_f = exp.ravel().astype(float, copy=False)

        min_len = min(len(ref_f), len(exp_f))
        ref_f, exp_f = ref_f[:min_len], exp_f[:min_len]

        correlation, pvalue = spearmanr(ref_f, exp_f)
        correlation = float(correlation) if not np.isnan(correlation) else 0.0

        return StatResult(
            {"spearman_correlation": correlation, "spearman_pvalue": float(pvalue)}
        )


class CosineReducer(Reducer):
    """Cosine similarity reducer."""

    def reduce(self, ref: np.ndarray, exp: np.ndarray) -> StatResult:
        ref_f = ref.ravel().astype(float, copy=False)
        exp_f = exp.ravel().astype(float, copy=False)

        min_len = min(len(ref_f), len(exp_f))
        ref_f, exp_f = ref_f[:min_len], exp_f[:min_len]

        dot_product = np.dot(ref_f, exp_f)
        norm_ref = np.linalg.norm(ref_f)
        norm_exp = np.linalg.norm(exp_f)

        if norm_ref == 0 or norm_exp == 0:
            similarity = 0.0
        else:
            similarity = dot_product / (norm_ref * norm_exp)

        return StatResult({"cosine_similarity": float(similarity)})


class JensenShannonReducer(Reducer):
    """Jensen-Shannon divergence reducer."""

    def __init__(self, bins: int = 50):
        self.bins = bins

    def reduce(self, ref: np.ndarray, exp: np.ndarray) -> StatResult:
        from scipy.spatial.distance import jensenshannon

        ref_f = ref.ravel().astype(float, copy=False)
        exp_f = exp.ravel().astype(float, copy=False)

        x_min, x_max = (
            min(np.min(ref_f), np.min(exp_f)),
            max(np.max(ref_f), np.max(exp_f)),
        )
        bin_edges = np.linspace(x_min, x_max, self.bins + 1)

        hist_ref, _ = np.histogram(ref_f, bins=bin_edges, density=True)
        hist_exp, _ = np.histogram(exp_f, bins=bin_edges, density=True)

        hist_ref = hist_ref / np.sum(hist_ref)
        hist_exp = hist_exp / np.sum(hist_exp)

        eps = 1e-10
        hist_ref = hist_ref + eps
        hist_exp = hist_exp + eps
        hist_ref = hist_ref / np.sum(hist_ref)
        hist_exp = hist_exp / np.sum(hist_exp)

        js_divergence = float(jensenshannon(hist_ref, hist_exp))
        return StatResult({"js_divergence": js_divergence})


class BhattacharyyaReducer(Reducer):
    """Bhattacharyya distance reducer."""

    def __init__(self, bins: int = 50):
        self.bins = bins

    def reduce(self, ref: np.ndarray, exp: np.ndarray) -> StatResult:
        ref_f = ref.ravel().astype(float, copy=False)
        exp_f = exp.ravel().astype(float, copy=False)

        x_min, x_max = (
            min(np.min(ref_f), np.min(exp_f)),
            max(np.max(ref_f), np.max(exp_f)),
        )
        bin_edges = np.linspace(x_min, x_max, self.bins + 1)

        hist_ref, _ = np.histogram(ref_f, bins=bin_edges, density=True)
        hist_exp, _ = np.histogram(exp_f, bins=bin_edges, density=True)

        hist_ref = hist_ref / np.sum(hist_ref)
        hist_exp = hist_exp / np.sum(hist_exp)

        eps = 1e-10
        hist_ref = hist_ref + eps
        hist_exp = hist_exp + eps

        bc = np.sum(np.sqrt(hist_ref * hist_exp))
        distance = float(-np.log(bc))
        return StatResult({"bhattacharyya_distance": distance})


class DTWReducer(Reducer):
    """Dynamic Time Warping distance reducer."""

    def reduce(self, ref: np.ndarray, exp: np.ndarray) -> StatResult:
        ref_f = ref.ravel().astype(float, copy=False)
        exp_f = exp.ravel().astype(float, copy=False)

        n, m = len(ref_f), len(exp_f)

        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(ref_f[i - 1] - exp_f[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]
                )

        dtw_distance = float(dtw_matrix[n, m])
        return StatResult({"dtw_distance": dtw_distance})


class HausdorffReducer(Reducer):
    """Hausdorff distance reducer (approximation for 1D curves)."""

    def reduce(self, ref: np.ndarray, exp: np.ndarray) -> StatResult:
        from scipy.spatial.distance import directed_hausdorff

        ref_f = ref.ravel().astype(float, copy=False)
        exp_f = exp.ravel().astype(float, copy=False)

        ref_points = np.column_stack((np.arange(len(ref_f)), ref_f))
        exp_points = np.column_stack((np.arange(len(exp_f)), exp_f))

        hausdorff_dist = max(
            directed_hausdorff(ref_points, exp_points)[0],
            directed_hausdorff(exp_points, ref_points)[0],
        )

        return StatResult({"hausdorff_distance": float(hausdorff_dist)})


class IntegratedAreaReducer(Reducer):
    """Integrated area difference using trapezoidal rule."""

    def reduce(self, ref: np.ndarray, exp: np.ndarray) -> StatResult:
        ref_f = ref.ravel().astype(float, copy=False)
        exp_f = exp.ravel().astype(float, copy=False)

        area_ref = float(np.trapezoid(ref_f))
        area_exp = float(np.trapezoid(exp_f))
        area_diff = abs(area_ref - area_exp)

        return StatResult(
            {"area_ref": area_ref, "area_exp": area_exp, "area_diff": area_diff}
        )


class TopologicalReducer(Reducer):
    """Persistent homology summary for 1D time series."""

    def reduce(self, ref: np.ndarray, exp: np.ndarray) -> StatResult:
        from scipy.signal import find_peaks

        def _compute_topology(signal):
            signal_f = signal.ravel().astype(float, copy=False)

            peaks, _ = find_peaks(signal_f)
            valleys, _ = find_peaks(-signal_f)

            peak_heights = signal_f[peaks] if len(peaks) > 0 else np.array([])
            valley_depths = signal_f[valleys] if len(valleys) > 0 else np.array([])

            return {
                "num_peaks": len(peaks),
                "num_valleys": len(valleys),
                "avg_peak_height": float(np.mean(peak_heights))
                if len(peak_heights) > 0
                else 0.0,
                "avg_valley_depth": float(np.mean(valley_depths))
                if len(valley_depths) > 0
                else 0.0,
                "peak_persistence": float(np.std(peak_heights))
                if len(peak_heights) > 1
                else 0.0,
                "valley_persistence": float(np.std(valley_depths))
                if len(valley_depths) > 1
                else 0.0,
                "total_variation": float(np.sum(np.abs(np.diff(signal_f)))),
            }

        ref_topo = _compute_topology(ref)
        exp_topo = _compute_topology(exp)

        results = {}
        for key in ref_topo:
            results[f"ref_{key}"] = ref_topo[key]
            results[f"exp_{key}"] = exp_topo[key]
            results[f"diff_{key}"] = abs(ref_topo[key] - exp_topo[key])

        return StatResult(results)


class MinMaxReducer(Reducer):
    """Min and max value differences."""

    def reduce(self, ref: np.ndarray, exp: np.ndarray) -> StatResult:
        ref_f = ref.ravel().astype(float, copy=False)
        exp_f = exp.ravel().astype(float, copy=False)

        ref_min, ref_max = float(np.min(ref_f)), float(np.max(ref_f))
        exp_min, exp_max = float(np.min(exp_f)), float(np.max(exp_f))

        min_diff = abs(ref_min - exp_min)
        max_diff = abs(ref_max - exp_max)

        return StatResult(
            {
                "ref_min": ref_min,
                "ref_max": ref_max,
                "exp_min": exp_min,
                "exp_max": exp_max,
                "min_diff": min_diff,
                "max_diff": max_diff,
            }
        )
