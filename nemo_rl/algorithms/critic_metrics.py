# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure NumPy metrics for scalar trajectory-value evaluation."""

from collections import defaultdict
from typing import Any

import numpy as np


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = average_ranks(scores)
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float(precision[sorted_labels].sum() / positives)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def group_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = np.asarray(
        [record["prediction"] for record in records], dtype=np.float64
    )
    targets = np.asarray([record["target"] for record in records], dtype=np.float64)
    clipped = np.clip(predictions, 0.0, 1.0)
    return {
        "count": len(records),
        "mse": float(np.mean((predictions - targets) ** 2)),
        "mae": float(np.mean(np.abs(predictions - targets))),
        "clipped_mse": float(np.mean((clipped - targets) ** 2)),
        "prediction_mean": float(predictions.mean()),
        "target_mean": float(targets.mean()),
    }


def _has_count_provenance(record: dict[str, Any]) -> bool:
    pass_count = record.get("pass_count")
    rollout_count = record.get("rollout_count")
    return (
        isinstance(pass_count, int)
        and not isinstance(pass_count, bool)
        and isinstance(rollout_count, int)
        and not isinstance(rollout_count, bool)
        and rollout_count > 0
        and 0 <= pass_count <= rollout_count
    )


def compute_critic_evaluation(
    records: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Compute point, ranking, calibration, provenance, and grouped metrics."""
    if not records:
        return {"num_valid_samples": 0.0}, {
            "calibration": [],
            "by_group": {},
            "observed_continuation": {},
        }

    predictions = np.asarray(
        [record["prediction"] for record in records], dtype=np.float64
    )
    targets = np.asarray([record["target"] for record in records], dtype=np.float64)
    clipped = np.clip(predictions, 0.0, 1.0)
    error = predictions - targets
    clipped_error = clipped - targets
    target_variance = float(np.var(targets))
    residual_variance = float(np.var(error))
    prediction_variance = float(np.var(predictions))
    mse = float(np.mean(error**2))
    if target_variance > 0.0:
        explained_variance = 1.0 - residual_variance / target_variance
        r2_vs_mean = 1.0 - mse / target_variance
    else:
        explained_variance = float("nan")
        r2_vs_mean = float("nan")
    exact_success = targets == 1.0
    exact_failure = targets == 0.0
    metrics = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "clipped_mse": float(np.mean(clipped_error**2)),
        "clipped_mae": float(np.mean(np.abs(clipped_error))),
        "target_variance": target_variance,
        "prediction_variance": prediction_variance,
        "residual_variance": residual_variance,
        "explained_variance": explained_variance,
        "r2_vs_mean": r2_vs_mean,
        "prediction_mean": float(predictions.mean()),
        "target_mean": float(targets.mean()),
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "out_of_range_rate": float(np.mean((predictions < 0) | (predictions > 1))),
        "pearson": correlation(predictions, targets),
        "spearman": correlation(average_ranks(predictions), average_ranks(targets)),
        "roc_auc_exact_success": binary_roc_auc(exact_success, predictions),
        "average_precision_exact_success": average_precision(
            exact_success, predictions
        ),
        "roc_auc_exact_failure": binary_roc_auc(exact_failure, -predictions),
        "average_precision_exact_failure": average_precision(
            exact_failure, -predictions
        ),
        "num_valid_samples": float(len(records)),
    }

    calibration: list[dict[str, Any]] = []
    calibration_errors: list[tuple[int, float]] = []
    bin_indices = np.minimum((clipped * 10).astype(np.int64), 9)
    for bin_index in range(10):
        mask = bin_indices == bin_index
        if not mask.any():
            continue
        predicted_mean = float(clipped[mask].mean())
        target_mean = float(targets[mask].mean())
        absolute_gap = abs(predicted_mean - target_mean)
        calibration_errors.append((int(mask.sum()), absolute_gap))
        calibration.append(
            {
                "bin_lower": bin_index / 10,
                "bin_upper": (bin_index + 1) / 10,
                "count": int(mask.sum()),
                "prediction_mean": predicted_mean,
                "target_mean": target_mean,
                "absolute_gap": absolute_gap,
            }
        )
    metrics["expected_calibration_error"] = float(
        sum(count * gap for count, gap in calibration_errors) / len(records)
    )
    metrics["maximum_calibration_error"] = float(
        max((gap for _, gap in calibration_errors), default=float("nan"))
    )

    group_values: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        for key, value in record.get("group_metadata", {}).items():
            group_values[key][str(value)].append(record)
    by_group = {
        key: {
            value: group_summary(group_records)
            for value, group_records in sorted(values.items())
        }
        for key, values in sorted(group_values.items())
    }

    count_records = [record for record in records if _has_count_provenance(record)]
    metrics["count_provenance_rate"] = len(count_records) / len(records)
    details: dict[str, Any] = {
        "calibration": calibration,
        "by_group": by_group,
        "observed_continuation": {
            "count": len(count_records),
            "five_bin_confusion": [],
            "by_pass_count": {},
        },
    }
    if count_records:
        observed_predictions = np.asarray(
            [record["prediction"] for record in count_records], dtype=np.float64
        )
        observed_clipped = np.clip(observed_predictions, 0.0, 1.0)
        pass_counts = np.asarray(
            [record["pass_count"] for record in count_records], dtype=np.int64
        )
        rollout_counts = np.asarray(
            [record["rollout_count"] for record in count_records], dtype=np.int64
        )
        all_success = pass_counts == rollout_counts
        all_failure = pass_counts == 0
        deterministic = all_success | all_failure
        deterministic_score = 2 * np.abs(observed_clipped - 0.5)
        predicted_pass_count = np.rint(observed_clipped * rollout_counts).astype(
            np.int64
        )
        metrics.update(
            {
                "observed/roc_auc_all_success": binary_roc_auc(
                    all_success, observed_predictions
                ),
                "observed/average_precision_all_success": average_precision(
                    all_success, observed_predictions
                ),
                "observed/roc_auc_all_failure": binary_roc_auc(
                    all_failure, -observed_predictions
                ),
                "observed/average_precision_all_failure": average_precision(
                    all_failure, -observed_predictions
                ),
                "observed/roc_auc_deterministic": binary_roc_auc(
                    deterministic, deterministic_score
                ),
                "observed/roc_auc_mixed": binary_roc_auc(
                    ~deterministic, 1 - deterministic_score
                ),
                "observed/nearest_pass_count_accuracy": float(
                    np.mean(predicted_pass_count == pass_counts)
                ),
            }
        )

        max_rollouts = int(rollout_counts.max())
        confusion = np.zeros((max_rollouts + 1, max_rollouts + 1), dtype=np.int64)
        for observed, predicted in zip(pass_counts, predicted_pass_count):
            if 0 <= observed <= max_rollouts and 0 <= predicted <= max_rollouts:
                confusion[observed, predicted] += 1
        details["observed_continuation"] = {
            "count": len(count_records),
            "five_bin_confusion": confusion.tolist(),
            "confusion_rows": "observed pass count",
            "confusion_columns": "nearest predicted pass count",
            "by_pass_count": {
                str(pass_count): group_summary(
                    [
                        record
                        for record in count_records
                        if record["pass_count"] == pass_count
                    ]
                )
                for pass_count in sorted(set(pass_counts.tolist()))
            },
        }
    return metrics, details


def compute_critic_evaluation_suites(
    records: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Evaluate dense supervision and paired anchor labels independently."""
    dense_records = [
        record
        for record in records
        if record.get("evaluation_suite", "dense_exp") == "dense_exp"
    ]
    dense_metrics, dense_details = compute_critic_evaluation(dense_records)

    # Keep legacy unprefixed metrics so old launchers and artifact readers remain
    # valid while exposing the explicit dense_exp namespace for new experiments.
    metrics = dict(dense_metrics)
    metrics.update({f"dense_exp/{key}": value for key, value in dense_metrics.items()})
    suite_details: dict[str, Any] = {"dense_exp": {"overall": dense_details}}

    for suite in ("anchor_raw", "anchor_exp"):
        suite_records = [
            record for record in records if record.get("evaluation_suite") == suite
        ]
        if not suite_records:
            continue
        overall_metrics, overall_details = compute_critic_evaluation(suite_records)
        metrics.update(
            {f"{suite}/overall/{key}": value for key, value in overall_metrics.items()}
        )
        by_anchor_kind: dict[str, Any] = {}
        anchor_kinds = sorted(
            {
                str(record["anchor_kind"])
                for record in suite_records
                if record.get("anchor_kind") is not None
            }
        )
        for anchor_kind in anchor_kinds:
            kind_records = [
                record
                for record in suite_records
                if record.get("anchor_kind") == anchor_kind
            ]
            kind_metrics, kind_details = compute_critic_evaluation(kind_records)
            metrics.update(
                {
                    f"{suite}/{anchor_kind}/{key}": value
                    for key, value in kind_metrics.items()
                }
            )
            by_anchor_kind[anchor_kind] = kind_details
        suite_details[suite] = {
            "overall": overall_details,
            "by_anchor_kind": by_anchor_kind,
        }

    details = dict(dense_details)
    details["evaluation_suites"] = suite_details
    return metrics, details
