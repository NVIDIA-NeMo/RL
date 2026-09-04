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

"""Input validation for the cross-DP scheduler.

These are pure argument checks with no scheduler state, kept apart so the
state machine itself reads as transitions rather than as a wall of guards.
"""

from __future__ import annotations

from typing import Any

from nemo_rl.models.generation.vllm.lfs.modes import (
    EXPLICIT_PROBE_SELECTION_SEMANTICS,
    IMPLICIT_PROBE_SELECTION_SEMANTICS,
    ONLINE_LFS_MODES,
    PREFERRED_DP_PINNED_PLACEMENT,
    SCHEDULER_SELECTED_DP_PLACEMENT,
)


def validate_scheduler_config(
    *,
    dp_size: int,
    max_num_seqs_per_dp: int,
    mode: str,
    lookahead_per_dp: int,
    dp_selection_mode: str,
    lfs_admission_fairness_interval: int,
) -> None:
    """Reject a scheduler configuration that could never be honoured."""
    if dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}")
    if max_num_seqs_per_dp <= 0:
        raise ValueError(
            "max_num_seqs_per_dp must be positive, "
            f"got {max_num_seqs_per_dp}"
        )
    if lookahead_per_dp < 0:
        raise ValueError(
            f"lookahead_per_dp must be non-negative, got {lookahead_per_dp}"
        )
    if mode not in (
        "fcfs",
        "lfs",
        "predicted_lfs",
        "history_lfs",
        "oracle_probe_lfs",
        "exact_length_lpt",
    ):
        raise ValueError(
            "mode must be 'fcfs', 'lfs', 'predicted_lfs', 'history_lfs', "
            "'oracle_probe_lfs', or 'exact_length_lpt', "
            f"got {mode!r}"
        )
    if dp_selection_mode not in ("static_cost", "inflight_count"):
        raise ValueError(
            "dp_selection_mode must be 'static_cost' or "
            f"'inflight_count', got {dp_selection_mode!r}"
        )
    if (
        type(lfs_admission_fairness_interval) is not int
        or lfs_admission_fairness_interval < 0
    ):
        raise ValueError(
            "lfs_admission_fairness_interval must be a non-negative "
            f"integer, got {lfs_admission_fairness_interval!r}"
        )
    if (
        lfs_admission_fairness_interval > 0
        and mode not in ONLINE_LFS_MODES
    ):
        raise ValueError(
            "lfs_admission_fairness_interval is only supported for "
            f"lfs/predicted_lfs/oracle_probe_lfs, got mode={mode!r}"
        )


def resolve_global_admission_limit(
    *, dp_size: int, admission_limit_per_dp: int, requested: int | None
) -> int:
    """Default the global cap to the aggregate per-DP cap, and bound it."""
    aggregate_admission_limit = dp_size * admission_limit_per_dp
    if requested is None:
        return aggregate_admission_limit
    if not 0 < requested <= aggregate_admission_limit:
        raise ValueError(
            "global_admission_limit must be in [1, "
            f"{aggregate_admission_limit}], got "
            f"{requested}"
        )
    return requested


def inspect_catalog(
    request_catalog: list[dict[str, Any]], *, mode: str, dp_size: int
) -> tuple[str, str, dict[str, str]]:
    """Validate a first-turn catalog and read its placement/probe conventions.

    Returns ``(dp_placement_mode, probe_selection_semantics,
    designated_probe_request_ids)``.
    """
    self = _ConfigView(mode=mode, dp_size=dp_size)

    preferred_dp_marker_presence = [
        "preferred_dp_idx" in item for item in request_catalog
    ]
    dp_placement_mode = SCHEDULER_SELECTED_DP_PLACEMENT
    if any(preferred_dp_marker_presence):
        if not all(preferred_dp_marker_presence):
            raise ValueError(
                "preferred_dp_idx must be present on every catalog request "
                "when exact-length DP pinning is used"
            )
        if self.mode != "exact_length_lpt":
            raise ValueError(
                "preferred_dp_idx is only supported for "
                f"exact_length_lpt, got mode={self.mode!r}"
            )
        invalid_preferred_dp_indices = [
            (str(item.get("request_id")), item["preferred_dp_idx"])
            for item in request_catalog
            if type(item["preferred_dp_idx"]) is not int
            or not 0 <= item["preferred_dp_idx"] < self.dp_size
        ]
        if invalid_preferred_dp_indices:
            raise ValueError(
                "preferred_dp_idx must be an int (not bool) in "
                f"[0, {self.dp_size}); invalid requests="
                f"{invalid_preferred_dp_indices}"
            )
        dp_placement_mode = PREFERRED_DP_PINNED_PLACEMENT

    marker_presence = [
        "is_designated_probe" in item for item in request_catalog
    ]
    designated_probe_request_ids: dict[str, str] = {}
    probe_selection_semantics = IMPLICIT_PROBE_SELECTION_SEMANTICS
    if any(marker_presence):
        if not all(marker_presence):
            raise ValueError(
                "is_designated_probe must be present on every catalog "
                "request when explicit probe designation is used"
            )
        invalid_markers = [
            str(item.get("request_id"))
            for item in request_catalog
            if type(item["is_designated_probe"]) is not bool
        ]
        if invalid_markers:
            raise ValueError(
                "is_designated_probe must be a bool for every catalog "
                f"request; invalid request_ids={invalid_markers}"
            )
        designated_counts: dict[str, int] = {}
        for item in request_catalog:
            group_id = str(item["group_id"])
            designated_counts.setdefault(group_id, 0)
            if item["is_designated_probe"]:
                designated_counts[group_id] += 1
                designated_probe_request_ids[group_id] = str(
                    item["request_id"]
                )
        invalid_groups = {
            group_id: count
            for group_id, count in designated_counts.items()
            if count != 1
        }
        if invalid_groups:
            raise ValueError(
                "explicit probe designation requires exactly one "
                "is_designated_probe request per group; "
                f"invalid_groups={invalid_groups}"
            )
        probe_selection_semantics = EXPLICIT_PROBE_SELECTION_SEMANTICS
    return (
        dp_placement_mode,
        probe_selection_semantics,
        designated_probe_request_ids,
    )


class _ConfigView:
    """Minimal stand-in so the moved checks keep reading ``self.mode``."""

    __slots__ = ("mode", "dp_size")

    def __init__(self, *, mode: str, dp_size: int) -> None:
        self.mode = mode
        self.dp_size = dp_size
