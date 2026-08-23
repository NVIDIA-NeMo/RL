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

"""Selected-rollout observations for online draft-update scheduling."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from nemo_rl.algorithms.draft_cadence_runtime import (
    CadenceTerminalEvidence,
    record_terminal_post_refit_observation,
)
from nemo_rl.algorithms.draft_update_schedule import (
    DraftUpdateDecision,
    DraftUpdateScheduler,
)
from nemo_rl.models.policy.draft_config import DraftUpdateScheduleConfig

ACCEPTED_KEY = "vllm/spec_num_accepted_tokens"
DRAFT_KEY = "vllm/spec_num_draft_tokens"
VERSION_KEY = "draft_schedule/applied_draft_version"


def stamp_selected_rollout_science(
    metrics: Mapping[str, object],
    *,
    enabled: bool,
    applied_draft_version: int,
) -> Mapping[str, object]:
    """Bind selected rollout metrics to the serving-draft version when enabled."""
    if not enabled:
        return metrics
    if type(applied_draft_version) is not int or applied_draft_version < 0:
        raise ValueError("selected serving version must be a nonnegative integer")
    return {**metrics, VERSION_KEY: applied_draft_version}


def acceptance_from_rollout_metric_batches(
    batches: Iterable[Mapping[str, object]],
) -> float | None:
    """Return count-weighted acceptance across every contributing batch."""
    accepted_total = 0.0
    draft_total = 0.0
    seen = False
    for metrics in batches:
        if ACCEPTED_KEY not in metrics or DRAFT_KEY not in metrics:
            return None
        try:
            accepted = float(metrics[ACCEPTED_KEY])
            draft = float(metrics[DRAFT_KEY])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(accepted) or not math.isfinite(draft):
            return None
        if accepted < 0.0 or draft < 0.0 or accepted > draft:
            return None
        accepted_total += accepted
        draft_total += draft
        seen = True
    if not seen or draft_total <= 0.0:
        return None
    return accepted_total / draft_total


def rollout_science_from_metric_batches(
    batches: Iterable[Mapping[str, object]],
    *,
    require_version: bool,
) -> tuple[float | None, int | None]:
    """Reconstruct acceptance and, when required, one selected version."""
    materialized = tuple(batches)
    acceptance = acceptance_from_rollout_metric_batches(materialized)
    if not require_version:
        return acceptance, None
    versions: set[int] = set()
    for metrics in materialized:
        value = metrics.get(VERSION_KEY)
        if type(value) is not int or value < 0:
            raise ValueError("selected serving version is absent or nonintegral")
        versions.add(value)
    if len(versions) != 1:
        raise ValueError("selected serving version is mixed across rollout batches")
    return acceptance, next(iter(versions))


def acceptance_observation_for_schedule(
    config: DraftUpdateScheduleConfig,
    acceptance: float | None,
) -> float | None:
    """Keep science visible to the adaptive state machine only."""
    return acceptance if config.mode == "adaptive" else None


@dataclass(frozen=True, slots=True)
class PreparedDraftDecision:
    decision: DraftUpdateDecision
    terminal_evidence: CadenceTerminalEvidence | None


def prepare_sync_draft_decision(
    scheduler: DraftUpdateScheduler,
    rollout_metric_batches: Iterable[Mapping[str, object]],
    *,
    cadence_runtime_enabled: bool,
    evidence: CadenceTerminalEvidence | None,
    global_step: int,
) -> PreparedDraftDecision:
    """Validate selected science before creating one synchronous decision."""
    needs_acceptance = scheduler.config.mode == "adaptive" or cadence_runtime_enabled
    if cadence_runtime_enabled != (evidence is not None):
        raise ValueError("cadence runtime evidence enablement mismatch")
    if needs_acceptance:
        science_acceptance, selected_version = rollout_science_from_metric_batches(
            rollout_metric_batches,
            require_version=cadence_runtime_enabled,
        )
    else:
        science_acceptance, selected_version = None, None
    if (
        cadence_runtime_enabled
        and selected_version != scheduler.state.applied_draft_version
    ):
        raise RuntimeError(
            "stale selected rollout: "
            f"selected={selected_version}, "
            f"current={scheduler.state.applied_draft_version}"
        )
    prior_refit_step = scheduler.state.last_applied_refit_step
    decision = scheduler.decide(
        global_step=global_step,
        acceptance=acceptance_observation_for_schedule(
            scheduler.config, science_acceptance
        ),
    )
    if cadence_runtime_enabled:
        assert evidence is not None
        evidence = record_terminal_post_refit_observation(
            evidence,
            decision=decision,
            last_applied_refit_step=prior_refit_step,
            acceptance_rate=science_acceptance,
        )
    return PreparedDraftDecision(decision=decision, terminal_evidence=evidence)
