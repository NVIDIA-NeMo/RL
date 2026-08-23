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

from unittest.mock import MagicMock

import pytest

from nemo_rl.algorithms.draft_cadence_runtime import CadenceTerminalEvidence
from nemo_rl.algorithms.draft_update_observation import (
    VERSION_KEY,
    acceptance_from_rollout_metric_batches,
    prepare_sync_draft_decision,
    stamp_selected_rollout_science,
)
from nemo_rl.algorithms.draft_update_schedule import DraftUpdateScheduler
from nemo_rl.models.policy.draft_config import (
    AdaptiveDraftUpdateScheduleConfig,
    AlwaysDraftUpdateScheduleConfig,
    FixedDraftUpdateScheduleConfig,
)


def test_acceptance_sums_counts_instead_of_averaging_rates() -> None:
    batches = [
        {
            "vllm/spec_num_accepted_tokens": 9.0,
            "vllm/spec_num_draft_tokens": 10.0,
        },
        {
            "vllm/spec_num_accepted_tokens": 1.0,
            "vllm/spec_num_draft_tokens": 90.0,
        },
    ]
    assert acceptance_from_rollout_metric_batches(batches) == pytest.approx(0.1)


def test_sync_science_stamp_is_opt_in_and_binds_reserved_version() -> None:
    metrics = {
        "vllm/spec_num_accepted_tokens": 9.0,
        "vllm/spec_num_draft_tokens": 10.0,
    }
    assert (
        stamp_selected_rollout_science(metrics, enabled=False, applied_draft_version=4)
        is metrics
    )
    stamped = stamp_selected_rollout_science(
        metrics, enabled=True, applied_draft_version=4
    )
    assert VERSION_KEY not in metrics
    assert stamped[VERSION_KEY] == 4


@pytest.mark.parametrize(
    "batches",
    [
        [],
        [{"vllm/spec_num_accepted_tokens": 1.0}],
        [
            {
                "vllm/spec_num_accepted_tokens": -1.0,
                "vllm/spec_num_draft_tokens": 2.0,
            }
        ],
        [
            {
                "vllm/spec_num_accepted_tokens": 0.0,
                "vllm/spec_num_draft_tokens": 0.0,
            }
        ],
    ],
)
def test_invalid_acceptance_counts_return_none(batches) -> None:
    assert acceptance_from_rollout_metric_batches(batches) is None


@pytest.mark.parametrize(
    "config",
    [
        AlwaysDraftUpdateScheduleConfig(),
        FixedDraftUpdateScheduleConfig(
            mode="fixed", action="sparse_update", fixed_interval=1
        ),
    ],
)
def test_default_nonadaptive_sync_does_not_read_science_metrics(config) -> None:
    unreadable_batches = MagicMock()
    prepared = prepare_sync_draft_decision(
        DraftUpdateScheduler.create(config, origin_step=0),
        unreadable_batches,
        cadence_runtime_enabled=False,
        evidence=None,
        global_step=1,
    )
    assert prepared.decision.observed_acceptance is None
    assert prepared.terminal_evidence is None
    unreadable_batches.__iter__.assert_not_called()


@pytest.mark.parametrize(
    "config",
    [
        AlwaysDraftUpdateScheduleConfig(),
        FixedDraftUpdateScheduleConfig(
            mode="fixed", action="sparse_update", fixed_interval=1
        ),
    ],
)
def test_experiment_sync_nonadaptive_collects_science_without_scheduler_input(
    config,
) -> None:
    scheduler = DraftUpdateScheduler.create(config, origin_step=0)
    evidence = CadenceTerminalEvidence({}, {})
    first = prepare_sync_draft_decision(
        scheduler,
        [
            {
                "vllm/spec_num_accepted_tokens": 8.0,
                "vllm/spec_num_draft_tokens": 10.0,
                "draft_schedule/applied_draft_version": 0,
            }
        ],
        cadence_runtime_enabled=True,
        evidence=evidence,
        global_step=1,
    )
    scheduler.record_outcome(
        first.decision,
        update_attempted=True,
        update_successful=True,
        draft_refit_attempted=True,
        draft_refit_successful=True,
    )
    assert first.terminal_evidence is not None
    assert first.accepted_tokens == 8.0
    assert first.draft_tokens == 10.0
    assert first.selected_version == 0
    second = prepare_sync_draft_decision(
        scheduler,
        [
            {
                "vllm/spec_num_accepted_tokens": 6.0,
                "vllm/spec_num_draft_tokens": 10.0,
                "draft_schedule/applied_draft_version": 1,
            }
        ],
        cadence_runtime_enabled=True,
        evidence=first.terminal_evidence,
        global_step=2,
    )
    assert second.decision.observed_acceptance is None
    assert second.terminal_evidence is not None
    assert second.terminal_evidence.observations_by_refit_step[1] == {
        "refit_step": 1,
        "observation_step": 2,
        "applied_draft_version": 1,
        "acceptance_rate": pytest.approx(0.6),
    }


@pytest.mark.parametrize(
    "batches",
    [
        [
            {
                "vllm/spec_num_accepted_tokens": 6.0,
                "vllm/spec_num_draft_tokens": 10.0,
            }
        ],
        [
            {
                "vllm/spec_num_accepted_tokens": 6.0,
                "vllm/spec_num_draft_tokens": 10.0,
                "draft_schedule/applied_draft_version": 0.5,
            }
        ],
        [
            {
                "vllm/spec_num_accepted_tokens": 3.0,
                "vllm/spec_num_draft_tokens": 5.0,
                "draft_schedule/applied_draft_version": 0,
            },
            {
                "vllm/spec_num_accepted_tokens": 3.0,
                "vllm/spec_num_draft_tokens": 5.0,
                "draft_schedule/applied_draft_version": 1,
            },
        ],
    ],
)
def test_experiment_sync_rejects_absent_nonintegral_or_mixed_versions(
    batches,
) -> None:
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    with pytest.raises(ValueError, match="selected serving version"):
        prepare_sync_draft_decision(
            scheduler,
            batches,
            cadence_runtime_enabled=True,
            evidence=CadenceTerminalEvidence({}, {}),
            global_step=1,
        )
    assert scheduler.state.next_decision_id == 1


def test_experiment_sync_rejects_stale_selected_version_before_decision() -> None:
    scheduler = DraftUpdateScheduler.create(
        AlwaysDraftUpdateScheduleConfig(), origin_step=0
    )
    with pytest.raises(RuntimeError, match="stale selected rollout"):
        prepare_sync_draft_decision(
            scheduler,
            [
                {
                    "vllm/spec_num_accepted_tokens": 6.0,
                    "vllm/spec_num_draft_tokens": 10.0,
                    "draft_schedule/applied_draft_version": 7,
                }
            ],
            cadence_runtime_enabled=True,
            evidence=CadenceTerminalEvidence({}, {}),
            global_step=1,
        )
    assert scheduler.state.next_decision_id == 1


def test_sync_adaptive_feeds_same_science_observation_to_scheduler() -> None:
    scheduler = DraftUpdateScheduler.create(
        AdaptiveDraftUpdateScheduleConfig(
            min_interval=1, max_interval=10, min_observations=1
        ),
        origin_step=0,
    )
    prepared = prepare_sync_draft_decision(
        scheduler,
        [
            {
                "vllm/spec_num_accepted_tokens": 6.0,
                "vllm/spec_num_draft_tokens": 10.0,
            }
        ],
        cadence_runtime_enabled=False,
        evidence=None,
        global_step=1,
    )
    assert prepared.decision.observed_acceptance == pytest.approx(0.6)
    assert prepared.accepted_tokens == 6.0
    assert prepared.draft_tokens == 10.0
    assert prepared.selected_version is None
    assert prepared.terminal_evidence is None
