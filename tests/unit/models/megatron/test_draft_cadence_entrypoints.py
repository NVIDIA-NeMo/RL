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

"""Behavioral contracts for controller-owned draft cadence decisions."""

from unittest.mock import MagicMock, patch

from nemo_rl.algorithms.draft_update_schedule import DraftUpdateDecision
from nemo_rl.algorithms.grpo_sync import _train_policy_from_meta
from nemo_rl.models.policy.tq_policy import TQPolicy, _aggregate_train_results


def _decision() -> DraftUpdateDecision:
    return DraftUpdateDecision(
        global_step=3,
        decision_id=7,
        update_requested=False,
        draft_refit_requested=False,
        reason="none",
        observed_acceptance=None,
    )


def _tq_policy_for_test() -> TQPolicy:
    policy = object.__new__(TQPolicy)
    policy.cfg = {"train_global_batch_size": 4, "train_micro_batch_size": 1}
    policy.flops_tracker = None
    policy.worker_group = MagicMock()
    policy.worker_group.run_all_workers_single_data.return_value = []
    return policy


@patch("nemo_rl.models.policy.tq_policy.ray.get")
def test_tq_begin_fans_out_identical_draft_decision(mock_get: MagicMock) -> None:
    policy = _tq_policy_for_test()
    decision = _decision()

    policy.begin_train_step(loss_fn=object(), draft_update_decision=decision)

    call = policy.worker_group.run_all_workers_single_data.call_args
    assert call.kwargs["draft_update_decision"] is decision


def test_cp1_controller_fans_out_identical_draft_decision() -> None:
    policy = MagicMock()
    decision = _decision()

    with patch(
        "nemo_rl.algorithms.grpo_sync._should_use_split_draft_training",
        return_value=False,
    ):
        _train_policy_from_meta(
            policy,
            MagicMock(),
            loss_fn=MagicMock(),
            timer=None,
            train_fields=("input_ids",),
            master_config=MagicMock(),
            draft_update_decision=decision,
        )

    call = policy.train_from_meta.call_args
    assert call.kwargs["draft_update_decision"] is decision
    policy.begin_train_step.assert_not_called()


def test_cp2_controller_fans_out_identical_draft_decision() -> None:
    policy = MagicMock()
    decision = _decision()

    with patch(
        "nemo_rl.algorithms.grpo_sync._should_use_split_draft_training",
        return_value=True,
    ):
        _train_policy_from_meta(
            policy,
            MagicMock(),
            loss_fn=MagicMock(),
            timer=None,
            train_fields=("input_ids",),
            master_config=MagicMock(),
            draft_update_decision=decision,
        )

    call = policy.begin_train_step.call_args
    assert call.kwargs["draft_update_decision"] is decision
    policy.train_from_meta.assert_not_called()


def test_cp1_disabled_cadence_preserves_legacy_call_shape() -> None:
    policy = MagicMock()

    with patch(
        "nemo_rl.algorithms.grpo_sync._should_use_split_draft_training",
        return_value=False,
    ):
        _train_policy_from_meta(
            policy,
            MagicMock(),
            loss_fn=MagicMock(),
            timer=None,
            train_fields=("input_ids",),
            master_config=MagicMock(),
        )

    assert "draft_update_decision" not in policy.train_from_meta.call_args.kwargs


def test_cp2_disabled_cadence_preserves_legacy_call_shape() -> None:
    policy = MagicMock()

    with patch(
        "nemo_rl.algorithms.grpo_sync._should_use_split_draft_training",
        return_value=True,
    ):
        _train_policy_from_meta(
            policy,
            MagicMock(),
            loss_fn=MagicMock(),
            timer=None,
            train_fields=("input_ids",),
            master_config=MagicMock(),
        )

    assert "draft_update_decision" not in policy.begin_train_step.call_args.kwargs


def test_split_finish_metrics_preserve_consensused_decision() -> None:
    decision = _decision()
    result = {
        "global_loss": 1.0,
        "grad_norm": 2.0,
        "all_mb_metrics": {},
        "draft_update_decision": decision,
    }

    aggregated = _aggregate_train_results([result])

    assert aggregated["draft_update_decision"] is decision
