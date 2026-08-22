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

"""Real two-rank decision-consensus behavior tests."""

import os
from unittest.mock import MagicMock
import pytest
import torch
import torch.distributed as dist

from nemo_rl.algorithms.draft_update_schedule import DraftUpdateDecision
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    validate_draft_enabled_consensus,
    validate_draft_update_decision_consensus,
)


def _init_torchrun_group() -> None:
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", init_method="env://")


def test_disabled_draft_uses_no_per_step_decision_collectives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_torchrun_group()
    all_reduce = MagicMock(side_effect=AssertionError("collective must not run"))
    monkeypatch.setattr(dist, "all_reduce", all_reduce)

    assert (
        validate_draft_update_decision_consensus(
            None,
            draft_enabled=False,
            globally_disabled=True,
            group=dist.group.WORLD,
            device=torch.device("cpu"),
        )
        is None
    )
    all_reduce.assert_not_called()


def test_startup_draft_enabled_mismatch_raises_on_every_rank() -> None:
    _init_torchrun_group()
    rank = int(os.environ["RANK"])

    with pytest.raises(RuntimeError, match="draft-enabled mode mismatch across ranks"):
        validate_draft_enabled_consensus(
            rank == 1,
            group=dist.group.WORLD,
            device=torch.device("cpu"),
        )


def test_draft_enabled_mismatch_raises_on_every_rank_after_collectives() -> None:
    _init_torchrun_group()
    rank = int(os.environ["RANK"])
    decision = (
        None
        if rank == 0
        else DraftUpdateDecision(
            global_step=3,
            decision_id=7,
            update_requested=True,
            draft_refit_requested=True,
            reason="always",
            observed_acceptance=None,
        )
    )

    with pytest.raises(RuntimeError, match="draft-enabled mode mismatch across ranks"):
        validate_draft_update_decision_consensus(
            decision,
            draft_enabled=rank == 1,
            group=dist.group.WORLD,
            device=torch.device("cpu"),
        )


def test_full_decision_mismatch_raises_on_every_rank() -> None:
    _init_torchrun_group()
    rank = int(os.environ["RANK"])
    decision = DraftUpdateDecision(
        global_step=3,
        decision_id=7 + rank,
        update_requested=True,
        draft_refit_requested=True,
        reason="always",
        observed_acceptance=0.8,
    )

    with pytest.raises(RuntimeError, match="decision mismatch across ranks"):
        validate_draft_update_decision_consensus(
            decision,
            draft_enabled=True,
            group=dist.group.WORLD,
            device=torch.device("cpu"),
        )


def test_missing_required_decision_raises_on_every_rank_after_collectives() -> None:
    _init_torchrun_group()
    rank = int(os.environ["RANK"])
    decision = (
        None
        if rank == 0
        else DraftUpdateDecision(
            global_step=3,
            decision_id=7,
            update_requested=True,
            draft_refit_requested=True,
            reason="always",
            observed_acceptance=None,
        )
    )

    with pytest.raises(RuntimeError, match="required.*missing on at least one rank"):
        validate_draft_update_decision_consensus(
            decision,
            draft_enabled=True,
            group=dist.group.WORLD,
            device=torch.device("cpu"),
        )
