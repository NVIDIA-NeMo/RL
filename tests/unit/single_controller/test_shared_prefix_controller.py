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
"""The train pump with a policy that shares prompts.

The train pump does not know which packing the policy uses. A step may hold any
number of prompt groups, and the driver plans the forwards of every rank for it.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import nemo_rl.algorithms.single_controller as single_controller
from nemo_rl.algorithms.async_utils.staleness_sampler import BaseSampler
from nemo_rl.data.packing.shared_prefix.row_metadata import (
    PROMPT_HASH_TAG,
    PROMPT_LENGTH_TAG,
    SHARED_PREFIX_TREE_IDS,
    UNSHARED_ROW,
    prompt_hash,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.preshard import shard_meta_for_dp
from nemo_rl.data_plane.schema import GROUP_ID_TAG, MICRO_BATCH_INDICES
from tests.unit.single_controller.test_single_controller_actor import (
    _EmptySampler,
    _NoOpTrainer,
    _train_pump_controller,
)

_DP_WORLD = 4
_ROWS_PER_GROUP = 4
_PROMPT_LENGTH = 80
_PACKING_ARGS: dict[str, Any] = {
    "algorithm": "modified_first_fit_decreasing",
    "max_tokens_per_microbatch": 1024,
    "sequence_length_pad_multiple": 1,
    "share_prefixes": True,
    "bound_by_full_length": False,
}


class _GroupSampler(_EmptySampler):
    """Returns ``min_prompt_groups`` new groups of four rows that share a prompt."""

    def __init__(self) -> None:
        self.next_group = 0

    async def select(self, **kwargs: Any) -> tuple[KVBatchMeta, int]:
        BaseSampler._validate_group_bounds(
            kwargs["min_prompt_groups"], kwargs["max_prompt_groups"]
        )
        count = kwargs["min_prompt_groups"]
        rows = [
            (f"group-{self.next_group + offset}", row)
            for offset in range(count)
            for row in range(_ROWS_PER_GROUP)
        ]
        self.next_group += count
        meta = KVBatchMeta(
            partition_id="rollout_data",
            task_name="train",
            sample_ids=[f"{group}_g{row}" for group, row in rows],
            fields=[],
            sequence_lengths=[_PROMPT_LENGTH + 10 + row for _, row in rows],
            tags=[
                {
                    "weight_version": 0,
                    GROUP_ID_TAG: group,
                    PROMPT_LENGTH_TAG: _PROMPT_LENGTH,
                    PROMPT_HASH_TAG: prompt_hash(list(group.encode())),
                }
                for group, _ in rows
            ],
        )
        return meta, count


class _PlanningTrainer(_NoOpTrainer):
    """Plans the forwards of every chunk as ``TQPolicy`` does when prompts are shared.

    The step loses one prompt group while its first chunk trains.
    """

    def __init__(self) -> None:
        self.ctrl: Any = None
        self.plans: list[list[KVBatchMeta]] = []

    def train_microbatches_from_meta(
        self, meta: KVBatchMeta, *, train_fields: tuple[str, ...]
    ) -> None:
        del train_fields
        if not self.plans:
            step = self.ctrl._trainer_version
            self.ctrl._rollout_runtime.batch_shortfall[step] = 1
        self.plans.append(
            shard_meta_for_dp(
                meta, dp_world=_DP_WORLD, sequence_packing_args=dict(_PACKING_ARGS)
            )
        )


def test_a_step_that_loses_a_group_still_trains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A step of 12 groups on 4 ranks trains on the 11 groups that are left."""
    trainer = _PlanningTrainer()
    ctrl = _train_pump_controller(sampler=_GroupSampler())
    trainer.ctrl = ctrl
    ctrl._trainer = trainer
    ctrl._algo_cfg.num_prompts_per_step = 12
    ctrl._async_cfg.min_groups_for_streaming_train = 4
    # Rollouts are still running, so the step closes on its target, not on exhaustion.
    ctrl._rollout_exhausted.clear()
    ctrl._buffer_capacity = asyncio.Semaphore(64)
    ctrl._sync_weights = AsyncMock(return_value=0)
    ctrl._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=5.0))

    assert ctrl._train_steps == 1
    train_metrics = ctrl._logger.log_metrics.call_args_list[0].args[0]
    assert train_metrics["dropped_prompt_groups"] == 1
    # The chunks hold 4, 4 and 3 groups: every row of the 11 groups is planned once.
    planned_rows = [
        sample_id
        for plan in trainer.plans
        for shard in plan
        for sample_id in shard.sample_ids
    ]
    assert len(planned_rows) == len(set(planned_rows)) == 11 * _ROWS_PER_GROUP
    assert [sum(len(shard.sample_ids) for shard in plan) for plan in trainer.plans] == [
        16,
        16,
        12,
    ]
    for plan in trainer.plans:
        assert len(plan) == _DP_WORLD
        forwards_per_rank = {
            len(shard.extra_info[MICRO_BATCH_INDICES][0]) for shard in plan
        }
        assert len(forwards_per_rank) == 1
        # Three groups on four ranks still share: the planner divides one tree.
        assert all(
            UNSHARED_ROW not in shard.extra_info[SHARED_PREFIX_TREE_IDS]
            for shard in plan
        )
