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
"""SingleController shared-prefix gating: DP alignment, sampler alignment, setup.

Shared-prefix execution assigns complete prompt groups to DP ranks, so every
training chunk must hold a multiple of the policy DP size. These tests pin
where that is enforced (before any chunk trains) and that the gate comes from
the policy config rather than from attributes on the trainer.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import nemo_rl.algorithms.single_controller as single_controller
from nemo_rl.algorithms.async_utils.staleness_sampler import (
    BaseSampler,
    InOrderSampler,
    InOrderSamplerConfig,
    ReadyFirstSampler,
    WeightFifoSampler,
    WindowedSampler,
)
from nemo_rl.algorithms.single_controller_utils.setup import setup_single_controller
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import GROUP_ID_TAG
from tests.unit.single_controller.test_sampler_interface import FakeBuffer, _run
from tests.unit.single_controller.test_setup import (  # noqa: F401
    _make_master_config,
    patched_factories,
)
from tests.unit.single_controller.test_single_controller_actor import (
    _EmptySampler,
    _NoOpTrainer,
    _train_pump_controller,
)


class _ShardedTrainer(_NoOpTrainer):
    """A trainer double with a DP axis and no shared-prefix attributes."""

    def __init__(self, dp: int) -> None:
        self.sharding_annotations = SimpleNamespace(get_axis_size=lambda axis: dp)
        self.train_calls = 0

    def train_microbatches_from_meta(self, meta, *, train_fields):
        del meta, train_fields
        self.train_calls += 1


class _MinSampler(_EmptySampler):
    """Returns exactly ``min_prompt_groups`` fresh groups, like a thin buffer."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.next_group = 0

    async def select(self, **kwargs):
        BaseSampler._validate_group_bounds(
            kwargs["min_prompt_groups"], kwargs["max_prompt_groups"]
        )
        self.calls.append(dict(kwargs))
        count = kwargs["min_prompt_groups"]
        group_ids = [f"grp{self.next_group + j}" for j in range(count)]
        self.next_group += count
        meta = KVBatchMeta(
            partition_id="rollout_data",
            task_name="train",
            sample_ids=[f"{group_id}_g0" for group_id in group_ids],
            fields=[],
            sequence_lengths=[1] * count,
            tags=[
                {"weight_version": 0, GROUP_ID_TAG: group_id} for group_id in group_ids
            ],
        )
        return meta, count


class _AligningMinSampler(_MinSampler):
    supports_prompt_group_multiple = True


class _LegacyWindowedSubclass(_MinSampler, WindowedSampler):
    """A custom subclass of a built-in that predates ``prompt_group_multiple``."""

    async def select(
        self, *, current_train_weight, min_prompt_groups, max_prompt_groups
    ):
        return await super().select(
            current_train_weight=current_train_weight,
            min_prompt_groups=min_prompt_groups,
            max_prompt_groups=max_prompt_groups,
        )


def _shared_prefix_controller(
    monkeypatch,
    sampler,
    *,
    num_prompts_per_step,
    dp,
    shared_prefix=True,
    min_groups_for_streaming_train=1,
):
    ctrl = _train_pump_controller(sampler=sampler)
    ctrl._shared_prefix_logprobs_enabled = shared_prefix
    ctrl._algo_cfg.num_prompts_per_step = num_prompts_per_step
    ctrl._async_cfg.min_groups_for_streaming_train = min_groups_for_streaming_train
    ctrl._rollout_exhausted.clear()
    ctrl._buffer_capacity = asyncio.Semaphore(64)
    ctrl._trainer = _ShardedTrainer(dp)
    ctrl._sync_weights = AsyncMock(return_value=0)
    ctrl._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})
    return ctrl


def test_controller_defaults_to_dense_without_init():
    controller_cls = single_controller.SingleControllerActor.__ray_metadata__
    assert controller_cls.modified_class._shared_prefix_logprobs_enabled is False


def test_unaligned_step_fails_before_any_chunk_trains(monkeypatch):
    sampler = _MinSampler()
    ctrl = _shared_prefix_controller(monkeypatch, sampler, num_prompts_per_step=6, dp=4)

    with pytest.raises(ValueError, match="multiple of the policy data-parallel"):
        asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=5.0))

    assert sampler.calls == []
    assert ctrl._trainer.train_calls == 0


def test_aligned_step_without_sampler_alignment_takes_exact_dp_chunks(monkeypatch):
    sampler = _MinSampler()
    ctrl = _shared_prefix_controller(monkeypatch, sampler, num_prompts_per_step=8, dp=4)

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=5.0))

    assert ctrl._train_steps == 1
    assert [
        (c["min_prompt_groups"], c["max_prompt_groups"]) for c in sampler.calls
    ] == [
        (4, 4),
        (4, 4),
    ]
    assert all("prompt_group_multiple" not in c for c in sampler.calls)


def test_streaming_chunks_round_up_to_whole_dp_groups(monkeypatch):
    # 12 groups at DP 4 with a 5-group streaming minimum: 8 (5 rounded up to
    # whole DP groups), then the 4-group remainder.
    sampler = _MinSampler()
    ctrl = _shared_prefix_controller(
        monkeypatch,
        sampler,
        num_prompts_per_step=12,
        dp=4,
        min_groups_for_streaming_train=5,
    )

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=5.0))

    assert ctrl._train_steps == 1
    assert [
        (c["min_prompt_groups"], c["max_prompt_groups"]) for c in sampler.calls
    ] == [
        (8, 8),
        (4, 4),
    ]
    assert ctrl._trainer.train_calls == 2


def test_dense_pump_never_aligns_to_dp(monkeypatch):
    # The default path keeps exact streaming chunks, even when the step is not
    # a multiple of DP, and needs no shared-prefix attribute on the trainer.
    sampler = _MinSampler()
    ctrl = _shared_prefix_controller(
        monkeypatch,
        sampler,
        num_prompts_per_step=6,
        dp=4,
        shared_prefix=False,
        min_groups_for_streaming_train=5,
    )

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=5.0))

    assert ctrl._train_steps == 1
    assert [
        (c["min_prompt_groups"], c["max_prompt_groups"]) for c in sampler.calls
    ] == [
        (5, 6),
        (1, 1),
    ]
    assert all("prompt_group_multiple" not in c for c in sampler.calls)


def test_aligning_sampler_gets_prompt_group_multiple(monkeypatch):
    sampler = _AligningMinSampler()
    ctrl = _shared_prefix_controller(monkeypatch, sampler, num_prompts_per_step=8, dp=4)

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=5.0))

    assert ctrl._train_steps == 1
    assert sampler.calls[0]["min_prompt_groups"] == 4
    assert sampler.calls[0]["max_prompt_groups"] == 8
    assert all(c["prompt_group_multiple"] == 4 for c in sampler.calls)


def test_builtin_subclass_does_not_inherit_prompt_group_multiple(monkeypatch):
    sampler = _LegacyWindowedSubclass()
    ctrl = _shared_prefix_controller(monkeypatch, sampler, num_prompts_per_step=8, dp=4)

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=5.0))

    assert ctrl._train_steps == 1
    assert [
        (c["min_prompt_groups"], c["max_prompt_groups"]) for c in sampler.calls
    ] == [
        (4, 4),
        (4, 4),
    ]


def test_builtin_samplers_declare_prompt_group_multiple():
    for sampler_cls in (
        WindowedSampler,
        ReadyFirstSampler,
        WeightFifoSampler,
        InOrderSampler,
    ):
        # Declared in each class body: the controller ignores inherited flags.
        assert sampler_cls.__dict__["supports_prompt_group_multiple"] is True

    class _CustomSampler(BaseSampler):
        async def admit(self, *, trainer_version_fn):
            return None

        async def select(self, **kwargs):
            return None, 0

    assert _CustomSampler.supports_prompt_group_multiple is False


@pytest.mark.parametrize(
    "make_sampler",
    [
        lambda buffer: WindowedSampler(buffer, max_staleness_versions=16),
        lambda buffer: WeightFifoSampler(buffer, max_staleness_versions=16),
        lambda buffer: InOrderSampler(buffer, max_lookahead_versions=16),
    ],
    ids=["windowed", "weight_fifo", "in_order"],
)
def test_builtin_samplers_select_greedy_aligned_prefix(make_sampler):
    buffer = FakeBuffer()
    for index in range(49):
        buffer.add(str(index), weight=0, target_step=0)
    sampler = make_sampler(buffer)

    meta, count = _run(
        sampler.select(
            current_train_weight=0,
            min_prompt_groups=16,
            max_prompt_groups=512,
            prompt_group_multiple=16,
        )
    )

    assert count == 48
    assert meta.sample_ids == [f"{i}_g0" for i in range(48)]
    assert [m.sample_ids for m in buffer.meta_list] == [["48_g0"]]


def test_shared_prefix_requires_token_capture_before_allocating_workers(
    patched_factories,  # noqa: F811
):
    mc = _make_master_config()
    mc.policy["shared_prefix_training"] = {"mode": "logprobs"}

    with pytest.raises(ValueError, match="requires token_capture.enabled=true"):
        setup_single_controller(mc, MagicMock(pad_token_id=0))

    patched_factories["_build_clusters"].assert_not_called()


@pytest.mark.parametrize(
    "budget", ["max_skipped_prompts", "max_consecutive_dropped_prompts"]
)
def test_shared_prefix_rejects_drop_budgets_before_allocating_workers(
    patched_factories,  # noqa: F811
    budget,
):
    mc = _make_master_config(sampler_cfg=InOrderSamplerConfig())
    mc.policy["shared_prefix_training"] = {"mode": "train"}
    mc.token_capture.enabled = True
    setattr(mc.async_rl.rollout_failure, budget, 1)

    with pytest.raises(ValueError, match="max_consecutive_dropped_prompts=0"):
        setup_single_controller(mc, MagicMock(pad_token_id=0))

    patched_factories["_build_clusters"].assert_not_called()
