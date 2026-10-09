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
"""Shared-prefix transport through the data plane and the train pump.

Covers the driver-side pieces between rollout and the Megatron worker:
complete-group DP sharding with prescribed execution slots, the reassembler's
prompt-length column and group tags, the rollout partition schema, and the
train pump's DP-aligned prompt-group selection.

Cross-module API assumptions. Other review fixes edit these modules
concurrently; update this file if one of them changes:

- ``shard_meta_for_dp(meta, dp_world=, batch_size=, sequence_packing_args=,
  shared_prefix_groups=True, shared_prefix_work_weights=)`` returns per-rank
  metas plus a permutation for ``BatchedDataDict.reorder_data`` (or None). It
  groups rows by prompt group: the PR head parses ``{group}_g{index}`` sample
  IDs and the review fix reads the ``GROUP_ID_TAG`` row tag. The metas here
  carry both, consistently, so the tests hold for either key.
- Work-weighted sharding reads a ``SHARED_PREFIX_PROMPT_LENGTHS`` tag per row.
- ``RolloutReassembler(include_shared_prefix_metadata=True)`` publishes the
  ``SHARED_PREFIX_PROMPT_LENGTHS`` column (verified prompt length, 0 for a
  placeholder) and a ``GROUP_ID_TAG`` tag on every row. Whether it also tags
  prompt lengths is not asserted (the review fix ships them as a column only).
- ``_register_single_controller_partitions(dp_client, master_config=,
  partition_id=, include_multimodal_fields=)``.
- The train pump gates DP alignment on the controller attribute
  ``_shared_prefix_logprobs_enabled`` (class default False, set from the policy
  config). At the PR head it reads ``trainer.shared_prefix_training_config``
  instead, so the pump tests fail there by design (review entry 4.2).
"""

from __future__ import annotations

import asyncio
import random
from collections import defaultdict
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from nemo_rl.data.packing.shared_prefix_metadata import (
    SHARED_PREFIX_EXECUTION_SLOT,
    SHARED_PREFIX_PROMPT_LENGTHS,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.preshard import shard_meta_for_dp
from nemo_rl.data_plane.schema import GROUP_ID_TAG
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

_PACKING_ARGS = {
    "max_tokens_per_microbatch": 256,
    "algorithm": "modified_first_fit_decreasing",
    "sequence_length_pad_multiple": 1,
    "input_key": "input_ids",
    "input_lengths_key": "input_lengths",
}
_GROUPS = 8
_GROUP_SIZE = 4


def _grouped_meta() -> KVBatchMeta:
    """Eight prompt groups of four rows each, interleaved across the batch."""
    rng = random.Random(0)
    rows = []
    for group in range(_GROUPS):
        prompt_length = rng.randint(20, 60)
        for index in range(_GROUP_SIZE):
            rows.append(
                (
                    f"grp{group}",
                    index,
                    prompt_length,
                    prompt_length + rng.randint(1, 90),
                )
            )
    rng.shuffle(rows)
    return KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[f"{group}_g{index}" for group, index, _, _ in rows],
        fields=["input_ids"],
        sequence_lengths=[length for *_, length in rows],
        extra_info={},
        tags=[
            {
                GROUP_ID_TAG: group,
                SHARED_PREFIX_PROMPT_LENGTHS: prompt_length,
                "owner": f"{group}_g{index}",
            }
            for group, index, prompt_length, _ in rows
        ],
    )


def _shard(meta: KVBatchMeta, dp_world: int, **kwargs: Any):
    return shard_meta_for_dp(
        meta,
        dp_world=dp_world,
        sequence_packing_args=dict(_PACKING_ARGS),
        shared_prefix_groups=True,
        **kwargs,
    )


@pytest.mark.mcore
@pytest.mark.parametrize("work_weights", [None, (4, 1)], ids=["rows", "work"])
@pytest.mark.parametrize("dp_world", [1, 2, 4, 8])
def test_shared_prefix_preshard_assigns_complete_groups_and_round_trips(
    dp_world, work_weights
):
    pytest.importorskip("megatron.rl.shared_prefix_metadata")
    from megatron.rl.shared_prefix_metadata import plan_fixed_execution_slots

    meta = _grouped_meta()
    shards, permutation = _shard(
        meta, dp_world, shared_prefix_work_weights=work_weights
    )

    assert len(shards) == dp_world
    expected_slots = plan_fixed_execution_slots(
        group_ids=[tag[GROUP_ID_TAG] for tag in meta.tags],
        sequence_lengths=meta.sequence_lengths,
        bin_capacity=_PACKING_ARGS["max_tokens_per_microbatch"],
        batch_size=None,
        sequence_length_pad_multiple=1,
    ).row_slot_ids
    original_row = {sample_id: row for row, sample_id in enumerate(meta.sample_ids)}
    ranks_by_group: dict[str, set[int]] = defaultdict(set)
    for rank, shard in enumerate(shards):
        assert shard.tags is not None
        slots = shard.extra_info[SHARED_PREFIX_EXECUTION_SLOT]
        assert len(slots) == len(shard.sample_ids)
        for sample_id, tag, slot in zip(shard.sample_ids, shard.tags, slots):
            # Tags, lengths and prescribed slots travel with their rows.
            assert tag["owner"] == sample_id
            assert slot == expected_slots[original_row[sample_id]]
            ranks_by_group[tag[GROUP_ID_TAG]].add(rank)
        assert shard.sequence_lengths == [
            meta.sequence_lengths[original_row[sample_id]]
            for sample_id in shard.sample_ids
        ]
    # Every group lands whole on one rank, and every rank gets the same count.
    assert len(ranks_by_group) == _GROUPS
    assert all(len(ranks) == 1 for ranks in ranks_by_group.values())
    groups_per_rank = [
        sum(1 for ranks in ranks_by_group.values() if rank in ranks)
        for rank in range(dp_world)
    ]
    assert groups_per_rank == [_GROUPS // dp_world] * dp_world

    flat = [sample_id for shard in shards for sample_id in shard.sample_ids]
    assert sorted(flat) == sorted(meta.sample_ids)
    if permutation is None:
        assert flat == meta.sample_ids
    else:
        restored = BatchedDataDict(
            {
                "ids": list(flat),
                "rows": torch.tensor([original_row[k] for k in flat]),
            }
        )
        restored.reorder_data(permutation)
        assert restored["ids"] == meta.sample_ids
        assert restored["rows"].tolist() == list(range(len(flat)))


@pytest.mark.mcore
def test_shared_prefix_preshard_rejects_groups_that_cannot_split_evenly():
    pytest.importorskip("megatron.rl.shared_prefix_metadata")
    with pytest.raises(ValueError, match="complete groups per rank"):
        _shard(_grouped_meta(), 3)


@pytest.mark.mcore
def test_shared_prefix_work_weights_cost_each_execution_slot(monkeypatch):
    cost = pytest.importorskip("megatron.rl.shared_prefix_cost")
    from megatron.rl.shared_prefix_metadata import plan_fixed_execution_slots

    estimate = cost.estimate_shared_prefix_row_work
    calls = []

    def record(**kwargs):
        calls.append(kwargs)
        return estimate(**kwargs)

    monkeypatch.setattr(cost, "estimate_shared_prefix_row_work", record)
    meta = _grouped_meta()
    _shard(meta, 2, shared_prefix_work_weights=(4, 1))

    # A group split into K slots stores its prompt K times.
    expected_slots = plan_fixed_execution_slots(
        group_ids=[tag[GROUP_ID_TAG] for tag in meta.tags],
        sequence_lengths=meta.sequence_lengths,
        bin_capacity=_PACKING_ARGS["max_tokens_per_microbatch"],
        batch_size=None,
        sequence_length_pad_multiple=1,
    ).row_slot_ids
    assert max(expected_slots) > 0
    assert len(calls) == 1 and calls[0]["row_slot_ids"] == expected_slots


@pytest.mark.mcore
def test_shared_prefix_work_weights_require_prompt_length_tags():
    pytest.importorskip("megatron.rl.shared_prefix_metadata")
    meta = _grouped_meta()
    for tag in meta.tags:
        del tag[SHARED_PREFIX_PROMPT_LENGTHS]
    with pytest.raises(ValueError, match="prompt length tags"):
        _shard(meta, 2, shared_prefix_work_weights=(4, 1))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {
                "shared_prefix_groups": True,
                "dynamic_batching_args": {"max_tokens_per_microbatch": 256},
            },
            "does not support dynamic batching",
        ),
        ({"shared_prefix_groups": True}, "requires sequence-packing arguments"),
        (
            {
                "shared_prefix_groups": True,
                "sequence_packing_args": {"algorithm": "first_fit_decreasing"},
            },
            "max_tokens_per_microbatch",
        ),
        (
            {
                "sequence_packing_args": dict(_PACKING_ARGS),
                "shared_prefix_work_weights": (4, 1),
            },
            "complete-group sharding",
        ),
    ],
    ids=[
        "dynamic_batching",
        "no_packing_args",
        "no_token_budget",
        "work_weights_without_groups",
    ],
)
def test_shared_prefix_preshard_rejects_unsupported_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        shard_meta_for_dp(_grouped_meta(), dp_world=2, **kwargs)


class _RecordingDataPlane:
    def __init__(self) -> None:
        self.fields: dict[str, list[str]] = {}

    def register_partition(self, *, partition_id: str, fields, **kwargs) -> None:
        del kwargs
        self.fields[partition_id] = list(fields)


@pytest.mark.parametrize(
    "mode, registered",
    [
        (None, False),
        ("disabled", False),
        ("dense", False),
        ("logprobs", True),
        ("train", True),
    ],
)
def test_rollout_partition_carries_prompt_lengths_only_when_sharing(mode, registered):
    from nemo_rl.algorithms.single_controller_utils.setup import (
        _register_single_controller_partitions,
    )
    from tests.unit.single_controller.test_setup import _make_master_config

    master_config = _make_master_config()
    master_config.token_capture.enabled = True
    if mode is not None:
        master_config.policy["shared_prefix_training"] = {"mode": mode}
    data_plane = _RecordingDataPlane()

    _register_single_controller_partitions(
        data_plane,
        master_config=master_config,
        partition_id="rollout_data",
        include_multimodal_fields=False,
    )

    rollout_fields = data_plane.fields["rollout_data"]
    assert (SHARED_PREFIX_PROMPT_LENGTHS in rollout_fields) is registered
    assert rollout_fields.count(SHARED_PREFIX_PROMPT_LENGTHS) <= 1


_STAGING_PARTITION = "rollout_staging_shared_prefix_test"
_CANONICAL_PARTITION = "rollout_data_shared_prefix_test"
_CANONICAL_FIELDS = [
    "input_ids",
    "input_lengths",
    "generation_logprobs",
    "token_mask",
    "sample_mask",
    "prompt_ids_for_adv",
    "total_reward",
    "mask_sample",
    "truncated",
    SHARED_PREFIX_PROMPT_LENGTHS,
]


@pytest.mark.nemo_gym
def test_reassembler_publishes_prompt_lengths_and_group_tags(tq_client):
    pytest.importorskip("nemo_gym.token_id_capture.staging")
    from nemo_rl.data_plane.tq_token_sink import STAGING_FIELDS, TQTokenSink
    from nemo_rl.experience.rollout_reassembler import RolloutReassembler
    from tests.unit.data_plane.token_capture_test_fixtures import (
        build_fixture_artifacts,
    )

    tq_client.register_partition(
        partition_id=_STAGING_PARTITION,
        fields=list(STAGING_FIELDS),
        num_samples=8,
        consumer_tasks=["finalize"],
    )
    tq_client.register_partition(
        partition_id=_CANONICAL_PARTITION,
        fields=_CANONICAL_FIELDS,
        num_samples=8,
        consumer_tasks=["train"],
    )
    try:
        group_id = "grp_shared"
        rollout_ids = [f"{group_id}_g0", f"{group_id}_g1"]
        records, receipt, expected = build_fixture_artifacts(
            "worked_example", rollout_id=rollout_ids[0]
        )
        sink = TQTokenSink(tq_client, staging_partition=_STAGING_PARTITION)
        for record in records:
            assert sink.stage(record).ok
        receipt = receipt.model_dump()
        receipt["rollout_id"] = rollout_ids[0]
        reassembler = RolloutReassembler(
            tq_client,
            partition_id=_CANONICAL_PARTITION,
            staging_partition=_STAGING_PARTITION,
            pad_token_id=0,
            max_seq_len=4096,
            include_shared_prefix_metadata=True,
        )

        finalized = reassembler.finalize_group(
            group_id,
            rollout_ids,
            [receipt, None],  # the second rollout lost its receipt: placeholder
            [1.0, 0.0],
            mask_sample=[True, False],
            fallback_weight_version=0,
            prompt_idx=3,
            loss_multiplier=1.0,
        )

        assert finalized.meta is not None
        assert finalized.meta.sample_ids == rollout_ids
        assert [tag[GROUP_ID_TAG] for tag in finalized.meta.tags] == [group_id] * 2
        published = tq_client.get_samples(
            sample_ids=rollout_ids,
            partition_id=_CANONICAL_PARTITION,
            select_fields=[SHARED_PREFIX_PROMPT_LENGTHS],
        )
        prompt_lengths = torch.as_tensor(published[SHARED_PREFIX_PROMPT_LENGTHS])
        # The verified capture boundary for the valid row; 0 forces the
        # placeholder onto the conventional fallback path.
        assert prompt_lengths.flatten().tolist() == [expected.prompt_len, 0]
    finally:
        tq_client.clear_samples(sample_ids=None, partition_id=_STAGING_PARTITION)
        tq_client.clear_samples(sample_ids=None, partition_id=_CANONICAL_PARTITION)


def _pump_controller(monkeypatch, *, num_prompts_per_step, dp_world, shared_prefix):
    """The repo's train-pump double with a DP axis and a min-groups sampler."""
    import nemo_rl.algorithms.single_controller as single_controller
    from nemo_rl.algorithms.async_utils.staleness_sampler import BaseSampler
    from tests.unit.single_controller.test_single_controller_actor import (
        _EmptySampler,
        _NoOpTrainer,
        _train_pump_controller,
    )

    class _ShardedTrainer(_NoOpTrainer):
        # No shared-prefix attributes: the gate must come from the controller.
        def __init__(self) -> None:
            self.sharding_annotations = SimpleNamespace(
                get_axis_size=lambda axis: dp_world
            )
            self.train_calls = 0

        def train_microbatches_from_meta(self, meta, *, train_fields):
            del meta, train_fields
            self.train_calls += 1

    class _MinGroupsSampler(_EmptySampler):
        """Returns exactly ``min_prompt_groups`` fresh groups."""

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.next_group = 0

        async def select(self, **kwargs):
            BaseSampler._validate_group_bounds(
                kwargs["min_prompt_groups"], kwargs["max_prompt_groups"]
            )
            self.calls.append(dict(kwargs))
            count = kwargs["min_prompt_groups"]
            groups = [f"grp{self.next_group + offset}" for offset in range(count)]
            self.next_group += count
            meta = KVBatchMeta(
                partition_id="rollout_data",
                task_name="train",
                sample_ids=[f"{group}_g0" for group in groups],
                fields=[],
                sequence_lengths=[1] * count,
                tags=[{"weight_version": 0, GROUP_ID_TAG: group} for group in groups],
            )
            return meta, count

    sampler = _MinGroupsSampler()
    controller = _train_pump_controller(sampler=sampler)
    controller._shared_prefix_logprobs_enabled = shared_prefix
    controller._algo_cfg.num_prompts_per_step = num_prompts_per_step
    controller._async_cfg.min_groups_for_streaming_train = 5
    controller._rollout_exhausted.clear()
    controller._buffer_capacity = asyncio.Semaphore(64)
    controller._trainer = _ShardedTrainer()
    controller._sync_weights = AsyncMock(return_value=0)
    controller._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})
    return controller, sampler


def test_train_pump_rounds_shared_prefix_chunks_up_to_whole_dp_groups(monkeypatch):
    # 12 groups at DP 4 with a 5-group streaming minimum: 8 (5 rounded up to
    # whole DP groups), then the 4-group remainder.
    controller, sampler = _pump_controller(
        monkeypatch, num_prompts_per_step=12, dp_world=4, shared_prefix=True
    )

    asyncio.run(asyncio.wait_for(controller._train_pump(), timeout=5.0))

    assert controller._train_steps == 1
    assert [
        (call["min_prompt_groups"], call["max_prompt_groups"]) for call in sampler.calls
    ] == [(8, 8), (4, 4)]
    assert controller._trainer.train_calls == 2


def test_train_pump_without_shared_prefix_never_aligns_to_dp(monkeypatch):
    # The default path keeps exact streaming chunks, even when the step is not
    # a multiple of DP, and needs no shared-prefix attribute on the trainer.
    controller, sampler = _pump_controller(
        monkeypatch, num_prompts_per_step=6, dp_world=4, shared_prefix=False
    )

    asyncio.run(asyncio.wait_for(controller._train_pump(), timeout=5.0))

    assert controller._train_steps == 1
    assert [
        (call["min_prompt_groups"], call["max_prompt_groups"]) for call in sampler.calls
    ] == [(5, 6), (1, 1)]
    assert all("prompt_group_multiple" not in call for call in sampler.calls)
