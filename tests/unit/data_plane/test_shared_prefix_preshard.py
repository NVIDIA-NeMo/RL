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
"""Shared-prefix transport through the data plane.

Covers the driver-side pieces between rollout and the Megatron worker:
complete-group DP sharding with prescribed execution slots, the reassembler's
prompt-length column and group tags, and the rollout partition schema. The
train pump's DP-aligned prompt-group selection is covered in
``tests/unit/single_controller/test_shared_prefix_controller.py``.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

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
    """Eight prompt groups of four rows each, interleaved across the batch.

    Only the ``GROUP_ID_TAG`` row tag names the group: sample ids are opaque and
    do not follow the ``{group}_g{index}`` rollout naming.
    """
    rng = random.Random(0)
    rows = []
    for group in range(_GROUPS):
        prompt_length = rng.randint(20, 60)
        for _ in range(_GROUP_SIZE):
            rows.append(
                (f"grp{group}", prompt_length, prompt_length + rng.randint(1, 90))
            )
    rng.shuffle(rows)
    sample_ids = [f"row{row}" for row in range(len(rows))]
    return KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=sample_ids,
        fields=["input_ids"],
        sequence_lengths=[length for *_, length in rows],
        extra_info={},
        tags=[
            {
                GROUP_ID_TAG: group,
                SHARED_PREFIX_PROMPT_LENGTHS: prompt_length,
                "owner": sample_id,
            }
            for sample_id, (group, prompt_length, _) in zip(sample_ids, rows)
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
        # Prompt lengths travel only as the column, never as row tags.
        assert all(
            SHARED_PREFIX_PROMPT_LENGTHS not in tag for tag in finalized.meta.tags
        )
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
