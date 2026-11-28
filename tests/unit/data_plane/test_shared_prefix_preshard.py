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
"""Shared-prefix planning on the driver.

``TQPolicy`` adds ``share_prefixes`` to the sequence-packing arguments.
``shard_meta_for_dp`` then plans every rank's forwards from row tags and ships
one tree id per row. A tree is the set of rows that run one copy of the prompt
they start with.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_rl.data.packing.shared_prefix.plan import plan_shared_prefix_shards
from nemo_rl.data.packing.shared_prefix.row_metadata import (
    PROMPT_HASH_TAG,
    PROMPT_LENGTH_TAG,
    SHARED_PREFIX_TREE_IDS,
    UNSHARED_ROW,
    prompt_hash,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.preshard import shard_meta_for_dp
from nemo_rl.data_plane.schema import (
    ELEM_COUNTS_PER_GB,
    GLOBAL_FORWARD_PAD_SEQLEN,
    GROUP_ID_TAG,
    MICRO_BATCH_INDICES,
    MICRO_BATCH_LENGTHS,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.tq_policy import TQPolicy

_POLICY_PACKING_ARGS: dict[str, Any] = {
    "algorithm": "modified_first_fit_decreasing",
    "sequence_length_pad_multiple": 1,
    "input_key": "input_ids",
    "input_lengths_key": "input_lengths",
}
_PACKING_ARGS: dict[str, Any] = {
    **_POLICY_PACKING_ARGS,
    "max_tokens_per_microbatch": 256,
}
_SHARED_ARGS: dict[str, Any] = {
    **_PACKING_ARGS,
    "share_prefixes": True,
    "bound_by_full_length": False,
}
_PAD_SEQLEN = 128


def _meta(rows: list[tuple[str, int, int]]) -> KVBatchMeta:
    """Build a batch from one ``(group, prompt length, row length)`` triple per row.

    Row ``i`` has sample id ``row{i}``, and its ``owner`` tag repeats that id,
    so a test can check that tags stay with their rows. The group name stands
    in for the prompt tokens that the prompt hash covers.
    """
    sample_ids = [f"row{row}" for row in range(len(rows))]
    return KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=sample_ids,
        fields=["input_ids"],
        sequence_lengths=[length for _, _, length in rows],
        extra_info={GLOBAL_FORWARD_PAD_SEQLEN: _PAD_SEQLEN},
        tags=[
            {
                GROUP_ID_TAG: group,
                PROMPT_LENGTH_TAG: prompt_length,
                PROMPT_HASH_TAG: prompt_hash(list(group.encode())),
                "owner": sample_id,
            }
            for sample_id, (group, prompt_length, _) in zip(
                sample_ids, rows, strict=True
            )
        ],
    )


def _grouped_rows(num_groups: int, group_size: int) -> list[tuple[str, int, int]]:
    """Rows of ``num_groups`` prompt groups, interleaved so no group is contiguous."""
    return [
        (f"group-{group}", 20 + 5 * group, 30 + 6 * group + 7 * row)
        for row in range(group_size)
        for group in range(num_groups)
    ]


def _shard(
    meta: KVBatchMeta,
    dp_world: int,
    *,
    batch_size: int | None = None,
    **packing_args: Any,
) -> list[KVBatchMeta]:
    return shard_meta_for_dp(
        meta,
        dp_world=dp_world,
        batch_size=batch_size,
        sequence_packing_args={**_SHARED_ARGS, **packing_args},
    )


def _row(sample_id: str) -> int:
    return int(sample_id.removeprefix("row"))


def _tree_ids(shards: list[KVBatchMeta]) -> list[int]:
    return [
        tree_id
        for shard in shards
        for tree_id in shard.extra_info[SHARED_PREFIX_TREE_IDS]
    ]


def _policy(**sequence_packing: Any) -> TQPolicy:
    """A ``TQPolicy`` with only the state that ``_packing_args`` reads."""
    policy = object.__new__(TQPolicy)
    policy.cfg = {
        "sequence_packing": {
            "enabled": True,
            "algorithm": "modified_first_fit_decreasing",
            "train_mb_tokens": 256,
            "logprob_mb_tokens": 512,
            **sequence_packing,
        },
        "megatron_cfg": {"enabled": True},
    }
    policy.use_sequence_packing = True
    policy.sequence_packing_args = dict(_POLICY_PACKING_ARGS)
    return policy


def _assert_packed_without_sharing(
    shards: list[KVBatchMeta], meta: KVBatchMeta
) -> None:
    """The shards equal what ordinary sequence packing gives for the same rows."""
    expected = shard_meta_for_dp(
        replace(meta, tags=None),
        dp_world=len(shards),
        sequence_packing_args=dict(_PACKING_ARGS),
    )
    for shard, dense in zip(shards, expected, strict=True):
        assert SHARED_PREFIX_TREE_IDS not in shard.extra_info
        assert shard.sample_ids == dense.sample_ids
        assert shard.extra_info == dense.extra_info


@pytest.mark.parametrize("dp_world", [1, 2, 4, 8])
def test_the_plan_is_built_from_row_tags(dp_world: int) -> None:
    meta = _meta(_grouped_rows(8, 4))
    planned = plan_shared_prefix_shards(
        sequence_lengths=meta.sequence_lengths,
        group_ids=[tag[GROUP_ID_TAG] for tag in meta.tags],
        prompt_lengths=[tag[PROMPT_LENGTH_TAG] for tag in meta.tags],
        prompt_hashes=[tag[PROMPT_HASH_TAG] for tag in meta.tags],
        num_shards=dp_world,
        batch_size=None,
        max_tokens_per_microbatch=256,
        sequence_length_pad_multiple=1,
        algorithm="modified_first_fit_decreasing",
        bound_by_full_length=False,
        largest_first=False,
    )

    shards = _shard(meta, dp_world)

    assert len(shards) == dp_world
    for shard, plan in zip(shards, planned, strict=True):
        assert [_row(sample_id) for sample_id in shard.sample_ids] == plan.row_indices
        assert shard.extra_info[SHARED_PREFIX_TREE_IDS] == plan.tree_ids
        assert shard.extra_info[MICRO_BATCH_INDICES] == plan.micro_batch_indices
        assert shard.extra_info[MICRO_BATCH_LENGTHS] == plan.micro_batch_lengths
        assert shard.extra_info[ELEM_COUNTS_PER_GB] == plan.elem_counts_per_gb
    # Every row of the eight groups runs in a tree.
    assert UNSHARED_ROW not in _tree_ids(shards)


def test_each_rank_gets_the_plan_next_to_the_callers_extra_info() -> None:
    meta = _meta(_grouped_rows(8, 4))
    for shard in _shard(meta, 4):
        assert set(shard.extra_info) == {
            GLOBAL_FORWARD_PAD_SEQLEN,
            MICRO_BATCH_INDICES,
            MICRO_BATCH_LENGTHS,
            ELEM_COUNTS_PER_GB,
            SHARED_PREFIX_TREE_IDS,
        }
        assert shard.extra_info[GLOBAL_FORWARD_PAD_SEQLEN] == _PAD_SEQLEN
        assert shard.partition_id == meta.partition_id
    assert meta.extra_info == {GLOBAL_FORWARD_PAD_SEQLEN: _PAD_SEQLEN}


def test_tree_ids_and_tags_stay_with_their_rows() -> None:
    # Eight groups that share a prompt, and four rows that share nothing.
    rows = _grouped_rows(8, 4) + [("single", 0, 40)] * 4
    shards = _shard(_meta(rows), 2)
    for shard in shards:
        tree_ids = shard.extra_info[SHARED_PREFIX_TREE_IDS]
        assert len(tree_ids) == len(shard.sample_ids)
        for sample_id, length, tag in zip(
            shard.sample_ids, shard.sequence_lengths, shard.tags, strict=True
        ):
            assert tag["owner"] == sample_id
            assert length == rows[_row(sample_id)][2]
        for start, end in shard.extra_info[MICRO_BATCH_INDICES][0]:
            groups_of_tree: dict[int, set[str]] = {}
            for tree_id, tag in zip(tree_ids[start:end], shard.tags[start:end]):
                groups_of_tree.setdefault(tree_id, set()).add(tag[GROUP_ID_TAG])
            if UNSHARED_ROW in groups_of_tree:
                assert groups_of_tree == {UNSHARED_ROW: {"single"}}
            else:
                assert all(len(groups) == 1 for groups in groups_of_tree.values())
    assert _tree_ids(shards).count(UNSHARED_ROW) == 4


def test_each_global_batch_is_planned_on_its_own_rows() -> None:
    # Two global batches of 16 rows. Each holds two rows of every group.
    shards = _shard(_meta(_grouped_rows(8, 4)), 2, batch_size=16)
    visited: list[int] = []
    for shard in shards:
        data = BatchedDataDict(
            {
                "row": torch.tensor(
                    [_row(sample_id) for sample_id in shard.sample_ids]
                ),
                SHARED_PREFIX_TREE_IDS: torch.tensor(
                    shard.extra_info[SHARED_PREFIX_TREE_IDS]
                ),
            }
        )
        data.micro_batch_indices = shard.extra_info[MICRO_BATCH_INDICES]
        data.micro_batch_lengths = shard.extra_info[MICRO_BATCH_LENGTHS]
        data.elem_counts_per_gb = shard.extra_info[ELEM_COUNTS_PER_GB]
        for global_batch in range(2):
            batch = data.get_batch(global_batch)
            microbatches = list(batch.make_microbatch_iterator_for_packable_sequences())
            rows = [
                row for microbatch in microbatches for row in microbatch["row"].tolist()
            ]
            # The forwards cover the rank's rows of this global batch once, in order.
            assert rows == batch["row"].tolist()
            assert all(row // 16 == global_batch for row in rows)
            for microbatch in microbatches:
                tree_ids = microbatch[SHARED_PREFIX_TREE_IDS].tolist()
                # A tree here is the two rows of one group in this global batch.
                assert all(tree_ids.count(tree_id) == 2 for tree_id in tree_ids)
            visited.extend(rows)
    assert sorted(visited) == list(range(32))


@pytest.mark.parametrize(
    "packing_args",
    [_PACKING_ARGS, {**_SHARED_ARGS, "share_prefixes": False}],
    ids=["absent", "false"],
)
def test_without_the_option_rows_are_packed_without_sharing(
    packing_args: dict[str, Any],
) -> None:
    meta = _meta(_grouped_rows(8, 4))
    shards = shard_meta_for_dp(
        meta, dp_world=2, sequence_packing_args=dict(packing_args)
    )
    _assert_packed_without_sharing(shards, meta)


@pytest.mark.parametrize("missing_tag", [PROMPT_LENGTH_TAG, PROMPT_HASH_TAG])
def test_a_row_without_a_prompt_tag_turns_sharing_off_for_the_batch(
    missing_tag: str,
) -> None:
    meta = _meta(_grouped_rows(8, 4))
    tags = [dict(tag) for tag in meta.tags]
    del tags[5][missing_tag]
    shards = _shard(replace(meta, tags=tags), 2)
    _assert_packed_without_sharing(shards, meta)


def test_a_batch_without_tags_is_packed_without_sharing() -> None:
    meta = _meta(_grouped_rows(8, 4))
    shards = _shard(replace(meta, tags=None), 2)
    _assert_packed_without_sharing(shards, meta)


@pytest.mark.parametrize(
    "num_groups, dp_world",
    [(7, 4), (3, 2), (1, 2)],
    ids=[
        "8_groups_minus_one_on_4_ranks",
        "4_groups_minus_one_on_2_ranks",
        "one_group_on_2_ranks",
    ],
)
def test_prompt_groups_need_not_divide_among_the_ranks(
    num_groups: int, dp_world: int
) -> None:
    """After a step loses a prompt group, the ranks still run equally many forwards."""
    shards = _shard(_meta(_grouped_rows(num_groups, 4)), dp_world)
    assert sorted(
        _row(sample_id) for shard in shards for sample_id in shard.sample_ids
    ) == list(range(4 * num_groups))
    assert len({len(shard.extra_info[MICRO_BATCH_INDICES][0]) for shard in shards}) == 1
    assert UNSHARED_ROW not in _tree_ids(shards)


def test_the_full_length_bound_reaches_the_planner() -> None:
    # Two trees of four rows. Each is 160 tokens packed and 400 tokens in full.
    meta = _meta([("a", 80, 100)] * 4 + [("b", 80, 100)] * 4)
    (packed,) = _shard(meta, 1, max_tokens_per_microbatch=400)
    (full,) = _shard(meta, 1, max_tokens_per_microbatch=400, bound_by_full_length=True)
    assert packed.extra_info[MICRO_BATCH_LENGTHS] == [[320]]
    assert full.extra_info[MICRO_BATCH_LENGTHS] == [[160, 160]]


def test_the_microbatch_order_reaches_the_planner() -> None:
    meta = _meta([("a", 0, 100), ("b", 0, 100), ("c", 0, 250), ("d", 0, 90)])
    (in_order,) = _shard(meta, 1, max_tokens_per_microbatch=300)
    (largest_first,) = _shard(
        meta, 1, max_tokens_per_microbatch=300, microbatch_order="largest_first"
    )
    assert in_order.extra_info[MICRO_BATCH_LENGTHS] == [[250, 290]]
    assert largest_first.extra_info[MICRO_BATCH_LENGTHS] == [[290, 250]]


def test_a_shared_batch_needs_a_group_tag_on_every_row() -> None:
    meta = _meta(_grouped_rows(2, 2))
    tags = [dict(tag) for tag in meta.tags]
    del tags[1][GROUP_ID_TAG]
    with pytest.raises(ValueError, match="carry no 'group_id' tag"):
        _shard(replace(meta, tags=tags), 1)


def test_a_row_longer_than_the_token_budget_is_rejected() -> None:
    meta = _meta([("a", 20, 257), ("a", 20, 30)])
    with pytest.raises(ValueError, match="must be nonempty and fit one forward"):
        _shard(meta, 1)


def test_the_full_length_choice_is_required() -> None:
    packing_args = {**_PACKING_ARGS, "share_prefixes": True}
    with pytest.raises(KeyError, match="bound_by_full_length"):
        shard_meta_for_dp(
            _meta(_grouped_rows(2, 2)), dp_world=1, sequence_packing_args=packing_args
        )


@pytest.mark.parametrize(
    "option", [{}, {"share_prefixes": False}], ids=["absent", "false"]
)
def test_tq_policy_leaves_the_packing_arguments_alone_without_the_option(
    option: dict[str, Any],
) -> None:
    policy = _policy(**option)
    assert policy._packing_args("train_mb_tokens") == (_PACKING_ARGS, None)


@pytest.mark.parametrize(
    "budget_key, max_tokens, full_length",
    [("logprob_mb_tokens", 512, False), ("train_mb_tokens", 256, True)],
)
def test_tq_policy_adds_the_choice_to_the_packing_arguments(
    budget_key: str, max_tokens: int, full_length: bool
) -> None:
    policy = _policy(share_prefixes=True)
    packing_args, dynamic_batching_args = policy._packing_args(budget_key)
    assert packing_args == {
        **_POLICY_PACKING_ARGS,
        "max_tokens_per_microbatch": max_tokens,
        "share_prefixes": True,
        "bound_by_full_length": full_length,
    }
    assert dynamic_batching_args is None
    # The arguments the policy keeps for its later dispatches are not edited.
    assert policy.sequence_packing_args == _POLICY_PACKING_ARGS


def test_tq_policy_packing_arguments_make_preshard_share_prompts() -> None:
    packing_args, _ = _policy(share_prefixes=True)._packing_args("logprob_mb_tokens")
    shards = shard_meta_for_dp(
        _meta(_grouped_rows(8, 4)), dp_world=2, sequence_packing_args=packing_args
    )
    assert UNSHARED_ROW not in _tree_ids(shards)


class _PreshardReached(Exception):
    """Ends a dispatch at its ``shard_meta_for_dp`` call."""


@pytest.mark.parametrize(
    "dispatch, full_length",
    [
        ("logprob_dispatch", False),
        ("train_from_meta", True),
        ("train_microbatches_from_meta", True),
    ],
)
def test_tq_policy_dispatches_hand_the_choice_to_preshard(
    dispatch: str, full_length: bool
) -> None:
    """Each dispatch packs with the arguments of its own stage."""
    policy = _policy(share_prefixes=True)
    policy._router_replay_enabled = False
    policy._opd_full_field = None
    policy._opd_full_teacher_index_field = None
    policy.sharding_annotations = MagicMock()
    policy.sharding_annotations.get_axis_size.return_value = 2
    meta = _meta(_grouped_rows(8, 4))
    preshard_arguments: dict[str, Any] = {}

    def reach_preshard(meta: KVBatchMeta, **arguments: Any) -> None:
        del meta
        preshard_arguments.update(arguments)
        raise _PreshardReached

    with (
        patch.object(TQPolicy, "_stamp_pad_seqlen"),
        patch(
            "nemo_rl.models.policy.tq_policy.shard_meta_for_dp",
            side_effect=reach_preshard,
        ),
        pytest.raises(_PreshardReached),
    ):
        if dispatch == "logprob_dispatch":
            policy._logprob_dispatch(
                meta,
                task_name="prev_lp",
                worker_method="get_logprobs_presharded",
                timer_prefix="prev_lp",
                timer=None,
                common_kwargs={},
            )
        elif dispatch == "train_from_meta":
            policy.train_from_meta(meta, loss_fn=MagicMock(), gbs=32, mbs=1)
        else:
            policy.train_microbatches_from_meta(meta)

    packing_args = preshard_arguments["sequence_packing_args"]
    assert packing_args["share_prefixes"] is True
    assert packing_args["bound_by_full_length"] is full_length
    assert preshard_arguments["dynamic_batching_args"] is None


def test_tq_policy_without_sequence_packing_reads_no_option() -> None:
    policy = object.__new__(TQPolicy)
    policy.cfg = {}
    assert policy._packing_args("train_mb_tokens") == (None, None)
