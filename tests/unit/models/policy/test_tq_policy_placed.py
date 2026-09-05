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

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import GLOBAL_FORWARD_PAD_SEQLEN
from nemo_rl.models.policy.tq_policy import TQPolicy


def _meta(rank: int, fields: list[str]) -> KVBatchMeta:
    return KVBatchMeta(
        partition_id="train",
        task_name="loader",
        sample_ids=[f"rank-{rank}-sample-0", f"rank-{rank}-sample-1"],
        fields=fields,
        sequence_lengths=[8 + rank, 16 + 8 * rank],
        extra_info={"logical_dp_rank": rank},
    )


def _policy() -> tuple[TQPolicy, MagicMock]:
    policy = object.__new__(TQPolicy)
    policy.cfg = {"train_global_batch_size": 4, "train_micro_batch_size": 1}
    policy._router_replay_enabled = False
    policy.flops_tracker = None
    policy.sharding_annotations = MagicMock()
    policy.sharding_annotations.get_axis_size.return_value = 2
    worker_group = MagicMock()
    policy.worker_group = worker_group
    return policy, worker_group


def test_train_placed_microbatches_keeps_fields_and_replica_delivery() -> None:
    policy, worker_group = _policy()
    dp_metas = [
        _meta(0, ["input_ids", "pixel_values"]),
        _meta(1, ["input_ids", "image_grid_thw"]),
    ]

    assert policy.train_placed_microbatches(dp_metas) is None

    dispatch = worker_group.run_all_workers_sharded_data.call_args
    dispatched = dispatch.kwargs["meta"]
    assert [meta.fields for meta in dispatched] == [
        ["input_ids", "pixel_values"],
        ["input_ids", "image_grid_thw"],
    ]
    assert [meta.task_name for meta in dispatched] == ["train", "train"]
    assert [meta.extra_info[GLOBAL_FORWARD_PAD_SEQLEN] for meta in dispatched] == [
        24,
        24,
    ]

    assert dispatch.args[0] == "train_microbatch_presharded"
    assert dispatch.kwargs["in_sharded_axes"] == ["data_parallel"]
    assert dispatch.kwargs["replicate_on_axes"] == [
        "context_parallel",
        "tensor_parallel",
        "pipeline_parallel",
    ]
    assert dispatch.kwargs["output_is_replicated"] == [
        "context_parallel",
        "tensor_parallel",
        "pipeline_parallel",
    ]
    worker_group.get_all_worker_results.assert_called_once()


def test_train_placed_microbatches_requires_one_batch_per_dp_rank() -> None:
    policy, worker_group = _policy()

    with pytest.raises(ValueError, match="one batch per DP rank"):
        policy.train_placed_microbatches([_meta(0, ["input_ids"])])

    worker_group.run_all_workers_sharded_data.assert_not_called()


def test_train_placed_microbatches_rejects_sequence_packing() -> None:
    policy, worker_group = _policy()
    policy.use_sequence_packing = True
    policy.sequence_packing_args = {"algorithm": "modified_first_fit_decreasing"}
    policy.cfg["sequence_packing"] = {"train_mb_tokens": 4096}

    with pytest.raises(ValueError, match="fixed batches only"):
        policy.train_placed_microbatches(
            [_meta(0, ["input_ids"]), _meta(1, ["input_ids"])]
        )

    worker_group.run_all_workers_sharded_data.assert_not_called()
