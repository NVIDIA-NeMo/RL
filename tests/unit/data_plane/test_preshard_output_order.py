# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from nemo_rl.data_plane.interfaces import KVBatchMeta
from nemo_rl.data_plane.preshard import shard_meta_for_dp
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@pytest.mark.parametrize("order", [(0, 1, 2, 3), (2, 0, 3, 1), (1, 2, 3, 0)])
def test_preshard_permutation_restores_original_rows(order):
    ids = [f"row-{i}" for i in range(4)]
    meta = KVBatchMeta(
        partition_id="test",
        task_name="logprobs",
        sample_ids=ids,
        sequence_lengths=[10, 20, 30, 40],
        tags=[{"row": i} for i in range(4)],
    )
    planned = [
        BatchedDataDict({"meta_idx": torch.tensor(order[:2])}),
        BatchedDataDict({"meta_idx": torch.tensor(order[2:])}),
    ]
    with patch.object(
        BatchedDataDict, "shard_by_batch_size", return_value=(planned, None)
    ):
        shards, permutation = shard_meta_for_dp(
            meta, dp_world=2, sequence_packing_args={"max_tokens_per_microbatch": 128}
        )
    result = BatchedDataDict(
        {
            "ids": [key for shard in shards for key in shard.sample_ids],
            "values": torch.tensor(order),
        }
    )
    if permutation is not None:
        result.reorder_data(permutation)
    assert result["ids"] == ids
    assert torch.equal(result["values"], torch.arange(4))
    assert [tag["row"] for shard in shards for tag in shard.tags] == list(order)
    assert (permutation is None) == (order == (0, 1, 2, 3))
