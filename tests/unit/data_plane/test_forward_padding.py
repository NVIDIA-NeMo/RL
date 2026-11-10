# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Model alignment survives jagged TQ storage and DP sharding."""

import pytest
import torch

from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.column_io import kv_first_write
from nemo_rl.data_plane.driver_mixin import TQDriverMixin
from nemo_rl.data_plane.interfaces import KVBatchMeta
from nemo_rl.data_plane.preshard import shard_meta_for_dp
from nemo_rl.data_plane.schema import GLOBAL_FORWARD_PAD_SEQLEN
from nemo_rl.data_plane.worker_mixin import TQWorkerMixin
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

from ._rollout_shapes import register_train_partition


@pytest.mark.parametrize(
    "producer_multiple,sequence_round,expected_length",
    [(1, 1, 2368), (96, 1, 2496), (16, 256, 2560)],
)
@pytest.mark.parametrize("task_name", ["prev_lp", "train"])
def test_model_alignment_applies_to_driver_and_all_dp_workers(
    producer_multiple, sequence_round, expected_length, task_name
):
    # 2336 triggered HybridEP's 64-token alignment assertion in live SC.
    lengths = torch.tensor([2200, 2336], dtype=torch.int32)
    batch = BatchedDataDict(
        input_ids=torch.ones(2, 2336, dtype=torch.long),
        input_lengths=lengths,
        token_mask=torch.ones(2, 2336),
        sample_mask=torch.ones(2),
    )
    client = NoOpDataPlaneClient()
    register_train_partition(client, num_samples=2, fields=list(batch))
    meta = kv_first_write(
        batch,
        sample_ids=["a", "b"],
        dp_client=client,
        partition_id="train",
        pad_to_multiple=producer_multiple,
    )
    driver = TQDriverMixin()
    driver.cfg = {
        "make_sequence_length_divisible_by": 64,
        "dynamic_batching": {"train_mb_tokens": 4096},
    }
    driver.use_dynamic_batches = sequence_round > 1
    driver.dynamic_batching_args = {"sequence_length_round": sequence_round}
    driver.use_sequence_packing = False
    driver.dp_client = client
    dispatch = driver._isolated_meta(meta, fields=list(batch), task_name=task_name)
    assert GLOBAL_FORWARD_PAD_SEQLEN not in meta.extra_info
    assert dispatch.extra_info[GLOBAL_FORWARD_PAD_SEQLEN] == expected_length
    fetched = driver.read_from_dataplane(dispatch, select_fields=list(batch))
    assert fetched["input_ids"].shape == (2, expected_length)
    shards, _ = shard_meta_for_dp(dispatch, dp_world=2)
    worker = TQWorkerMixin()
    worker._dp_client = client
    for shard in shards:
        local = worker._fetch(shard)
        assert local["input_ids"].shape == (1, expected_length)
        valid = int(local["input_lengths"][0])
        assert torch.all(local["input_ids"][0, :valid] == 1)
        assert torch.count_nonzero(local["input_ids"][0, valid:]) == 0
        assert torch.count_nonzero(local["token_mask"][0, valid:]) == 0


def test_each_model_mints_its_own_alignment_without_reusing_a_stale_target():
    meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=["a"],
        fields=["input_ids"],
        sequence_lengths=[2336],
        extra_info={GLOBAL_FORWARD_PAD_SEQLEN: 4096},
    )
    driver = TQDriverMixin()
    driver.cfg = {"make_sequence_length_divisible_by": 64}
    policy = driver._isolated_meta(meta, fields=["input_ids"], task_name="prev_lp")
    driver.cfg = {"make_sequence_length_divisible_by": 32}
    critic = driver._isolated_meta(meta, fields=["input_ids"], task_name="values")
    assert policy.extra_info[GLOBAL_FORWARD_PAD_SEQLEN] == 2368
    assert critic.extra_info[GLOBAL_FORWARD_PAD_SEQLEN] == 2336
    assert meta.extra_info[GLOBAL_FORWARD_PAD_SEQLEN] == 4096
