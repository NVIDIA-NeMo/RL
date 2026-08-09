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

"""CPU tests for MegatronGeneration DP prompt sharding (driver-side dispatch)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import numpy as np

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.models.generation.megatron.megatron_generation import MegatronGeneration


def _make_megatron_generation(dp_size: int) -> MegatronGeneration:
    mg = MegatronGeneration.__new__(MegatronGeneration)
    mg.cfg = {"_pad_token_id": 0}
    mg.current_generate_dp_shard_idx = 0
    mg._policy = SimpleNamespace(
        sharding_annotations=NamedSharding(
            layout=np.arange(dp_size).reshape(1, dp_size, 1, 1),
            names=[
                "pipeline_parallel",
                "data_parallel",
                "context_parallel",
                "tensor_parallel",
            ],
        ),
        worker_group=MagicMock(),
    )
    mg._policy.worker_group.dp_size = dp_size
    mg._policy.worker_group.dp_leader_worker_indices = list(range(0, dp_size * 2, 2))
    mg._policy.worker_group.get_dp_leader_worker_idx.side_effect = (
        lambda dp_idx: dp_idx * 2
    )
    return mg


@pytest.mark.mcore
def test_generate_shards_batch_across_dp_leaders():
    """MegatronGeneration.generate must fan out like vLLM, not pin worker 0."""
    mg = _make_megatron_generation(dp_size=2)
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
            "input_lengths": torch.tensor([2, 2, 2, 2]),
        }
    )
    shard_a = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "input_lengths": torch.tensor([2, 2]),
        }
    )
    shard_b = BatchedDataDict(
        {
            "input_ids": torch.tensor([[5, 6], [7, 8]]),
            "input_lengths": torch.tensor([2, 2]),
        }
    )
    out_a = BatchedDataDict(
        {
            "output_ids": torch.tensor([[1, 2, 9], [3, 4, 10]]),
            "generation_lengths": torch.tensor([1, 1]),
            "unpadded_sequence_lengths": torch.tensor([3, 3]),
            "logprobs": torch.zeros(2, 3),
        }
    )
    out_b = BatchedDataDict(
        {
            "output_ids": torch.tensor([[5, 6, 11], [7, 8, 12]]),
            "generation_lengths": torch.tensor([1, 1]),
            "unpadded_sequence_lengths": torch.tensor([3, 3]),
            "logprobs": torch.zeros(2, 3),
        }
    )

    with patch.object(
        BatchedDataDict,
        "shard_by_batch_size",
        return_value=[shard_a, shard_b],
    ) as shard_mock:
        future_bundle = object()
        mg._policy.worker_group.run_all_workers_sharded_data.return_value = (
            future_bundle
        )
        mg._policy.worker_group.get_all_worker_results.return_value = [out_a, out_b]

        result = mg.generate(data, greedy=False)

    shard_mock.assert_called_once_with(2, allow_uneven_shards=True)
    mg._policy.worker_group.run_all_workers_sharded_data.assert_called_once_with(
        "generate",
        data=[shard_a, shard_b],
        in_sharded_axes=["data_parallel"],
        replicate_on_axes=None,
        output_is_replicated=None,
        common_kwargs={"greedy": False},
    )
    mg._policy.worker_group.get_all_worker_results.assert_called_once_with(
        future_bundle
    )
    assert result.size == 4
    assert torch.equal(result["output_ids"], torch.tensor([[1, 2, 9], [3, 4, 10], [5, 6, 11], [7, 8, 12]]))
