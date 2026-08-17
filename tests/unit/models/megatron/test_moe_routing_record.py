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

import torch

from nemo_rl.models.megatron.moe_routing_record import (
    align_routing_rows_to_token_count,
    build_routed_experts_batch,
    coerce_routing_to_3d,
)


def test_coerce_routing_to_3d_accepts_tokens_layers_topk():
    routing = torch.zeros(5, 2, 4, dtype=torch.int32)
    assert coerce_routing_to_3d(routing).shape == (5, 2, 4)


def test_align_routing_rows_to_token_count_pads_with_last_row():
    routing = torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int32)
    aligned = align_routing_rows_to_token_count(routing, num_tokens=4)
    assert aligned.shape == (4, 1, 2)
    assert aligned[2].tolist() == [[3, 4]]
    assert aligned[3].tolist() == [[3, 4]]


def test_build_routed_experts_batch_right_pads_to_seq_dim():
    routing_a = torch.ones(3, 2, 1, dtype=torch.int32)
    routing_b = torch.full((2, 2, 1), 7, dtype=torch.int32)
    seq_lengths = torch.tensor([3, 2])
    batch = build_routed_experts_batch(
        [routing_a, routing_b],
        seq_lengths,
        seq_dim=5,
    )
    assert batch is not None
    assert batch.shape == (2, 5, 2, 1)
    assert batch[0, :3, 0, 0].tolist() == [1, 1, 1]
    assert batch[0, 3:, 0, 0].tolist() == [0, 0]
    assert batch[1, :2, 0, 0].tolist() == [7, 7]
