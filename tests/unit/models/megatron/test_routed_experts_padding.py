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
"""Per-microbatch padding of per-row routed_experts."""

import torch

from nemo_rl.data_plane.codec import pad_batch
from nemo_rl.data_plane.schema import MICROBATCH_PADDED_FIELDS
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL


def test_pad_batch_pads_jagged_routes_to_the_old_table():
    lengths = [3, 0, 5]
    rows = [torch.arange(n * 4, dtype=torch.int16).reshape(n, 2, 2) for n in lengths]
    width = 8
    expected = torch.full(
        (3, width, 2, 2), ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL, dtype=torch.int16
    )
    for i, row in enumerate(rows):
        expected[i, : len(row)] = row
    data = BatchedDataDict(
        {
            "input_ids": torch.zeros(3, width, dtype=torch.long),
            "routed_experts": torch.nested.as_nested_tensor(rows, layout=torch.jagged),
        }
    )

    unpadded = pad_batch(data.copy(), None, width, skip=MICROBATCH_PADDED_FIELDS)
    assert unpadded["routed_experts"].is_nested
    assert MICROBATCH_PADDED_FIELDS["routed_experts"] == (
        ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL
    )
    pad_batch(data, MICROBATCH_PADDED_FIELDS, data["input_ids"].shape[1])

    assert torch.equal(data["routed_experts"], expected)
