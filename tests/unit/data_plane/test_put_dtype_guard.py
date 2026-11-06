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
"""put_samples rejects dtypes torch supports only partially (uint16/32/64)."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.codec import check_put_dtypes


@pytest.mark.parametrize("dtype", [torch.uint16, torch.uint32, torch.uint64])
def test_check_put_dtypes_rejects_limited_support_dtype(dtype):
    # e.g. vLLM routed-expert ids for >256 experts arrive as uint16.
    fields = TensorDict(
        {"routed_experts": torch.zeros(2, 3, 22, dtype=dtype)}, batch_size=[2]
    )
    with pytest.raises(TypeError, match="routed_experts"):
        check_put_dtypes(fields)


def test_check_put_dtypes_accepts_supported_and_jagged_fields():
    fields = TensorDict(
        {
            "input_ids": torch.nested.nested_tensor(
                [torch.ones(3, dtype=torch.int64), torch.ones(5, dtype=torch.int64)],
                layout=torch.jagged,
            ),
            "routed_experts": torch.zeros(2, 3, 22, dtype=torch.int16),
            "digest_bytes": torch.zeros(2, 32, dtype=torch.uint8),
            "sample_mask": torch.ones(2, dtype=torch.float32),
        },
        batch_size=[2],
    )
    check_put_dtypes(fields)


def test_put_samples_rejects_uint16_before_writing():
    client = NoOpDataPlaneClient()
    client.register_partition(
        partition_id="p",
        fields=["routed_experts"],
        num_samples=2,
        consumer_tasks=["train"],
    )
    bad = TensorDict(
        {"routed_experts": torch.zeros(2, 3, 22, dtype=torch.uint16)}, batch_size=[2]
    )
    with pytest.raises(TypeError, match="routed_experts"):
        client.put_samples(["a", "b"], "p", fields=bad)
    assert client.list_sample_ids("p") == []
