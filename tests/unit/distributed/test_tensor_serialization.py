# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Portable RPC tensor payloads preserve dtypes, shapes, and writable storage."""

import pytest
import torch

from nemo_rl.distributed.tensor_serialization import (
    tensor_from_payload,
    tensor_to_payload,
)


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float32, torch.int64, torch.bool]
)
@pytest.mark.parametrize("shape", [(), (0, 3), (2, 3)])
def test_tensor_payload_round_trip(dtype, shape):
    tensor = torch.ones(shape, dtype=dtype)
    if tensor.ndim == 2:
        tensor = tensor.T
    restored = tensor_from_payload(tensor_to_payload(tensor))
    torch.testing.assert_close(restored, tensor)
    restored.zero_()
    assert tensor.eq(1).all()


def test_tensor_payload_detaches_autograd():
    tensor = torch.tensor(2.0, requires_grad=True)
    restored = tensor_from_payload(tensor_to_payload(tensor))
    assert restored.item() == 2.0
    assert not restored.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tensor_payload_stages_cuda_tensor_to_cpu():
    tensor = torch.arange(6, device="cuda", dtype=torch.float32).reshape(2, 3)
    restored = tensor_from_payload(tensor_to_payload(tensor))
    assert restored.device.type == "cpu"
    torch.testing.assert_close(restored, tensor.cpu())
