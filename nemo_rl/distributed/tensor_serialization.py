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
"""Backend-independent tensor payloads for policy worker RPC results."""

import numpy as np
import torch

TensorPayload = tuple[np.ndarray, torch.dtype, tuple[int, ...]]


def tensor_to_payload(tensor: torch.Tensor) -> TensorPayload:
    """Encode an RPC tensor without invoking Torch's storage pickler.

    MCore can replace that pickler's loader with a Megatron function, which
    cannot be imported in the Ray driver. Byte arrays also preserve BF16.
    CUDA tensors are staged to host memory and, like all actor return tensors
    using this transport, are restored as CPU tensors in the driver.
    """
    host_tensor = tensor.detach().to(device="cpu").contiguous()
    data = host_tensor.reshape(-1).view(torch.uint8).numpy()
    return data, tensor.dtype, tuple(tensor.shape)


def tensor_from_payload(payload: TensorPayload) -> torch.Tensor:
    """Restore a writable CPU tensor without any backend dependencies."""
    data, dtype, shape = payload
    if data.size == 0:
        return torch.empty(0, dtype=dtype).reshape(shape)
    return torch.from_numpy(data.copy()).view(dtype).reshape(shape)


def register_policy_tensor_serializer() -> None:
    """Register serialization for actor return tensors in this worker process.

    Ray invokes this after an actor method returns and before the result enters
    the object store. Registration is process-local and does not change the
    driver's serializer or MCore's checkpoint loading behavior.
    """
    # Ray is optional in tensor packing/round-trip unit tests.
    import ray.util

    ray.util.register_serializer(
        torch.Tensor, serializer=tensor_to_payload, deserializer=tensor_from_payload
    )
