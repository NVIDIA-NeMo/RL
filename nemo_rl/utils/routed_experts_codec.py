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
"""Compact wire codec for router-replay routed-expert indices.

Routed experts for a long-context sample are millions of ints (shape
[tokens, num_moe_layers, topk]). Serialized as nested JSON lists they cost
~1s of single-threaded CPU per serialize/parse hop, and every HTTP hop on
the NeMo Gym path (model server, agent, resources server) pays that again
for pydantic validation and re-serialization. Encoded as a single base64
string the payload stays one opaque Python object end to end, so
intermediate hops only pay a string copy.

Envelope format (version 1):
    "nrlre1:<dtype>:<S>x<L>x<K>:<base64 of C-contiguous array bytes>"

This module must stay importable inside the NeMo Gym actor, so it may only
depend on numpy and torch.
"""

import base64
from typing import Any, Union

import numpy as np
import torch

_MAGIC = "nrlre1"
_NP_DTYPES = {"int8": np.int8, "int16": np.int16, "int32": np.int32}
_TORCH_DTYPE_NAMES = {
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
}


def encode_routed_experts(routed_experts: torch.Tensor) -> str:
    """Encode a [tokens, num_moe_layers, topk] tensor as a base64 envelope.

    The tensor's own dtype (int8/int16/int32, as resolved by
    ``resolve_routed_experts_dtype``) is preserved on the wire.
    """
    if routed_experts.dim() != 3:
        raise ValueError(
            "routed_experts must have shape [tokens, num_moe_layers, topk], "
            f"got {tuple(routed_experts.shape)}."
        )
    dtype_name = _TORCH_DTYPE_NAMES.get(routed_experts.dtype)
    if dtype_name is None:
        raise ValueError(
            f"Unsupported routed_experts dtype {routed_experts.dtype}; "
            f"expected one of {sorted(_NP_DTYPES)}."
        )
    arr = routed_experts.detach().cpu().contiguous().numpy()
    tokens, num_layers, topk = arr.shape
    # memoryview feeds the buffer to b64encode zero-copy; tobytes() would
    # materialize a full-size intermediate copy of the payload.
    data = base64.b64encode(memoryview(arr)).decode("ascii")
    return f"{_MAGIC}:{dtype_name}:{tokens}x{num_layers}x{topk}:{data}"


def decode_routed_experts(payload: Union[str, Any], dtype: torch.dtype) -> torch.Tensor:
    """Decode routed experts into a tensor of the requested dtype.

    Accepts the base64 envelope produced by ``encode_routed_experts`` or the
    legacy nested-list format.
    """
    if not isinstance(payload, str):
        return torch.as_tensor(payload, dtype=dtype)
    parts = payload.split(":", 3)
    if len(parts) != 4 or parts[0] != _MAGIC:
        raise ValueError(
            "routed_experts string payload is not a valid "
            f"'{_MAGIC}:<dtype>:<SxLxK>:<base64>' envelope."
        )
    _, dtype_name, shape_str, data = parts
    if dtype_name not in _NP_DTYPES:
        raise ValueError(f"Unsupported routed_experts dtype '{dtype_name}'.")
    shape = tuple(int(dim) for dim in shape_str.split("x"))
    if len(shape) != 3:
        raise ValueError(
            f"routed_experts envelope shape '{shape_str}' is not 3-dimensional."
        )
    arr = np.frombuffer(base64.b64decode(data), dtype=_NP_DTYPES[dtype_name])
    expected = shape[0] * shape[1] * shape[2]
    if arr.size != expected:
        raise ValueError(
            f"routed_experts envelope has {arr.size} elements, expected "
            f"{expected} for shape {shape}."
        )
    # frombuffer views are read-only; copy() yields a writable array.
    return torch.from_numpy(arr.reshape(shape).copy()).to(dtype)
