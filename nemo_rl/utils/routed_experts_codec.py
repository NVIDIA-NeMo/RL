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
depend on torch.
"""

import base64
from dataclasses import dataclass, field
from typing import Any, Union

import torch

_MAGIC = "nrlre1"
_WIRE_TORCH_DTYPES = {
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
}
_TORCH_DTYPE_NAMES = {v: k for k, v in _WIRE_TORCH_DTYPES.items()}

# Opaque framework attachment; never part of capture extras or the wire format.
ROUTED_EXPERTS_STAGING_ATTACHMENT = "nrl_routed_experts_staging"


@dataclass(frozen=True, init=False)
class RoutedExpertsStagingPayload:
    """Own a CPU route snapshot and the exact envelope committed by its record.

    The capture record still digests the ordinary envelope. The local sink can
    reuse its tensor instead of decoding that envelope. No global cache or
    request lookup is involved; the attachment lives only through staging.
    """

    encoded: str = field(repr=False)
    _routes: torch.Tensor = field(repr=False)
    _version: int = field(repr=False)

    def __init__(self, routes: torch.Tensor) -> None:
        # Own the bytes: later engine/cache mutations cannot change a staged
        # record. Disable inference mode so mutation checks also work when the
        # source tensor was created under inference_mode().
        with torch.inference_mode(False):
            snapshot = routes.detach().to("cpu").clone().contiguous()
        object.__setattr__(self, "encoded", encode_routed_experts(snapshot))
        object.__setattr__(self, "_routes", snapshot)
        object.__setattr__(self, "_version", snapshot._version)

    def tensor_for_payload(self, encoded: object) -> torch.Tensor:
        """Return read-only-by-contract routes bound to this record's envelope.

        Raises:
            ValueError: The record names another envelope, or a consumer has
                modified the owned tensor since its envelope was computed.
        """
        if not isinstance(encoded, str) or encoded != self.encoded:
            raise ValueError("route staging attachment does not match record payload")
        if self._routes._version != self._version:
            raise ValueError("route staging attachment was modified after encoding")
        return self._routes


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
            f"expected one of {sorted(_WIRE_TORCH_DTYPES)}."
        )
    arr = routed_experts.detach().cpu().contiguous().numpy()
    tokens, num_layers, topk = arr.shape
    # memoryview feeds the buffer to b64encode zero-copy; tobytes() would
    # materialize a full-size intermediate copy of the payload.
    data = base64.b64encode(memoryview(arr)).decode("ascii")
    return f"{_MAGIC}:{dtype_name}:{tokens}x{num_layers}x{topk}:{data}"


def decode_routed_experts(
    payload: Union[str, Any], dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Decode routed experts into a tensor of ``dtype``.

    Accepts the base64 envelope produced by ``encode_routed_experts`` or the
    legacy nested-list format. With ``dtype=None`` the envelope keeps the
    dtype it was encoded with, and a nested list decodes as int16.
    """
    if not isinstance(payload, str):
        return torch.as_tensor(payload, dtype=dtype or torch.int16)
    parts = payload.split(":", 3)
    if len(parts) != 4 or parts[0] != _MAGIC:
        raise ValueError(
            "routed_experts string payload is not a valid "
            f"'{_MAGIC}:<dtype>:<SxLxK>:<base64>' envelope."
        )
    _, dtype_name, shape_str, data = parts
    wire_dtype = _WIRE_TORCH_DTYPES.get(dtype_name)
    if wire_dtype is None:
        raise ValueError(f"Unsupported routed_experts dtype '{dtype_name}'.")
    shape = tuple(int(dim) for dim in shape_str.split("x"))
    if len(shape) != 3:
        raise ValueError(
            f"routed_experts envelope shape '{shape_str}' is not 3-dimensional."
        )
    # Decode into a writable bytearray so torch.frombuffer can share its
    # memory directly (frombuffer writability follows the underlying buffer;
    # the bytearray construction is the single unavoidable copy).
    raw = bytearray(base64.b64decode(data, validate=True))
    expected = shape[0] * shape[1] * shape[2]
    if dtype is None:
        dtype = wire_dtype
    if len(raw) != expected * wire_dtype.itemsize:
        raise ValueError(
            f"routed_experts envelope has {len(raw) // wire_dtype.itemsize} "
            f"elements, expected {expected} for shape {shape}."
        )
    if expected == 0:
        return torch.empty(shape, dtype=dtype)
    tensor = torch.frombuffer(raw, dtype=wire_dtype).reshape(shape)
    return tensor if tensor.dtype == dtype else tensor.to(dtype)
