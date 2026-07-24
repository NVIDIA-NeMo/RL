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

import pytest
import torch

from nemo_rl.utils.routed_experts_codec import (
    decode_routed_experts,
    encode_routed_experts,
)


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32])
def test_round_trip_preserves_values_and_wire_dtype(dtype):
    routes = torch.randint(0, 100, (7, 3, 2), dtype=torch.int32)
    routes[2, 1, 0] = -1  # missing-route sentinel survives signed dtypes
    routes = routes.to(dtype)

    payload = encode_routed_experts(routes)
    assert isinstance(payload, str)
    assert payload.startswith(f"nrlre1:{str(dtype).removeprefix('torch.')}:7x3x2:")

    decoded = decode_routed_experts(payload, dtype=torch.int32)
    assert decoded.dtype == torch.int32
    assert torch.equal(decoded, routes.to(torch.int32))


def test_decode_accepts_legacy_nested_lists():
    decoded = decode_routed_experts([[[1, 2]], [[3, 4]]], dtype=torch.int16)
    assert decoded.dtype == torch.int16
    assert decoded.tolist() == [[[1, 2]], [[3, 4]]]


def test_decode_rejects_malformed_payloads():
    with pytest.raises(ValueError, match="envelope"):
        decode_routed_experts("not-an-envelope", dtype=torch.int32)

    good = encode_routed_experts(torch.zeros(2, 3, 2, dtype=torch.int16))
    magic, dtype_name, _, data = good.split(":", 3)
    with pytest.raises(ValueError, match="expected"):
        # Element count does not match the declared shape.
        decode_routed_experts(f"{magic}:{dtype_name}:5x3x2:{data}", dtype=torch.int32)
    with pytest.raises(ValueError, match="dtype"):
        decode_routed_experts(f"{magic}:int64:2x3x2:{data}", dtype=torch.int32)


def test_encode_rejects_bad_inputs():
    with pytest.raises(ValueError, match="shape"):
        encode_routed_experts(torch.zeros(3, 2, dtype=torch.int16))
    with pytest.raises(ValueError, match="dtype"):
        encode_routed_experts(torch.zeros(2, 3, 2, dtype=torch.int64))
