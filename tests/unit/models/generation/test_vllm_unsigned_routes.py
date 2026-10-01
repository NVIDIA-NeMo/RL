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

import numpy as np
import pytest
import torch

from nemo_rl.models.generation.vllm import utils


@pytest.mark.parametrize("dtype", [torch.int16, torch.int32])
def test_uint16_routes_preserve_expert_ids(monkeypatch, dtype):
    monkeypatch.setattr(utils, "G_ROUTED_EXPERTS_RANGE_CHECKED", False)
    routes = np.array([[0, 127], [128, 255]], dtype=np.uint16)

    result = utils._as_routed_experts_tensor(
        routes, device=torch.device("cpu"), dtype=dtype
    )

    assert result.dtype == dtype
    assert result.tolist() == routes.tolist()


def test_uint16_routes_reject_overflow_before_narrowing(monkeypatch):
    monkeypatch.setattr(utils, "G_ROUTED_EXPERTS_RANGE_CHECKED", False)
    routes = np.array([[0, 128]], dtype=np.uint16)

    with pytest.raises(ValueError, match="exceeds the resolved carry dtype"):
        utils._as_routed_experts_tensor(
            routes, device=torch.device("cpu"), dtype=torch.int8
        )
