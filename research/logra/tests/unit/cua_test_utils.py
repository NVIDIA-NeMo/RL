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

"""Small shared optimizer fixture for CUA numerical tests."""

import torch
from logra.config import LoGRAConfig
from logra.optimizer import LoGRAOptimizer
from torch import nn


def create_optimizer(
    dtype: torch.dtype = torch.float32,
) -> tuple[nn.Sequential, LoGRAOptimizer]:
    torch.manual_seed(7)
    net = nn.Sequential(nn.Linear(2, 3, bias=False, dtype=dtype))
    optimizer = LoGRAOptimizer(
        net,
        torch.optim.AdamW(net.parameters(), lr=0.01, weight_decay=0),
        LoGRAConfig(
            rank=2, optimizer="sgd", target_modules=["0"], freeze_non_target=True
        ),
    )
    return net, optimizer
