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

"""Shared utilities for aligning value-model predictions."""

from typing import Optional

import torch

from nemo_rl.distributed.model_utils import allgather_cp_sharded_tensor


def right_shift_values(values: torch.Tensor) -> torch.Tensor:
    """Align next-token value-head outputs with their input states.

    The first column becomes zero and column ``t`` (for ``t >= 1``) takes the
    prediction from column ``t - 1``. The output shape matches the input shape.
    """
    return torch.cat([torch.zeros_like(values[:, :1]), values[:, :-1]], dim=1)


def gather_and_right_shift_values(
    values: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    sequence_dim: int = 1,
) -> torch.Tensor:
    """Restore global CP sequence order before value temporal alignment."""
    if cp_group is not None and torch.distributed.get_world_size(cp_group) > 1:
        values = allgather_cp_sharded_tensor(values, cp_group, seq_dim=sequence_dim)
    return right_shift_values(values)
