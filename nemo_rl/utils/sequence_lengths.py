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

from collections.abc import Sequence

import torch

CpuIntTuple = tuple[int, ...]


def to_cpu_int_tuple(values: torch.Tensor | Sequence[int]) -> CpuIntTuple:
    """Normalize sequence metadata at the host/device API boundary.

    A CUDA tensor incurs one synchronization here. Callers should therefore
    invoke this before the model forward whenever the original CPU values are
    available. Code below this boundary accepts only :class:`CpuIntTuple`.
    """
    if torch.is_tensor(values):
        return tuple(int(value) for value in values.tolist())
    return tuple(int(value) for value in values)
