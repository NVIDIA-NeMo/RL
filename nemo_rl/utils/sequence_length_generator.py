# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Callable, Any, Optional

import numpy as np


def get_sequence_length_generator(constant_length_or_length_distribution: Optional[int | dict[str, Any]]) -> Callable:
    """Returns a callable that samples sequence lengths from a normal distribution.

    Args:
        constant_length_or_length_distribution: A constant length or a dict with keys 'mean' and 'std' for the normal distribution.

    Returns:
        A callable that when invoked returns a sampled sequence length (int >= 1).
    """
    if constant_length_or_length_distribution is None:
        return lambda _: None
    if isinstance(constant_length_or_length_distribution, int):
        return lambda _: max(1, constant_length_or_length_distribution)
    if isinstance(constant_length_or_length_distribution, dict):
        mean = constant_length_or_length_distribution["mean"]
        std = constant_length_or_length_distribution["std"]
        def sample_length(sample_idx: int | None = None) -> int:
            length = int(np.round(np.random.normal(mean, std)))
            return max(1, length)
        return sample_length
    if callable(constant_length_or_length_distribution):
        return constant_length_or_length_distribution
    raise ValueError(f"Invalid constant_length_or_length_distribution: {constant_length_or_length_distribution}")
