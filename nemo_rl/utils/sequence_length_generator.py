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
from typing import Callable

import numpy as np


def get_sequence_length_generator(sequence_length_generator_cfg: dict) -> Callable:
    mean = sequence_length_generator_cfg["mean"]
    std = sequence_length_generator_cfg["std"]
    length = int(np.round(np.random.normal(mean, std)))
    length = max(1, length)
    return length
