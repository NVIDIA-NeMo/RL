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

import math


def batch_invariant_token_multiple(configured_multiple: int, tp_size: int) -> int:
    """Return a token multiple compatible with batch-invariant MCore inference."""
    if configured_multiple < 1:
        raise ValueError("configured_multiple must be positive.")
    if tp_size < 1:
        raise ValueError("tp_size must be positive.")

    # Import lazily so non-Megatron policy drivers do not import Megatron-Core.
    # TOKEN_ROUNDER is MCore's single source of truth for eager and graphed
    # batch-invariant inference token alignment.
    from megatron.core.inference.batch_dimensions_utils import TOKEN_ROUNDER

    inference_multiple = ((TOKEN_ROUNDER + tp_size - 1) // tp_size) * tp_size
    return math.lcm(configured_multiple, inference_multiple)
