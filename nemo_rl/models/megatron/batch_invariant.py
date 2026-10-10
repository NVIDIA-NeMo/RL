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

# Mirrors megatron.core.inference.batch_dimensions_utils.TOKEN_ROUNDER; the driver
# venv has no Megatron-Core, so it cannot be imported here.
MCORE_TOKEN_ROUNDER = 64


def batch_invariant_token_multiple(tp_size: int) -> int:
    """Smallest token multiple compatible with batch-invariant MCore inference."""
    if tp_size < 1:
        raise ValueError("tp_size must be positive.")
    return ((MCORE_TOKEN_ROUNDER + tp_size - 1) // tp_size) * tp_size
