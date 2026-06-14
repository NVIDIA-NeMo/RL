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

"""Checkpoint-engine public interface.

Backend implementations live under :mod:`nemo_rl.utils.checkpoint_engines`.
This module preserves the original import path while keeping backend-specific
code out of the public facade.
"""

from nemo_rl.utils.checkpoint_engines import (
    CheckpointEngine,
    CheckpointEngineRegistry,
    TensorMeta,
    create_checkpoint_engine,
    merge_weight_chunk_batches,
    split_weight_chunks,
)

__all__ = [
    "CheckpointEngine",
    "CheckpointEngineRegistry",
    "TensorMeta",
    "create_checkpoint_engine",
    "merge_weight_chunk_batches",
    "split_weight_chunks",
]
