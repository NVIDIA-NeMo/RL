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

"""Length-aware (LFS) admission scheduling for vLLM rollouts."""

from nemo_rl.models.generation.vllm.lfs.groups import build_async_cross_dp_group_ids
from nemo_rl.models.generation.vllm.lfs.modes import (
    LFS_ADMISSION_FAIRNESS_SEMANTICS,
    CrossDpMode,
    DpSelectionMode,
)
from nemo_rl.models.generation.vllm.lfs.scheduler import CrossDpSchedulerState

__all__ = [
    "LFS_ADMISSION_FAIRNESS_SEMANTICS",
    "CrossDpMode",
    "CrossDpSchedulerState",
    "DpSelectionMode",
    "build_async_cross_dp_group_ids",
]
