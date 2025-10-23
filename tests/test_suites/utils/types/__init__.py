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

from typing import Literal, Union

# Import MasterConfig types from all algorithms
from nemo_rl.algorithms.distillation import MasterConfig as DistillationMasterConfig
from nemo_rl.algorithms.dpo import MasterConfig as DPOMasterConfig
from nemo_rl.algorithms.grpo import MasterConfig as GRPOMasterConfig
from nemo_rl.algorithms.rm import MasterConfig as RMMasterConfig
from nemo_rl.algorithms.sft import MasterConfig as SFTMasterConfig

# Type aliases for algorithms, model classes, and configs
Algorithm = Literal["sft", "grpo", "dpo", "distillation", "rm"]
ModelClass = Literal["llm", "vlm"]
MasterConfigUnion = Union[
    SFTMasterConfig,
    GRPOMasterConfig,
    DPOMasterConfig,
    DistillationMasterConfig,
    RMMasterConfig,
]

__all__ = [
    "Algorithm",
    "ModelClass",
    "MasterConfigUnion",
    "DistillationMasterConfig",
    "DPOMasterConfig",
    "GRPOMasterConfig",
    "RMMasterConfig",
    "SFTMasterConfig",
]
