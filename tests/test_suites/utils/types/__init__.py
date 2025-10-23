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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Union

import numpy as np
import torch

from nemo_rl.algorithms.distillation import MasterConfig as DistillationMasterConfig
from nemo_rl.algorithms.dpo import MasterConfig as DPOMasterConfig
from nemo_rl.algorithms.grpo import MasterConfig as GRPOMasterConfig
from nemo_rl.algorithms.rm import MasterConfig as RMMasterConfig
from nemo_rl.algorithms.sft import MasterConfig as SFTMasterConfig

Algorithm = Literal["sft", "grpo", "dpo", "distillation", "rm"]
ModelClass = Literal["llm", "vlm"]
Backend = Literal["fsdp2", "dtensor", "megatron"]
MasterConfigUnion = Union[
    SFTMasterConfig,
    GRPOMasterConfig,
    DPOMasterConfig,
    DistillationMasterConfig,
    RMMasterConfig,
]

# Functional testing types
TensorLike = Union[np.ndarray, torch.Tensor, List, float, int]


@dataclass
class StatResult:
    """Output of a reducer (pure statistics; no pass/fail)."""

    results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CriterionResult:
    """Judgment after applying a Criterion to a StatResult."""

    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "Algorithm",
    "ModelClass",
    "Backend",
    "MasterConfigUnion",
    "DistillationMasterConfig",
    "DPOMasterConfig",
    "GRPOMasterConfig",
    "RMMasterConfig",
    "SFTMasterConfig",
    "TensorLike",
    "StatResult",
    "CriterionResult",
]
