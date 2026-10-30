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

"""Project-local configuration, independent of the NeMo RL policy schema."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class LoGRAConfig(BaseModel, extra="forbid"):
    enabled: bool = True
    rank: Annotated[int, Field(gt=0)] = 256
    seed: int = 42
    distribution: Literal["rademacher", "gaussian"] = "rademacher"
    refresh: bool = True
    optimizer: Literal["row_adam", "sgd"] = "row_adam"
    beta2: Annotated[float, Field(ge=0, lt=1)] = 0.95
    epsilon: Annotated[float, Field(gt=0)] = 1e-8
    target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    freeze_non_target: bool = False
