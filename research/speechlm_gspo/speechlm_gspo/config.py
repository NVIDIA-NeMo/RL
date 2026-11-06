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

"""Native SpeechLM model contract carried in ``automodel_cfg.automodel_kwargs``."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class SpeechLMNativeConfig(BaseModel):
    """Exact trainable surface and audio context limit of the native model."""

    model_config = ConfigDict(extra="forbid")

    trainable_names: tuple[str, ...]
    expected_trainable_tensors: PositiveInt
    expected_trainable_scalars: PositiveInt
    max_expanded_sequence_length: PositiveInt
    model_config_overrides: dict[str, Any] = Field(default_factory=dict)
    activation_checkpointing_perception: bool = True
    deterministic_perception: bool = True
