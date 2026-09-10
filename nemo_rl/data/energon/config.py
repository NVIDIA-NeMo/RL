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

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

from nemo_rl.data.energon.multimodal.model_families import ModelFamily


class EnergonSourceConfig(BaseModel, extra="allow"):
    """One prepared Energon dataset split."""

    path: str
    split: str
    virtual_epoch_length: Annotated[int, Field(ge=0)] = 0
    limit: Annotated[int, Field(ge=1)] | None = None


class EnergonTaskEncoderConfig(BaseModel, extra="allow"):
    """One task encoder selected by registry key."""

    name: str = "generic_sft"
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _from_registry_key(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"name": value}
        return value


class EnergonCookerConfig(BaseModel, extra="allow"):
    """One source cooker selected by registry key."""

    name: str = "generic_conversation"
    options: dict[str, Any] = Field(default_factory=dict)
    has_subflavors: dict[str, str | int | float | bool | None] | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_registry_key(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"name": value}
        return value


class EnergonLoaderConfig(BaseModel, extra="allow"):
    """Shared Energon settings for driver- and worker-owned SFT loaders."""

    model_family: ModelFamily = Field(
        description="Model family used to validate cooker and task-encoder support."
    )
    num_workers: Annotated[int, Field(ge=0)] = 8
    shuffle_buffer_size: Annotated[int, Field(ge=0)] = 1000
    max_samples_per_sequence: None = None
    packing_buffer_size: None = None
    batch_grouping: Literal["auto"] = "auto"
    processor_adapter: Literal["hf_multimodal"] = "hf_multimodal"
    topology_mapper: Literal["default"] = "default"
    task_encoder: EnergonTaskEncoderConfig = Field(
        default_factory=EnergonTaskEncoderConfig
    )
    cookers: list[EnergonCookerConfig] = Field(
        default_factory=lambda: [EnergonCookerConfig()]
    )
    seed_offset: int = 0
    prefetch_factor: Annotated[int, Field(ge=1)] = 2
    checkpoint_every_sec: Annotated[float, Field(gt=0)] = 60.0
    watchdog_timeout_seconds: Annotated[float, Field(gt=0)] | None = 60.0
