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

"""Configuration for refitting generation from a shared HF checkpoint."""

from pathlib import Path
from pydantic import BaseModel, Field, model_validator


class HfCheckpointRefitConfig(BaseModel, extra="forbid"):
    """Shared storage used to publish immutable policy exports."""

    checkpoint_dir: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_checkpoint_dir(self) -> "HfCheckpointRefitConfig":
        if not Path(self.checkpoint_dir).expanduser().is_absolute():
            raise ValueError("checkpoint_dir must be an absolute shared path")
        return self
