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

"""Settings for an existing CoreWeave Dedicated Inference deployment."""

from pydantic import BaseModel, Field


class CoreWeaveConfig(BaseModel, extra="forbid"):
    """Rollout and checkpoint settings; credentials are read from mounted files/environment."""

    base_url: str
    model: str
    deployment_id: str
    run_id: str
    state_dir: str
    max_model_len: int = Field(gt=0)
    api_key_file: str | None = None
    concurrency: int = Field(default=32, gt=0)
    timeout: float = Field(default=600.0, gt=0, allow_inf_nan=False)
    max_attempts: int = Field(default=3, gt=0)
    full_every: int = Field(default=1, ge=1, le=63)
    upload_workers: int = Field(default=8, gt=0)
    wait: bool = True
    verify: bool = True
    prompt_cache_policy: str = "reset_all"
    require_logprobs: bool = True
    importance_sampling: bool = True

    def validate_training(self) -> None:
        """Require a completed update and exact rollout probabilities."""
        if not self.wait or not self.verify or not self.require_logprobs:
            raise ValueError(
                "CoreWeave requires wait, verify, and require_logprobs=true"
            )
        if self.prompt_cache_policy != "reset_all":
            raise ValueError(
                "CoreWeave synchronous GRPO requires prompt_cache_policy=reset_all"
            )
