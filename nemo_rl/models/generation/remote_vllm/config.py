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

"""Configuration for an externally managed stock vLLM rollout server."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, model_validator


class RemoteVllmRefitConfig(BaseModel, extra="forbid"):
    """Disk-checkpoint refit settings for vLLM's development control API."""

    checkpoint_dir: str = Field(min_length=1)
    inflight_policy: Literal["abort", "wait", "keep"] = "keep"
    timeout_s: PositiveFloat = 1800.0

    @model_validator(mode="after")
    def validate_checkpoint_dir(self) -> "RemoteVllmRefitConfig":
        if not Path(self.checkpoint_dir).expanduser().is_absolute():
            raise ValueError(
                "checkpoint_dir must be an absolute path shared by the policy "
                "and external vLLM containers"
            )
        return self


class RemoteVllmServiceConfig(BaseModel, extra="forbid"):
    """Connection and control settings for one external vLLM deployment.

    ``base_url`` may include the OpenAI ``/v1`` prefix. ``control_base_url``
    can point at a separate control-plane fan-out proxy when the generation
    endpoint is a router that does not expose vLLM's development endpoints.
    When omitted, control requests use ``base_url`` for compatibility with a
    single vLLM server or the original NeMo RL proxy deployment.
    """

    base_url: str = Field(min_length=1)
    control_base_url: str | None = None
    served_model_name: str = Field(min_length=1)
    max_model_len: PositiveInt
    health_path: str = "/health"
    models_path: str = "/v1/models"
    server_info_path: str = "/server_info"
    pause_path: str = "/pause"
    resume_path: str = "/resume"
    collective_rpc_path: str = "/collective_rpc"
    reset_prefix_cache_path: str = "/reset_prefix_cache"
    request_timeout_s: PositiveFloat = 600.0
    connect_timeout_s: PositiveFloat = 5.0
    api_key_env_var: str | None = None
    refit: RemoteVllmRefitConfig

    @model_validator(mode="after")
    def validate_urls_and_timeouts(self) -> "RemoteVllmServiceConfig":
        for field_name in ("base_url", "control_base_url"):
            url = getattr(self, field_name)
            if url is not None and not url.startswith(("http://", "https://")):
                raise ValueError(f"{field_name} must start with http:// or https://")
        path_fields = (
            "health_path",
            "models_path",
            "server_info_path",
            "pause_path",
            "resume_path",
            "collective_rpc_path",
            "reset_prefix_cache_path",
        )
        for field_name in path_fields:
            path = getattr(self, field_name)
            if not path.startswith("/"):
                raise ValueError(f"{field_name} must be an absolute HTTP path")
        if self.connect_timeout_s > self.request_timeout_s:
            raise ValueError("connect_timeout_s cannot exceed request_timeout_s")
        return self

    @property
    def effective_control_base_url(self) -> str:
        """Return the endpoint used for pause/refit/reset/resume operations."""
        return self.control_base_url or self.base_url


class RemoteVllmServiceInfo(BaseModel, extra="forbid"):
    """Non-mutating facts verified during setup-time preflight."""

    model: str
    dev_mode_control_api: bool
    server_info: dict[str, Any]
