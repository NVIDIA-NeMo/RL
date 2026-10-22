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

from typing import Any, Literal

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, model_validator


class RemoteVllmRefitConfig(BaseModel, extra="forbid"):
    """NCCL refit settings for one external vLLM 0.29.0 deployment.

    ``nccl_engine_world_size`` counts all TP/PP/DP GPU workers in that one
    deployment; the HTTP frontend and router are not NCCL participants.
    Independent deployments and control fan-out are not supported yet.
    """

    inflight_policy: Literal["abort", "wait", "keep"] = "keep"
    timeout_s: PositiveFloat = 1800.0
    transport: Literal["nccl"] = "nccl"
    nccl_engine_world_size: PositiveInt = 1
    nccl_num_engines: Literal[1] = 1
    nccl_packed_buffer_size_bytes: PositiveInt = 1024**3
    nccl_packed_num_buffers: PositiveInt = 2


class RemoteVllmServiceConfig(BaseModel, extra="forbid"):
    """Connection and control settings for one external vLLM deployment.

    ``base_url`` may include the OpenAI ``/v1`` prefix. Control paths are
    resolved against the URL origin so the same endpoint can be handed to
    NeMo-Gym and used for vLLM's development endpoints.
    """

    base_url: str = Field(min_length=1)
    served_model_name: str = Field(min_length=1)
    max_model_len: PositiveInt
    health_path: str = "/health"
    models_path: str = "/v1/models"
    server_info_path: str = "/server_info?config_format=json"
    pause_path: str = "/pause"
    resume_path: str = "/resume"
    reset_prefix_cache_path: str = "/reset_prefix_cache"
    request_timeout_s: PositiveFloat = 600.0
    connect_timeout_s: PositiveFloat = 5.0
    api_key_env_var: str | None = None
    refit: RemoteVllmRefitConfig

    @model_validator(mode="after")
    def validate_urls_and_timeouts(self) -> "RemoteVllmServiceConfig":
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        path_fields = (
            "health_path",
            "models_path",
            "server_info_path",
            "pause_path",
            "resume_path",
            "reset_prefix_cache_path",
        )
        for field_name in path_fields:
            path = getattr(self, field_name)
            if not path.startswith("/"):
                raise ValueError(f"{field_name} must be an absolute HTTP path")
        if self.connect_timeout_s > self.request_timeout_s:
            raise ValueError("connect_timeout_s cannot exceed request_timeout_s")
        return self


class RemoteVllmServiceInfo(BaseModel, extra="forbid"):
    """Non-mutating facts verified during setup-time preflight."""

    model: str
    dev_mode_control_api: bool
    server_info: dict[str, Any]
