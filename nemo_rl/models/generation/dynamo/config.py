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

"""Configuration for serving NeMo RL's vLLM workers behind Dynamo.

``policy.generation.dynamo: true`` on any vLLM config is the whole switch;
``policy.generation.dynamo_cfg`` only tunes the frontend and registration, and
every field below carries its default.
"""

from typing import Any, Literal, Mapping

from pydantic import BaseModel, Field, PositiveFloat


class DynamoFrontendArgs(BaseModel, extra="forbid"):
    """Arguments for the managed ``dynamo.frontend`` process."""

    # "kv" routes each request to the engine holding most of its prefix, fed by
    # the engines' vLLM KV-cache events. No "direct": it needs a worker ID on
    # every request, and NeMo RL's rollouts name none.
    router_mode: Literal[
        "round-robin",
        "random",
        "power-of-two",
        "kv",
        "least-loaded",
        "device-aware-weighted",
    ] = "kv"
    extra_cli_args: list[str] = Field(default_factory=list)


class DynamoCfg(BaseModel, extra="forbid"):
    """``policy.generation.dynamo_cfg``: the frontend and engine registration."""

    # Bounds etcd/NATS/frontend start-up and every engine's registration.
    startup_timeout_s: PositiveFloat = 600.0
    # Deadline for one NeMo RL rollout through the frontend. None waits as long
    # as generation takes, as native generate_async does; on the SingleController,
    # async_rl.rollout_failure.native.generation_timeout_s bounds both paths.
    request_timeout_s: PositiveFloat | None = None
    frontend_args: DynamoFrontendArgs = Field(default_factory=DynamoFrontendArgs)


def dynamo_enabled(generation_config: Mapping[str, Any]) -> bool:
    """Whether ``policy.generation.dynamo`` is on for a vLLM generation config."""
    return bool(generation_config.get("dynamo"))


def dynamo_cfg(generation_config: Mapping[str, Any]) -> DynamoCfg:
    """The validated ``dynamo_cfg`` block; an absent block means all defaults."""
    return DynamoCfg.model_validate(generation_config.get("dynamo_cfg") or {})
