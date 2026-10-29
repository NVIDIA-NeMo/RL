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

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt


class DynamoFrontendArgs(BaseModel, extra="forbid"):
    """Arguments for the managed ``dynamo.frontend`` process."""

    tokenizer: Literal["default", "fastokens"] = "default"
    tokenizer_cache: bool = False
    tokenizer_cache_bytes: PositiveInt = 50 * 1024**2
    # "kv" routes each request to the engine holding most of its prefix, fed by
    # the engines' vLLM KV-cache events.
    router_mode: Literal[
        "round-robin",
        "random",
        "power-of-two",
        "kv",
        "direct",
        "least-loaded",
        "device-aware-weighted",
    ] = "kv"
    extra_cli_args: list[str] = Field(default_factory=list)


class DynamoCfg(BaseModel, extra="forbid"):
    """``policy.generation.dynamo_cfg``: the frontend and engine registration."""

    # Bounds etcd/NATS/frontend start-up and every engine's registration.
    startup_timeout_s: PositiveFloat = 600.0
    # Timeout for one NeMo RL rollout routed through the frontend, in place of
    # NRL_VLLM_ASYNC_TIMEOUT_SECONDS. A timeout fails the attempt as a native one
    # does (GenerationUnavailable), and the rollout retry policy re-dispatches it
    # (SingleController: async_rl.rollout_failure.max_infra_attempts_per_prompt).
    request_timeout_s: PositiveFloat = 900.0
    frontend_args: DynamoFrontendArgs = Field(default_factory=DynamoFrontendArgs)


def dynamo_enabled(generation_config: Mapping[str, Any]) -> bool:
    """Whether ``policy.generation.dynamo`` is on for a vLLM generation config."""
    return bool(generation_config.get("dynamo"))


def dynamo_cfg(generation_config: Mapping[str, Any]) -> DynamoCfg:
    """The validated ``dynamo_cfg`` block; an absent block means all defaults."""
    return DynamoCfg.model_validate(generation_config.get("dynamo_cfg") or {})
