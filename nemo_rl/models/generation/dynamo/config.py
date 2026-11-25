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

"""Configuration for serving NeMo RL's generation workers behind Dynamo.

``policy.generation.dynamo: true`` on a vLLM or Megatron config is the switch;
``policy.generation.dynamo_cfg`` only tunes the frontend, and every field below
carries its default. NeMo-Gym's chat calls are rendered and parsed with the
native server's own settings (``vllm_cfg.http_server_serving_chat_kwargs``).
"""

import os
from typing import Annotated, Any, Literal, Mapping

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, model_validator

from nemo_rl.telemetry.setup import (
    telemetry_enabled_in_env,
    vllm_native_tracing_requested,
)


class DynamoFrontendArgs(BaseModel, extra="forbid"):
    """Arguments for the managed ``dynamo.frontend`` process."""

    # "kv" routes each request to the engine holding most of its prefix, fed by
    # the engines' vLLM KV-cache events. No "direct": it needs a worker ID on
    # every request, and NeMo RL's rollouts name none.
    router_mode: Literal[
        "round-robin",
        "random",
        "kv",
        "least-loaded",
        "device-aware-weighted",
    ] = "kv"
    # Idle seconds a NeMo-Gym token-capture session stays on one engine, as native
    # pins each Gym session to one server; None disables it.
    session_affinity_ttl_secs: Annotated[int, Field(gt=0, le=31_536_000)] | None = 300
    # Most sessions the affinity table holds; a new one past it gets HTTP 529.
    # Every frontend holds the whole fleet's sessions, so size it for all of them.
    # Entries are allocated as sessions start and reaped after the TTL, so the cap
    # only bounds memory; a large run holds more sessions than Dynamo's own 65,536.
    session_affinity_max_entries: PositiveInt = 16_777_216
    # Frontends, spread round-robin over the generation nodes, each with its own
    # router. More than one adds --router-replica-sync; with router_mode kv, every
    # router then sees the requests the others have in flight. NeMo-Gym keeps each
    # session on one.
    num_frontends: PositiveInt = 1
    # Processes that render NeMo-Gym's chat calls and build their replies (the
    # checkpoint's template, tokenizer, HF processor and vLLM's parsers); each
    # uses one core. Every frontend runs its own. See
    # docs/guides/dynamo-generation.md for sizing.
    render_workers: PositiveInt = 8
    extra_cli_args: list[str] = Field(default_factory=list)
    # Extra environment for every frontend, for settings Dynamo reads only from
    # the environment, e.g. DYN_TCP_LISTEN_BACKLOG. It may not set a variable
    # the runtime manages.
    extra_env: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _chat_is_nemo_rls(self) -> "DynamoFrontendArgs":
        # NeMo-Gym's chat calls are rendered and parsed by NeMo RL's chat
        # engine, never by Dynamo's own renderer or chat processor.
        for flag in (
            "--dyn-chat-processor",
            "--dyn-chat-processor-config",
            "--custom-jinja-template",
        ):
            if any(arg.split("=", 1)[0] == flag for arg in self.extra_cli_args):
                raise ValueError(
                    f"dynamo_cfg.frontend_args.extra_cli_args may not set {flag}: "
                    "NeMo RL's chat engine renders and parses chat calls with "
                    "vllm_cfg.http_server_serving_chat_kwargs, as the native server "
                    "does."
                )
        return self


class DynamoCfg(BaseModel, extra="forbid"):
    """``policy.generation.dynamo_cfg``: the managed frontend."""

    # Bounds etcd/NATS/frontend start-up and every engine's registration.
    startup_timeout_s: PositiveFloat = 600.0
    # Deadline for one NeMo RL rollout through the frontend. None waits as long
    # as generation takes, as native generate_async does; on the SingleController,
    # async_rl.rollout_failure.native.generation_timeout_s bounds both paths.
    # NeMo-Gym's own calls keep Gym's timeouts.
    request_timeout_s: PositiveFloat | None = None
    frontend_args: DynamoFrontendArgs = Field(default_factory=DynamoFrontendArgs)


def dynamo_enabled(generation_config: Mapping[str, Any]) -> bool:
    """Whether ``policy.generation.dynamo`` is on, for vLLM or Megatron generation."""
    return bool(generation_config.get("dynamo"))


def dynamo_cfg(generation_config: Mapping[str, Any]) -> DynamoCfg:
    """The validated ``dynamo_cfg`` block; an absent block means all defaults."""
    return DynamoCfg.model_validate(generation_config.get("dynamo_cfg") or {})


def trace_export_env() -> dict[str, str]:
    """Dynamo's trace-context switch, on when the engines trace natively.

    Dynamo carries a call's trace context from the frontend to the engine only
    while its OTLP export is on, and reads the switch when a process first loads
    it, so the frontends and the vLLM workers need it in their environment from
    the start. The condition is the one that turns on vLLM's own tracing
    (``_maybe_enable_vllm_native_tracing``), so the engine has a tracer for the
    trace headers it is handed. Dynamo reads the collector from the same
    ``OTEL_EXPORTER_OTLP_*`` variables.
    """
    endpoint = (
        os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "").strip()
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    )
    if telemetry_enabled_in_env() and vllm_native_tracing_requested() and endpoint:
        return {"OTEL_EXPORT_ENABLED": "1"}
    return {}
