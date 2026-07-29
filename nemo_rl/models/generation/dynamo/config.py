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

from typing import Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel, PositiveInt, model_validator

from nemo_rl.models.generation.interfaces import GenerationConfig


class DynamoWorkerArgsConfig(TypedDict):
    """Structured arguments passed to each managed ``dynamo.vllm`` worker."""

    tool_call_parser: str | None
    reasoning_parser: str | None
    exclude_tools_when_tool_choice_none: bool
    enable_structural_tag: bool
    structural_tag_scope: Literal["auto", "always"]
    structural_tag_schema: Literal["auto", "strict"]
    custom_jinja_template: str | None
    endpoint_types: list[Literal["chat", "completions"]]
    extra_cli_args: list[str]


class DynamoFrontendArgsConfig(TypedDict):
    """Structured arguments passed to the managed Dynamo frontend."""

    tokenizer: Literal["default", "fastokens"]
    tokenizer_cache: bool
    tokenizer_cache_bytes: int
    router_mode: Literal[
        "round-robin",
        "random",
        "power-of-two",
        "kv",
        "direct",
        "least-loaded",
        "device-aware-weighted",
    ]
    router_reset_states: bool
    extra_cli_args: list[str]


class DynamoRuntimeConfig(TypedDict):
    """Configuration for the driver-owned Dynamo service and worker fleet."""

    engine_world_size: int
    namespace: str | None
    frontend_port: int
    dynamo_python: str
    startup_timeout_s: float
    request_timeout_s: float | None
    etcd_port: int
    etcd_peer_port: int
    nats_port: int
    system_port_base: int
    worker_args: DynamoWorkerArgsConfig
    frontend_args: DynamoFrontendArgsConfig
    metrics_include_prefixes: list[str] | None
    metrics_exclude_prefixes: list[str] | None


class DynamoConfig(GenerationConfig):
    """Generation configuration for a Ray-managed Dynamo vLLM fleet."""

    dynamo_cfg: DynamoRuntimeConfig
    vllm_cfg: NotRequired[dict[str, Any]]
    vllm_kwargs: NotRequired[dict[str, Any]]


class DynamoWorkerArgs(BaseModel, extra="forbid"):
    """Validated managed ``dynamo.vllm`` arguments."""

    tool_call_parser: str | None
    reasoning_parser: str | None
    exclude_tools_when_tool_choice_none: bool
    enable_structural_tag: bool
    structural_tag_scope: Literal["auto", "always"]
    structural_tag_schema: Literal["auto", "strict"]
    custom_jinja_template: str | None
    endpoint_types: list[Literal["chat", "completions"]]
    extra_cli_args: list[str]

    @model_validator(mode="after")
    def _validate_endpoint_types(self) -> "DynamoWorkerArgs":
        if not self.endpoint_types:
            raise ValueError("endpoint_types must contain at least one endpoint.")
        if len(self.endpoint_types) != len(set(self.endpoint_types)):
            raise ValueError("endpoint_types must not contain duplicates.")
        return self


class DynamoFrontendArgs(BaseModel, extra="forbid"):
    """Validated arguments for the managed Dynamo frontend."""

    tokenizer: Literal["default", "fastokens"]
    tokenizer_cache: bool
    tokenizer_cache_bytes: PositiveInt
    router_mode: Literal[
        "round-robin",
        "random",
        "power-of-two",
        "kv",
        "direct",
        "least-loaded",
        "device-aware-weighted",
    ]
    router_reset_states: bool
    extra_cli_args: list[str]


class DynamoCfg(BaseModel, extra="forbid"):
    """Validated driver-owned Dynamo runtime configuration.

    Defaults intentionally live in exemplar YAML rather than this model.
    """

    engine_world_size: PositiveInt
    namespace: str | None
    frontend_port: int
    dynamo_python: str
    startup_timeout_s: float
    request_timeout_s: float | None
    etcd_port: int
    etcd_peer_port: int
    nats_port: int
    system_port_base: int
    worker_args: DynamoWorkerArgs
    frontend_args: DynamoFrontendArgs
    metrics_include_prefixes: list[str] | None
    metrics_exclude_prefixes: list[str] | None

    @model_validator(mode="after")
    def _validate_runtime(self) -> "DynamoCfg":
        if self.startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be positive.")
        for field_name in ("frontend_port", "etcd_port", "etcd_peer_port", "nats_port"):
            value = getattr(self, field_name)
            if not (0 <= value <= 65535):
                raise ValueError(
                    f"{field_name} must be 0 (automatic) or between 1 and 65535."
                )
        if not (1 <= self.system_port_base <= 65535):
            raise ValueError("system_port_base must be between 1 and 65535.")
        if not self.dynamo_python:
            raise ValueError("dynamo_python must not be empty.")
        if self.request_timeout_s is not None and self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive when configured.")
        return self
