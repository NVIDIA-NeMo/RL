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

from pydantic import BaseModel, Field, PositiveFloat, model_validator


class DynamoWorkerArgs(BaseModel, extra="forbid"):
    """How each vLLM engine registers with Dynamo (``WorkerConfig`` fields)."""

    # Dynamo's built-in parser names (dynamo-parsers). None follows the native
    # server's vllm_cfg.http_server_serving_chat_kwargs (see dynamo_parsers).
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    # False, as vLLM's OpenAIServingChat (the native server) defaults it.
    exclude_tools_when_tool_choice_none: bool = False
    enable_structural_tag: bool = False
    structural_tag_scope: Literal["auto", "always"] = "auto"
    structural_tag_schema: Literal["auto", "strict"] = "auto"
    # A Jinja chat template file that replaces the model's own. None follows
    # vllm_cfg.http_server_serving_chat_kwargs.chat_template.
    custom_jinja_template: str | None = None
    endpoint_types: list[Literal["chat", "completions"]] = ["chat", "completions"]

    @model_validator(mode="after")
    def _validate_endpoint_types(self) -> "DynamoWorkerArgs":
        if not self.endpoint_types:
            raise ValueError("endpoint_types must contain at least one endpoint")
        if len(self.endpoint_types) != len(set(self.endpoint_types)):
            raise ValueError("endpoint_types must not contain duplicates")
        return self


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
    worker_args: DynamoWorkerArgs = Field(default_factory=DynamoWorkerArgs)
    frontend_args: DynamoFrontendArgs = Field(default_factory=DynamoFrontendArgs)


# vLLM parser names whose Dynamo built-in parser goes by another name. nano_v3 is
# NeMo RL's vLLM plugin: DeepSeek-R1 reasoning parsing, with the whole output as
# content when chat_template_kwargs.enable_thinking is false, as Dynamo's
# nemotron_v3 parser does.
_DYNAMO_REASONING_PARSER_NAMES = {"nano_v3": "nemotron_v3"}
_DYNAMO_TOOL_PARSER_NAMES = {"openai": "harmony"}


def _dynamo_parser(
    configured: str | None,
    native: str | None,
    plugin: str | None,
    renames: Mapping[str, str],
    *,
    kind: str,
    worker_arg: str,
) -> str | None:
    if configured is not None or native is None:
        return configured
    if native in renames:
        return renames[native]
    if plugin:
        raise ValueError(
            f"policy.generation.vllm_cfg.{kind}_parser_plugin loads a Python parser "
            "into vLLM's OpenAI server, which Dynamo's Rust frontend cannot run. Set "
            f"policy.generation.dynamo_cfg.worker_args.{worker_arg} to the Dynamo "
            f"built-in parser for {native!r}."
        )
    return native


def dynamo_parsers(
    generation_config: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    """The ``(tool_call_parser, reasoning_parser)`` Dynamo's frontend applies.

    ``dynamo_cfg.worker_args`` names them explicitly. Otherwise they follow the
    native server's ``vllm_cfg.http_server_serving_chat_kwargs``, renamed where
    Dynamo calls the same parser differently. A parser plugin without such a
    rename has no Dynamo equivalent and is rejected.
    """
    worker_args = dynamo_cfg(generation_config).worker_args
    vllm_cfg = generation_config["vllm_cfg"]
    serving = vllm_cfg.get("http_server_serving_chat_kwargs") or {}
    tool_call_parser = _dynamo_parser(
        worker_args.tool_call_parser,
        serving.get("tool_parser"),
        vllm_cfg.get("tool_parser_plugin"),
        _DYNAMO_TOOL_PARSER_NAMES,
        kind="tool",
        worker_arg="tool_call_parser",
    )
    reasoning_parser = _dynamo_parser(
        worker_args.reasoning_parser,
        serving.get("reasoning_parser"),
        vllm_cfg.get("reasoning_parser_plugin"),
        _DYNAMO_REASONING_PARSER_NAMES,
        kind="reasoning",
        worker_arg="reasoning_parser",
    )
    return tool_call_parser, reasoning_parser


def check_dynamo_serving_chat_kwargs(generation_config: Mapping[str, Any]) -> None:
    """Reject native chat-server settings Dynamo's frontend has no counterpart for.

    The frontend renders templates and parses tool calls itself: it always
    parses ``tool_choice="auto"`` calls when a tool parser is set, and has no
    content-format setting. The native defaults (``enable_auto_tools: true``,
    ``chat_template_content_format: auto``) match it; other values would
    silently change what the policy sees.
    """
    serving = generation_config["vllm_cfg"].get("http_server_serving_chat_kwargs") or {}
    content_format = serving.get("chat_template_content_format")
    if content_format not in (None, "auto"):
        raise ValueError(
            "policy.generation.vllm_cfg.http_server_serving_chat_kwargs."
            f"chat_template_content_format={content_format!r} is not supported "
            "with policy.generation.dynamo=true: Dynamo's frontend renders chat "
            "templates itself and has no content-format setting. Remove it or "
            "set it to 'auto'."
        )
    if serving.get("enable_auto_tools") is False:
        raise ValueError(
            "policy.generation.vllm_cfg.http_server_serving_chat_kwargs."
            "enable_auto_tools=false is not supported with "
            "policy.generation.dynamo=true: Dynamo's frontend always parses "
            "tool_choice='auto' calls when a tool parser is set. Remove it, or "
            "remove the tool parser."
        )


def dynamo_chat_template(generation_config: Mapping[str, Any]) -> str | None:
    """The chat template the frontend renders with: a file path or literal Jinja.

    ``dynamo_cfg.worker_args.custom_jinja_template``, else the native server's
    ``http_server_serving_chat_kwargs.chat_template``; None uses the model's own.
    """
    configured = dynamo_cfg(generation_config).worker_args.custom_jinja_template
    if configured is not None:
        return configured
    serving = generation_config["vllm_cfg"].get("http_server_serving_chat_kwargs") or {}
    return serving.get("chat_template")


def dynamo_enabled(generation_config: Mapping[str, Any]) -> bool:
    """Whether ``policy.generation.dynamo`` is on for a vLLM generation config."""
    return bool(generation_config.get("dynamo"))


def dynamo_cfg(generation_config: Mapping[str, Any]) -> DynamoCfg:
    """The validated ``dynamo_cfg`` block; an absent block means all defaults."""
    return DynamoCfg.model_validate(generation_config.get("dynamo_cfg") or {})
