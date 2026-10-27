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

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, model_validator


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
    # Idle TTL in seconds for keeping one session on one engine, or None to
    # disable it. NeMo-Gym names the session on every call (its rollout id, else
    # its own session), as the native backend pins each Gym session to one
    # server; the engine splices a call's echoed or staged prefix after routing,
    # so the KV router alone cannot see where that prefix is cached.
    session_affinity_ttl_secs: PositiveInt | None = 1800
    extra_cli_args: list[str] = Field(default_factory=list)


class DynamoCfg(BaseModel, extra="forbid"):
    """``policy.generation.dynamo_cfg``: the frontend and engine registration."""

    # Bounds etcd/NATS/frontend start-up and every engine's registration.
    startup_timeout_s: PositiveFloat = 600.0
    # Timeout for one NeMo RL rollout routed through the frontend, in place of
    # NRL_VLLM_ASYNC_TIMEOUT_SECONDS. A timeout fails the attempt as a native one
    # does (GenerationUnavailable), and the rollout retry policy re-dispatches it
    # (SingleController: async_rl.rollout_failure.max_infra_attempts_per_prompt).
    # NeMo-Gym's own calls keep Gym's timeouts.
    request_timeout_s: PositiveFloat = 900.0
    worker_args: DynamoWorkerArgs = Field(default_factory=DynamoWorkerArgs)
    frontend_args: DynamoFrontendArgs = Field(default_factory=DynamoFrontendArgs)


# vLLM parser names whose Dynamo built-in parser goes by another name. nano_v3 is
# NeMo RL's vLLM plugin: DeepSeek-R1 reasoning parsing, with the whole output as
# content when chat_template_kwargs.enable_thinking is false, as Dynamo's
# nemotron_v3 parser does.
_DYNAMO_REASONING_PARSER_NAMES = {"nano_v3": "nemotron_v3"}
_DYNAMO_TOOL_PARSER_NAMES = {"openai": "harmony"}


# (kind, the dynamo_cfg.worker_args field naming it, Dynamo's renames).
_PARSER_KINDS = (
    ("tool", "tool_call_parser", _DYNAMO_TOOL_PARSER_NAMES),
    ("reasoning", "reasoning_parser", _DYNAMO_REASONING_PARSER_NAMES),
)


def dynamo_parsers(
    generation_config: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    """The ``(tool_call_parser, reasoning_parser)`` Dynamo's frontend applies.

    ``dynamo_cfg.worker_args`` names them explicitly. Otherwise they follow the
    native server's ``vllm_cfg.http_server_serving_chat_kwargs``, renamed where
    Dynamo calls the same parser differently. A name that a parser plugin
    provides has no Dynamo equivalent; the engine refuses it at registration
    (see ``plugin_parser_names``).
    """
    worker_args = dynamo_cfg(generation_config).worker_args
    serving = generation_config["vllm_cfg"].get("http_server_serving_chat_kwargs") or {}
    names = []
    for kind, worker_arg, renames in _PARSER_KINDS:
        configured = getattr(worker_args, worker_arg)
        native = serving.get(f"{kind}_parser")
        if configured is not None or native is None:
            names.append(configured)
        else:
            names.append(renames.get(native, native))
    tool_call_parser, reasoning_parser = names
    return tool_call_parser, reasoning_parser


def plugin_parser_names(
    generation_config: Mapping[str, Any],
) -> list[tuple[str, str, str, str]]:
    """``(kind, name, plugin, worker_arg)`` for each parser to check against a plugin.

    These are the native parser names Dynamo takes as they are while
    ``vllm_cfg.<kind>_parser_plugin`` is set. The plugin may provide the name,
    and then Dynamo's Rust frontend cannot run it, or the name may be a vLLM
    built-in that Dynamo also ships. Only the engine has vLLM to tell which, so
    it checks these at registration.
    """
    worker_args = dynamo_cfg(generation_config).worker_args
    vllm_cfg = generation_config["vllm_cfg"]
    serving = vllm_cfg.get("http_server_serving_chat_kwargs") or {}
    checks = []
    for kind, worker_arg, renames in _PARSER_KINDS:
        native = serving.get(f"{kind}_parser")
        plugin = vllm_cfg.get(f"{kind}_parser_plugin")
        if (
            getattr(worker_args, worker_arg) is None
            and native is not None
            and native not in renames
            and plugin
        ):
            checks.append((kind, native, plugin, worker_arg))
    return checks


def _media_items_allowed(generation_config: Mapping[str, Any]) -> int | None:
    """The most items of one modality a prompt may carry; None when unbounded.

    vLLM allows 999 of a modality ``vllm_kwargs.limit_mm_per_prompt`` leaves
    out, so an unset limit is unbounded.
    """
    limits = (generation_config.get("vllm_kwargs") or {}).get("limit_mm_per_prompt")
    if not limits:
        return None
    return max(
        limit["count"] if isinstance(limit, Mapping) else limit
        for limit in limits.values()
    )


def check_dynamo_serving_chat_kwargs(generation_config: Mapping[str, Any]) -> None:
    """Reject native chat-server settings Dynamo's frontend would not reproduce.

    The frontend renders templates and parses output itself. Its renderer keeps
    string content as strings and passes content arrays to the template, which
    for text and for one media item renders what vLLM's ``string`` and
    ``openai`` formats do. It differs only for several media items in one
    message: vLLM's ``string`` puts their placeholders first, while the template
    places them. It always parses ``tool_choice="auto"`` calls when a tool parser
    is set. And its reasoning parsers return a non-thinking turn as content,
    where vLLM's ``deepseek_r1`` returns it as reasoning.
    """
    serving = generation_config["vllm_cfg"].get("http_server_serving_chat_kwargs") or {}
    if serving.get("chat_template_content_format") == "string":
        items = _media_items_allowed(generation_config)
        if items is None or items > 1:
            raise ValueError(
                "policy.generation.vllm_cfg.http_server_serving_chat_kwargs."
                "chat_template_content_format='string' with "
                "policy.generation.dynamo=true needs "
                "policy.generation.vllm_kwargs.limit_mm_per_prompt to allow at "
                "most one item of each modality (vLLM allows 999 of a modality it "
                "leaves out). With several items in one message, vLLM's 'string' "
                "format puts their placeholders first, while Dynamo's frontend "
                "lets the chat template place them. Lower the limits, or remove "
                "the content format."
            )
    if serving.get("enable_auto_tools") is False:
        raise ValueError(
            "policy.generation.vllm_cfg.http_server_serving_chat_kwargs."
            "enable_auto_tools=false is not supported with "
            "policy.generation.dynamo=true: Dynamo's frontend always parses "
            "tool_choice='auto' calls when a tool parser is set. Remove it, or "
            "remove the tool parser."
        )
    thinking = native_default_chat_template_kwargs(generation_config).get(
        "enable_thinking"
    )
    if serving.get("reasoning_parser") == "deepseek_r1" and thinking is False:
        raise ValueError(
            "http_server_serving_chat_kwargs.reasoning_parser='deepseek_r1' with "
            "default_chat_template_kwargs.enable_thinking=false is not supported "
            "with policy.generation.dynamo=true: vLLM's deepseek_r1 parser returns "
            "a non-thinking turn as reasoning, and Dynamo's reasoning parsers "
            "return it as content. Use the nano_v3 reasoning parser, which returns "
            "it as content natively too, or enable thinking."
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


def native_default_chat_template_kwargs(
    generation_config: Mapping[str, Any],
) -> dict[str, Any]:
    """The template kwargs the native server merges under every request's own.

    ``default_chat_template_kwargs``, or the older ``chat_template_kwargs``
    spelling, of ``vllm_cfg.http_server_serving_chat_kwargs``.
    """
    serving = generation_config["vllm_cfg"].get("http_server_serving_chat_kwargs") or {}
    kwargs = serving.get("default_chat_template_kwargs")
    if kwargs is None:
        kwargs = serving.get("chat_template_kwargs")
    return dict(kwargs or {})


def dynamo_enabled(generation_config: Mapping[str, Any]) -> bool:
    """Whether ``policy.generation.dynamo`` is on for a vLLM generation config."""
    return bool(generation_config.get("dynamo"))


def dynamo_cfg(generation_config: Mapping[str, Any]) -> DynamoCfg:
    """The validated ``dynamo_cfg`` block; an absent block means all defaults."""
    return DynamoCfg.model_validate(generation_config.get("dynamo_cfg") or {})
