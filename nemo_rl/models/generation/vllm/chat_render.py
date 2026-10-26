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
"""Render a chat request into engine prompt ids outside an engine process.

The disaggregation frontend's ``frontend_tokenize`` moves the chat-template
rendering, tokenization and on-policy prefix splice off the prefill engine's
adapter thread and onto the CPU-only frontend actors. The ids it produces are
fed to the engine verbatim, so they must be *bit-identical* to what the
engine's own ``NeMoRLOnlineRenderer.preprocess_chat`` would compute: a
divergence silently trains on off-policy tokens. This module therefore builds
the same vLLM objects the engine uses (``OnlineRenderer`` over the model's
``BaseRenderer``) rather than re-implementing chat templating in HF terms, and
applies the same two-render splice (``openai_server_utils.splice_prefix_tokens``).

Everything vLLM is imported lazily: the module is also imported by the driver,
whose environment does not carry vLLM.
"""

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from nemo_rl.models.generation.openai_server_utils import splice_prefix_tokens

logger = logging.getLogger(__name__)

__all__ = ["FrontendChatRenderer", "RenderSettings", "last_assistant_prefix"]


@dataclass(frozen=True)
class RenderSettings:
    """The rendering inputs the engine's HTTP server was built with.

    Mirrors ``vllm_cfg.http_server_serving_chat_kwargs`` after the normalization
    in ``vllm_worker_async._setup_vllm_openai_api_server`` so both sides render
    the same prompt.
    """

    model_name: str
    chat_template: str | None
    chat_template_content_format: str
    default_chat_template_kwargs: dict[str, Any]
    enable_auto_tools: bool
    tool_parser: str | None
    reasoning_parser: str | None
    tool_parser_plugin: str | None
    reasoning_parser_plugin: str | None
    # ``vllm_kwargs`` keys that shape the HF config the renderer sees.
    hf_overrides: dict[str, Any]
    max_model_len: int
    config_format: str | None


def last_assistant_prefix(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The conversation up to and including the last assistant message.

    This is the slice whose template rendering is spliced against the model's
    own previously generated ids; see ``splice_prefix_tokens``. Without an
    assistant message the whole conversation is the prefix.
    """
    for i in reversed(range(len(messages))):
        if messages[i].get("role") == "assistant":
            return messages[: i + 1]
    return messages


class FrontendChatRenderer:
    """Renders chat requests into the exact prompt ids the engine would use.

    Built once per frontend process. CPU only: vLLM's ``ModelConfig`` and the
    HF tokenizer are loaded, no engine is. Multimodal models are refused up
    front -- the engine-side renderer additionally runs the multimodal
    processor, which this does not replicate.
    """

    def __init__(self, settings: RenderSettings) -> None:
        # Deferred: vLLM lives in the generation venv only.
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.multimodal import MULTIMODAL_REGISTRY
        from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
        from vllm.renderers.online_renderer import OnlineRenderer
        from vllm.renderers.registry import RENDERER_REGISTRY
        from vllm.tokenizers.registry import (
            cached_tokenizer_from_config,
            tokenizer_args_from_config,
        )
        from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

        if settings.tool_parser_plugin:
            ToolParserManager.import_tool_parser(settings.tool_parser_plugin)
        if settings.reasoning_parser_plugin:
            ReasoningParserManager.import_reasoning_parser(
                settings.reasoning_parser_plugin
            )

        engine_args_kwargs: dict[str, Any] = {
            "model": settings.model_name,
            "trust_remote_code": True,
            "hf_overrides": dict(settings.hf_overrides),
            "max_model_len": settings.max_model_len,
            "skip_tokenizer_init": False,
        }
        if settings.config_format is not None:
            engine_args_kwargs["config_format"] = settings.config_format
        model_config = AsyncEngineArgs(**engine_args_kwargs).create_model_config()
        if MULTIMODAL_REGISTRY.supports_multimodal_inputs(model_config):
            raise ValueError(
                "disaggregation.frontend_tokenize does not support multimodal "
                f"models ({settings.model_name}): the frontend renders text only. "
                "Disable frontend_tokenize to render on the prefill engines."
            )

        tokenizer = cached_tokenizer_from_config(model_config)
        renderer_mode, *_ = tokenizer_args_from_config(model_config)
        # The engine's server resolves a chat-template file path to its
        # content before handing it to the renderer; do the same.
        chat_template = settings.chat_template
        if chat_template is not None:
            from vllm.entrypoints.chat_utils import load_chat_template

            chat_template = load_chat_template(chat_template)
        # BaseRenderer wants the engine's VllmConfig but, for a text-only model,
        # reads only these attributes of it (model_config for rendering, the
        # API-process rank for logging, the prefix-caching flag for multimodal
        # uuids that never run here). A full VllmConfig would need a device,
        # which this CPU actor does not have.
        renderer_config = SimpleNamespace(
            model_config=model_config,
            parallel_config=SimpleNamespace(_api_process_rank=0, _api_process_count=1),
            observability_config=None,
            cache_config=SimpleNamespace(enable_prefix_caching=True),
        )
        base_renderer = RENDERER_REGISTRY.load_renderer(
            renderer_mode, renderer_config, tokenizer
        )
        self._model_config = model_config
        self._tokenizer = tokenizer
        self._online_renderer = OnlineRenderer(
            model_config=model_config,
            renderer=base_renderer,
            request_logger=None,
            chat_template=chat_template,
            chat_template_content_format=settings.chat_template_content_format,  # type: ignore[arg-type]
            enable_auto_tools=settings.enable_auto_tools,
            tool_parser=settings.tool_parser,
            reasoning_parser=settings.reasoning_parser,
            default_chat_template_kwargs=settings.default_chat_template_kwargs,
        )
        self._settings = settings

    @property
    def model_config(self) -> Any:
        return self._model_config

    async def _render(
        self,
        request: Any,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> list[int]:
        renderer = self._online_renderer
        req = request.model_copy(
            update={"add_generation_prompt": add_generation_prompt}
        )
        # tool_dicts the same way OpenAIServingChat derives them before calling
        # the renderer: the request's tools as plain dicts, or None.
        tool_dicts = (
            None if not req.tools else [tool.model_dump() for tool in req.tools]
        )
        _conversation, engine_inputs = await renderer.preprocess_chat(
            request=req,
            messages=messages,
            default_template=renderer.chat_template,
            default_template_content_format=renderer.chat_template_content_format,
            default_template_kwargs=renderer.default_chat_template_kwargs,
            tool_dicts=tool_dicts,
            parser=renderer.parser,
        )
        return list(engine_inputs[0]["prompt_token_ids"])

    async def prompt_token_ids(self, body: dict[str, Any]) -> list[int]:
        """The engine prompt ids for a chat-completions request body.

        Renders the full conversation; when the body carries the model's own
        previously generated ids (``required_prefix_token_ids``, or the Gym
        token fields on the last assistant message), renders the prefix up to
        that message with ``add_generation_prompt=False`` and splices the
        model's ids in -- the two-render splice the engine performs.
        """
        # Deferred: vLLM lives in the generation venv only.
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )

        # Only the fields that shape rendering reach the vLLM request model;
        # the Gym-only ones (ng_capture, required_prefix_token_ids, ...) are
        # read here and would be rejected or ignored there.
        render_fields = {
            k: v
            for k, v in body.items()
            if k
            in (
                "model",
                "messages",
                "tools",
                "tool_choice",
                "chat_template",
                "chat_template_kwargs",
                "add_generation_prompt",
                "continue_final_message",
                "add_special_tokens",
                "documents",
                "cache_salt",
            )
        }
        render_fields.setdefault("model", self._settings.model_name)
        messages = [dict(m) for m in body.get("messages") or []]
        for message in messages:
            if message.get("tool_calls"):
                message["tool_calls"] = list(message["tool_calls"])
        request = ChatCompletionRequest(**render_fields)

        full_ids = await self._render(
            request, messages, add_generation_prompt=request.add_generation_prompt
        )

        required_prefix = body.get("required_prefix_token_ids")
        if required_prefix is None:
            # The engine's request model derives the prefix from the Gym token
            # fields on the newest message that carries them.
            for message in reversed(messages):
                if "prompt_token_ids" in message:
                    required_prefix = list(message["prompt_token_ids"]) + list(
                        message["generation_token_ids"]
                    )
                    break
        if not required_prefix:
            return full_ids

        prefix_ids = await self._render(
            request, last_assistant_prefix(messages), add_generation_prompt=False
        )
        return splice_prefix_tokens(
            tokenizer=self._tokenizer,
            model_prefix_token_ids=list(required_prefix),
            template_prefix_token_ids=prefix_ids,
            template_token_ids=full_ids,
        ).token_ids
