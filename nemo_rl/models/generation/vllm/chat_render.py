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
divergence silently trains on off-policy tokens. This module therefore runs
the engine's own steps over the same vLLM objects: the chat server's render
(``chat_server.NeMoRLOpenAIServingMixin._render_chat_call``) and its splice
(``prompt_and_capture.prepare_prompt``), built from the chat server's settings
(``chat_server.serving_chat_settings``).

Everything vLLM is imported lazily: the module is also imported by the driver,
whose environment does not carry vLLM.
"""

import copy
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = ["FrontendChatRenderer", "FrontendRender", "RenderSettings"]


@dataclass(frozen=True)
class RenderSettings:
    """What a frontend needs to render exactly like the engines' HTTP servers."""

    model_name: str
    # The part of ``policy.generation`` the engines' chat server is built
    # from: ``stop_token_ids`` and, under ``vllm_cfg``,
    # ``http_server_serving_chat_kwargs`` and the parser plugins.
    generation_cfg: dict[str, Any]
    # The prefill engines' effective ``vllm_kwargs``. ``AsyncEngineArgs`` is
    # built from them exactly as the engine does, so every key that shapes the
    # HF config or the tokenizer (``hf_overrides``, ``tokenizer``,
    # ``tokenizer_mode``, ``revision``, ``config_format``, ...) reaches the
    # renderer's ``ModelConfig``.
    engine_kwargs: dict[str, Any]
    max_model_len: int


@dataclass(frozen=True)
class FrontendRender:
    """One chat request rendered by a frontend."""

    # The engine prompt, prefix spliced in.
    token_ids: list[int]
    # Whether reasoning is over before the call generates, read from the
    # template render (``chat_server.reasoning_ended``). The engines cannot
    # answer it from the spliced ids, so the legs carry it to them.
    reasoning_ended: Optional[bool]


def _no_staged_prefix(_admission: Any) -> list[int]:
    raise RuntimeError(
        "the disaggregation frontend renders without a capture admission"
    )


class FrontendChatRenderer:
    """Renders chat requests into the exact prompt ids the engine would use.

    Built once per frontend process. CPU only: vLLM's ``ModelConfig`` and the
    HF tokenizer are loaded, no engine is. Multimodal models are refused up
    front -- the engine would run on the pixels its own processor produced,
    which a relayed id list does not carry.
    """

    def __init__(self, settings: RenderSettings) -> None:
        # Deferred: vLLM lives in the generation venv only.
        import vllm.platforms

        # A frontend actor is given no GPU; vLLM only renders here. Its
        # explicit CPU target selects the CPU platform where it finds no
        # accelerator, as Dynamo's render workers do (render_pool._import_vllm).
        if vllm.platforms.resolve_current_platform_cls_qualname().endswith(
            ".UnspecifiedPlatform"
        ):
            os.environ["VLLM_TARGET_DEVICE"] = "cpu"

        from vllm.config import CacheConfig, LoadConfig, VllmConfig
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.renderers.registry import renderer_from_config
        from vllm.utils.mistral import is_mistral_tokenizer

        from nemo_rl.models.generation.openai_server_utils import turn_end_token_ids
        from nemo_rl.models.generation.vllm import chat_server
        from nemo_rl.models.generation.vllm.utils import model_stop_token_ids

        cfg = settings.generation_cfg
        chat_server.register_parser_plugins(cfg)

        # The engine builds AsyncEngineArgs(**vllm_kwargs) (vllm_worker_async),
        # so the renderer's ModelConfig comes from the same arguments and every
        # tokenizer- and config-shaping key reaches it. compilation_config is
        # engine-only and may still be a dict here (the worker converts it).
        engine_args_kwargs: dict[str, Any] = {
            key: value
            for key, value in settings.engine_kwargs.items()
            if key != "compilation_config"
        }
        engine_args_kwargs.update(
            model=settings.model_name,
            trust_remote_code=True,
            max_model_len=settings.max_model_len,
            skip_tokenizer_init=False,
            # The engine's memory management; vLLM's CPU platform refuses it.
            enable_sleep_mode=False,
        )
        model_config = AsyncEngineArgs(**engine_args_kwargs).create_model_config()
        if model_config.supports_multimodal_inputs:
            raise ValueError(
                "disaggregation.frontend_tokenize does not support multimodal "
                f"models ({settings.model_name}): the frontend renders text only. "
                "Disable frontend_tokenize to render on the prefill engines."
            )
        renderer = renderer_from_config(
            VllmConfig(
                model_config=model_config,
                load_config=LoadConfig(load_format="dummy"),
                cache_config=CacheConfig(),
            )
        )
        tokenizer = renderer.tokenizer
        if is_mistral_tokenizer(tokenizer):
            raise ValueError(
                "disaggregation.frontend_tokenize does not support Mistral "
                f"tokenizers ({settings.model_name}): the engine's render path "
                "runs Mistral-specific request rewriting (tool-call "
                "serialization, id truncation) that the frontend does not "
                "replicate. Disable frontend_tokenize to render on the prefill "
                "engines."
            )
        serving_chat_kwargs, default_chat_template_kwargs = (
            chat_server.serving_chat_settings(cfg)
        )
        self._online_renderer = chat_server.FrontendRenderer(
            model_config=model_config,
            renderer=renderer,
            request_logger=None,
            chat_template=serving_chat_kwargs["chat_template"],
            chat_template_content_format=serving_chat_kwargs[
                "chat_template_content_format"
            ],
            enable_auto_tools=serving_chat_kwargs["enable_auto_tools"],
            tool_parser=serving_chat_kwargs.get("tool_parser"),
            reasoning_parser=serving_chat_kwargs.get("reasoning_parser"),
            default_chat_template_kwargs=default_chat_template_kwargs,
        )
        if self._online_renderer.use_harmony:
            raise ValueError(
                "disaggregation.frontend_tokenize does not support gpt-oss "
                f"(harmony) models ({settings.model_name}): the engine renders "
                "them through vLLM's harmony path, which bypasses the chat "
                "template and the prefix splice the frontend performs. Disable "
                "frontend_tokenize to render on the prefill engines."
            )
        self._model_config = model_config
        self._tokenizer = tokenizer
        # Every id a turn can end on: the splice cuts by them.
        self._turn_end_token_ids = turn_end_token_ids(
            model_stop_token_ids(model_config, cfg), tokenizer
        )
        self._settings = settings

    @property
    def model_config(self) -> Any:
        return self._model_config

    async def render(self, body: dict[str, Any]) -> FrontendRender:
        """The engine prompt for a chat-completions request body.

        The engine's two steps: the chat server's render (the full
        conversation and, when the body carries the model's own previously
        generated ids, the history up to the last assistant message) and the
        splice of the model's ids into it. The body's ``ng_capture`` is not
        resolved: the decode engine captures the call.
        """
        # Deferred: vLLM lives in the generation venv only.
        from nemo_rl.models.generation.vllm.chat_server import (
            NeMoRLChatCompletionRequest,
        )
        from nemo_rl.models.generation.vllm.prompt_and_capture import prepare_prompt

        # Every field the engine's request model declares reaches it, so
        # whatever shapes rendering there (tools, chat_template_kwargs,
        # reasoning_effort, truncation, required_prefix_token_ids, ...) shapes
        # it here.
        render_fields = {
            k: v
            for k, v in body.items()
            if k in NeMoRLChatCompletionRequest.model_fields and k != "ng_capture"
        }
        render_fields.setdefault("model", self._settings.model_name)
        # A deep copy: vLLM's renderer rewrites the conversation in place
        # (``_postprocess_messages`` turns each tool call's JSON-string
        # ``arguments`` into a dict for the chat template), and the body is
        # relayed to the prefill engine afterwards, whose request validation
        # requires the string form.
        render_fields["messages"] = copy.deepcopy(body.get("messages") or [])
        request = NeMoRLChatCompletionRequest.model_validate(render_fields)

        renderer = self._online_renderer
        # The arguments vLLM's render_chat hands preprocess_chat.
        rendered = await renderer._render_chat_call(
            request,
            request.messages,
            dict(
                default_template=renderer.chat_template,
                default_template_content_format=renderer.chat_template_content_format,
                default_template_kwargs=renderer.default_chat_template_kwargs,
                tool_dicts=(
                    None
                    if request.tools is None
                    or (
                        request.tool_choice == "none"
                        and renderer.exclude_tools_when_tool_choice_none
                    )
                    else [tool.model_dump() for tool in request.tools]
                ),
                parser=renderer.parser,
                skip_mm_cache=False,
            ),
            admission=None,
        )
        plan = await prepare_prompt(
            rendered,
            resolve_prefix=_no_staged_prefix,
            turn_end_token_ids=self._turn_end_token_ids,
            max_model_len=self._model_config.max_model_len,
            # The engine clamps the output budget to the ids it is handed.
            measure_only=True,
            tokenizer=self._tokenizer,
        )
        return FrontendRender(
            token_ids=list(plan.prompt_token_ids),
            reasoning_ended=rendered.reasoning_ended,
        )
