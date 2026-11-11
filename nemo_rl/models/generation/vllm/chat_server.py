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

"""NeMo RL's OpenAI chat and tokenize server over a vLLM engine.

The vLLM async worker builds it for its HTTP server. The classes live here, not
inside the worker, so another host of the same chat path (Dynamo's frontend)
can build the same renderer and chat serving. ``host`` is the worker: it owns
token capture. This module imports vLLM's OpenAI server stack; import it only
where vLLM is installed.
"""

import asyncio
import copy
import logging
from copy import deepcopy
from dataclasses import dataclass, replace
from logging import Filter as LoggingFilter
from logging import LogRecord
from typing import Any, Literal, Optional, Union

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import PrivateAttr
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels

# vLLM 0.29 moved this out of the openai package (vllm-project/vllm#54492).
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.tokenize.protocol import (
    TokenizeChatRequest,
    TokenizeCompletionRequest,
    TokenizeResponse,
)
from vllm.entrypoints.serve.tokenize.serving import ServingTokenization
from vllm.entrypoints.serve.utils.api_utils import with_cancellation
from vllm.exceptions import VLLMValidationError
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.renderers.hf import HfRenderer
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.v1.engine.async_llm import logger as vllm_async_llm_logger

from nemo_rl.models.generation.capture_context import resolve_capture_input
from nemo_rl.models.generation.gym_call import GymCallResult, GymRequestFields
from nemo_rl.models.generation.openai_server_utils import turn_end_token_ids
from nemo_rl.models.generation.prefix_recovery.restore import (
    _full_generator_signature,
    _RestoredPrefixTerminal,
    restore_reply_text,
    restore_reply_usage,
    wrap_parser_for_restore,
)
from nemo_rl.models.generation.vllm.captured_chat_logprobs import (
    CapturedChatLogprobsMixin,
)
from nemo_rl.models.generation.vllm.prompt_and_capture import (
    abort_request_capture,
    start_capture,
    abort_capture,
    save_capture,
    prepare_prompt,
    set_max_tokens,
)
from nemo_rl.models.generation.vllm.generation_cut_route import (
    register_generation_cut_route,
)
from nemo_rl.models.generation.vllm.utils import (
    attach_routed_experts_to_chat_response_choices,
    attach_sampler_topk_logprobs_to_chat_response_choices,
    attach_token_information_to_chat_response_choices,
    model_dump_chat_response_with_dynamic_message_fields,
    model_stop_token_ids,
    on_policy_sampling,
)

LOGGER = logging.getLogger(__name__)


class NeMoRLOpenAIChatRequestMixin:
    # Declared by the request models this mixes into.
    required_prefix_token_ids: Optional[list[int]]
    messages: Any

    def model_post_init(self, context):
        # NeMo-Gym specific processing. This is just how NeMo-Gym returns the extra token information.
        if self.required_prefix_token_ids is None:
            for message in reversed(self.messages):
                if "prompt_token_ids" in message:
                    self.required_prefix_token_ids = (
                        message["prompt_token_ids"] + message["generation_token_ids"]
                    )
                    break

        return super().model_post_init(context)


# This MRO is necessary i.e. NeMoRLOpenAIChatRequestMixin > ChatCompletionRequest
class NeMoRLChatCompletionRequest(
    NeMoRLOpenAIChatRequestMixin, GymRequestFields, ChatCompletionRequest
):
    # Set when the call resumes from the tokens a checkpoint cut saved.
    _restored_prefix_terminal: _RestoredPrefixTerminal | None = PrivateAttr(
        default=None
    )
    _restored_generation_token_ids: tuple[int, ...] = PrivateAttr(default=())
    # The engine prompt of a restored call: the capture prefix, then the
    # saved rows; empty otherwise.
    _restored_prompt_token_ids: tuple[int, ...] = PrivateAttr(default=())
    # The client's min_tokens before it is reduced by a restored prefix;
    # replaying the prefix's stop handling needs the original window.
    _original_min_tokens: int | None = PrivateAttr(default=None)


class NeMoRLMeasureChatRequest(NeMoRLChatCompletionRequest):
    """A chat call Gym measures, counted as ``/tokenize`` counts it.

    vLLM's tokenize request sets no context limit, so a prompt longer than the
    context is counted instead of refused.
    """

    def build_tok_params(self, model_config: Any) -> Any:
        return replace(super().build_tok_params(model_config), max_total_tokens=None)


def chat_request_from_render(request: dict[str, Any]) -> NeMoRLChatCompletionRequest:
    """The chat request a render produced, rebuilt from its JSON without messages.

    vLLM's tool parser sets ``structured_outputs`` during the render (a forced
    tool's schema), which the request's own validators reject beside a named
    ``tool_choice``; it is set back after validation, as the render left it.
    """
    structured_outputs = request.get("structured_outputs")
    chat = NeMoRLChatCompletionRequest.model_validate(
        {
            **request,
            "structured_outputs": None,
            "messages": [{"role": "user", "content": ""}],
        }
    )
    if structured_outputs is not None:
        chat.structured_outputs = StructuredOutputsParams(
            **{k: v for k, v in structured_outputs.items() if not k.startswith("_")}
        )
    return chat


# This MRO is necessary i.e. NeMoRLOpenAIChatRequestMixin > TokenizeRequest
class NeMoRLTokenizeChatRequest(
    NeMoRLOpenAIChatRequestMixin, GymRequestFields, TokenizeChatRequest
):
    ng_measure_only: Literal[True] = True


NeMoRLTokenizeRequest = Union[TokenizeCompletionRequest, NeMoRLTokenizeChatRequest]


class NeMoRLOpenAIServingMixin:
    """The chat preprocessing NeMo RL adds: prefix splice and token capture."""

    async def _render_chat_call(
        self, request, messages, render_kwargs: dict[str, Any], *, admission
    ) -> "RenderedChat":
        """Render a chat call and decide how its capture prefix is served.

        The steps every host of the chat path shares, before any TQ read: the
        render with the checkpoint's template and processor, the capture
        decision for a Gym candidate, and, when a prefix will be
        spliced, the render of the history up to the last assistant turn that
        the splice cuts against. A chat request renders with an output budget
        of 1, so vLLM's pre-tokenization length check passes; start_capture sets the
        budget clamped to the final prompt.
        """
        request_max_tokens = None
        if isinstance(request, ChatCompletionRequest):
            request_max_tokens = (
                request.max_completion_tokens
                if request.max_completion_tokens is not None
                else request.max_tokens
            )
            if request_max_tokens is not None:
                set_max_tokens(request, 1)
        for message in messages:
            if message.get("tool_calls"):
                message["tool_calls"] = list(message["tool_calls"])
        messages_for_replace_prefix_tokens = deepcopy(messages)

        try:
            conversation, (engine_prompt,) = await super().preprocess_chat(
                request=request, messages=messages, **render_kwargs
            )
        except (ValueError, VLLMValidationError) as e:
            if "maximum context length" in str(e):
                LOGGER.warning("Prompt exceeds max_model_len: %s", e)
            raise

        # Token capture: the admission is built once by the host and resolved
        # here, before branching; the ``ng_capture`` dict is never mutated.
        serving_admission = admission
        verify_retained_media = False
        if admission is not None and (
            admission.mode == "candidate" or admission.request_replay is not None
        ):
            # Inline assistant token fields cannot override RL's decision.
            request.required_prefix_token_ids = None
            decision = resolve_capture_input(
                admission,
                messages=messages_for_replace_prefix_tokens,
                conversation=conversation,
                renderer_is_hf=type(self.renderer) is HfRenderer,
            )
            admission = decision.storage
            serving_admission = decision.serving
            verify_retained_media = decision.verify_retained_media

        history_token_ids = None
        if getattr(request, "required_prefix_token_ids", None) is not None or (
            serving_admission is not None and serving_admission.mode == "token_in"
        ):
            # The splice cuts the full render at the turn this history ends.
            last_assistant_message_idx = None
            for i in reversed(range(len(messages_for_replace_prefix_tokens))):
                if messages_for_replace_prefix_tokens[i]["role"] == "assistant":
                    last_assistant_message_idx = i
                    break
            if last_assistant_message_idx is None:
                messages_to_last_assistant_message = messages_for_replace_prefix_tokens
            else:
                messages_to_last_assistant_message = messages_for_replace_prefix_tokens[
                    : last_assistant_message_idx + 1
                ]
            modified_request = request.model_copy(
                update={"add_generation_prompt": False}
            )
            _, (history_prompt,) = await super().preprocess_chat(
                request=modified_request,
                messages=messages_to_last_assistant_message,
                **render_kwargs,
            )
            history_token_ids = history_prompt["prompt_token_ids"]

        return RenderedChat(
            template_token_ids=list(engine_prompt["prompt_token_ids"]),
            history_token_ids=history_token_ids,
            required_prefix_token_ids=getattr(
                request, "required_prefix_token_ids", None
            ),
            admission=admission,
            serving_admission=serving_admission,
            verify_retained_media=verify_retained_media,
            request_max_tokens=request_max_tokens,
            conversation=conversation,
            engine_prompt=engine_prompt,
        )

    # vLLM 0.25 moved chat preprocessing to
    # OnlineRenderer.preprocess_chat (tool_parser/reasoning_parser were
    # folded into a single `parser`), so this override now applies via
    # the renderer subclass.
    async def preprocess_chat(
        self,
        request,
        messages,
        default_template,
        default_template_content_format,
        default_template_kwargs,
        tool_dicts=None,
        parser=None,
        *,
        skip_mm_cache: bool = False,
    ):
        host = self._host
        admission = host._capture_admission(request)
        # #4124: processor-only cache reads retain concrete pixels even
        # when the engine's sender cache would return references.
        if host._capture_media and admission is not None:
            skip_mm_cache = True

        rendered = await self._render_chat_call(
            request,
            messages,
            dict(
                default_template=default_template,
                default_template_content_format=default_template_content_format,
                default_template_kwargs=default_template_kwargs,
                tool_dicts=tool_dicts,
                parser=parser,
                skip_mm_cache=skip_mm_cache,
            ),
            admission=admission,
        )
        plan = await prepare_prompt(
            rendered,
            resolve_prefix=host._resolve_admission_prefix,
            turn_end_token_ids=turn_end_token_ids(
                self._model_stop_token_ids, self.renderer.tokenizer
            ),
            max_model_len=self.model_config.max_model_len,
            # /tokenize counts a prompt of any length, so only chat calls clamp.
            clamp=isinstance(request, NeMoRLChatCompletionRequest),
            tokenizer=self.renderer.tokenizer,
        )
        await start_capture(
            host,
            request,
            rendered.engine_prompt,
            plan,
            max_model_len=self.model_config.max_model_len,
        )
        return rendered.conversation, [rendered.engine_prompt]


@dataclass(frozen=True)
class RenderedChat:
    """One chat call rendered for the engine, before its prefix is spliced in.

    A :class:`~nemo_rl.models.generation.vllm.prompt_and_capture.RenderedPrompt`.
    """

    template_token_ids: list[int]
    # The render up to the last assistant turn, when a prefix will be spliced.
    history_token_ids: Optional[list[int]]
    # The request's own prefix (Gym's echo) after the capture decision.
    required_prefix_token_ids: Optional[list[int]]
    # What capture stores, after the candidate decision; None without capture.
    admission: Any
    # How the prompt is served: "token_in" reads the prefix from TQ.
    serving_admission: Any
    verify_retained_media: bool
    # The output budget the request asked for, before the render lowered it.
    request_max_tokens: Optional[int]
    conversation: list[Any]
    # The full render, multimodal inputs included; its ids are the template's.
    engine_prompt: dict[str, Any]


class NeMoRLOnlineRenderer(NeMoRLOpenAIServingMixin, OnlineRenderer):
    def __init__(
        self, *args, host: Any, model_stop_token_ids: frozenset[int], **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._host = host
        # vLLM adds the generation config's EOS to every request's stop set
        # regardless of the configured stops, so a turn can end on either.
        self._model_stop_token_ids = model_stop_token_ids


class FrontendRenderer(NeMoRLOpenAIServingMixin, OnlineRenderer):
    """The render step for NeMo RL's chat engine in Dynamo's frontend.

    Runs only the shared step (:meth:`_render_chat_call`): the frontend reads
    the TQ prefix and splices, and the engine captures. Media is always
    processed here, never served from a processor cache, since the engine runs
    on the pixels the frontend sends.
    """

    # The render the overridden preprocess_chat is asked for, and its result.
    _admission: Any = None
    _rendered: Any = None

    async def render_for_frontend(self, request, *, admission) -> "RenderedChat":
        """Render ``request``; the caller restores its lowered output budget."""
        if self.use_harmony:
            # vLLM renders gpt-oss's Harmony format without preprocess_chat, so
            # NeMo RL's render, capture decision and splice would never run.
            raise ValueError(
                "gpt-oss (Harmony) chat calls are not served through NeMo RL's "
                "Dynamo chat engine; use policy.generation.dynamo=false"
            )
        self._admission = admission
        result = await self.render_chat(request, skip_mm_cache=True)
        if not isinstance(result, tuple):
            raise ValueError(result.error.message)
        return self._rendered

    async def preprocess_chat(
        self,
        request,
        messages,
        default_template,
        default_template_content_format,
        default_template_kwargs,
        tool_dicts=None,
        parser=None,
        *,
        skip_mm_cache: bool = False,
    ):
        self._rendered = await self._render_chat_call(
            request,
            messages,
            dict(
                default_template=default_template,
                default_template_content_format=default_template_content_format,
                default_template_kwargs=default_template_kwargs,
                tool_dicts=tool_dicts,
                parser=parser,
                skip_mm_cache=True,
            ),
            admission=self._admission,
        )
        return self._rendered.conversation, [self._rendered.engine_prompt]


_FULL_GENERATOR_SIGNATURE = _full_generator_signature(
    OpenAIServingChat.chat_completion_full_generator
)


class ResumedCallReplyMixin:
    """vLLM's reply to a call resumed from a checkpoint cut, as the whole call's.

    The engine ran the restored tokens as prompt and generated only the rest.
    Before vLLM parses the reply, the restored text goes in front of the final
    output and the parser sees the restored tokens first; after, usage counts
    them as completion. It reads the request's restore fields, which
    ``resolve_restored_call`` sets on the worker and the render pool sets from
    the engine's reply. The native serving chat and the render pool's both use
    it.
    """

    async def chat_completion_full_generator(
        self,
        request,
        result_generator,
        *args,
        **kwargs,
    ):
        restored_token_ids = request._restored_generation_token_ids

        async def restored_results():
            final_res = None
            async for res in result_generator:
                final_res = res
            if final_res is None:
                return
            # Before vLLM parses the reply, so tool calls and reasoning are
            # parsed from the whole call's text.
            restore_reply_text(
                request,
                final_res,
                prefix_token_ids=restored_token_ids,
                engine_prompt_token_ids=request._restored_prompt_token_ids,
                tokenizer=self.renderer.tokenizer,
            )
            yield final_res

        if request._restored_prompt_token_ids:
            result_generator = restored_results()
        # Bound by name: vLLM passes request_metadata by position and
        # parser by keyword, and either may move.
        bound = _FULL_GENERATOR_SIGNATURE.bind_partial(
            self, request, result_generator, *args, **kwargs
        )
        if bound.arguments.get("parser") is not None:
            bound.arguments["parser"] = wrap_parser_for_restore(
                bound.arguments["parser"], restored_token_ids
            )
        response = await super().chat_completion_full_generator(
            *bound.args[1:], **bound.kwargs
        )
        if not isinstance(response, ChatCompletionResponse):
            return response
        terminal = request._restored_prefix_terminal
        if not isinstance(terminal, _RestoredPrefixTerminal):
            terminal = None
        if restore_reply_usage(
            response,
            restored_token_count=len(restored_token_ids),
            terminal=terminal,
        ):
            request_metadata = bound.arguments.get("request_metadata")
            if request_metadata is not None:
                request_metadata.final_usage_info = response.usage
        return response


class NeMoRLOpenAIServingChatMixin:
    async def chat_completion_full_generator(
        self,
        request,
        result_generator,
        *args,
        **kwargs,
    ):
        host = self._host
        return_as_token_id = (
            request.return_tokens_as_token_ids
            if request.return_tokens_as_token_ids is not None
            else self.return_tokens_as_token_ids
        )

        final_res = None
        capture_routes = request in host._capture_registry
        cancelled = False

        def _without_routes(res):
            # vLLM encodes choice-level routes in its response. Keep
            # engine arrays for our native capture, and hide them only
            # in the superclass's shallow view.
            res = copy.copy(res)
            res.outputs = [copy.copy(output) for output in res.outputs]
            for output in res.outputs:
                output.routed_experts = None
            return res

        async def capture_result_generator():
            nonlocal final_res, cancelled
            try:
                async for res in result_generator:
                    # Recorded before asking for the next output, which
                    # releases vLLM's finished state for this request.
                    host._prefix_recovery.observe(request, res)
                    final_res = res
                if final_res is None:
                    return
                yield (_without_routes(final_res) if capture_routes else final_res)
            except asyncio.CancelledError:
                cancelled = True
                raise

        # Capture stages the raw arrays below and strips OpenAI logprobs
        # after the terminal PUT. Skip only their object construction;
        # changing request.logprobs would also change engine sampling.
        bypass_logprobs = bool(
            not request.stream
            and request.logprobs
            and return_as_token_id
            and request in host._capture_registry
        )
        with self._bypass_chat_logprobs(enabled=bypass_logprobs):
            response = await super().chat_completion_full_generator(
                request, capture_result_generator(), *args, **kwargs
            )
        if cancelled:
            # vLLM's chat_completion_full_generator catches the
            # CancelledError that with_cancellation throws into the
            # handler task on a client disconnect and returns
            # ErrorResponse("Client disconnected") instead of
            # re-raising. Nobody reads that response (the wrapper has
            # already returned), and letting it through would make the
            # route label the capture "error_response". Re-raise so the
            # task finishes cancelled and the route sees the disconnect.
            raise asyncio.CancelledError()
        if not isinstance(response, ChatCompletionResponse) or final_res is None:
            return response

        if request.logprobs and return_as_token_id:
            response = attach_token_information_to_chat_response_choices(
                response,
                final_res,
            )
            if request.ng_capture and host._capture_top_logprobs:
                # Extraction scales with completion length and K.
                response = await asyncio.to_thread(
                    attach_sampler_topk_logprobs_to_chat_response_choices,
                    response,
                    final_res,
                    top_k=host._capture_top_logprobs,
                )

        if host._return_routed_experts_enabled():
            response = attach_routed_experts_to_chat_response_choices(
                response,
                final_res,
                device=torch.device("cpu"),
                logger=LOGGER,
                routed_experts_dtype=host.routed_experts_dtype,
                encode_for_wire=not capture_routes,
            )

        return response


class NeMoRLOpenAIServingChat(
    NeMoRLOpenAIServingChatMixin,
    CapturedChatLogprobsMixin,
    ResumedCallReplyMixin,
    OpenAIServingChat,
):
    def __init__(self, *args, host: Any, **kwargs):
        super().__init__(*args, **kwargs)
        self._host = host


class FrontendServingChat(ResumedCallReplyMixin, OpenAIServingChat):
    """The reply step for NeMo RL's chat engine in Dynamo's frontend.

    Builds the reply from the engine's output and never generates.
    """


# Tokenize path delegates to OnlineRenderer.preprocess_chat,
# where the prefix-token override lives.
class NeMoRLServingTokenization(ServingTokenization):
    pass


@dataclass(frozen=True)
class ChatServer:
    """The serving objects behind /v1/chat/completions and /tokenize."""

    online_renderer: NeMoRLOnlineRenderer
    serving_chat: NeMoRLOpenAIServingChat
    serving_tokenization: NeMoRLServingTokenization


def register_parser_plugins(cfg: dict[str, Any]) -> None:
    """Import the tool and reasoning parser plugins ``policy.generation`` names."""
    maybe_tool_parser_plugin = cfg["vllm_cfg"].get("tool_parser_plugin")
    if maybe_tool_parser_plugin:
        ToolParserManager.import_tool_parser(maybe_tool_parser_plugin)

    maybe_reasoning_parser_plugin = cfg["vllm_cfg"].get("reasoning_parser_plugin")
    if maybe_reasoning_parser_plugin:
        ReasoningParserManager.import_reasoning_parser(maybe_reasoning_parser_plugin)


def serving_chat_settings(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The chat serving's settings and default chat-template kwargs.

    One copy for every host of the chat path, so each renders and parses
    alike. Returns the ``OpenAIServingChat`` kwargs (template loaded,
    ``default_chat_template_kwargs`` normalized) and those template kwargs.
    """
    serving_chat_default_kwargs: dict[str, Any] = {
        "response_role": "assistant",
        "request_logger": None,
        "chat_template": None,
        "chat_template_content_format": "auto",
        "enable_auto_tools": True,
    }
    serving_chat_kwargs = serving_chat_default_kwargs | cfg["vllm_cfg"].get(
        "http_server_serving_chat_kwargs", dict()
    )
    # The embedded server is constructed directly instead of through
    # vLLM's CLI, where chat-template file paths are normally loaded.
    # OnlineRenderer expects literal Jinja content; passing a path makes
    # Transformers render the path itself and drops multimodal
    # placeholders such as <image>.
    configured_chat_template = serving_chat_kwargs.get("chat_template")
    if configured_chat_template is not None:
        serving_chat_kwargs["chat_template"] = load_chat_template(
            configured_chat_template
        )
    # Recipes may name the parameter either way: ``default_chat_template_kwargs``
    # is vLLM's own spelling, ``chat_template_kwargs`` is accepted for recipes
    # written against the older name. Normalize onto the native key rather
    # than popping it: OnlineRenderer, OpenAIServingChat and ServingTokenization
    # each keep their *own* copy and read it independently -- the chat serving
    # builds its reasoning parser from it, and the tokenize path passes its own
    # into preprocess_chat -- so the renderer's copy does not reach either.
    # vLLM's api_server hands the same value to all three for that reason.
    #
    # Popped separately, not `A or B`: short-circuiting on a truthy A would
    # leave B in the bag and OpenAIServingChat(**kwargs) would reject it.
    _legacy_chat_template_kwargs = serving_chat_kwargs.pop("chat_template_kwargs", None)
    if serving_chat_kwargs.get("default_chat_template_kwargs") is None:
        serving_chat_kwargs["default_chat_template_kwargs"] = (
            _legacy_chat_template_kwargs
        )
    default_chat_template_kwargs: dict[str, Any] = (
        serving_chat_kwargs["default_chat_template_kwargs"] or {}
    )
    return serving_chat_kwargs, default_chat_template_kwargs


def build_chat_server(
    host: Any, *, engine_client: Any, model_config: Any, cfg: dict[str, Any]
) -> ChatServer:
    """Build the chat serving the worker's HTTP server answers with.

    Args:
        host: The vLLM worker; it owns token capture.
        engine_client: The engine the chat serving generates with.
        model_config: The engine's vLLM ``ModelConfig``.
        cfg: ``policy.generation``.
    """
    register_parser_plugins(cfg)

    base_model_paths = [
        BaseModelPath(
            name=model_config.served_model_name, model_path=model_config.model
        ),
        BaseModelPath(name=model_config.model, model_path=model_config.model),
    ]

    openai_serving_models_kwargs = dict(
        engine_client=engine_client,
        base_model_paths=base_model_paths,
        lora_modules=None,
    )
    openai_serving_models = OpenAIServingModels(**openai_serving_models_kwargs)

    serving_chat_kwargs, default_chat_template_kwargs = serving_chat_settings(cfg)
    online_renderer = NeMoRLOnlineRenderer(
        model_config=engine_client.model_config,
        renderer=engine_client.renderer,
        request_logger=serving_chat_kwargs["request_logger"],
        chat_template=serving_chat_kwargs["chat_template"],
        chat_template_content_format=serving_chat_kwargs[
            "chat_template_content_format"
        ],
        enable_auto_tools=serving_chat_kwargs["enable_auto_tools"],
        # Keep the renderer's parser consistent with any parser overrides
        # passed to OpenAIServingChat via http_server_serving_chat_kwargs.
        tool_parser=serving_chat_kwargs.get("tool_parser"),
        reasoning_parser=serving_chat_kwargs.get("reasoning_parser"),
        # vLLM merges these into every render, with request-supplied keys
        # winning (preprocess_chat's default_template_kwargs). The renderer
        # is shared by /v1/chat/completions and /tokenize, so setting it
        # here keeps the two endpoints rendering identical prompts.
        default_chat_template_kwargs=default_chat_template_kwargs,
        host=host,
        model_stop_token_ids=model_stop_token_ids(model_config, cfg),
    )
    serving_chat_kwargs.update(
        dict(
            engine_client=engine_client,
            models=openai_serving_models,
            online_renderer=online_renderer,
            return_tokens_as_token_ids=True,
        )
    )
    openai_serving_chat = NeMoRLOpenAIServingChat(**serving_chat_kwargs, host=host)

    serving_tokenization_kwargs = dict(
        request_logger=serving_chat_kwargs["request_logger"],
        chat_template=serving_chat_kwargs["chat_template"],
        chat_template_content_format=serving_chat_kwargs[
            "chat_template_content_format"
        ],
        models=serving_chat_kwargs["models"],
        online_renderer=online_renderer,
        # ServingTokenization reads its own copy in preprocess_chat rather
        # than the renderer's, so /tokenize would otherwise render with {}
        # and diverge from /v1/chat/completions under multi-turn.
        default_chat_template_kwargs=default_chat_template_kwargs,
    )
    openai_serving_tokenization = NeMoRLServingTokenization(
        **serving_tokenization_kwargs
    )
    return ChatServer(online_renderer, openai_serving_chat, openai_serving_tokenization)


def register_routes(
    app: FastAPI, server: ChatServer, *, host: Any, generation_config: dict[str, Any]
) -> None:
    """Add /v1/chat/completions and /tokenize to ``app``."""
    openai_serving_chat = server.serving_chat
    openai_serving_tokenization = server.serving_tokenization

    class HostedChatCompletionRequest(NeMoRLChatCompletionRequest):
        def to_sampling_params(self, *args, **kwargs):
            sampling_params = super().to_sampling_params(*args, **kwargs)
            if self.ng_capture is not None:
                host._prefix_recovery.on_sampling_params(self, sampling_params)
            return sampling_params

    register_generation_cut_route(app, lambda: host._prefix_recovery)

    # The create_chat_completion and tokenize methods are taken from vllm/entrypoints/openai/api_server.py
    @app.post("/v1/chat/completions")
    # Like vLLM's own routes: cancel the handler (and with it the engine
    # request) when the client disconnects, instead of generating for nobody.
    @with_cancellation
    async def create_chat_completion(
        request: HostedChatCompletionRequest, raw_request: Request
    ):
        # Accept train and validation profiles; reject off-policy requests.
        # vLLM fills an unset top_k/min_p/repetition_penalty from
        # generation_config.json when vllm_kwargs sets generation_config="auto",
        # so serve the matched profile's top_k (as
        # vllm_worker.py::BaseVllmGenerationWorker::_build_sampling_params does)
        # with min_p and the repetition penalty off; the trainer replays none of
        # them, nor the penalties or logit bias a request may therefore not set.
        try:
            sampling = on_policy_sampling(
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                min_p=request.min_p,
                repetition_penalty=request.repetition_penalty,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                logit_bias=request.logit_bias,
                n=request.n,
                logprobs=request.logprobs,
                # vLLM defaults an unset top_logprobs to 0; Gym must set it.
                top_logprobs=(
                    request.top_logprobs
                    if "top_logprobs" in request.model_fields_set
                    else None
                ),
                generation_config=generation_config,
            )
            for field, value in sampling.items():
                setattr(request, field, value)
            generator = await openai_serving_chat.create_chat_completion(
                request, raw_request
            )
        except BaseException as error:
            # A cancel before generation starts propagates here directly; one
            # that lands mid-generation is swallowed by vLLM and re-raised by
            # NeMoRLOpenAIServingChatMixin.chat_completion_full_generator.
            failure = await abort_capture(host, request, error)
            if failure is None:
                raise
            return JSONResponse(
                content={
                    "error": {
                        "message": failure.message,
                        "type": failure.error_type,
                        "param": failure.param,
                        "code": failure.code,
                    }
                },
                status_code=failure.status,
                # Sent after abort_capture drained the capture state.
                headers=(
                    host._context_refusal_headers(request)
                    if failure.acknowledge
                    else None
                ),
            )

        if isinstance(generator, ErrorResponse):
            await abort_request_capture(host, request, "error_response")
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.error.code
            )

        elif isinstance(generator, ChatCompletionResponse):
            content = model_dump_chat_response_with_dynamic_message_fields(generator)
            # Token capture: stage the delta and ride the coords on the
            # response; strips logprobs/ids (no-op when capture is off).
            content = await save_capture(host, request, content)
            return JSONResponse(content=content)

        await abort_request_capture(host, request, "streaming_response")
        return StreamingResponse(content=generator, media_type="text/event-stream")

    @app.post("/tokenize")
    async def tokenize(request: NeMoRLTokenizeRequest, raw_request: Request):
        generator = await openai_serving_tokenization.create_tokenize(
            request, raw_request
        )

        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.error.code
            )
        elif isinstance(generator, TokenizeResponse):
            if (
                isinstance(request, NeMoRLTokenizeChatRequest)
                and request.ng_capture is not None
            ):
                return JSONResponse(
                    content=GymCallResult(
                        prompt_token_count=generator.count, ng_context_measured=True
                    ).model_dump(exclude_none=True)
                )
            return JSONResponse(content=generator.model_dump())


class _CleanLoggingFilter(LoggingFilter):
    def filter(self, record: LogRecord) -> bool:
        msg = record.getMessage()

        # vLLM does not accept `strict` tool definitions and reporting it to the user is not useful either.
        return (
            "Added request" not in msg
            and "The following fields were present in the request but ignored: {'strict'}"
            not in msg
        )


# Suppress the noisy vLLM traceback when a prompt exceeds max_model_len.
# This is expected during multi-turn rollouts; we log a clean one-line
# warning from preprocess_chat instead.
class _MaxContextLengthFilter(LoggingFilter):
    def filter(self, record: LogRecord) -> bool:
        if record.exc_info and record.exc_info[1]:
            if "maximum context length" in str(record.exc_info[1]):
                return False
        return True


def install_log_filters() -> None:
    """Keep vLLM's per-request and context-overflow noise out of the logs."""
    print(
        "Adding a vLLM logging filter so that the logs aren't spammed with not useful messages like `Added request ...`. This is to help errors pop up better and filter out noise."
    )
    vllm_async_llm_logger.addFilter(_CleanLoggingFilter())
    logging.getLogger("vllm.entrypoints.openai.engine.protocol").addFilter(
        _CleanLoggingFilter()
    )
    logging.getLogger("vllm.entrypoints.openai.chat_completion.serving").addFilter(
        _MaxContextLengthFilter()
    )
