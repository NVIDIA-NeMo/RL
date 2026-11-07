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
from dataclasses import dataclass
from logging import Filter as LoggingFilter
from logging import LogRecord
from typing import Any, List, Literal, Optional, Union

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
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
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.v1.engine.async_llm import logger as vllm_async_llm_logger

from nemo_rl.data.captured_media import MediaCaptureRejected
from nemo_rl.models.generation.capture_context import resolve_capture_input
from nemo_rl.models.generation.openai_server_utils import (
    model_stop_token_ids,
    splice_prefix_tokens,
    turn_end_token_ids,
)
from nemo_rl.models.generation.vllm.captured_chat_logprobs import (
    CapturedChatLogprobsMixin,
)
from nemo_rl.models.generation.vllm.utils import (
    attach_routed_experts_to_chat_response_choices,
    attach_sampler_topk_logprobs_to_chat_response_choices,
    attach_token_information_to_chat_response_choices,
    model_dump_chat_response_with_dynamic_message_fields,
    on_policy_sampling,
    remap_multimodal_placeholders,
)

LOGGER = logging.getLogger(__name__)


class NeMoRLOpenAIChatRequestMixin:
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
class NeMoRLChatCompletionRequest(NeMoRLOpenAIChatRequestMixin, ChatCompletionRequest):
    required_prefix_token_ids: Optional[List[int]] = None
    # Ledger-authoritative token capture: the call identity the ledger
    # attaches (capture_key, call_id, parent_call_id, prev_len, mode).
    ng_capture: Optional[dict[str, Any]] = None


# This MRO is necessary i.e. NeMoRLOpenAIChatRequestMixin > TokenizeRequest
class NeMoRLTokenizeChatRequest(NeMoRLOpenAIChatRequestMixin, TokenizeChatRequest):
    required_prefix_token_ids: Optional[List[int]] = None
    ng_capture: Optional[dict[str, Any]] = None
    ng_measure_only: Literal[True] = True


NeMoRLTokenizeRequest = Union[TokenizeCompletionRequest, NeMoRLTokenizeChatRequest]


class NeMoRLOpenAIServingMixin:
    """The chat preprocessing NeMo RL adds: prefix splice and token capture."""

    @staticmethod
    def _set_max_tokens(request, max_tokens: int) -> None:
        """Set the request's max output tokens.

        Mutates the request in place. Handles both max_completion_tokens (newer OpenAI API)
        and max_tokens (deprecated but still supported by vLLM).
        """
        if request.max_completion_tokens is not None:
            request.max_completion_tokens = max_tokens
        elif request.max_tokens is not None:
            request.max_tokens = max_tokens

    def _clamp_max_tokens(
        self, request, request_max_tokens: int, prompt_token_ids: list[int]
    ) -> None:
        """Clamp the request's max output tokens so that input + output <= max_model_len."""
        remaining = self.model_config.max_model_len - len(prompt_token_ids)
        if remaining <= 0:
            # preserve the literal "context length" in this message to match Gym's overflow handling
            message = (
                f"Prompt length ({len(prompt_token_ids)}) fills or exceeds "
                f"this model's maximum context length ({self.model_config.max_model_len}). "
                f"No room for output tokens."
            )
            LOGGER.warning("Prompt exceeds max_model_len: %s", message)
            raise VLLMValidationError(
                message,
                parameter="input_tokens",
                value=len(prompt_token_ids),
            )
        max_tokens = min(request_max_tokens, remaining)
        self._set_max_tokens(request, max_tokens)

    async def _render_chat_call(
        self, request, messages, render_kwargs: dict[str, Any], *, admission
    ) -> "RenderedChat":
        """Render a chat call and decide how its capture prefix is served.

        The steps every host of the chat path shares, before any TQ read: the
        render with the checkpoint's template and processor, the capture
        decision for a Gym candidate (C26 on #4979), and, when a prefix will be
        spliced, the render of the history up to the last assistant turn that
        the splice cuts against.
        """
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
            conversation=conversation,
            engine_prompt=engine_prompt,
            admission=admission,
            serving_admission=serving_admission,
            verify_retained_media=verify_retained_media,
            history_token_ids=history_token_ids,
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

        # Temporarily set to 1 so vLLM's pre-tokenization length check passes;
        # the actual value will be set through _clamp_max_tokens later.
        actual_request_max_tokens = None
        if isinstance(request, NeMoRLChatCompletionRequest):
            actual_request_max_tokens = (
                request.max_completion_tokens
                if request.max_completion_tokens is not None
                else request.max_tokens
            )
            # If max_completion_tokens or max_tokens is not set, we don't need to do _clamp_max_tokens.
            # So we don't need to set the request's max output tokens to 1 here.
            if actual_request_max_tokens is not None:
                self._set_max_tokens(request, 1)

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
        res = (rendered.conversation, [rendered.engine_prompt])
        engine_prompt = rendered.engine_prompt
        admission = rendered.admission
        serving_admission = rendered.serving_admission
        verify_retained_media = rendered.verify_retained_media

        # Resolve the admission's prefix (staging_chain -> cached TQ read).
        # Off-loop: the chain fetch is a blocking TQ read, and Gym's staging
        # protocol requires the serving host to move blocking staging I/O off
        # its event loop explicitly. The adapter then attaches the prefix to
        # the request, so the inline-prefix branch below is the single splice
        # path for staged and inline prefixes.
        capture_prefix_token_ids: list[int] | None = None
        if serving_admission is not None and serving_admission.mode == "token_in":
            capture_prefix_token_ids = await asyncio.to_thread(
                host._resolve_admission_prefix, serving_admission
            )
            host._enter_request_prefix(request, capture_prefix_token_ids)

        if (
            not hasattr(request, "required_prefix_token_ids")
            or request.required_prefix_token_ids is None
        ):
            # Clamp the request's max output tokens so that input + output <= max_model_len.
            if actual_request_max_tokens is not None:
                self._clamp_max_tokens(
                    request,
                    actual_request_max_tokens,
                    engine_prompt["prompt_token_ids"],
                )
            # Token capture, text mode: the full render is the exact
            # engine prompt.
            media = await asyncio.to_thread(
                host._capture_request_media,
                engine_prompt,
                admission=admission,
                verify_retained_media=verify_retained_media,
            )
            host._begin_request_capture(
                request,
                engine_prompt["prompt_token_ids"],
                admission=admission,
                media=media,
            )
            return res

        splice = splice_prefix_tokens(
            tokenizer=self.renderer.tokenizer,
            model_prefix_token_ids=list(request.required_prefix_token_ids),
            template_prefix_token_ids=rendered.history_token_ids,
            template_token_ids=engine_prompt["prompt_token_ids"],
            eos_token_id=turn_end_token_ids(
                self._model_stop_token_ids, self.renderer.tokenizer
            ),
        )
        final_prompt_token_ids = splice.token_ids
        # Reject overflows before media validation can mask the length error.
        if actual_request_max_tokens is not None:
            self._clamp_max_tokens(
                request,
                actual_request_max_tokens,
                final_prompt_token_ids,
            )

        media = await asyncio.to_thread(
            host._capture_request_media,
            engine_prompt,
            admission=serving_admission,
            splice=splice,
            verify_retained_media=verify_retained_media,
        )
        # Must precede the reassignment below: the remap reads the
        # current value of prompt_token_ids as the template-coordinate
        # sequence the placeholders were computed against. Reordering
        # these two statements silently degrades to a no-op.
        # Successful media capture already remapped these ranges.
        if media is None and (mm_placeholders := engine_prompt.get("mm_placeholders")):
            engine_prompt["mm_placeholders"] = remap_multimodal_placeholders(
                template_token_ids=engine_prompt["prompt_token_ids"],
                final_token_ids=final_prompt_token_ids,
                mm_placeholders=mm_placeholders,
                splice=splice,
            )

        engine_prompt["prompt_token_ids"] = final_prompt_token_ids

        # Token capture, token-in mode: the spliced prompt is the
        # exact engine prompt; begin_call re-checks the prefix it
        # was spliced from against the admission.
        host._begin_request_capture(
            request,
            final_prompt_token_ids,
            admission=admission,
            prefix_token_ids=(
                capture_prefix_token_ids
                if admission is not None and admission.mode == "token_in"
                else None
            ),
            media=media,
        )

        return res


@dataclass(frozen=True)
class RenderedChat:
    """One chat call rendered for the engine, before its prefix is spliced in."""

    conversation: list[Any]
    # The full render, multimodal inputs included; its ids are the template's.
    engine_prompt: dict[str, Any]
    # What capture stores, after the candidate decision; None without capture.
    admission: Any
    # How the prompt is served: "token_in" reads the prefix from TQ.
    serving_admission: Any
    verify_retained_media: bool
    # The render up to the last assistant turn, when a prefix will be spliced.
    history_token_ids: Optional[list[int]]


class NeMoRLOnlineRenderer(NeMoRLOpenAIServingMixin, OnlineRenderer):
    def __init__(
        self, *args, host: Any, model_stop_token_ids: frozenset[int], **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._host = host
        # vLLM adds the generation config's EOS to every request's stop set
        # regardless of the configured stops, so a turn can end on either.
        self._model_stop_token_ids = model_stop_token_ids


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
        if request.logprobs and return_as_token_id and request.top_logprobs is None:
            raise VLLMValidationError(
                "`top_logprobs` must be set when requesting token "
                "information from the NeMo-RL chat endpoint.",
                parameter="top_logprobs",
            )

        final_res = None
        capture_routes = id(request) in host._capture_calls
        cancelled = False

        async def capture_result_generator():
            nonlocal final_res, cancelled
            try:
                async for res in result_generator:
                    final_res = res
                    if capture_routes:
                        # vLLM encodes choice-level routes in its response.
                        # Keep engine arrays for our native capture, and
                        # hide them only in the superclass's shallow view.
                        res = copy.copy(res)
                        res.outputs = [copy.copy(output) for output in res.outputs]
                        for output in res.outputs:
                            output.routed_experts = None
                    yield res
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
            and id(request) in host._capture_calls
        )
        with self._bypass_chat_logprobs(enabled=bypass_logprobs):
            response = await super().chat_completion_full_generator(
                request,
                capture_result_generator(),
                *args,
                **kwargs,
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
    OpenAIServingChat,
):
    def __init__(self, *args, host: Any, **kwargs):
        super().__init__(*args, **kwargs)
        self._host = host


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
    maybe_tool_parser_plugin = cfg["vllm_cfg"].get("tool_parser_plugin")
    if maybe_tool_parser_plugin:
        ToolParserManager.import_tool_parser(maybe_tool_parser_plugin)

    maybe_reasoning_parser_plugin = cfg["vllm_cfg"].get("reasoning_parser_plugin")
    if maybe_reasoning_parser_plugin:
        ReasoningParserManager.import_reasoning_parser(maybe_reasoning_parser_plugin)

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

    serving_chat_default_kwargs = dict(
        response_role="assistant",
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        enable_auto_tools=True,
    )
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

    # The create_chat_completion and tokenize methods are taken from vllm/entrypoints/openai/api_server.py
    @app.post("/v1/chat/completions")
    # Like vLLM's own routes: cancel the handler (and with it the engine
    # request) when the client disconnects, instead of generating for nobody.
    @with_cancellation
    async def create_chat_completion(
        request: NeMoRLChatCompletionRequest, raw_request: Request
    ):
        # Accept train and validation profiles; reject off-policy requests.
        # vLLM fills an unset top_k/min_p/repetition_penalty from
        # generation_config.json when vllm_kwargs sets generation_config="auto",
        # so serve the matched profile's top_k (as
        # vllm_worker.py::BaseVllmGenerationWorker::_build_sampling_params does)
        # with min_p and the repetition penalty off; the trainer replays none of
        # them, nor the penalties or logit bias a request may therefore not set.
        for field, value in on_policy_sampling(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            repetition_penalty=request.repetition_penalty,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            logit_bias=request.logit_bias,
            generation_config=generation_config,
        ).items():
            setattr(request, field, value)

        try:
            generator = await openai_serving_chat.create_chat_completion(
                request, raw_request
            )
        except VLLMValidationError as e:
            # vLLM raises VLLMValidationError for prompts exceeding
            # max_model_len during tokenization, instead of returning an
            # ErrorResponse. Convert to HTTP 400 so the Gym proxy can
            # detect context-length overflow and handle it gracefully.
            host._abort_request_capture(request, reason="context_length")
            return JSONResponse(
                content={
                    "error": {
                        "message": str(e),
                        "type": "invalid_request_error",
                        "param": e.parameter,
                        "code": 400,
                    }
                },
                status_code=400,
                headers=host._context_refusal_headers(request),
            )
        except MediaCaptureRejected as e:
            # Raised inside preprocess_chat before begin_call, so no capture
            # state exists yet and the abort below is a no-op kept for
            # symmetry. Return a 400 carrying a stable code so Gym and the
            # worker log can distinguish a retained-media re-tile from a
            # real engine error (which stays a 500 below).
            host._abort_request_capture(request, reason=e.code)
            LOGGER.warning(
                "Rejected captured call before inference (%s): %s", e.code, e
            )
            return JSONResponse(
                content={
                    "error": {
                        "message": str(e),
                        "type": "invalid_request_error",
                        "param": "messages",
                        "code": e.code,
                    }
                },
                status_code=400,
            )
        except ValueError as e:
            # vLLM's get_max_tokens raises a plain ValueError ("Input length
            # (N) exceeds model's maximum context length (M).") after
            # preprocess_chat, e.g. when a token-in splice grows the prompt
            # past max_model_len on a request with no max_tokens to clamp.
            # Convert only that overflow to the same 400 as above; any other
            # ValueError is a server bug and stays a 500. Must follow
            # MediaCaptureRejected, which is also a ValueError.
            if "maximum context length" not in str(e):
                # Re-raising here bypasses the BaseException handler below.
                host._abort_request_capture(request, reason="engine_error")
                raise
            LOGGER.warning("Prompt exceeds max_model_len: %s", e)
            host._abort_request_capture(request, reason="context_length")
            return JSONResponse(
                content={
                    "error": {
                        "message": str(e),
                        "type": "invalid_request_error",
                        "param": "input_tokens",
                        "code": 400,
                    }
                },
                status_code=400,
                # Only vLLM's known pre-generation get_max_tokens refusal
                # can settle capture; an arbitrary ValueError cannot.
                headers=(
                    host._context_refusal_headers(request)
                    if str(e).startswith("Input length (")
                    and "exceeds model's maximum context length" in str(e)
                    else None
                ),
            )
        except asyncio.CancelledError:
            # Client disconnected (see with_cancellation above). A cancel
            # before generation starts propagates here directly; one that
            # lands mid-generation is swallowed by vLLM and re-raised by
            # NeMoRLOpenAIServingChatMixin.chat_completion_full_generator.
            host._abort_request_capture(request, reason="client_disconnect")
            raise
        except BaseException:
            host._abort_request_capture(request, reason="engine_error")
            raise

        if isinstance(generator, ErrorResponse):
            host._abort_request_capture(request, reason="error_response")
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.error.code
            )

        elif isinstance(generator, ChatCompletionResponse):
            content = model_dump_chat_response_with_dynamic_message_fields(generator)
            # Token capture: stage the delta and ride the coords on the
            # response; strips logprobs/ids (no-op when capture is off).
            # Off-loop: the sink write inside complete_call is a blocking
            # TQ round trip (see the staging protocol's serving-host rule).
            try:
                content = await asyncio.to_thread(
                    host._finish_request_capture, request, content
                )
            except asyncio.CancelledError:
                # Queued executor work may never start after cancellation.
                # If staging already claimed the capture, abort is a no-op.
                host._abort_request_capture(request, reason="client_disconnect")
                raise
            return JSONResponse(content=content)

        host._abort_request_capture(request, reason="streaming_response")
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
                    content={
                        "prompt_token_count": generator.count,
                        "ng_context_measured": True,
                    }
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
