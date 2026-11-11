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

"""Render and answer NeMo-Gym chat calls in worker processes of Dynamo's frontend.

Rendering (template, tokenizer, the checkpoint's HF processor) and building the
reply (detokenized text through vLLM's reasoning and tool parsers) are
CPU-bound Python; on the frontend's one event loop they would serialize every
call in flight. Each worker process builds the same vLLM objects NeMo RL's
native chat server uses, from the engine's own engine args, and runs:

- ``render``: the native server's render step (``_render_chat_call``: the
  render, the capture decision, the history render the splice cuts against);
- ``respond``: vLLM's own non-streaming response builder over the engine's
  output, so the reply is the one the native server builds.

Jobs queue FIFO in the pool; nothing renders on the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import pickle
import uuid
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypedDict

LOGGER = logging.getLogger(__name__)

# Set in each worker process by _init_worker.
_WORKER: Optional["_Worker"] = None


class RestoredReply(TypedDict):
    """What the engine resumed a call from, after a checkpoint cut it.

    The engine swaps the frontend's prompt for the saved rows and generates
    only the rest, so the reply needs these to read as the whole call's.
    """

    # The prompt the engine ran: the capture prefix, then the saved rows.
    prompt_token_ids: list[int]
    # The saved rows' generated tokens, which the reply's text and usage count.
    token_ids: list[int]
    # The saved prefix's finish, when it had already ended the call:
    # ``_RestoredPrefixTerminal``'s fields without the prompt.
    terminal: Optional[dict[str, Any]]


@dataclass(frozen=True)
class RenderedCall:
    """What the frontend needs from a render to plan, route and serve a call.

    A :class:`~nemo_rl.models.generation.vllm.prompt_and_capture.RenderedPrompt`.
    """

    template_token_ids: list[int]
    # The render up to the last assistant turn, when a prefix will be spliced.
    history_token_ids: Optional[list[int]]
    # The request's own prefix (Gym's echo) after the capture decision.
    required_prefix_token_ids: Optional[list[int]]
    # Capture admissions: what is stored, and how the prompt is served.
    admission: Any
    serving_admission: Any
    verify_retained_media: bool
    # The rendered engine input's multimodal part (pickled), or None for text.
    multimodal: Optional[bytes]
    # The chat request as rendered (parsers may adjust it), without messages:
    # the engine builds its SamplingParams from it, as the native server does.
    request: dict[str, Any]
    # The output-token budget the request asked for, before the clamp.
    request_max_tokens: Optional[int]
    # What vLLM's chat serving hands the engine besides the prompt.
    reasoning_ended: Optional[bool]
    reasoning_parser_kwargs: Optional[dict[str, Any]]


class _Worker:
    """The vLLM objects one worker process renders and responds with."""

    def __init__(self, spec: dict[str, Any], model_dir: str):
        import vllm.platforms

        # A frontend host may have no accelerator; vLLM only renders here, as in
        # Dynamo's own frontend. vLLM's explicit CPU target selects its CPU
        # platform. An accelerator host keeps its own, which validates the
        # engine's args (quantization, for one) as the engine does.
        if vllm.platforms.resolve_current_platform_cls_qualname().endswith(
            ".UnspecifiedPlatform"
        ):
            os.environ["VLLM_TARGET_DEVICE"] = "cpu"

        from types import SimpleNamespace

        from vllm.config import CacheConfig, LoadConfig, VllmConfig
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.entrypoints.openai.models.protocol import BaseModelPath
        from vllm.entrypoints.openai.models.serving import OpenAIServingModels
        from vllm.renderers.registry import renderer_from_config
        from vllm.v1.engine.input_processor import InputProcessor

        from nemo_rl.models.generation.openai_server_utils import turn_end_token_ids
        from nemo_rl.models.generation.vllm import chat_server
        from nemo_rl.models.generation.vllm.utils import model_stop_token_ids
        from nemo_rl.models.generation.vllm.video_utils import (
            register_vllm_video_loader,
        )

        cfg = spec["generation"]
        chat_server.register_parser_plugins(cfg)
        # This worker decodes the call's videos; the engine runs its pixels.
        register_vllm_video_loader(cfg)
        # The engine's own args, with the weight-free snapshot Dynamo fetched.
        # Sleep mode is the engine's memory management, and vLLM's CPU platform
        # refuses it.
        args = AsyncEngineArgs(
            **{
                **spec["engine_args"],
                "model": model_dir,
                "tokenizer": model_dir,
                "enable_sleep_mode": False,
            }
        )
        self.model_config = args.create_model_config()
        vllm_config = VllmConfig(
            model_config=self.model_config,
            load_config=LoadConfig(load_format="dummy"),
            cache_config=CacheConfig(),
        )
        renderer = renderer_from_config(vllm_config)
        self.tokenizer = renderer.tokenizer
        serving_chat_kwargs, default_chat_template_kwargs = (
            chat_server.serving_chat_settings(cfg)
        )
        self.online_renderer = chat_server.FrontendRenderer(
            model_config=self.model_config,
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
        # vLLM's chat serving builds replies only here; it never generates.
        engine_client = SimpleNamespace(
            model_config=self.model_config,
            renderer=renderer,
            vllm_config=vllm_config,
            input_processor=InputProcessor(vllm_config),
            io_processor=None,
            errored=False,
        )
        models = OpenAIServingModels(
            engine_client=engine_client,
            base_model_paths=[
                BaseModelPath(
                    name=self.model_config.served_model_name, model_path=model_dir
                )
            ],
            lora_modules=None,
        )
        self.serving_chat = chat_server.FrontendServingChat(
            **{
                **serving_chat_kwargs,
                "engine_client": engine_client,
                "models": models,
                "online_renderer": self.online_renderer,
                "return_tokens_as_token_ids": True,
            }
        )
        self.loop = asyncio.new_event_loop()
        # Every id a turn can end on: the splice in the frontend cuts by them.
        self.turn_end_token_ids = sorted(
            turn_end_token_ids(
                model_stop_token_ids(self.model_config, cfg), self.tokenizer
            )
        )

    def _parser(self, chat: Any) -> Any:
        parser_cls = self.serving_chat.parser_cls
        if parser_cls is None:
            return None
        return parser_cls(
            self.tokenizer,
            chat.tools,
            chat_template_kwargs=self.serving_chat._effective_chat_template_kwargs(
                chat
            ),
            model_config=self.model_config,
        )

    def render(self, body: dict[str, Any], *, measure: bool = False) -> RenderedCall:
        from nemo_rl.models.generation.vllm.prompt_and_capture import set_max_tokens
        from nemo_rl.models.generation.vllm.chat_server import (
            NeMoRLChatCompletionRequest,
            NeMoRLMeasureChatRequest,
        )

        request_cls = (
            NeMoRLMeasureChatRequest if measure else NeMoRLChatCompletionRequest
        )
        chat = request_cls.model_validate(body)
        admission = None
        if chat.ng_capture is not None:
            # Deferred: only captured calls need it, and capture already imports nemo_gym.
            from nemo_gym.token_id_capture.staging.records import CaptureAdmission

            admission = CaptureAdmission.model_validate(chat.ng_capture)
        rendered = self.loop.run_until_complete(
            self.online_renderer.render_for_frontend(chat, admission=admission)
        )
        # The serving reads these back from the request; restore the budget
        # the render lowered for vLLM's pre-tokenization length check.
        if rendered.request_max_tokens is not None:
            set_max_tokens(chat, rendered.request_max_tokens)
        parser = self._parser(chat)
        reasoning_ended = None
        reasoning_parser_kwargs = None
        if not chat.include_reasoning or chat._grammar_from_parser:
            reasoning_ended = True
        elif parser is not None and parser.reasoning_parser is not None:
            reasoning_ended = parser.is_reasoning_end(
                rendered.engine_prompt["prompt_token_ids"]
            )
        if parser is not None and parser.reasoning_parser is not None:
            reasoning_parser_kwargs = {
                "chat_template_kwargs": self.serving_chat._engine_chat_template_kwargs(
                    self.serving_chat._effective_chat_template_kwargs(chat)
                )
            }
        multimodal = None
        if rendered.engine_prompt.get("mm_kwargs"):
            multimodal = pickle.dumps(
                {
                    key: value
                    for key, value in rendered.engine_prompt.items()
                    if key != "prompt_token_ids"
                },
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return RenderedCall(
            template_token_ids=list(rendered.engine_prompt["prompt_token_ids"]),
            history_token_ids=rendered.history_token_ids,
            required_prefix_token_ids=rendered.required_prefix_token_ids,
            admission=rendered.admission,
            serving_admission=rendered.serving_admission,
            verify_retained_media=rendered.verify_retained_media,
            multimodal=multimodal,
            request=chat.model_dump(mode="json", exclude={"messages"}),
            request_max_tokens=rendered.request_max_tokens,
            reasoning_ended=reasoning_ended,
            reasoning_parser_kwargs=reasoning_parser_kwargs,
        )

    def respond(
        self,
        request: dict[str, Any],
        prompt_token_ids: list[int],
        output: dict[str, Any],
        restored: Optional[RestoredReply] = None,
    ) -> dict[str, Any]:
        from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionResponse,
        )
        from vllm.logprobs import Logprob
        from vllm.outputs import CompletionOutput, RequestOutput

        from nemo_rl.models.generation.prefix_recovery.restore import (
            _RestoredPrefixTerminal,
        )
        from nemo_rl.models.generation.vllm.chat_server import (
            chat_request_from_render,
        )

        chat = chat_request_from_render(request)
        if restored is not None:
            # The engine resumed this call from the tokens a checkpoint cut
            # saved: it ran this prompt and generated only the rest. The
            # serving chat's reply fixes read these fields, as natively.
            prompt_token_ids = restored["prompt_token_ids"]
            chat._restored_prompt_token_ids = tuple(prompt_token_ids)
            chat._restored_generation_token_ids = tuple(restored["token_ids"])
            # The engine lowered its own copy's min_tokens, never this one.
            chat._original_min_tokens = chat.min_tokens
            if restored["terminal"] is not None:
                chat._restored_prefix_terminal = _RestoredPrefixTerminal(
                    prompt_token_ids=tuple(prompt_token_ids), **restored["terminal"]
                )
        logprobs = None
        if output.get("logprobs") is not None:
            logprobs = [
                {
                    int(token_id): Logprob(
                        logprob=entry[0], rank=entry[1], decoded_token=entry[2]
                    )
                    for token_id, entry in position.items()
                }
                for position in output["logprobs"]
            ]
        request_output = RequestOutput(
            request_id=f"chatcmpl-{uuid.uuid4().hex}",
            prompt=None,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text=output["text"],
                    token_ids=output["token_ids"],
                    cumulative_logprob=None,
                    logprobs=logprobs,
                    finish_reason=output["finish_reason"],
                    stop_reason=output.get("stop_reason"),
                )
            ],
            finished=True,
        )

        async def results():
            yield request_output

        response = self.loop.run_until_complete(
            self.serving_chat.chat_completion_full_generator(
                chat,
                results(),
                request_output.request_id,
                request["model"],
                [],
                self.tokenizer,
                RequestResponseMetadata(request_id=request_output.request_id),
                parser=self._parser(chat),
            )
        )
        if not isinstance(response, ChatCompletionResponse):
            raise ValueError(f"vLLM could not build the reply: {response}")
        return response.model_dump(mode="json", exclude_none=True)


def _init_worker(spec: dict[str, Any], model_dir: str) -> None:
    global _WORKER
    _WORKER = _Worker(spec, model_dir)


def _render(body: dict[str, Any], measure: bool) -> RenderedCall:
    assert _WORKER is not None
    return _WORKER.render(body, measure=measure)


def _respond(
    request: dict[str, Any],
    prompt_token_ids: list[int],
    output: dict[str, Any],
    restored: Optional[RestoredReply],
) -> dict[str, Any]:
    assert _WORKER is not None
    return _WORKER.respond(request, prompt_token_ids, output, restored)


def _info() -> dict[str, Any]:
    assert _WORKER is not None
    return {
        "turn_end_token_ids": _WORKER.turn_end_token_ids,
        "max_model_len": _WORKER.model_config.max_model_len,
    }


class RenderPool:
    """``workers`` processes that render and answer chat calls, FIFO."""

    def __init__(self, spec: dict[str, Any], model_dir: str, *, workers: int):
        self._spec = spec
        self._model_dir = model_dir
        self._workers = workers
        self._executor = self._start()
        self._info: Optional[dict[str, Any]] = None

    def _start(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=self._workers,
            # Fork would copy the frontend's Rust runtime and event loop.
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(self._spec, self._model_dir),
        )

    async def _submit(self, fn: Callable[..., Any], *args: Any) -> Any:
        executor = self._executor
        try:
            return await asyncio.wrap_future(executor.submit(fn, *args))
        except BrokenProcessPool as error:
            if self._info is None:
                # No worker ever served a job: most often _init_worker failed (a
                # config error, which a restart cannot fix), so do not restart.
                raise RuntimeError(
                    "the render pool's workers died before serving a call; see "
                    "this log for an 'Exception in initializer' traceback or a "
                    "kill (e.g. OOM)"
                ) from error
            if self._executor is executor:
                # A worker died (e.g. OOM-killed). The calls in flight fail;
                # later calls get a fresh pool instead of this broken one.
                LOGGER.error("A render-pool worker died; restarting the pool")
                executor.shutdown(wait=False, cancel_futures=True)
                self._executor = self._start()
            raise

    async def info(self) -> dict[str, Any]:
        """The model's turn-end ids and context length, from a started worker.

        Workers start on first use, so a run that sends no chat call never
        builds them.
        """
        if self._info is None:
            self._info = await self._submit(_info)
        return self._info

    async def render(
        self, body: dict[str, Any], *, measure: bool = False
    ) -> RenderedCall:
        """Render a call; a measure call is counted past the context, as natively."""
        return await self._submit(_render, body, measure)

    async def respond(
        self,
        request: dict[str, Any],
        prompt_token_ids: list[int],
        output: dict[str, Any],
        restored: Optional[RestoredReply] = None,
    ) -> dict[str, Any]:
        """Build the reply; ``restored`` is the engine's ``RestoredReply``, if any."""
        return await self._submit(_respond, request, prompt_token_ids, output, restored)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
