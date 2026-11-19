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

"""vLLM's side of NeMo RL's chat engine in Dynamo's frontend.

:data:`CHAT_BACKEND` is what the frontend's chat engine
(:mod:`~nemo_rl.models.generation.dynamo.chat_engine`) runs for vLLM engines,
and :class:`VllmRenderer` its render-pool worker. Every step is the native vLLM
chat server's own: the render and reply through vLLM's chat serving, the
on-policy sampling check, the clamp and the error statuses.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import pickle
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any, Optional

from nemo_rl.models.generation.dynamo.render_pool import RenderedCall, RestoredReply
from nemo_rl.models.generation.gym_prompt import PreparedPrompt


def _import_vllm() -> None:
    """Import everything a worker builds and serves with."""
    import vllm.platforms

    # A frontend host may have no accelerator; vLLM only renders here, as in
    # Dynamo's own frontend. vLLM's explicit CPU target selects its CPU
    # platform. An accelerator host keeps its own, which validates the
    # engine's args (quantization, for one) as the engine does.
    if vllm.platforms.resolve_current_platform_cls_qualname().endswith(
        ".UnspecifiedPlatform"
    ):
        os.environ["VLLM_TARGET_DEVICE"] = "cpu"
    # The chat server imports the rest of vLLM that a worker uses.
    importlib.import_module("nemo_rl.models.generation.vllm.chat_server")


class VllmRenderer:
    """The vLLM objects one render-pool worker process renders and responds with.

    A :class:`~nemo_rl.models.generation.dynamo.render_pool.RenderWorker`; each
    process builds the same vLLM objects NeMo RL's native chat server uses, from
    the engine's own engine args: ``render`` is the native server's render step
    (``_render_chat_call``), and ``respond`` vLLM's own non-streaming response
    builder over the engine's output.
    """

    prepare_process = staticmethod(_import_vllm)

    def __init__(self, spec: dict[str, Any], model_dir: str):
        started = time.monotonic()
        _import_vllm()

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
        # As vLLM's API server does at startup: compiles the chat template and
        # warms the multimodal processor, so the first render does neither.
        self.online_renderer.warmup()
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
        self.build_seconds = time.monotonic() - started

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
        from nemo_rl.models.generation.vllm.chat_server import (
            NeMoRLChatCompletionRequest,
            NeMoRLMeasureChatRequest,
        )
        from nemo_rl.models.generation.vllm.prompt_and_capture import set_max_tokens

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
        reasoning_parser_kwargs = None
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
            reasoning_ended=rendered.reasoning_ended,
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

    def info(self) -> dict[str, Any]:
        return {
            "turn_end_token_ids": self.turn_end_token_ids,
            "max_model_len": self.model_config.max_model_len,
            "build_seconds": self.build_seconds,
        }


class VllmChatBackend:
    """The chat engine's vLLM steps; vLLM is imported only on the call."""

    renderer = f"{__name__}:VllmRenderer"
    # prompt_and_capture imports vLLM.
    warm_modules = ("nemo_rl.models.generation.vllm.prompt_and_capture",)

    def serves_chat(self, generation: Mapping[str, Any]) -> bool:
        # NeMo-Gym's calls need this flag, and the native worker starts its
        # chat server on it.
        return bool(generation["vllm_cfg"].get("expose_http_server"))

    def checked_sampling(
        self,
        body: dict[str, Any],
        *,
        on_policy: bool,
        generation: Mapping[str, Any],
    ) -> dict[str, Any]:
        from nemo_rl.models.generation.vllm.utils import (
            OffPolicySamplingError,
            on_policy_sampling,
        )

        if (body.get("n") or 1) != 1:
            # Capture stages one choice per call, and the trainer replays one.
            raise OffPolicySamplingError(
                f"request n={body['n']} is not supported; NeMo-Gym calls take n=1"
            )
        if body.get("stream"):
            # vLLM's delta outputs would leave this path only the last piece.
            raise OffPolicySamplingError(
                "request stream=true is not supported; NeMo-Gym calls do not stream"
            )
        if not on_policy:
            return {}
        return on_policy_sampling(
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            top_k=body.get("top_k"),
            min_p=body.get("min_p"),
            repetition_penalty=body.get("repetition_penalty"),
            presence_penalty=body.get("presence_penalty"),
            frequency_penalty=body.get("frequency_penalty"),
            logit_bias=body.get("logit_bias"),
            generation_config=generation,
        )

    async def prepare_prompt(
        self,
        rendered: RenderedCall,
        *,
        resolve_prefix: Callable[[Any], list[int]],
        render_info: Mapping[str, Any],
        measure_only: bool,
    ) -> PreparedPrompt:
        from nemo_rl.models.generation.vllm.prompt_and_capture import prepare_prompt

        return await prepare_prompt(
            rendered,
            resolve_prefix=resolve_prefix,
            turn_end_token_ids=render_info["turn_end_token_ids"],
            max_model_len=render_info["max_model_len"],
            measure_only=measure_only,
        )

    def error_reply(self, error: Exception) -> Any:
        from nemo_rl.models.generation.vllm.prompt_and_capture import error_reply

        return error_reply(error)

    def engine_options(
        self, body: dict[str, Any], rendered: RenderedCall
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # The adapter builds the engine's SamplingParams from the rendered
        # request, as the native server does; these feed only Dynamo's router.
        del rendered
        return {"temperature": body["temperature"], "top_p": body["top_p"]}, {}


CHAT_BACKEND = VllmChatBackend()
