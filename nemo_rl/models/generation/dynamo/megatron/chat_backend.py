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

"""Megatron's side of NeMo RL's chat engine in Dynamo's frontend.

:data:`CHAT_BACKEND` is what the frontend's chat engine
(:mod:`~nemo_rl.models.generation.dynamo.chat_engine`) runs for Megatron
prefill/decode engines, and :class:`MegatronRenderer` its render-pool worker.
Both are thin wrappers over Megatron-LM's ``ChatRenderer``, the render and
reply half of Megatron-LM's own chat endpoint
(``dynamic_text_gen_server/endpoints/chat_completions.py``), so a Gym call is
rendered, stitched and answered as that endpoint would:

- the render worker calls ``ChatRenderer.render`` (options, message and tool
  sanitizing, chat template, tokenization) without its BOS rule, and its
  ``build_response`` (detokenization, ``mcore_generation_config.parsers``, the
  OpenAI body) for the reply;
- the frontend splices the model's own earlier turns into the render
  (:func:`~nemo_rl.models.generation.gym_prompt.prepare_prompt`), cutting by
  ``ChatRenderer.eos_token_ids``, and only then applies
  ``ChatRenderer.apply_bos_policy`` with the request's ``add_BOS``, which is
  the order Megatron-LM's endpoint stitches and applies the rule in.

The engine never sees chat: it gets the final token ids, and its adapter
(:mod:`~nemo_rl.models.generation.dynamo.megatron.adapter`) generates and
stages the call.

Megatron-LM is imported only in the render workers and on the call, never
when this module loads. ``ChatRenderer`` is in Megatron-LM from
NVIDIA/Megatron-LM#9579 on; older pins fail when the render workers start.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

from nemo_rl.models.generation.dynamo.render_pool import RenderedCall, RestoredReply
from nemo_rl.models.generation.gym_prompt import PreparedPrompt, prepare_prompt

# Megatron-LM's public chat rendering: ChatRenderer, ChatGeneration.
_CHAT_RENDERER_MODULE = (
    "megatron.core.inference.text_generation_server.dynamic_text_gen_server"
)
# The policy.generation keys the frontend renders, checks and replies with. Only
# these are published, so prefill and decode engines (whose own configs carry
# their role and layout) publish identical settings.
_CHAT_GENERATION_KEYS = (
    "model_name",
    "temperature",
    "top_p",
    "top_k",
    "val_temperature",
    "val_top_p",
    "val_top_k",
)
_CHAT_MCORE_KEYS = ("expose_http_server", "parsers", "max_model_len")


def megatron_chat_spec(generation: Mapping[str, Any]) -> dict[str, Any]:
    """The chat settings Megatron engines publish, from ``policy.generation``.

    The same on the driver (before the engines load) and on every engine (at
    registration), so the render workers the driver's copy built are kept.
    """
    mcore = generation["mcore_generation_config"]
    return {
        "generation": {
            **{key: generation.get(key) for key in _CHAT_GENERATION_KEYS},
            "mcore_generation_config": {key: mcore[key] for key in _CHAT_MCORE_KEYS},
        }
    }


class MegatronCallRejected(ValueError):
    """A NeMo-Gym call the Megatron engines cannot serve as asked."""


def _clamp_max_tokens(
    request_max_tokens: Optional[int], prompt_len: int, max_model_len: int
) -> Optional[int]:
    """The output budget that keeps input + output within ``max_model_len``."""
    remaining = max_model_len - prompt_len
    if remaining <= 0:
        # Gym detects a context overflow by the literal "maximum context length".
        raise MegatronCallRejected(
            f"Prompt length ({prompt_len}) fills or exceeds this model's maximum "
            f"context length ({max_model_len}). No room for output tokens."
        )
    if request_max_tokens is None:
        return None
    return min(request_max_tokens, remaining)


@dataclass(frozen=True)
class _Refusal:
    """A :class:`~nemo_rl.models.generation.dynamo.chat_engine.CallRefusal`."""

    status: int
    message: str
    acknowledge: bool


def _build_tokenizer(model_dir: str) -> Any:
    """The Megatron tokenizer the engines build (``models/megatron/setup.py``)."""
    from megatron.core.tokenizers import MegatronTokenizer

    return MegatronTokenizer.from_pretrained(
        tokenizer_path=model_dir,
        metadata_path={"library": "huggingface"},
        trust_remote_code=True,
        use_fast=True,
    )


class MegatronRenderer:
    """One render-pool worker: Megatron-LM's ``ChatRenderer`` over the engines' tokenizer.

    A :class:`~nemo_rl.models.generation.dynamo.render_pool.RenderWorker`.
    """

    @staticmethod
    def prepare_process() -> None:
        importlib.import_module("megatron.core.tokenizers")
        importlib.import_module(_CHAT_RENDERER_MODULE)

    def __init__(self, spec: dict[str, Any], model_dir: str):
        started = time.monotonic()
        from megatron.core.tokenizers.text.parsers import PARSER_MAPPING

        chat = importlib.import_module(_CHAT_RENDERER_MODULE)
        mcore = spec["generation"]["mcore_generation_config"]
        parsers = list(mcore["parsers"] or [])
        # Megatron-LM checks the names only when it parses a reply; a typo
        # fails the render workers' start instead.
        unknown = sorted(set(parsers) - PARSER_MAPPING.keys())
        if unknown:
            raise ValueError(
                f"mcore_generation_config.parsers {unknown} are not Megatron-LM "
                f"parsers; known: {sorted(PARSER_MAPPING)}"
            )
        self._chat_generation = chat.ChatGeneration
        # As the engines' own server is started (MegatronGenerationMixin.
        # _setup_openai_api_server): no server chat template, not eval mode.
        self.chat = chat.ChatRenderer(_build_tokenizer(model_dir), parsers)
        self.max_model_len = int(mcore["max_model_len"])
        self.build_seconds = time.monotonic() - started

    def info(self) -> dict[str, Any]:
        return {
            # What Megatron-LM's endpoint stitches previous turns by.
            "turn_end_token_ids": sorted(self.chat.eos_token_ids()),
            # The frontend applies the BOS rule after the splice.
            "bos_token_id": getattr(self.chat.tokenizer, "bos", None),
            "max_model_len": self.max_model_len,
            "build_seconds": self.build_seconds,
        }

    def _history_token_ids(self, rendered: Any) -> list[int]:
        """The render up to the last assistant turn, which a splice cuts against.

        ``ChatRenderer`` renders it only when there is an assistant turn; a
        prefix spliced into a conversation with none cuts against the whole
        conversation without the generation prompt, as the vLLM path does.
        """
        if rendered.history_token_ids is not None:
            return rendered.history_token_ids
        if not self.chat.has_chat_template(rendered.options):
            raise MegatronCallRejected(
                "a token-capture prefix needs a chat template to splice into"
            )
        return self.chat.tokenize(
            rendered.template_messages,
            rendered.media_slots,
            tools=rendered.template_tools,
            chat_template_kwargs=rendered.options.chat_template_kwargs,
            add_generation_prompt=False,
        )

    def render(self, body: dict[str, Any], *, measure: bool = False) -> RenderedCall:
        from nemo_rl.models.generation.capture_context import resolve_capture_input

        required_prefix_token_ids = body.get("required_prefix_token_ids")
        # Only a call that may splice renders its history.
        may_splice = (
            required_prefix_token_ids is not None or body.get("ng_capture") is not None
        )
        try:
            # add_bos=None: the BOS rule comes after the splice, in the frontend.
            rendered = self.chat.render(body, with_history=may_splice, add_bos=None)
        except ValueError as error:
            # Megatron-LM's endpoint answers a malformed request with 400.
            raise MegatronCallRejected(str(error)) from error
        if rendered.multi_modal_data is not None or rendered.media_slots:
            raise MegatronCallRejected(
                "Megatron engines behind Dynamo serve text chat only; the call "
                "carried media"
            )
        admission = None
        if body.get("ng_capture") is not None:
            # Deferred: only captured calls need it, and capture already imports nemo_gym.
            from nemo_gym.token_id_capture.staging.records import CaptureAdmission

            admission = CaptureAdmission.model_validate(body["ng_capture"])
        serving_admission = admission
        verify_retained_media = False
        if admission is not None and (
            admission.mode == "candidate" or admission.request_replay is not None
        ):
            # Inline assistant token fields cannot override RL's decision.
            required_prefix_token_ids = None
            # The template saw ChatRenderer's sanitized messages, every content
            # a string, so null and empty tool content render alike, as for
            # vLLM's HF renderer.
            decision = resolve_capture_input(
                admission,
                messages=body["messages"],
                conversation=rendered.template_messages,
                renderer_is_hf=True,
            )
            admission = decision.storage
            serving_admission = decision.serving
            verify_retained_media = decision.verify_retained_media
        history_token_ids = None
        if required_prefix_token_ids is not None or (
            serving_admission is not None and serving_admission.mode == "token_in"
        ):
            history_token_ids = self._history_token_ids(rendered)
        request_max_tokens = body.get("max_completion_tokens")
        if request_max_tokens is None:
            request_max_tokens = body.get("max_tokens")
        if measure:
            request_max_tokens = None
        return RenderedCall(
            template_token_ids=rendered.prompt_token_ids,
            history_token_ids=history_token_ids,
            required_prefix_token_ids=required_prefix_token_ids,
            admission=admission,
            serving_admission=serving_admission,
            verify_retained_media=verify_retained_media,
            multimodal=None,
            request={key: value for key, value in body.items() if key != "messages"},
            request_max_tokens=request_max_tokens,
            reasoning_ended=None,
            reasoning_parser_kwargs=None,
        )

    def respond(
        self,
        request: dict[str, Any],
        prompt_token_ids: list[int],
        output: dict[str, Any],
        restored: Optional[RestoredReply] = None,
    ) -> dict[str, Any]:
        """``ChatRenderer.build_response`` for the engine's output."""
        if restored is not None:
            raise RuntimeError(
                "Megatron engines take no generation cuts, so no call is restored"
            )
        from megatron.core.inference.sampling_params import SamplingParams

        # The two fields build_response reads, as the endpoint's
        # parse_sampling_params sets them for a chat request.
        sampling_params = SamplingParams(
            return_log_probs=bool(request.get("logprobs")),
            detokenize_stop_sequence=bool(request.get("include_stop_str_in_output")),
        )
        generation = self._chat_generation(
            generated_tokens=list(output["token_ids"]),
            finish_reason=output["finish_reason"],
            prompt_tokens=prompt_token_ids,
            generated_log_probs=output.get("log_probs"),
        )
        return self.chat.build_response(
            [generation], sampling_params, self.chat.parse_options(request)
        )


@dataclass(frozen=True)
class _BosTokenizer:
    """The one tokenizer field ``ChatRenderer.apply_bos_policy`` reads.

    The frontend holds no tokenizer; the render workers report its BOS id.
    """

    bos: Optional[int]


class MegatronChatBackend:
    """The chat engine's Megatron steps."""

    renderer = f"{__name__}:MegatronRenderer"
    # prepare_prompt applies Megatron-LM's BOS rule.
    warm_modules: tuple[str, ...] = (_CHAT_RENDERER_MODULE,)

    def __init__(self) -> None:
        # The frontend's ChatRenderer, for its BOS rule only; built on the
        # first call from the render workers' BOS id.
        self._bos_renderer: Any = None

    def serves_chat(self, generation: Mapping[str, Any]) -> bool:
        return bool(generation["mcore_generation_config"]["expose_http_server"])

    def checked_sampling(
        self,
        body: dict[str, Any],
        *,
        on_policy: bool,
        generation: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Megatron-LM's endpoint defaults for the sampling a call leaves unset.

        As the native server is started (``_http_sampling_defaults``): an
        omitted temperature, top_p or top_k takes ``policy.generation``'s.
        """
        if (body.get("n") or 1) != 1:
            # Capture stages one choice per call, and the trainer replays one.
            raise MegatronCallRejected(
                f"request n={body['n']} is not supported; NeMo-Gym calls take n=1"
            )
        if body.get("stream"):
            raise MegatronCallRejected(
                "request stream=true is not supported; NeMo-Gym calls do not stream"
            )
        if body.get("logprobs") and body.get("top_logprobs"):
            # The Megatron Dynamo adapter returns the sampled tokens' only.
            raise MegatronCallRejected(
                "top_logprobs is not supported for Megatron engines behind Dynamo"
            )
        # DRAFT: the vLLM backend refuses a trained call whose sampling is off
        # the train or validation profile (on_policy_sampling); Megatron's
        # native server only fills omitted fields, which this keeps.
        del on_policy
        top_p = generation["top_p"]
        top_k = generation["top_k"]
        defaults = {
            "temperature": float(generation["temperature"]),
            "top_p": 1.0 if top_p is None else float(top_p),
            "top_k": -1 if top_k is None or top_k < 0 else int(top_k),
        }
        return {key: value for key, value in defaults.items() if body.get(key) is None}

    def _bos_policy(
        self, bos_token_id: Optional[int], add_bos: bool
    ) -> Callable[[list[int], Optional[int]], list[int]]:
        """Megatron-LM's BOS rule, as its endpoint applies it after the splice."""
        tokenizer = _BosTokenizer(bos_token_id)
        if self._bos_renderer is None or self._bos_renderer.tokenizer != tokenizer:
            chat = importlib.import_module(_CHAT_RENDERER_MODULE)
            self._bos_renderer = chat.ChatRenderer(tokenizer)
        renderer = self._bos_renderer

        def apply(prompt_token_ids: list[int], prefix_len: Optional[int]) -> list[int]:
            final = renderer.apply_bos_policy(list(prompt_token_ids), add_bos)
            if prefix_len is not None and final != prompt_token_ids:
                # The model's own prefix is the prompt the engine ran last
                # turn, so it already follows the rule; a change means the
                # call asks for add_BOS other than its earlier turns did.
                raise MegatronCallRejected(
                    f"add_BOS={add_bos} would change the leading BOS of the model's "
                    "own prefix; send the add_BOS its earlier turns were served with"
                )
            return final

        return apply

    async def prepare_prompt(
        self,
        rendered: RenderedCall,
        *,
        resolve_prefix: Callable[[Any], list[int]],
        render_info: Mapping[str, Any],
        measure_only: bool,
    ) -> PreparedPrompt:
        """Splice, then Megatron-LM's BOS rule, then the clamp.

        ``ChatRenderer`` renders with the template's own BOS; Megatron-LM's
        endpoint stitches the previous turn into that render and only then
        applies the BOS rule with the request's ``add_BOS``, so the rule runs
        here, on the spliced prompt, before the output budget is measured.
        """
        return await prepare_prompt(
            rendered,
            resolve_prefix=resolve_prefix,
            turn_end_token_ids=render_info["turn_end_token_ids"],
            max_model_len=render_info["max_model_len"],
            measure_only=measure_only,
            clamp_max_tokens=_clamp_max_tokens,
            finish_prompt=self._bos_policy(
                render_info["bos_token_id"],
                # As the endpoint's parse_sampling_params reads it.
                bool(rendered.request.get("add_BOS")),
            ),
        )

    def error_reply(self, error: Exception) -> Optional[_Refusal]:
        if isinstance(error, MegatronCallRejected):
            overflow = "maximum context length" in str(error)
            return _Refusal(status=400, message=str(error), acknowledge=overflow)
        return None

    def engine_options(
        self, body: dict[str, Any], rendered: RenderedCall
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # The engine's adapter builds its sampling from the rendered request
        # (adapter.gym_sampling_request); these feed only Dynamo's router.
        del rendered
        return {"temperature": body["temperature"], "top_p": body["top_p"]}, {}


CHAT_BACKEND = MegatronChatBackend()
