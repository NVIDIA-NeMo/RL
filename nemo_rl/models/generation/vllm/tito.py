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
"""Request-local incremental rendering for verified captured Qwen3 histories."""

import asyncio
import copy
import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol

from nemo_rl.models.generation.capture_context import (
    CaptureInputDecision,
    decide_capture_input,
)

if TYPE_CHECKING:
    from nemo_gym.token_id_capture.staging.records import CaptureAdmission

LOGGER = logging.getLogger(__name__)

# Official Qwen/Qwen3-0.6B tokenizer_config.json chat_template. An exact
# template contract prevents custom templates from silently changing the suffix.
QWEN3_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"  # pragma: allowlist secret


class TitoNotApplicable(ValueError):
    """The request needs the existing full-history preprocessing path."""


def qwen3_suffix_text(messages: list[dict[str, Any]], *, enable_thinking: bool) -> str:
    """Render appended observations using the verified Qwen3 template contract."""
    if not messages or any(
        message.get("role") not in ("user", "tool")
        or not isinstance(message.get("content"), str)
        for message in messages
    ):
        raise TitoNotApplicable("unsupported_observation")
    # Generation stops at im_end; its following template newline is context.
    parts = ["\n"]
    for index, message in enumerate(messages):
        role, content = message["role"], message["content"]
        if role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        else:
            if index == 0 or messages[index - 1]["role"] != "tool":
                parts.append("<|im_start|>user")
            parts.append(f"\n<tool_response>\n{content}\n</tool_response>")
            if index == len(messages) - 1 or messages[index + 1]["role"] != "tool":
                parts.append("<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    if not enable_thinking:
        parts.append("<think>\n\n</think>\n\n")
    return "".join(parts)


@lru_cache(maxsize=16)
def _template_digest(template: str) -> str:
    return hashlib.sha256(template.encode()).hexdigest()


class IncrementalTokenizer(Protocol):
    """Model-specific qualification and suffix encoding for captured histories."""

    def qualify_template(self, template: str | None) -> None: ...

    def validate_prefix(self, tokenizer: Any, prefix: list[int]) -> None: ...

    def encode_suffix(
        self, tokenizer: Any, messages: list[dict[str, Any]], *, enable_thinking: bool
    ) -> list[int]: ...


@dataclass(frozen=True)
class Qwen3IncrementalTokenizer:
    """Incremental observations for the qualified Qwen3 chat template."""

    def qualify_template(self, template: str | None) -> None:
        if template is None or _template_digest(template) != QWEN3_TEMPLATE_SHA256:
            raise TitoNotApplicable("unqualified_template")

    def validate_prefix(self, tokenizer: Any, prefix: list[int]) -> None:
        end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if not prefix or prefix[-1] != end_id or end_id != tokenizer.eos_token_id:
            raise TitoNotApplicable("unfinished_generation")

    def encode_suffix(
        self, tokenizer: Any, messages: list[dict[str, Any]], *, enable_thinking: bool
    ) -> list[int]:
        suffix = qwen3_suffix_text(messages, enable_thinking=enable_thinking)
        return tokenizer.encode(suffix, add_special_tokens=False)


INCREMENTAL_TOKENIZERS: dict[str, IncrementalTokenizer] = {
    "qwen3": Qwen3IncrementalTokenizer(),
}


@dataclass(frozen=True)
class CapturedContinuation:
    """A verified Gym-selected parent and its request-owned original tokens."""

    decision: CaptureInputDecision
    prefix_token_ids: list[int]
    suffix_messages: list[dict[str, Any]]


async def resolve_captured_continuation(
    admission: "CaptureAdmission",
    *,
    messages: list[dict[str, Any]],
    conversation: list[dict[str, Any]],
    resolve_prefix: Callable[[Any], list[int]],
) -> CapturedContinuation:
    """Resolve the selected parent without imposing latest-call-only semantics.

    Gym owns source matching and ambiguity resolution. This entry point verifies
    the converted cut and retrieves the same selected prefix for rendering and
    capture registration. Storage errors propagate to the caller.
    """
    decision = decide_capture_input(
        admission,
        messages=messages,
        allow_empty_tool_content=bool(conversation)
        and all(isinstance(item.get("content"), str) for item in conversation),
    )
    serving = decision.serving
    if serving is None or serving.mode != "token_in":
        raise TitoNotApplicable("unverified_boundary")
    assert admission.request_replay is not None
    assert admission.candidate_replay is not None
    suffix_count = (
        len(admission.request_replay.items) - admission.candidate_replay.item_count
    )
    cut = len(messages) - suffix_count
    if (
        len(conversation) != len(messages)
        or conversation[cut - 1]["role"] != "assistant"
    ):
        raise TitoNotApplicable("parsed_boundary_changed")
    suffix = conversation[cut:]
    if any(message.get("role") not in ("user", "tool") for message in suffix):
        raise TitoNotApplicable("unsupported_observation")
    prefix = await asyncio.to_thread(resolve_prefix, serving)
    if len(prefix) != serving.prev_len:
        raise ValueError("Captured TITO prefix length differs from the selected parent")
    return CapturedContinuation(decision, prefix, suffix)


@dataclass
class TitoTrace:
    """Per-attempt measurements emitted as a structured debug log."""

    outcome: str = "fallback"
    reason: str = ""
    reused_tokens: int = 0
    suffix_tokens: int = 0
    history_seconds: float = 0.0
    suffix_seconds: float = 0.0
    prepare_seconds: float = 0.0
    total_seconds: float = 0.0


@dataclass(frozen=True)
class TitoPreprocessResult:
    """Explicit handoff from request preprocessing to capture registration."""

    preprocessed: Any | None
    continuation: CapturedContinuation | None


@dataclass
class CapturedTitoRenderer:
    """Request-local bridge between native vLLM and incremental tokenization."""

    renderer: Any
    worker: Any
    admission: "CaptureAdmission"
    incremental_tokenizer: IncrementalTokenizer
    trace: TitoTrace
    continuation: CapturedContinuation | None = field(default=None, init=False)

    @property
    def tokenizer(self) -> Any:
        return self.renderer.tokenizer

    def get_tokenizer(self) -> Any:
        return self.renderer.get_tokenizer()

    async def render_chat_async(
        self,
        conversations: list[list[dict[str, Any]]],
        params: Any,
        tok_params: Any,
        *,
        prompt_extras: dict[str, Any],
        skip_mm_cache: bool,
    ) -> Any:
        # vLLM is optional outside the serving worker.
        from vllm.renderers.chat_utils import parse_chat_messages_async
        from vllm.renderers.hf import (
            resolve_chat_template,
            resolve_chat_template_content_format,
        )

        # Checkpoint restore owns interrupted output tokens and remaining budgets.
        if self.admission.generation_cut is not None:
            raise TitoNotApplicable("generation_restore")
        if len(conversations) != 1:
            raise TitoNotApplicable("multiple_conversations")
        messages = conversations[0]
        kwargs = params.get_apply_chat_template_kwargs()
        template = resolve_chat_template(
            self.tokenizer,
            chat_template=kwargs.get("chat_template"),
            tools=kwargs.get("tools"),
            model_config=self.renderer.model_config,
        )
        self.incremental_tokenizer.qualify_template(template)
        if kwargs.get("tokenize") or kwargs.get("continue_final_message"):
            raise TitoNotApplicable("continuing_template")
        if not kwargs.get("add_generation_prompt"):
            raise TitoNotApplicable("missing_generation_prompt")
        if any(
            message.get("role") not in ("system", "user", "assistant", "tool")
            or (
                message.get("content") is not None
                and not isinstance(message["content"], str)
            )
            for message in messages
        ):
            raise TitoNotApplicable("unsupported_content")
        content_format = resolve_chat_template_content_format(
            chat_template=params.chat_template,
            tools=params.chat_template_kwargs.get("tools"),
            given_format=params.chat_template_content_format,
            tokenizer=self.tokenizer,
            model_config=self.renderer.model_config,
        )
        if content_format != "string":
            raise TitoNotApplicable("non_string_renderer")
        conversation, media, media_ids = await parse_chat_messages_async(
            messages,
            self.renderer.model_config,
            content_format=content_format,
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        if media is not None or media_ids is not None:
            raise TitoNotApplicable("media")
        started = time.perf_counter()
        continuation = await resolve_captured_continuation(
            self.admission,
            messages=messages,
            conversation=conversation,
            resolve_prefix=self.worker._resolve_admission_prefix,
        )
        self.trace.history_seconds = time.perf_counter() - started
        prefix = continuation.prefix_token_ids
        self.incremental_tokenizer.validate_prefix(self.tokenizer, prefix)
        started = time.perf_counter()
        suffix_ids = await asyncio.to_thread(
            self.incremental_tokenizer.encode_suffix,
            self.tokenizer,
            continuation.suffix_messages,
            enable_thinking=kwargs.get("enable_thinking") is not False,
        )
        self.trace.suffix_seconds = time.perf_counter() - started
        started = time.perf_counter()
        inputs = await self.renderer.render_cmpl_async(
            [{"prompt_token_ids": prefix + suffix_ids}],
            tok_params,
            prompt_extras=prompt_extras,
            skip_mm_cache=skip_mm_cache,
        )
        self.trace.prepare_seconds = time.perf_counter() - started
        self.trace.reused_tokens = len(prefix)
        self.trace.suffix_tokens = len(suffix_ids)
        self.continuation = continuation
        return [conversation], inputs


async def preprocess_captured_tito(
    *,
    model: str,
    worker: Any,
    online_renderer: Any,
    request: Any,
    messages: list[Any],
    default_template: Any,
    default_template_content_format: Any,
    default_template_kwargs: Any,
    tool_dicts: Any,
    parser: Any,
    skip_mm_cache: bool,
) -> TitoPreprocessResult:
    """Use native vLLM request adjustment with a request-local token renderer."""
    # These imports are only needed when the optional TITO serving path is used.
    import vllm
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.renderers.hf import HfRenderer
    from vllm.renderers.online_renderer import OnlineRenderer

    started = time.perf_counter()
    trace = TitoTrace()

    def finish(
        outcome: str,
        reason: str,
        result: Any | None,
        continuation: CapturedContinuation | None,
    ) -> TitoPreprocessResult:
        trace.outcome = outcome
        trace.reason = reason
        trace.total_seconds = time.perf_counter() - started
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("Captured TITO %s", json.dumps(asdict(trace), sort_keys=True))
        return TitoPreprocessResult(result, continuation)

    if vllm.__version__ != "0.33.0":
        raise ValueError("Captured TITO requires vLLM 0.33.0")
    incremental_tokenizer = INCREMENTAL_TOKENIZERS[model]
    if not isinstance(request, ChatCompletionRequest):
        return finish("fallback", "unsupported_request", None, None)
    if type(online_renderer.renderer) is not HfRenderer:
        return finish("fallback", "unsupported_renderer", None, None)
    if online_renderer.model_config.enable_prompt_embeds:
        return finish("fallback", "prompt_embeds", None, None)
    for option in (
        "stream",
        "echo",
        "continue_final_message",
        "return_assistant_tokens_mask",
        "add_special_tokens",
        "return_prompt_text",
        "return_token_offsets",
        "kv_transfer_params",
    ):
        if getattr(request, option, None):
            return finish("fallback", option, None, None)
    if getattr(request, "n", 1) not in (None, 1):
        return finish("fallback", "multiple_choices", None, None)
    if getattr(request, "truncate_prompt_tokens", None) is not None:
        return finish("fallback", "prompt_truncation", None, None)
    admission = worker._capture_admission(request)
    if admission is None:
        return finish("no_history", "uncaptured", None, None)
    if admission.mode != "candidate":
        reason = admission.root_reason or "non_candidate"
        outcome = "initial" if reason == "first_call" else "no_history"
        return finish(outcome, reason, None, None)
    adapter = CapturedTitoRenderer(
        online_renderer.renderer, worker, admission, incremental_tokenizer, trace
    )
    local = copy.copy(online_renderer)
    local.renderer = adapter
    try:
        result = await OnlineRenderer.preprocess_chat(
            local,
            request=request,
            messages=messages,
            default_template=default_template,
            default_template_content_format=default_template_content_format,
            default_template_kwargs=default_template_kwargs,
            tool_dicts=tool_dicts,
            parser=parser,
            skip_mm_cache=skip_mm_cache,
        )
    except TitoNotApplicable as error:
        return finish("fallback", str(error), None, None)
    assert adapter.continuation is not None
    return finish("incremental", "matched", result, adapter.continuation)
