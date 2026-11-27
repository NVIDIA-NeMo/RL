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
"""Incremental continuation of captured chat requests on the vLLM HTTP server."""

import asyncio
import json
import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from nemo_rl.models.generation.capture_context import (
    CaptureInputDecision,
    decide_capture_input,
)
from nemo_rl.models.generation.tito import (
    IncrementalTokenizer,
    TitoNotApplicable,
)

LOGGER = logging.getLogger(__name__)

_OUTCOME_SUMMARY_INTERVAL = 1000

# vLLM's name for prompt token IDs that a chat request carries past rendering.
REUSED_PROMPT_FIELD = "prompt_token_ids"

# Request options that change the prompt or need the rendered conversation.
_BRIDGED_REQUEST_OPTIONS = (
    "stream",
    "echo",
    "continue_final_message",
    "return_assistant_tokens_mask",
    "add_special_tokens",
    "return_prompt_text",
    "return_token_offsets",
    "kv_transfer_params",
    "chat_template",
    "reasoning_effort",
)


@dataclass(frozen=True)
class IncrementalContinuation:
    """A captured parent and the tokens that continue it without a render."""

    decision: CaptureInputDecision
    parent_token_ids: list[int]
    suffix_token_ids: list[int]

    @property
    def prompt_token_ids(self) -> list[int]:
        return self.parent_token_ids + self.suffix_token_ids


@dataclass
class TitoTrace:
    """Per-attempt measurements emitted as a structured debug log."""

    outcome: str = "bridge"
    reason: str = ""
    reused_tokens: int = 0
    suffix_tokens: int = 0
    history_seconds: float = 0.0
    suffix_seconds: float = 0.0
    total_seconds: float = 0.0


@dataclass
class TitoOutcomes:
    """Tally of preprocessing outcomes, summarized periodically in the log."""

    counts: Counter[str] = field(default_factory=Counter)

    def record(self, trace: TitoTrace) -> None:
        self.counts[f"{trace.outcome}/{trace.reason}"] += 1
        total = self.counts.total()
        if total % _OUTCOME_SUMMARY_INTERVAL == 0:
            LOGGER.info(
                "Captured TITO outcomes over %d requests: %s",
                total,
                json.dumps(dict(self.counts), sort_keys=True),
            )


# One HTTP server process owns one tally.
_OUTCOMES = TitoOutcomes()


async def _plan(
    *,
    incremental_tokenizer: IncrementalTokenizer,
    tokenizer: Any,
    worker: Any,
    request: Any,
    messages: list[dict[str, Any]],
    default_template_kwargs: dict[str, Any] | None,
    trace: TitoTrace,
) -> IncrementalContinuation:
    for option in _BRIDGED_REQUEST_OPTIONS:
        if getattr(request, option, None):
            raise TitoNotApplicable(option)
    if getattr(request, "n", 1) not in (None, 1):
        raise TitoNotApplicable("multiple_choices")
    if getattr(request, "truncate_prompt_tokens", None) is not None:
        raise TitoNotApplicable("prompt_truncation")
    if not getattr(request, "add_generation_prompt", False):
        raise TitoNotApplicable("missing_generation_prompt")
    admission = worker._capture_admission(request)
    if admission is None:
        raise TitoNotApplicable("uncaptured")
    if admission.mode != "candidate":
        raise TitoNotApplicable(admission.root_reason or "non_candidate")
    if admission.generation_cut is not None:
        raise TitoNotApplicable("generation_restore")
    if any(
        message.get("role") not in ("system", "user", "assistant", "tool")
        or (
            message.get("content") is not None
            and not isinstance(message["content"], str)
        )
        for message in messages
    ):
        raise TitoNotApplicable("unsupported_content")
    decision = decide_capture_input(admission, messages=messages)
    serving = decision.serving
    if serving is None or serving.mode != "token_in":
        raise TitoNotApplicable("unverified_boundary")
    assert admission.request_replay is not None
    assert admission.candidate_replay is not None
    suffix_count = (
        len(admission.request_replay.items) - admission.candidate_replay.item_count
    )
    started = time.perf_counter()
    parent = await asyncio.to_thread(worker._resolve_admission_prefix, serving)
    trace.history_seconds = time.perf_counter() - started
    if len(parent) != serving.prev_len:
        raise ValueError(
            f"Parent {serving.parent_call_id} resolved to {len(parent)} tokens "
            f"but the admission records {serving.prev_len}"
        )
    incremental_tokenizer.validate_prefix(tokenizer, parent)
    template_kwargs = {
        **(default_template_kwargs or {}),
        **(getattr(request, "chat_template_kwargs", None) or {}),
    }
    started = time.perf_counter()
    suffix = await asyncio.to_thread(
        incremental_tokenizer.encode_suffix,
        tokenizer,
        messages[len(messages) - suffix_count :],
        enable_thinking=template_kwargs.get("enable_thinking") is not False,
    )
    trace.suffix_seconds = time.perf_counter() - started
    trace.reused_tokens = len(parent)
    trace.suffix_tokens = len(suffix)
    return IncrementalContinuation(decision, parent, suffix)


async def plan_incremental_continuation(
    *,
    incremental_tokenizer: IncrementalTokenizer,
    tokenizer: Any,
    worker: Any,
    request: Any,
    messages: list[dict[str, Any]],
    default_template_kwargs: dict[str, Any] | None,
) -> IncrementalContinuation | None:
    """Build a captured chat continuation without rendering its history.

    Args:
        incremental_tokenizer: The suffix encoder of the served model family.
        tokenizer: The server's Hugging Face tokenizer.
        worker: The generation worker that hosts token capture.
        request: The chat completion request.
        messages: The request's messages as the client sent them.
        default_template_kwargs: The server's default chat-template arguments.

    Returns:
        The parent's tokens and the tokens that continue it, or ``None`` when
        the request has to be rendered and bridged.

    Raises:
        ValueError: The parent's stored tokens differ in length from the
            admission.
    """
    started = time.perf_counter()
    trace = TitoTrace()
    continuation = None
    try:
        continuation = await _plan(
            incremental_tokenizer=incremental_tokenizer,
            tokenizer=tokenizer,
            worker=worker,
            request=request,
            messages=messages,
            default_template_kwargs=default_template_kwargs,
            trace=trace,
        )
        trace.outcome, trace.reason = "incremental", "matched"
    except TitoNotApplicable as reason:
        trace.reason = str(reason)
    trace.total_seconds = time.perf_counter() - started
    _OUTCOMES.record(trace)
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("Captured TITO %s", json.dumps(asdict(trace), sort_keys=True))
    return continuation
