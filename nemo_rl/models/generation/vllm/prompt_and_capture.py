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

"""The steps of one Gym chat call after its render, shared by every serving path.

1. :func:`prepare_prompt` turns a render into a :class:`PreparedPrompt`: the
   token-capture prefix read from TQ, the splice, and the output budget
   (:mod:`~nemo_rl.models.generation.gym_prompt`, with vLLM's clamp).
2. On the engine's worker, :func:`start_capture` takes the prepared prompt to
   ``begin_call``: a restore from a checkpoint cut, the budget, the media
   snapshot, the placeholder remap and the final prompt; after generation,
   :func:`save_capture` stages the call in one TQ write.
3. :func:`abort_capture` fails the capture of a call that raised, and
   :func:`error_reply` says how to answer it.

The native vLLM server runs all of them in the vLLM worker. Behind Dynamo,
the frontend's chat engine prepares the prompt (its KV router needs the final
token ids before it picks an engine) and the engine's adapter runs the rest,
with the prepared prompt sent across as JSON (:meth:`PreparedPrompt.to_json`).
A restore is resolved on the worker: its weight-version check needs the
engine's own.

This module imports vLLM; import it only where vLLM is installed.
"""

import asyncio
import logging
from dataclasses import dataclass, replace
from typing import Any, Callable, Collection, Optional

import torch
from vllm.entrypoints.serve.exception_handling.error_response import (
    create_error_response,
)
from vllm.exceptions import GracefulHTTPError, VLLMClientError, VLLMValidationError

from nemo_rl.data.captured_media import MediaCaptureRejected
from nemo_rl.models.generation import gym_prompt
from nemo_rl.models.generation.gym_prompt import PreparedPrompt, RenderedPrompt
from nemo_rl.models.generation.prefix_recovery.restore import (
    _classify_restored_prefix_terminal,
    _restored_prefix_terminal_routes,
    _RestoredPrefixTerminal,
    mark_restored_terminal,
    plan_restore,
)
from nemo_rl.models.generation.vllm.utils import (
    OffPolicySamplingError,
    remap_multimodal_placeholders,
)

LOGGER = logging.getLogger(__name__)


def set_max_tokens(request: Any, max_tokens: int) -> None:
    """Set the request's max output tokens.

    Mutates the request in place. Handles both max_completion_tokens (newer OpenAI API)
    and max_tokens (deprecated but still supported by vLLM). A request
    that set neither gets max_tokens: otherwise vLLM would fall back to
    the generation config's budget, e.g. for a restored call whose
    remaining budget is smaller.
    """
    if request.max_completion_tokens is not None:
        request.max_completion_tokens = max_tokens
    else:
        request.max_tokens = max_tokens


def clamp_max_tokens(
    request_max_tokens: Optional[int], prompt_len: int, max_model_len: int
) -> Optional[int]:
    """The output budget that keeps input + output within ``max_model_len``.

    Refuses a prompt that leaves no room for output, whether or not the call set
    a budget, and lowers the budget only when it did.
    """
    remaining = max_model_len - prompt_len
    if remaining <= 0:
        # preserve the literal "context length" in this message to match Gym's overflow handling
        message = (
            f"Prompt length ({prompt_len}) fills or exceeds "
            f"this model's maximum context length ({max_model_len}). "
            f"No room for output tokens."
        )
        LOGGER.warning("Prompt exceeds max_model_len: %s", message)
        raise VLLMValidationError(message, parameter="input_tokens", value=prompt_len)
    if request_max_tokens is None:
        return None
    return min(request_max_tokens, remaining)


@dataclass(frozen=True)
class RestoredCall:
    """A call that resumes from the tokens a checkpoint cut saved."""

    # The verified saved rows, as capture's begin_call takes them.
    saved: Any
    # The engine prompt: the capture prefix, then the saved rows.
    prompt_token_ids: list[int]
    # The saved rows' generated tokens alone.
    resumed_generation_token_ids: list[int]
    # The whole output budget, as the cut call resolved it.
    output_limit: int
    # What is left of that budget; None keeps the request's own.
    max_tokens: Optional[int]
    # Router replay: the saved prefix's routes; None when cuts stage no routes.
    routed_experts: Optional[torch.Tensor] = None
    # Score centering: the saved prefix's sampler top-k (token ids, logprobs);
    # None when cuts stage none.
    sampler_topk: Optional[tuple[torch.Tensor, torch.Tensor]] = None


async def resolve_restored_call(
    host: Any, request: Any, serving_admission: Any, prefix_token_ids: list[int]
) -> tuple[Any, Optional[RestoredCall]]:
    """Resolve the prefix a checkpoint cut saved for this call.

    Returns the admission to serve and the restore. A prefix the worker cannot
    use comes back as the admission without its cut and None, so the call
    regenerates. For a restore, the request gets its restore fields (which the
    reply reads) and its remaining ``min_tokens``.
    """
    if serving_admission.generation_cut is None:
        return serving_admission, None
    # Off-loop: the saved rows are a blocking TQ read.
    resolved_admission, resolved = await asyncio.to_thread(
        host._prefix_recovery.resolve_restore, serving_admission, prefix_token_ids
    )
    if resolved is None:
        return resolved_admission, None
    saved = resolved.snapshot
    original_min_tokens = getattr(request, "min_tokens", None)
    plan = plan_restore(
        prefix_token_ids=prefix_token_ids,
        saved=saved,
        continuation=resolved_admission.generation_cut,
        min_tokens=original_min_tokens,
    )
    request._restored_generation_token_ids = plan.resumed_generation_token_ids
    request._restored_prompt_token_ids = tuple(plan.engine_prefix_token_ids)
    request._original_min_tokens = original_min_tokens
    if plan.min_tokens is not None:
        request.min_tokens = plan.min_tokens
    return resolved_admission, RestoredCall(
        saved=saved,
        prompt_token_ids=plan.engine_prefix_token_ids,
        resumed_generation_token_ids=list(plan.resumed_generation_token_ids),
        output_limit=plan.output_limit,
        max_tokens=plan.max_tokens,
        routed_experts=resolved.routed_experts,
        sampler_topk=resolved.sampler_topk,
    )


def classify_restored_terminal(
    restored: RestoredCall,
    prompt_token_ids: list[int],
    admission: Any,
    *,
    max_model_len: int,
) -> Optional[_RestoredPrefixTerminal]:
    """The terminal of a saved prefix that already finished its call, else None."""
    continuation = admission.generation_cut
    try:
        return _classify_restored_prefix_terminal(
            prompt_token_ids=prompt_token_ids,
            prefix_token_count=continuation.prefix_token_count,
            requested_output_tokens=restored.output_limit,
            model_max_tokens=max_model_len,
            terminal_finish_reason=continuation.terminal_finish_reason,
            terminal_stop_reason=continuation.terminal_stop_reason,
        )
    except ValueError as error:
        raise VLLMValidationError(
            str(error),
            parameter="generation_prefix",
            value=continuation.prefix_token_count,
        ) from error


def serve_restored_terminal(
    request: Any,
    engine_prompt: Any,
    terminal: _RestoredPrefixTerminal,
    admission: Any,
    *,
    max_model_len: int,
) -> None:
    """Have the engine client answer a finished prefix without decoding.

    Call it in the task that submits the call to the engine: the marker is a
    context variable.
    """
    # One token is the smallest budget vLLM accepts, so a missed marker costs
    # one token, not the whole call.
    request.min_tokens = 0
    set_max_tokens(request, 1)
    mark_restored_terminal(terminal)
    prompt_token_ids = engine_prompt["prompt_token_ids"]
    if len(prompt_token_ids) == max_model_len:
        engine_prompt["prompt_token_ids"] = prompt_token_ids[:-1]
    request._restored_prefix_terminal = terminal
    LOGGER.info(
        "generation prefix already terminal: "
        "rollout_id=%s model_call_id=%s prefix_tokens=%d reason=%s",
        admission.capture_key,
        admission.model_call_id,
        terminal.prefix_token_count,
        terminal.reason,
    )


async def abort_request_capture(host: Any, request: Any, reason: str) -> None:
    """Fail a request's capture without blocking the event loop.

    An uncontended abort finishes here, so it never queues behind
    executor work. Only one that must wait out a generation cut or a
    terminal write holding the call's lifecycle lock moves off the loop.
    """
    if host._abort_request_capture(request, reason=reason, blocking=False):
        return
    # Shielded: these paths often run after the handler was cancelled,
    # and a second cancel must not drop the abort before it starts,
    # leaving the call neither completed nor failed.
    await asyncio.shield(
        asyncio.to_thread(host._abort_request_capture, request, reason=reason)
    )


async def prepare_prompt(
    rendered: RenderedPrompt,
    *,
    resolve_prefix: Callable[[Any], list[int]],
    turn_end_token_ids: Collection[int],
    max_model_len: int,
    measure_only: bool,
    tokenizer: Any = None,
) -> PreparedPrompt:
    """:func:`~nemo_rl.models.generation.gym_prompt.prepare_prompt` with vLLM's clamp.

    Raises:
        VLLMValidationError: A generating call's prompt leaves no room for
            output. Raised before any media check, so the length error is the
            one the call answers with.
    """
    return await gym_prompt.prepare_prompt(
        rendered,
        resolve_prefix=resolve_prefix,
        turn_end_token_ids=turn_end_token_ids,
        max_model_len=max_model_len,
        measure_only=measure_only,
        clamp_max_tokens=clamp_max_tokens,
        tokenizer=tokenizer,
    )


async def start_capture(
    host: Any,
    request: Any,
    engine_prompt: dict[str, Any],
    plan: PreparedPrompt,
    *,
    max_model_len: int,
) -> None:
    """Take a prepared prompt to the start of its capture, on the engine's worker.

    In order: resolve a restore from a checkpoint cut (its saved rows replace
    the prepared prompt), set the output budget, snapshot the call's media for
    capture, make ``engine_prompt`` the final prompt (its media placeholders
    moved with the splice), begin the call's capture, and have the engine
    answer a restored call that had already finished.

    Args:
        host: The vLLM worker; it owns token capture.
        request: The chat request the engine serves.
        engine_prompt: The render's engine input, in the template's
            coordinates; updated in place.
        plan: The call's prepared prompt.
        max_model_len: The engine's context length.
    """
    admission = plan.admission
    serving_admission = plan.serving_admission
    if admission is not None and admission.mode == "candidate":
        raise RuntimeError(
            "a candidate admission reached the engine undecided; NeMo RL's "
            "render must decide it before capture"
        )
    prompt_token_ids = plan.prompt_token_ids
    splice = plan.splice
    max_tokens = plan.max_tokens
    restored = None
    terminal = None
    if serving_admission is not None:
        # A cut this worker cannot use comes back as None: the prepared prompt
        # stands, and the call regenerates.
        resolved_admission, restored = await resolve_restored_call(
            host,
            request,
            serving_admission,
            # The model's own prefix is the prepared prompt's first ids.
            prompt_token_ids[: plan.prefix_len]
            if serving_admission.mode == "token_in"
            else [],
        )
        if serving_admission is admission:
            admission = resolved_admission
        serving_admission = resolved_admission
    if restored is not None:
        # The saved rows are the exact engine prompt at the cut: never spliced or
        # truncated again (test_vllm_progress_read_layout.py).
        splice = None
        prompt_token_ids = restored.prompt_token_ids
        terminal = classify_restored_terminal(
            restored, prompt_token_ids, admission, max_model_len=max_model_len
        )
        if terminal is not None and restored.routed_experts is not None:
            # The engine never runs for a terminal prefix, so its output
            # carries the restored routes instead.
            terminal = replace(
                terminal,
                routed_experts=_restored_prefix_terminal_routes(
                    restored.routed_experts, prev_len=admission.prev_len
                ),
            )
        # A restored terminal prefix generates nothing more.
        max_tokens = None
        if not plan.measure_only and terminal is None:
            max_tokens = clamp_max_tokens(
                restored.max_tokens
                if restored.max_tokens is not None
                else plan.request_max_tokens,
                len(prompt_token_ids),
                max_model_len,
            )
    if max_tokens is not None:
        set_max_tokens(request, max_tokens)
    media = None
    media_admission = admission if splice is None else serving_admission
    if restored is None and media_admission is not None:
        # Off-loop: a retained-media check reads staged calls.
        media = await asyncio.to_thread(
            host._capture_request_media,
            engine_prompt,
            admission=media_admission,
            splice=splice,
            verify_retained_media=plan.verify_retained_media,
        )
    elif restored is not None:
        # The saved rows are the prompt, but its media still sits where the
        # prepared prompt's splice puts it. Off-loop: it hashes the pixels.
        media = await asyncio.to_thread(
            host._capture_restored_media,
            engine_prompt,
            admission=admission,
            splice=plan.splice,
            snapshot=restored.saved,
        )
    if splice is not None or restored is not None:
        # Successful media capture already remapped these ranges. The remap
        # reads the template's ids, so it runs before they are replaced.
        if media is None and (placeholders := engine_prompt.get("mm_placeholders")):
            engine_prompt["mm_placeholders"] = remap_multimodal_placeholders(
                template_token_ids=engine_prompt["prompt_token_ids"],
                final_token_ids=prompt_token_ids,
                mm_placeholders=placeholders,
                splice=splice,
            )
        engine_prompt["prompt_token_ids"] = prompt_token_ids
    host._begin_request_capture(
        request,
        prompt_token_ids,
        admission=admission,
        prefix_token_ids=plan.begin_call_prefix_token_ids,
        media=media,
        generation_cut=None if restored is None else restored.saved,
        generation_cut_routed_experts=(
            None if restored is None else restored.routed_experts
        ),
        generation_cut_sampler_topk=None if restored is None else restored.sampler_topk,
        resumed_generation_token_ids=(
            None if restored is None else restored.resumed_generation_token_ids
        ),
    )
    if terminal is not None:
        # The caller's task also submits the call: the engine client answers it.
        serve_restored_terminal(
            request, engine_prompt, terminal, admission, max_model_len=max_model_len
        )


async def save_capture(host: Any, request: Any, content: dict) -> dict:
    """Stage a finished call and ride its coords on ``content``.

    A no-op without capture. Off the loop: the sink write is a blocking TQ
    round trip.
    """
    try:
        return await asyncio.to_thread(host._finish_request_capture, request, content)
    except asyncio.CancelledError:
        # Queued executor work may never start after cancellation.
        # If staging already claimed the capture, abort is a no-op.
        await abort_request_capture(host, request, "client_disconnect")
        raise


@dataclass(frozen=True)
class ErrorReply:
    """How a call that failed before its reply is answered, on every path."""

    status: int
    # The OpenAI error body.
    message: str
    error_type: str
    param: Optional[str]
    code: Any
    # What the call's capture is failed with.
    abort_reason: str
    # Whether Gym is told its capture call was refused before it was staged.
    acknowledge: bool


def error_reply(error: Exception) -> Optional[ErrorReply]:
    """The answer to a call that raised ``error``; None for a server error (500)."""
    message = str(error)
    # Gym detects a context overflow by this text.
    overflow = "maximum context length" in message
    if isinstance(error, MediaCaptureRejected):
        # First, since it is a ValueError too. A stable code lets Gym and the
        # worker log tell a retained-media re-tile from an engine error.
        return ErrorReply(
            status=400,
            message=message,
            error_type="invalid_request_error",
            param="messages",
            code=error.code,
            abort_reason=error.code,
            acknowledge=False,
        )
    if isinstance(error, OffPolicySamplingError):
        return ErrorReply(
            status=400,
            message=message,
            error_type="invalid_request_error",
            param="sampling",
            code=400,
            abort_reason="invalid_request",
            acknowledge=False,
        )
    if isinstance(error, VLLMValidationError):
        return ErrorReply(
            status=400,
            message=message,
            error_type="invalid_request_error",
            param=error.parameter,
            code=400,
            abort_reason="context_length" if overflow else "invalid_request",
            acknowledge=True,
        )
    if isinstance(error, (VLLMClientError, GracefulHTTPError)):
        # vLLM's own status: 503 for a full queue, so Gym retries elsewhere.
        info = create_error_response(error).error
        return ErrorReply(
            status=info.code,
            message=info.message,
            error_type=info.type,
            param=info.param,
            code=info.code,
            abort_reason=(
                "invalid_request"
                if isinstance(error, VLLMClientError)
                else "engine_error"
            ),
            acknowledge=False,
        )
    if isinstance(error, ValueError) and overflow:
        # vLLM's get_max_tokens refuses an over-long prompt with a plain ValueError
        # ("Input length (N) exceeds ..."); only that refusal settles capture.
        return ErrorReply(
            status=400,
            message=message,
            error_type="invalid_request_error",
            param="input_tokens",
            code=400,
            abort_reason="context_length",
            acknowledge=message.startswith("Input length (")
            and "exceeds model's maximum context length" in message,
        )
    return None


async def abort_capture(
    host: Any, request: Any, error: BaseException
) -> Optional[ErrorReply]:
    """Fail the capture of a call that raised ``error``; say how to answer it.

    Returns None when the call fails as a server error or a disconnect: the
    caller re-raises. A no-op on capture once the call was staged.
    """
    if isinstance(error, asyncio.CancelledError):
        await abort_request_capture(host, request, "client_disconnect")
        return None
    failure = error_reply(error) if isinstance(error, Exception) else None
    await abort_request_capture(
        host, request, "engine_error" if failure is None else failure.abort_reason
    )
    if failure is not None and failure.status < 500:
        LOGGER.warning(
            "Refused a chat call before it ran (%s): %s",
            failure.abort_reason,
            failure.message,
        )
    return failure
