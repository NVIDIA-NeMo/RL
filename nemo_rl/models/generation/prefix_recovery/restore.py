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

"""Serve a restored call: plan its prompt and budgets, and assemble its reply.

Plain functions over a call's saved tokens, so any serving path can resume a
cut call: the native vLLM chat server today, and the Dynamo frontend (#4979).
The vLLM-specific pieces are the detokenizer replay, the finished-call output
and the parser wrap, which follow vLLM's own types.
"""

from __future__ import annotations

import contextvars
import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal, Optional


def _remaining_generation_limits_after_prefix(
    *,
    max_tokens: int,
    min_tokens: int | None,
    prefix_token_count: int,
) -> tuple[int, int | None]:
    """Return output limits for the suffix after restoring generated tokens."""
    return (
        max_tokens - prefix_token_count,
        None if min_tokens is None else max(0, min_tokens - prefix_token_count),
    )


def _bad_words_can_straddle_a_cut(sampling_params: Any, tokenizer: Any) -> bool:
    """Whether any bad word vLLM enforces spans more than one token.

    vLLM bans a one-token bad word at every step, whatever came before it, so a
    restored tail bans it exactly as the original call did. A longer one is
    banned only after its leading tokens appear in the call's own output, which
    after a restore no longer holds the prefix. The expansion of each word into
    token sequences (with and without a leading space) is vLLM's own.
    """
    params = sampling_params.clone()
    try:
        params.update_from_tokenizer(tokenizer)
    except Exception:  # noqa: BLE001 - vLLM rejects the request itself later
        return True
    return any(len(token_ids) > 1 for token_ids in params._bad_words_token_ids or ())


def _generation_cut_blockers(sampling_params: Any, tokenizer: Any) -> list[str]:
    """Sampling settings a resumed tail cannot honor across a generation cut.

    After a restore the prefix is part of the prompt, but vLLM matches stop
    strings and multi-token bad words over the output tokens only. A resumed
    tail would therefore miss a stop string or bad word that straddles the cut.
    (The chat route refuses presence and frequency penalties outright.)

    A request-level ``seed`` gives the call its own random generator, seeded
    when vLLM admits it. The restored call would start that generator over and
    reuse the draws that chose its prefix. Greedy decoding draws nothing, and
    an unseeded call draws from the engine's shared generator, so neither is
    affected; vLLM's ``SamplingType.RANDOM_SEED`` is exactly the affected case.
    """
    blockers = []
    if getattr(sampling_params, "stop", None):
        blockers.append("stop strings")
    if getattr(sampling_params, "bad_words", None) and _bad_words_can_straddle_a_cut(
        sampling_params, tokenizer
    ):
        blockers.append("multi-token bad_words")
    if getattr(sampling_params, "seed", None) is not None:
        from vllm.sampling_params import SamplingType

        if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
            blockers.append("seed")
    return blockers


# A restored call whose saved prefix already finished. Set the same way, when
# the chat is prepared, and read by the engine client, which answers it without
# decoding. It never rides in vLLM's prompt, which vLLM may rebuild.
_RESTORED_PREFIX_TERMINAL: contextvars.ContextVar[
    Optional["_RestoredPrefixTerminal"]
] = contextvars.ContextVar("nemo_rl_restored_prefix_terminal", default=None)


@dataclass(frozen=True)
class _RestoredPrefixTerminal:
    """A restored call whose durable output already reached a terminal limit."""

    prompt_token_ids: tuple[int, ...]
    prefix_token_count: int
    reason: str
    finish_reason: Literal["stop", "length"]
    stop_reason: str | int | None = None

    @property
    def original_prompt_token_count(self) -> int:
        return len(self.prompt_token_ids) - self.prefix_token_count


def _count_restored_prefix_as_completion(
    usage: Any,
    prompt_token_ids: Sequence[int] | None,
    *,
    restored_token_count: int,
    terminal: _RestoredPrefixTerminal | None,
) -> list[int] | None:
    """Report a restored call's usage the way the uninterrupted call would.

    The engine served the restored prefix as part of the prompt, so vLLM counts
    it as prompt tokens and only the regenerated tail as completion tokens,
    while the reasoning count covers prefix and tail. Gym adds usage across
    turns, so the prefix moves back to the completion side. Updates ``usage``
    in place and returns ``prompt_token_ids`` without the prefix.
    """
    if terminal is not None:
        # Nothing was generated after the restore: the prefix is the completion.
        prompt_tokens = terminal.original_prompt_token_count
        completion_tokens = terminal.prefix_token_count
    else:
        prompt_tokens = usage.prompt_tokens - restored_token_count
        completion_tokens = usage.completion_tokens + restored_token_count
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None and details.cached_tokens is not None:
        details.cached_tokens = min(details.cached_tokens, prompt_tokens)
    if prompt_token_ids is None:
        return None
    return list(prompt_token_ids[:prompt_tokens])


def _classify_restored_prefix_terminal(
    *,
    prompt_token_ids: list[int],
    prefix_token_count: int,
    requested_output_tokens: int | None,
    model_max_tokens: int,
    terminal_finish_reason: Literal["stop", "length"] | None = None,
    terminal_stop_reason: str | int | None = None,
) -> _RestoredPrefixTerminal | None:
    """Classify an exact terminal prefix or reject an incompatible restore.

    Gym's continuation model already guarantees a non-negative prefix count and
    a stop reason only alongside a finish reason.
    """
    prompt_token_count = len(prompt_token_ids)
    if prefix_token_count > prompt_token_count:
        raise ValueError(
            "Durable generation prefix contains more generated tokens than the "
            "restored engine prompt."
        )
    if (
        requested_output_tokens is not None
        and prefix_token_count > requested_output_tokens
    ):
        raise ValueError(
            "Durable generation prefix token count "
            f"({prefix_token_count}) exceeds the restored request output "
            f"budget ({requested_output_tokens})."
        )
    if prompt_token_count > model_max_tokens:
        raise ValueError(
            "Durable generation prefix prompt length "
            f"({prompt_token_count}) exceeds restored model capacity "
            f"({model_max_tokens})."
        )
    if terminal_finish_reason is not None:
        return _RestoredPrefixTerminal(
            prompt_token_ids=tuple(prompt_token_ids),
            prefix_token_count=prefix_token_count,
            reason=f"observed_{terminal_finish_reason}",
            finish_reason=terminal_finish_reason,
            stop_reason=terminal_stop_reason,
        )
    reached_output_limit = (
        requested_output_tokens is not None
        and prefix_token_count == requested_output_tokens
    )
    reached_model_capacity = prompt_token_count == model_max_tokens
    if not reached_output_limit and not reached_model_capacity:
        return None
    return _RestoredPrefixTerminal(
        prompt_token_ids=tuple(prompt_token_ids),
        prefix_token_count=prefix_token_count,
        reason=(
            "output_and_model_limit"
            if reached_output_limit and reached_model_capacity
            else "output_limit"
            if reached_output_limit
            else "model_limit"
        ),
        finish_reason="length",
    )


def _build_restored_prefix_terminal_output(
    request_id: str, terminal: _RestoredPrefixTerminal
) -> Any:
    """Build the final vLLM output for a prefix that needs no more decoding."""
    from vllm.outputs import CompletionOutput, RequestOutput

    return RequestOutput(
        request_id=request_id,
        prompt=None,
        prompt_token_ids=list(terminal.prompt_token_ids),
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="",
                token_ids=[],
                cumulative_logprob=0.0,
                logprobs=[],
                finish_reason=terminal.finish_reason,
                stop_reason=terminal.stop_reason,
            )
        ],
        finished=True,
    )


def _detokenize_restored_prefix(
    tokenizer: Any,
    *,
    original_prompt_token_ids: Sequence[int],
    prefix_token_ids: Sequence[int],
    terminal: _RestoredPrefixTerminal | None,
    stop: str | Sequence[str] | None,
    min_tokens: int,
    include_stop_str_in_output: bool,
    skip_special_tokens: bool,
    spaces_between_special_tokens: bool,
) -> str:
    """Return the text vLLM produced for a restored prefix the first time.

    The prefix is replayed through vLLM's own incremental detokenizer, primed
    on the call's original prompt: a standalone decode drops the leading space
    of SentencePiece tokenizers. Replaying with the call's stop settings also
    reapplies what vLLM stripped from an output that had already stopped: the
    final stop token (EOS or a ``stop_token_ids`` entry) and a matched stop
    string.
    """
    # Deferred: vLLM is only installed in the generation worker's environment.
    from vllm.v1.engine.detokenizer import IncrementalDetokenizer

    stop_token_terminated = (
        terminal is not None
        and terminal.finish_reason == "stop"
        and not isinstance(terminal.stop_reason, str)
    )
    # The detokenizer reads only these request and sampling fields.
    sampling_params = SimpleNamespace(
        stop=stop,
        min_tokens=min_tokens,
        include_stop_str_in_output=include_stop_str_in_output,
        skip_special_tokens=skip_special_tokens,
        spaces_between_special_tokens=spaces_between_special_tokens,
    )
    detokenizer = IncrementalDetokenizer.from_new_request(
        tokenizer,
        SimpleNamespace(
            request_id="restored-prefix",
            prompt_token_ids=list(original_prompt_token_ids),
            prompt_embeds=None,
            sampling_params=sampling_params,
        ),
    )
    detokenizer.update(list(prefix_token_ids), stop_token_terminated)
    return detokenizer.get_next_output_text(finished=True, delta=False)


@dataclass(frozen=True)
class _RestoredPrefixParser:
    """Give vLLM's parser the complete output while capture keeps tail deltas."""

    delegate: Any
    prefix_token_ids: tuple[int, ...]

    def parse(
        self,
        model_output: str,
        request: Any,
        *,
        enable_auto_tools: bool,
        model_output_token_ids: list[int],
    ) -> Any:
        return self.delegate.parse(
            model_output,
            request,
            enable_auto_tools=enable_auto_tools,
            model_output_token_ids=[
                *self.prefix_token_ids,
                *model_output_token_ids,
            ],
        )

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        return self.delegate.count_reasoning_tokens(
            [*self.prefix_token_ids, *token_ids]
        )

    def __getattr__(self, name: str) -> Any:
        # Anything else vLLM reads from its parser is the delegate's.
        if name == "delegate":
            raise AttributeError(name)
        return getattr(self.delegate, name)


# chat_completion_full_generator arguments a restored call rewrites or reads.
_FULL_GENERATOR_RESTORE_PARAMETERS = ("parser", "request_metadata")


def _full_generator_signature(method: Callable[..., Any]) -> inspect.Signature:
    """The signature of vLLM's ``chat_completion_full_generator``, checked at setup.

    A restored call wraps ``parser`` so tool calls and reasoning are parsed
    from the whole output, and writes the corrected usage to
    ``request_metadata``. Both are bound by name, so a vLLM that renames or
    drops either fails here instead of silently on restored calls.
    """
    signature = inspect.signature(method)
    missing = [
        name
        for name in _FULL_GENERATOR_RESTORE_PARAMETERS
        if name not in signature.parameters
    ]
    if missing:
        raise RuntimeError(
            "vLLM's OpenAIServingChat.chat_completion_full_generator has no "
            f"{missing} parameter, which restored generation prefixes rewrite; "
            "update NeMoRLOpenAIServingChatMixin.chat_completion_full_generator"
        )
    return signature


@dataclass(frozen=True)
class RestorePlan:
    """How to serve a call that resumes from its saved generation prefix."""

    # The prompt the engine decodes from: the call's context, then the saved
    # rows (the context's prompt tokens and the generated prefix).
    engine_prefix_token_ids: list[int]
    # The generated prefix alone, which the reply's text and usage count.
    resumed_generation_token_ids: tuple[int, ...]
    # The call's whole output budget, as the engine resolved it originally.
    output_limit: int
    # What is left to decode: None keeps the request's own max_tokens.
    max_tokens: int | None
    # None leaves an unset min_tokens unset.
    min_tokens: int | None


def plan_restore(
    *,
    prefix_token_ids: Sequence[int],
    saved: Any,
    continuation: Any,
    min_tokens: int | None,
) -> RestorePlan:
    """Plan a restored call's engine prompt and remaining budgets.

    Args:
        prefix_token_ids: The call's context, resolved as for any captured call.
        saved: The verified saved rows (``token_ids_delta``/``token_mask_delta``).
        continuation: The admission's generation cut (``effective_output_limit``
            and ``prefix_token_count``).
        min_tokens: The request's own ``min_tokens``.
    """
    remaining_output, remaining_min = _remaining_generation_limits_after_prefix(
        max_tokens=continuation.effective_output_limit,
        min_tokens=min_tokens,
        prefix_token_count=continuation.prefix_token_count,
    )
    return RestorePlan(
        engine_prefix_token_ids=[*prefix_token_ids, *saved.token_ids_delta],
        resumed_generation_token_ids=tuple(
            token_id
            for token_id, mask in zip(
                saved.token_ids_delta, saved.token_mask_delta, strict=True
            )
            if mask == 1.0
        ),
        output_limit=continuation.effective_output_limit,
        # A prefix that used the whole budget is answered without decoding.
        max_tokens=remaining_output if remaining_output > 0 else None,
        min_tokens=remaining_min,
    )


def mark_restored_terminal(terminal: _RestoredPrefixTerminal) -> None:
    """Tell the engine client this call's saved prefix already finished it."""
    _RESTORED_PREFIX_TERMINAL.set(terminal)


def restored_terminal() -> _RestoredPrefixTerminal | None:
    """The finished restored prefix this request task serves, if any."""
    return _RESTORED_PREFIX_TERMINAL.get()


def restore_reply_text(
    request: Any,
    request_output: Any,
    *,
    prefix_token_ids: Sequence[int],
    engine_prompt_token_ids: Sequence[int],
    tokenizer: Any,
) -> None:
    """Prepend a resumed call's restored text to its final output's text.

    The engine already detokenized the tail with the request's settings and
    stripped EOS and any matched stop string from it. Re-decoding the whole
    sequence would restore both, so only the prefix is replayed, with the same
    settings, and prepended to the processed tail. A call that had already
    stopped has no tail, so its stop is stripped here.
    """
    outputs = getattr(request_output, "outputs", None)
    if not outputs:
        return
    output = outputs[0]
    # The engine prompt ends with the spliced prefix; what precedes it is the
    # prompt the engine decoded the prefix against the first time.
    if len(engine_prompt_token_ids) < len(prefix_token_ids):
        raise RuntimeError(
            "restored engine prompt is shorter than its generation prefix"
        )
    original_min_tokens = getattr(request, "_original_min_tokens", None)
    prefix_text = _detokenize_restored_prefix(
        tokenizer,
        original_prompt_token_ids=list(
            engine_prompt_token_ids[
                : len(engine_prompt_token_ids) - len(prefix_token_ids)
            ]
        ),
        prefix_token_ids=list(prefix_token_ids),
        terminal=getattr(request, "_restored_prefix_terminal", None),
        stop=getattr(request, "stop", None),
        min_tokens=(
            original_min_tokens
            if original_min_tokens is not None
            else getattr(request, "min_tokens", None) or 0
        ),
        include_stop_str_in_output=bool(
            getattr(request, "include_stop_str_in_output", False)
        ),
        skip_special_tokens=getattr(request, "skip_special_tokens", True),
        spaces_between_special_tokens=getattr(
            request, "spaces_between_special_tokens", True
        ),
    )
    output.text = prefix_text + (getattr(output, "text", None) or "")


def restore_reply_usage(
    response: Any,
    *,
    restored_token_count: int,
    terminal: _RestoredPrefixTerminal | None,
) -> bool:
    """Count a resumed call's restored tokens as completion in its reply.

    Returns whether the usage changed, so the caller can pass it on.
    """
    if response.usage is None or not (terminal is not None or restored_token_count):
        return False
    prompt_token_ids = response.prompt_token_ids
    if prompt_token_ids is not None and terminal is not None:
        # Its engine prompt may have dropped a token to fit max_model_len; the
        # terminal record keeps the full one.
        prompt_token_ids = terminal.prompt_token_ids
    prompt_token_ids = _count_restored_prefix_as_completion(
        response.usage,
        prompt_token_ids,
        restored_token_count=restored_token_count,
        terminal=terminal,
    )
    if prompt_token_ids is not None:
        response.prompt_token_ids = prompt_token_ids
    return True


def wrap_parser_for_restore(parser: Any, prefix_token_ids: Sequence[int]) -> Any:
    """Give the response parser the whole output, restored prefix first."""
    if not prefix_token_ids or parser is None:
        return parser
    return _RestoredPrefixParser(
        delegate=parser, prefix_token_ids=tuple(prefix_token_ids)
    )
