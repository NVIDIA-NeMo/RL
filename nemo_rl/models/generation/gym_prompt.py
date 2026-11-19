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

"""A NeMo-Gym chat call's prompt between its render and the engine, on every backend.

:func:`prepare_prompt` turns a render into a :class:`PreparedPrompt`: the
token-capture prefix read from TQ, the splice, and the output budget. The
native vLLM server runs it in the worker; behind Dynamo the frontend's chat
engine runs it for every backend, since its KV router needs the final token
ids before it picks an engine. Each backend passes its own
``clamp_max_tokens``, which raises the error type that backend answers a
context overflow with, and may pass a ``finish_prompt`` step that runs on the
spliced prompt before the clamp (Megatron's BOS rule).
"""

import asyncio
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from nemo_rl.models.generation.openai_server_utils import (
    PrefixSplice,
    splice_prefix_tokens,
)

# The output budget that keeps input + output within the context:
# (request_max_tokens, prompt_len, max_model_len) -> budget; raises when the
# prompt leaves no room for output.
ClampMaxTokens = Callable[[Optional[int], int, int], Optional[int]]
# A backend's last step on the spliced prompt, before the clamp:
# (prompt_token_ids, prefix_len) -> the ids the engine runs. With a model
# prefix (prefix_len not None) it must return the prompt unchanged or raise,
# since the splice's offsets and the capture's prefix index the prompt.
FinishPrompt = Callable[[list[int], Optional[int]], list[int]]


class RenderedPrompt(Protocol):
    """What :func:`prepare_prompt` reads from a call's render.

    Read-only members, so each path's frozen render dataclass fits.
    """

    # The full render's token ids, in the template's coordinates.
    @property
    def template_token_ids(self) -> list[int]: ...

    # The render up to the last assistant turn, when a prefix will be spliced.
    @property
    def history_token_ids(self) -> Optional[list[int]]: ...

    # The request's own prefix (Gym's echo) after the capture decision.
    @property
    def required_prefix_token_ids(self) -> Optional[list[int]]: ...

    # Capture admissions: what is stored, and how the prompt is served; None
    # without capture.
    @property
    def admission(self) -> Any: ...

    @property
    def serving_admission(self) -> Any: ...

    @property
    def verify_retained_media(self) -> bool: ...

    # The output budget the request asked for; None when it set none.
    @property
    def request_max_tokens(self) -> Optional[int]: ...


@dataclass(frozen=True)
class PreparedPrompt:
    """A call's prompt as the engine runs it, its output budget, and how capture admits it."""

    prompt_token_ids: list[int]
    # How the model's own prefix replaced the template's history; None if not.
    splice: Optional[PrefixSplice]
    # The model's own prefix is the prompt's first ``prefix_len`` ids.
    prefix_len: Optional[int]
    # Whether Gym only measures the prompt; such a call never generates.
    measure_only: bool
    # The output budget the request asked for, and the one after the clamp;
    # None when the request set none.
    request_max_tokens: Optional[int]
    max_tokens: Optional[int]
    admission: Any
    serving_admission: Any
    verify_retained_media: bool

    @property
    def begin_call_prefix_token_ids(self) -> Optional[list[int]]:
        """The prefix ``begin_call`` checks a token-in admission against."""
        if self.admission is None or self.admission.mode != "token_in":
            return None
        return self.prompt_token_ids[: self.prefix_len]

    def to_json(self) -> dict[str, Any]:
        """As JSON, without the prompt (it rides as the call's token ids)."""
        return {
            "splice": None
            if self.splice is None
            else [self.splice.model_cut_end, self.splice.template_cut_start],
            "prefix_len": self.prefix_len,
            "request_max_tokens": self.request_max_tokens,
            "max_tokens": self.max_tokens,
            "admission": _admission_json(self.admission),
            "serving_admission": _admission_json(self.serving_admission),
            "verify_retained_media": self.verify_retained_media,
        }

    @classmethod
    def from_json(
        cls,
        wire: dict[str, Any],
        *,
        prompt_token_ids: list[int],
        parse_admission: Callable[[Any], Any],
    ) -> "PreparedPrompt":
        """What :meth:`to_json` sent, for the prompt it rides beside."""
        splice = None
        if wire["splice"] is not None:
            splice = PrefixSplice(prompt_token_ids, *wire["splice"])
        admission = parse_admission(wire["admission"])
        # One object when storage and serving are one admission, as natively.
        serving_admission = admission
        if wire["serving_admission"] != wire["admission"]:
            serving_admission = parse_admission(wire["serving_admission"])
        return cls(
            prompt_token_ids=prompt_token_ids,
            splice=splice,
            prefix_len=wire["prefix_len"],
            # A measure call is answered where it is prepared.
            measure_only=False,
            request_max_tokens=wire["request_max_tokens"],
            max_tokens=wire["max_tokens"],
            admission=admission,
            serving_admission=serving_admission,
            verify_retained_media=wire["verify_retained_media"],
        )


def _admission_json(admission: Any) -> Optional[dict[str, Any]]:
    return None if admission is None else admission.model_dump(mode="json")


async def prepare_prompt(
    rendered: RenderedPrompt,
    *,
    resolve_prefix: Callable[[Any], list[int]],
    turn_end_token_ids: Collection[int],
    max_model_len: int,
    measure_only: bool,
    clamp_max_tokens: ClampMaxTokens,
    finish_prompt: Optional[FinishPrompt] = None,
    tokenizer: Any = None,
) -> PreparedPrompt:
    """Prepare a rendered call's prompt: the ids the engine runs, and its output budget.

    Args:
        rendered: The call's render.
        resolve_prefix: Reads a token-in admission's prefix from TQ. It blocks,
            so it runs off the event loop.
        turn_end_token_ids: Every id a turn can end on; the splice cuts by them.
        max_model_len: The engine's context length.
        measure_only: Whether Gym only measures the prompt. Such a call counts
            a prompt of any length; a generating call's budget is clamped to
            the context.
        clamp_max_tokens: The backend's clamp; it raises the backend's context
            overflow error for a prompt that leaves no room for output, before
            any media check, so the length error is the one the call answers
            with.
        finish_prompt: The backend's step on the spliced prompt, before the
            clamp; None leaves the prompt as spliced.
        tokenizer: Only for the splice's error message; may be None.
    """
    prefix = rendered.required_prefix_token_ids
    serving = rendered.serving_admission
    if serving is not None and serving.mode == "token_in":
        prefix = await asyncio.to_thread(resolve_prefix, serving)
    splice = None
    prompt_token_ids = rendered.template_token_ids
    if prefix is not None:
        # The render includes the history whenever a prefix will be spliced.
        assert rendered.history_token_ids is not None
        splice = splice_prefix_tokens(
            tokenizer=tokenizer,
            model_prefix_token_ids=list(prefix),
            template_prefix_token_ids=rendered.history_token_ids,
            template_token_ids=rendered.template_token_ids,
            eos_token_id=turn_end_token_ids,
        )
        prompt_token_ids = splice.token_ids
    prefix_len = None if prefix is None else len(prefix)
    if finish_prompt is not None:
        finished = finish_prompt(prompt_token_ids, prefix_len)
        if prefix_len is not None and finished != prompt_token_ids:
            raise RuntimeError(
                "finish_prompt changed a prompt with a model prefix; it must "
                "refuse the call instead"
            )
        prompt_token_ids = finished
    max_tokens = None
    if not measure_only:
        max_tokens = clamp_max_tokens(
            rendered.request_max_tokens, len(prompt_token_ids), max_model_len
        )
    return PreparedPrompt(
        prompt_token_ids=prompt_token_ids,
        splice=splice,
        prefix_len=prefix_len,
        measure_only=measure_only,
        request_max_tokens=rendered.request_max_tokens,
        max_tokens=max_tokens,
        admission=rendered.admission,
        serving_admission=rendered.serving_admission,
        verify_retained_media=rendered.verify_retained_media,
    )
