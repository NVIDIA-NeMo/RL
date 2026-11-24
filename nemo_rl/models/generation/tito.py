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
"""Token-in/token-out inference contract shared by generation backends.

The core accepts token IDs and returns token IDs. A chat adapter in front of it
renders messages, bridges a continuation onto its captured parent and parses the
reply. See ``docs/design-docs/tito.md``.
"""

from collections.abc import Collection
from typing import Any

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt

TITO_GENERATE_ROUTE = "/v1/tito/generate"

QUALIFICATION_HISTORY: list[dict[str, Any]] = [
    {"role": "system", "content": "Fix the repository."},
    {"role": "user", "content": "Run the tests."},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "bash", "arguments": {"cmd": "pytest"}},
            }
        ],
    },
]
QUALIFICATION_OBSERVATIONS: list[list[dict[str, Any]]] = [
    [{"role": "tool", "content": "1 failed"}],
    [{"role": "tool", "content": "first output"}, {"role": "tool", "content": ""}],
    [{"role": "user", "content": "try another patch"}],
    [
        {"role": "tool", "content": "first output"},
        {"role": "user", "content": "continue"},
        {"role": "tool", "content": "second output"},
    ],
]
# Templates may drop earlier reasoning once the conversation grows, so the
# parent's tokens and the re-rendered history differ before the turn close.
_QUALIFICATION_REASONING = "The tests have not been run yet."
_QUALIFICATION_HISTORY_TEXT = (
    "Fix the repository.",
    "Run the tests.",
    "pytest",
    _QUALIFICATION_REASONING,
)


class TitoBridgeError(ValueError):
    """The rendered history has no turn close where the captured parent ends."""


class TitoRootRejected(ValueError):
    """A call would start a root for a reason the server is configured to reject."""

    code = "tito_root_rejected"

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"The call would re-render its history from text ({reason}), which "
            "vllm_cfg.tito_reject_root_reasons rejects"
        )


class TitoGenerateRequest(BaseModel, extra="forbid"):
    """One token-in/token-out model call.

    The engine prompt is the parent's prompt and generated tokens followed by
    ``token_ids``. ``ng_capture`` is Gym's capture admission: a ``token_in``
    admission names the parent, a ``text`` admission starts a root. Without
    ``ng_capture`` the call is not recorded and ``token_ids`` is the full prompt.
    """

    token_ids: list[NonNegativeInt] = Field(min_length=1)
    ng_capture: dict[str, Any] | None = None
    temperature: float
    top_p: float
    top_k: int | None = None
    max_tokens: PositiveInt | None = None
    seed: int | None = None


class TitoGenerateResponse(BaseModel):
    """Tokens the engine sampled for one :class:`TitoGenerateRequest`."""

    call_id: str | None
    prompt_token_count: int
    token_ids: list[int]
    logprobs: list[float]
    finish_reason: str | None
    stop_reason: int | str | None
    ng_commit_coords: dict[str, Any] | None


def bridge_suffix(
    *,
    parent_token_ids: list[int],
    rendered_token_ids: list[int],
    turn_close_token_ids: Collection[int],
) -> list[int]:
    """Return the tokens that continue a captured parent in a re-rendered history.

    The parent ends its assistant turn on a turn-close token, or without one when
    generation was cut short. ``rendered_token_ids`` is the chat template's
    rendering of the same history plus the appended messages. The turn that the
    parent closes is located by counting turn-close tokens, so tokens the
    template rewrites before that close, such as dropped reasoning, do not move
    the cut.

    Args:
        parent_token_ids: The parent's prompt followed by its generated tokens.
        rendered_token_ids: The full chat-template rendering of the request.
        turn_close_token_ids: Every token ID that can end a turn.

    Returns:
        The tokens to append after ``parent_token_ids``. They start with the
        template's turn close when the parent did not generate one.

    Raises:
        TitoBridgeError: The rendering has fewer turn closes than the parent.
    """
    closes = frozenset(turn_close_token_ids)
    if not closes:
        raise ValueError("turn_close_token_ids must not be empty")
    if not parent_token_ids:
        raise ValueError("parent_token_ids must not be empty")
    parent_closed = parent_token_ids[-1] in closes
    needed = sum(token_id in closes for token_id in parent_token_ids)
    if not parent_closed:
        needed += 1
    seen = 0
    for position, token_id in enumerate(rendered_token_ids):
        if token_id in closes:
            seen += 1
            if seen == needed:
                start = position + 1 if parent_closed else position
                return rendered_token_ids[start:]
    raise TitoBridgeError(
        f"The rendered history has {seen} turn-close tokens but the captured "
        f"parent ends at turn close {needed}"
    )


def _render(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    chat_template: str | None,
    template_kwargs: dict[str, Any],
    add_generation_prompt: bool,
) -> list[int]:
    text = tokenizer.apply_chat_template(
        messages,
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **template_kwargs,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def _closed_turn(token_ids: list[int], closes: frozenset[int]) -> list[int]:
    """Cut a rendering that ends in an assistant turn right after its close."""
    for position in range(len(token_ids) - 1, -1, -1):
        if token_ids[position] in closes:
            return token_ids[: position + 1]
    raise ValueError("the chat template does not close an assistant turn")


def qualify_bridge(
    *,
    tokenizer: Any,
    chat_template: str | None,
    template_kwargs: dict[str, Any],
    turn_close_token_ids: Collection[int],
) -> None:
    """Reject a chat template that :func:`bridge_suffix` cannot continue exactly.

    Replays fixed conversations and requires of every one that the bridged
    tokens hold the appended messages in order and nothing of the history, that
    they do not depend on what the parent generated, and that a parent which
    ended without a turn close gets the template's close.

    Args:
        tokenizer: The server's Hugging Face tokenizer.
        chat_template: The server's chat template; ``None`` uses the tokenizer's.
        template_kwargs: The server's default chat-template keyword arguments.
        turn_close_token_ids: Every token ID that can end a turn.

    Raises:
        ValueError: The template fails one of the properties.
    """
    closes = frozenset(turn_close_token_ids)

    def render(
        messages: list[dict[str, Any]], *, add_generation_prompt: bool
    ) -> list[int]:
        return _render(
            tokenizer,
            messages,
            chat_template=chat_template,
            template_kwargs=template_kwargs,
            add_generation_prompt=add_generation_prompt,
        )

    reasoning_history = [
        *QUALIFICATION_HISTORY[:-1],
        {**QUALIFICATION_HISTORY[-1], "reasoning_content": _QUALIFICATION_REASONING},
    ]
    parent = _closed_turn(
        render(QUALIFICATION_HISTORY, add_generation_prompt=False), closes
    )
    reasoning_parent = _closed_turn(
        render(reasoning_history, add_generation_prompt=False), closes
    )
    for observations in QUALIFICATION_OBSERVATIONS:
        roles = [message["role"] for message in observations]
        rendered = render(
            QUALIFICATION_HISTORY + observations, add_generation_prompt=True
        )
        try:
            suffix = bridge_suffix(
                parent_token_ids=parent,
                rendered_token_ids=rendered,
                turn_close_token_ids=closes,
            )
            unclosed_suffix = bridge_suffix(
                parent_token_ids=parent[:-1],
                rendered_token_ids=rendered,
                turn_close_token_ids=closes,
            )
            reasoning_suffix = bridge_suffix(
                parent_token_ids=reasoning_parent,
                rendered_token_ids=render(
                    reasoning_history + observations, add_generation_prompt=True
                ),
                turn_close_token_ids=closes,
            )
        except TitoBridgeError as error:
            raise ValueError(
                f"The chat template cannot be bridged for appended {roles}: {error}"
            ) from error
        text = tokenizer.decode(suffix)
        position = 0
        for message in observations:
            position = text.find(message["content"], position)
            if position < 0:
                raise ValueError(
                    f"The tokens bridged for appended {roles} do not hold the "
                    "appended messages in order"
                )
            position += len(message["content"])
        if any(history_text in text for history_text in _QUALIFICATION_HISTORY_TEXT):
            raise ValueError(
                f"The tokens bridged for appended {roles} repeat the history"
            )
        if reasoning_suffix != suffix:
            raise ValueError(
                f"The tokens bridged for appended {roles} depend on the parent's "
                "generated tokens"
            )
        if len(unclosed_suffix) != len(suffix) + 1 or unclosed_suffix[1:] != suffix:
            raise ValueError(
                f"Bridging appended {roles} after an unfinished turn does not "
                "add the chat template's turn close"
            )
