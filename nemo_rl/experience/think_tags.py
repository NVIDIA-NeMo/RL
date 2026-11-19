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
"""Where the malformed-think-tag penalty reads a turn's thinking mode.

The penalty infers whether a turn was generated with thinking on from the think-tag tokens in
the prompt that precedes it: one more open than close tag means the chat template opened
thinking (its header ``<|im_start|>assistant`` ends in ``<think>``), so the generation must close it exactly once;
equal counts mean thinking off. Only the generation prompt the template appends for this turn
sets that mode. The rest of the prompt delta -- tool results, files the agent read -- can carry
tag tokens of its own (a grep that matched a transcript, a subagent's answer that starts with
``<think>``), and counting those would flag correct turns. So the counts are taken from the
prompt's last turn header on.

``think_tag_violation`` is the per-turn rule, shared by the finalizer's rollout-level penalty and
``malformed_think_turns``, which finds the malformed turns of a trainer row for
``<algo>.malformed_think_credit``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, cast

import numpy as np

from nemo_rl.experience.tool_failures import single_token_id

# ChatML's turn-start token; reward_penalties.token_ids.turn_start overrides it.
DEFAULT_TURN_START = "<|im_start|>"

# How a turn breaks the format (think_tag_violation).
PROMPT_MODE = "prompt_mode"  # the generation prompt is neither thinking on nor off
STRAY_OPEN = "stray_open"  # an open tag in the generation
CLOSE_WITHOUT_THINKING = "close_without_thinking"  # a close tag in a thinking-off turn
DOUBLE_CLOSE = "double_close"  # thinking closed two or more times
UNCLOSED = "unclosed"  # thinking never closed in a turn that ended normally
# Thinking never closed in a generation stopped by a length limit (the context or the token budget):
# the turn was cut, not malformed.
CUT_UNCLOSED = "cut_unclosed"
VIOLATIONS = (PROMPT_MODE, STRAY_OPEN, CLOSE_WITHOUT_THINKING, DOUBLE_CLOSE, UNCLOSED)


def generation_prompt_start(
    prompt_ids: Sequence[int], turn_start: Optional[int]
) -> int:
    """Index of the prompt's last ``turn_start`` token, where this turn's generation prompt begins.

    0 (the whole prompt) without a ``turn_start`` id or when the prompt has none.
    """
    if turn_start is None:
        return 0
    for index in range(len(prompt_ids) - 1, -1, -1):
        if prompt_ids[index] == turn_start:
            return index
    return 0


def think_tag_violation(
    prompt_open: int, prompt_close: int, gen_open: int, gen_close: int, *, cut: bool
) -> Optional[str]:
    """How one turn's think-tag counts break the format, or None when they do not.

    The counts are the open / close tag tokens in the turn's generation prompt (from its last turn
    header) and in its generation. Balanced prompt tags mean thinking off (no tags generated); one
    trailing open tag means thinking on (exactly one close tag generated). ``cut``: the generation
    was stopped by a length limit, so a missing close tag is ``CUT_UNCLOSED`` rather than
    ``UNCLOSED``.
    """
    if prompt_open == prompt_close:
        expected = (0, 0)
    elif prompt_open == prompt_close + 1:
        expected = (0, 1)
    else:
        return PROMPT_MODE
    if (gen_open, gen_close) == expected:
        return None
    if gen_open:
        return STRAY_OPEN
    if expected == (0, 0):
        return CLOSE_WITHOUT_THINKING
    if gen_close:
        return DOUBLE_CLOSE
    return CUT_UNCLOSED if cut else UNCLOSED


@dataclass(frozen=True)
class ThinkTagIds:
    """Single-token ids ``malformed_think_turns`` scans for."""

    think_open: int
    think_close: int
    turn_start: int
    # Tokens a normally ended generation ends with (end of turn, end of sequence).
    turn_end: frozenset[int]


@dataclass
class RowMalformedThink:
    """``malformed_think_turns`` result for one row."""

    # Token-aligned bool array: True on every generated token of a malformed turn.
    mask: np.ndarray
    turns: int = 0
    malformed_turns: int = 0
    # Turns per violation (``VIOLATIONS`` and ``CUT_UNCLOSED``).
    kinds: dict[str, int] = field(default_factory=dict)


def malformed_think_turns(
    token_ids: Sequence[int], token_mask: Sequence[float], *, ids: ThinkTagIds
) -> RowMalformedThink:
    """Mark the generated tokens of every turn in one row whose think tags break the format.

    A turn is a run of generated tokens and its generation prompt is the stretch of
    non-generated tokens before it, from its last turn header on (the whole stretch when it has
    none), as the finalizer's per-call check reads them. A turn is malformed on any of
    ``VIOLATIONS``; a ``CUT_UNCLOSED`` turn (its last token is not in ``ids.turn_end``) is counted
    but not marked.

    Args:
        token_ids: the row's token ids (prompt, turns and environment replies).
        token_mask: 1 on generated tokens, 0 elsewhere.
        ids: the tag token ids.

    Returns:
        The token-aligned mask and the row's counts.
    """
    tokens = np.asarray(token_ids, dtype=np.int64)
    generated = np.asarray(token_mask, dtype=np.float32) > 0
    n = int(tokens.shape[0])
    result = RowMalformedThink(mask=np.zeros(n, dtype=bool))
    edges = np.diff(np.concatenate(([0], generated.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    result.turns = int(starts.shape[0])
    if not result.turns:
        return result
    opens = np.concatenate(([0], np.cumsum(tokens == ids.think_open)))
    closes = np.concatenate(([0], np.cumsum(tokens == ids.think_close)))
    headers = np.flatnonzero(~generated & (tokens == ids.turn_start))
    ending = np.isin(tokens[ends - 1], list(ids.turn_end))
    prompt_from = 0
    for t in range(result.turns):
        start, end = int(starts[t]), int(ends[t])
        h = int(np.searchsorted(headers, start)) - 1
        if h >= 0 and headers[h] >= prompt_from:
            prompt_from = int(headers[h])
        kind = think_tag_violation(
            int(opens[start] - opens[prompt_from]),
            int(closes[start] - closes[prompt_from]),
            int(opens[end] - opens[start]),
            int(closes[end] - closes[start]),
            cut=not bool(ending[t]),
        )
        prompt_from = end
        if kind is None:
            continue
        result.kinds[kind] = result.kinds.get(kind, 0) + 1
        if kind != CUT_UNCLOSED:
            result.mask[start:end] = True
            result.malformed_turns += 1
    return result


def resolve_think_tag_ids(tokenizer: Any) -> ThinkTagIds:
    """Ids of the tags ``malformed_think_turns`` scans for; each tag must be one token.

    A turn ends with the end-of-turn tag or the tokenizer's end-of-sequence token.
    """
    tags = {
        "think_open": "<think>",
        "think_close": "</think>",
        "turn_start": DEFAULT_TURN_START,
        "turn_end": "<|im_end|>",
    }
    ids = {key: single_token_id(tokenizer, tag) for key, tag in tags.items()}
    missing = [tags[key] for key, tid in ids.items() if tid is None]
    if missing:
        raise ValueError(
            "malformed_think_credit needs these tags to be single tokens of the policy "
            f"tokenizer: {missing}. Turns are checked by counting sampled token ids."
        )
    turn_end = {cast(int, ids["turn_end"])}
    if tokenizer.eos_token_id is not None:
        turn_end.add(int(tokenizer.eos_token_id))
    return ThinkTagIds(
        think_open=cast(int, ids["think_open"]),
        think_close=cast(int, ids["think_close"]),
        turn_start=cast(int, ids["turn_start"]),
        turn_end=frozenset(turn_end),
    )
