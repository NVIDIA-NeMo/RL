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
r"""Per-turn THINKING / ACTION / CONTENT token classes for token-capture rows.

On the token-capture path the trainer row is assembled from captured model calls: every run of
generated tokens (``token_mask`` == 1) is one assistant turn (one model call). ``turn_text_columns``
labels each generated token of a turn, and the result becomes the token-aligned
``assistant_turn_id`` / ``thinking_mask`` / ``action_mask`` columns that the per-turn scopes of
``grpo.adv_estimator.positive_length_reweight`` read.

The thinking state of a turn comes from the chat template's generation prompt: with thinking on
it ends with ``<think>\n`` (the last think tag before the turn is ``<think>``), with thinking off
with ``<think></think>``.
"""

from __future__ import annotations

from collections.abc import Sequence

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
END_OF_TURN = "<|im_end|>"


def turn_text_columns(
    token_ids: Sequence[int],
    token_mask: Sequence[float],
    *,
    think_open_id: int | None,
    think_close_id: int | None,
    tool_call_open_id: int,
    tool_call_close_id: int,
    end_of_turn_id: int,
) -> tuple[list[int], list[bool], list[bool]]:
    """Token-aligned ``(assistant_turn_id, thinking, action)`` columns for one row.

    The per-turn scopes of ``grpo.adv_estimator.positive_length_reweight`` read these. Every run
    of generated tokens (``token_mask`` == 1) is one assistant turn, numbered from 0; every other
    position gets turn -1 and no class. Within a turn each generated token is exactly one of:

    * THINKING: from the turn's first token through the last ``</think>`` of the turn, inclusive,
      when the turn was generated with thinking on (read from the last ``<think>`` / ``</think>``
      tag before it). Everything before that ``</think>`` is reasoning, including tool-call
      markup and earlier ``</think>`` tags. A thinking turn with no ``</think>`` is all reasoning
      (its tool-call markup is never executed). With ``think_close_id`` None or thinking off
      there is no thinking span.
    * ACTION: each ``<tool_call>`` ... ``</tool_call>`` span after the thinking, inclusive (an
      unclosed span runs to the end of the turn), and the turn's last token when it is the
      end-of-turn token (in the final turn, the decision to end the episode).
    * CONTENT: every other generated token (free text before, between or after tool calls).

    All tags must be single tokens (``RolloutReassembler`` checks this against the tokenizer):
    on the capture path the row holds the sampled token ids, so the scan is a pure id match.
    """
    n = len(token_ids)
    turn_id = [-1] * n
    thinking = [False] * n
    action = [False] * n
    last_tag_open = True
    turn = 0
    i = 0
    while i < n:
        if not token_mask[i]:
            if think_open_id is not None and token_ids[i] == think_open_id:
                last_tag_open = True
            elif think_close_id is not None and token_ids[i] == think_close_id:
                last_tag_open = False
            i += 1
            continue
        j = i
        while j < n and token_mask[j]:
            j += 1
        for k in range(i, j):
            turn_id[k] = turn
        ends_turn = token_ids[j - 1] == end_of_turn_id
        body_end = (
            j - 1 if ends_turn else j
        )  # exclusive; the end-of-turn token is ACTION
        start = i
        if think_close_id is not None and (think_open_id is None or last_tag_open):
            close = next(
                (
                    k
                    for k in range(body_end - 1, i - 1, -1)
                    if token_ids[k] == think_close_id
                ),
                None,
            )
            stop = body_end if close is None else close + 1
            for k in range(i, stop):
                thinking[k] = True
            start = stop
        inside = False
        for k in range(start, body_end):
            if not inside and token_ids[k] == tool_call_open_id:
                inside = True
            if inside:
                action[k] = True
                if token_ids[k] == tool_call_close_id:
                    inside = False
        if ends_turn:
            action[j - 1] = True
        turn += 1
        i = j
    return turn_id, thinking, action
