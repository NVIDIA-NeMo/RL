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
"""Tests for the per-turn THINKING / ACTION / CONTENT token classes (``turn_text_columns``)."""

import pytest

from nemo_rl.experience.rollout_reassembler import resolve_turn_text_ids
from nemo_rl.experience.turn_text import turn_text_columns

# Single-token tag ids, as in the Nemotron tokenizer; ordinary text is any id >= 100.
IM_END, THINK, ETHINK, TC, ETC = 11, 12, 13, 14, 15
IDS = dict(
    think_open_id=THINK,
    think_close_id=ETHINK,
    tool_call_open_id=TC,
    tool_call_close_id=ETC,
    end_of_turn_id=IM_END,
)


def _seq(*segments):
    """Concatenate (generated: bool, ids) segments into token_ids / token_mask."""
    ids, mask = [], []
    for generated, part in segments:
        ids.extend(part)
        mask.extend([1.0 if generated else 0.0] * len(part))
    return ids, mask


def _classes(ids, mask, **overrides):
    turn, think, act = turn_text_columns(ids, mask, **{**IDS, **overrides})
    out = []
    for t, th, ac, m in zip(turn, think, act, mask):
        out.append("-" if t < 0 else ("T" if th else "A" if ac else "C"))
        assert (t >= 0) == bool(m)  # turns cover exactly the generated tokens
        assert not (th and ac)  # classes are disjoint
    return "".join(out), turn


PROMPT = [100, 101, THINK]  # generation prompt ends with <think>: thinking on


def test_thinking_then_tool_call_then_end_of_turn():
    ids, mask = _seq(
        (False, PROMPT), (True, [200, 201, ETHINK, TC, 300, 301, ETC, IM_END])
    )
    classes, turn = _classes(ids, mask)
    assert classes == "---TTTAAAAA"
    assert turn == [-1, -1, -1] + [0] * 8


def test_content_before_and_between_two_tool_calls():
    gen = [200, ETHINK, 400, 401, TC, 300, ETC, 402, TC, 301, ETC, 403, IM_END]
    ids, mask = _seq((False, PROMPT), (True, gen))
    classes, _ = _classes(ids, mask)
    assert classes == "---" + "TT" + "CC" + "AAA" + "C" + "AAA" + "C" + "A"


def test_final_turn_without_tool_call_and_several_turns():
    ids, mask = _seq(
        (False, PROMPT),
        (True, [200, ETHINK, TC, 300, ETC, IM_END]),
        (False, [500, 501, THINK]),  # tool output + next generation prompt
        (True, [201, 202, ETHINK, 403, 404, IM_END]),
    )
    classes, turn = _classes(ids, mask)
    assert classes == "---" + "TTAAAA" + "---" + "TTTCCA"
    assert turn == [-1] * 3 + [0] * 6 + [-1] * 3 + [1] * 6


def test_thinking_without_content_and_no_closing_tag():
    # A turn of only thinking and its close tag, then end of turn.
    ids, mask = _seq((False, PROMPT), (True, [200, 201, ETHINK, IM_END]))
    assert _classes(ids, mask)[0] == "---TTTA"
    # No </think>: the reasoning parser reads it all as reasoning (its call markup never runs).
    ids, mask = _seq((False, PROMPT), (True, [200, TC, 300, ETC, 201, IM_END]))
    assert _classes(ids, mask)[0] == "---TTTTTA"


def test_tool_call_markup_inside_thinking_stays_thinking():
    ids, mask = _seq(
        (False, PROMPT), (True, [200, TC, 300, ETC, ETHINK, TC, 301, ETC, IM_END])
    )
    assert _classes(ids, mask)[0] == "---TTTTTAAAA"


def test_thinking_runs_to_the_last_think_close_of_the_turn():
    # Two </think>: everything through the second one is reasoning, tool-call markup included.
    gen = [200, ETHINK, 400, TC, 300, ETC, 201, ETHINK, TC, 301, ETC, IM_END]
    ids, mask = _seq((False, PROMPT), (True, gen))
    assert _classes(ids, mask)[0] == "---" + "T" * 8 + "AAAA"
    # Content after the last </think> is still content.
    gen = [200, ETHINK, 400, ETHINK, 401, TC, 300, ETC, IM_END]
    ids, mask = _seq((False, PROMPT), (True, gen))
    assert _classes(ids, mask)[0] == "---" + "TTTT" + "C" + "AAA" + "A"


def test_last_think_close_is_per_turn():
    # The second turn's </think> never extends the first turn's thinking, and vice versa.
    ids, mask = _seq(
        (False, PROMPT),
        (True, [200, ETHINK, 400, TC, 300, ETC, IM_END]),
        (False, [500, THINK]),
        (True, [201, ETHINK, 202, ETHINK, 401, IM_END]),
    )
    classes, turn = _classes(ids, mask)
    assert classes == "---" + "TTCAAAA" + "--" + "TTTTCA"
    assert turn == [-1] * 3 + [0] * 7 + [-1] * 2 + [1] * 6


def test_think_close_as_the_last_token_before_end_of_turn():
    ids, mask = _seq((False, PROMPT), (True, [200, TC, 300, ETC, ETHINK, IM_END]))
    assert _classes(ids, mask)[0] == "---TTTTTA"


def test_unclosed_tool_call_runs_to_the_end_of_the_turn():
    ids, mask = _seq((False, PROMPT), (True, [200, ETHINK, 400, TC, 300, 301, IM_END]))
    assert _classes(ids, mask)[0] == "---TTCAAAA"


def test_cut_off_turn_has_no_end_of_turn_action():
    ids, mask = _seq((False, PROMPT), (True, [200, ETHINK, 400, 401]))
    assert _classes(ids, mask)[0] == "---TTCC"


def test_thinking_off_turns_have_no_thinking_span():
    prompt_off = [100, THINK, ETHINK]  # template's empty thinking: thinking off
    ids, mask = _seq(
        (False, prompt_off), (True, [400, ETHINK, 401, TC, 300, ETC, IM_END])
    )
    assert _classes(ids, mask)[0] == "---CCCAAAA"
    # Without thinking tags configured there is never a thinking span.
    ids, mask = _seq((False, PROMPT), (True, [200, ETHINK, 401, IM_END]))
    assert _classes(ids, mask, think_open_id=None, think_close_id=None)[0] == "---CCCA"


def test_columns_align_with_token_positions_and_padding():
    ids, mask = _seq(
        (False, PROMPT), (True, [200, ETHINK, TC, 300, ETC, IM_END]), (False, [500])
    )
    turn, think, act = turn_text_columns(ids, mask, **IDS)
    assert len(turn) == len(think) == len(act) == len(ids)
    assert turn[-1] == -1 and not think[-1] and not act[-1]
    assert [i for i, t in enumerate(think) if t] == [3, 4]
    assert [i for i, a in enumerate(act) if a] == [5, 6, 7, 8]


class _FakeTok:
    unk_token_id = 0

    def __init__(
        self,
        single=("<think>", "</think>", "<tool_call>", "</tool_call>", "<|im_end|>"),
    ):
        self.vocab = {tag: 10 + i for i, tag in enumerate(single)}

    def convert_tokens_to_ids(self, tag):
        return self.vocab.get(tag, self.unk_token_id)

    def encode(self, tag, add_special_tokens=False):
        return [self.vocab[tag]] if tag in self.vocab else [101, 102]


def _resolve(tok, require=False):
    return resolve_turn_text_ids(
        tok,
        think_open="<think>",
        think_close="</think>",
        tool_call_open="<tool_call>",
        tool_call_close="</tool_call>",
        end_of_turn="<|im_end|>",
        require_thinking_tags=require,
    )


def test_resolve_turn_text_ids():
    ids = _resolve(_FakeTok())
    assert ids == {
        "think_open": 10,
        "think_close": 11,
        "tool_call_open": 12,
        "tool_call_close": 13,
        "end_of_turn": 14,
    }
    # Thinking tags are optional for prose_per_turn, required for thinking_per_turn.
    no_think = _FakeTok(single=("<tool_call>", "</tool_call>", "<|im_end|>"))
    assert _resolve(no_think)["think_close"] is None
    with pytest.raises(ValueError, match="</think>"):
        _resolve(no_think, require=True)
    # A tool-call tag split into several tokens cannot be located.
    with pytest.raises(ValueError, match="<tool_call>"):
        _resolve(_FakeTok(single=("<think>", "</think>", "</tool_call>", "<|im_end|>")))
