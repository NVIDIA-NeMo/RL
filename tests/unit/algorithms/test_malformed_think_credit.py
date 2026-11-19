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
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from nemo_rl.algorithms.single_controller_utils.malformed_think_credit import (
    MalformedThinkCreditConfig,
    MalformedThinkCreditStats,
    MalformedThinkTurnDetector,
    malformed_think_credit_config,
    cap_advantage_on_malformed_turns,
)
from nemo_rl.experience.think_tags import (
    ThinkTagIds,
    malformed_think_turns,
    resolve_think_tag_ids,
    think_tag_violation,
)

# Toy ids: 10 <|im_start|>, 11 <|im_end|>, 2 </s>, 12 <think>, 13 </think>, 1 any other token.
IDS = ThinkTagIds(
    think_open=12, think_close=13, turn_start=10, turn_end=frozenset({11, 2})
)


@pytest.mark.parametrize(
    ("counts", "cut", "kind"),
    [
        ((1, 0, 0, 1), False, None),  # thinking on, closed once
        ((0, 0, 0, 0), False, None),  # thinking off, no tags
        ((1, 1, 0, 0), False, None),  # thinking off (balanced prompt), no tags
        ((1, 0, 0, 2), False, "double_close"),
        ((1, 0, 0, 3), True, "double_close"),
        ((1, 0, 1, 1), False, "stray_open"),
        ((1, 0, 1, 0), True, "stray_open"),
        ((1, 0, 0, 0), False, "unclosed"),
        ((1, 0, 0, 0), True, "cut_unclosed"),
        ((0, 0, 0, 1), False, "close_without_thinking"),
        ((2, 0, 0, 1), False, "prompt_mode"),
        ((0, 1, 0, 0), False, "prompt_mode"),
    ],
)
def test_think_tag_violation(counts, cut, kind):
    assert think_tag_violation(*counts, cut=cut) == kind


def _row(*turns):
    """A row of (carry, generation) turns, and its generated-token mask."""
    ids, mask = [], []
    for carry, generation in turns:
        ids += carry + generation
        mask += [0] * len(carry) + [1] * len(generation)
    return ids, mask


def test_marks_malformed_turns_and_reads_the_mode_from_the_last_header():
    header = [10, 1, 12]  # <|im_start|>assistant\n<think>
    ids, mask = _row(
        ([10, 1, 11] + header, [1, 13, 1, 11]),  # fine
        ([1] + header, [1, 13, 1, 13, 11]),  # closed twice
        (
            [1, 12, 1, 13, 13] + header,
            [1, 13, 2],
        ),  # tool result with tags: fine, </s> ends it
        (header, [1, 1, 11]),  # ended without closing thinking
        (header + [13], [1, 11]),  # thinking off
        (header + [13], [1, 13, 11]),  # a close tag with thinking off
        (header, [1, 12, 1, 13, 11]),  # thinking opened again
        (header, [1, 1, 1]),  # cut before closing thinking: counted, not marked
    )
    found = malformed_think_turns(ids, mask, ids=IDS)
    assert found.turns == 8 and found.malformed_turns == 4
    assert found.kinds == {
        "double_close": 1,
        "unclosed": 1,
        "close_without_thinking": 1,
        "stray_open": 1,
        "cut_unclosed": 1,
    }
    marked = [i for i, m in enumerate(found.mask) if m]
    starts = [i for i in range(len(mask)) if mask[i] and (i == 0 or not mask[i - 1])]
    ends = [
        i + 1
        for i in range(len(mask))
        if mask[i] and (i + 1 == len(mask) or not mask[i + 1])
    ]
    expected = [i for t in (1, 3, 5, 6) for i in range(starts[t], ends[t])]
    assert marked == expected


def test_a_carry_without_a_header_is_read_whole():
    # No turn header in the second carry: all its tags set the mode, as in the finalizer's check.
    ids, mask = _row(([10, 1, 12], [1, 13, 11]), ([1, 12], [1, 13, 11]))
    assert malformed_think_turns(ids, mask, ids=IDS).malformed_turns == 0
    ids, mask = _row(([10, 1, 12], [1, 13, 11]), ([1, 12, 13], [1, 13, 11]))
    assert malformed_think_turns(ids, mask, ids=IDS).kinds == {
        "close_without_thinking": 1
    }


def test_row_without_generated_tokens():
    found = malformed_think_turns([10, 1, 12], [0, 0, 0], ids=IDS)
    assert found.turns == 0 and not found.mask.any()


def _detector():
    detector = MalformedThinkTurnDetector({"name": "fake"})
    detector._ids = IDS
    return detector


@pytest.mark.parametrize("nested", [False, True])
def test_detector_marks_malformed_turns_in_jagged_or_padded_rows(nested):
    rows = [
        [10, 12, 1, 13, 13, 11, 1, 10, 12, 1, 13, 11],  # turn 1 closes twice
        [10, 12, 1, 13, 11, 1, 10, 12, 1, 11],  # turn 2 never closes
        [10, 12, 1, 13, 13, 11],  # malformed, but no trained tokens: not scanned
    ]
    gen = [
        [0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    ]
    width = 12
    token_mask = torch.tensor(gen, dtype=torch.float32)
    if nested:
        input_ids = torch.nested.nested_tensor(
            [torch.tensor(r) for r in rows], layout=torch.jagged
        )
    else:
        input_ids = torch.tensor([r + [0] * (width - len(r)) for r in rows])
    malformed, stats = _detector().mask(
        input_ids, token_mask, scan_rows=torch.tensor([True, True, False])
    )
    assert malformed.shape == (3, width) and malformed.dtype == torch.bool
    assert malformed[0].tolist() == [False, False, True, True, True, True] + [False] * 6
    assert malformed[1].tolist() == [False] * 8 + [True, True, False, False]
    assert not malformed[2].any()
    assert stats.turns == 4 and stats.malformed_turns == 2
    assert stats.kinds == {"double_close": 1, "unclosed": 1}


def test_caps_the_advantage_of_trained_tokens_in_malformed_turns():
    advantages = torch.tensor(
        [
            [
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],  # rewarded rollout: its malformed turn drops to -1
            [0.0, -0.5, -0.5, -0.5, -0.5],  # failed rollout: -1 there too
            [0.0, -3.0, -3.0, -3.0, -3.0],  # already below the cap: kept
            [
                0.0,
                2.0,
                2.0,
                2.0,
                0.0,
            ],  # flagged tokens outside the train mask are left alone
        ]
    )
    malformed = torch.tensor([[False, True, True, False, False]] * 4)
    train = torch.tensor(
        [[0, 1, 1, 1, 1], [0, 1, 1, 1, 1], [0, 1, 1, 1, 1], [0, 1, 0, 1, 0]],
        dtype=torch.float32,
    )
    out, stats = cap_advantage_on_malformed_turns(
        advantages, malformed_turn_mask=malformed, train_mask=train, max_advantage=-1.0
    )
    expected = advantages.clone()
    expected[0, 1:3] = -1.0
    expected[1, 1:3] = -1.0
    expected[3, 1] = -1.0
    torch.testing.assert_close(out, expected)
    assert out.dtype == advantages.dtype
    assert stats.trained_tokens == 14 and stats.flagged_tokens == 7
    assert stats.flagged_positive_tokens == 3 and stats.lowered_tokens == 5
    assert stats.trained_rows == 4 and stats.flagged_rows == 4
    m = stats.to_metrics()
    assert m["malformed_think_credit/flagged_token_share"] == pytest.approx(0.5)
    assert m["malformed_think_credit/flagged_positive_token_share"] == pytest.approx(
        3 / 14
    )
    assert m["malformed_think_credit/lowered_token_share"] == pytest.approx(5 / 14)


def test_shape_mismatch_fails_loud():
    with pytest.raises(ValueError, match="one shape"):
        cap_advantage_on_malformed_turns(
            torch.zeros(2, 3),
            malformed_turn_mask=torch.zeros(2, 4, dtype=torch.bool),
            train_mask=torch.ones(2, 3),
            max_advantage=-1.0,
        )


def test_stats_merge_and_metrics():
    assert MalformedThinkCreditStats().to_metrics() == {}
    s = MalformedThinkCreditStats(
        trained_tokens=10, turns=8, malformed_turns=1, kinds={"double_close": 1}
    )
    s.merge(
        MalformedThinkCreditStats(
            trained_tokens=10,
            turns=12,
            malformed_turns=1,
            kinds={"unclosed": 1, "cut_unclosed": 2},
        )
    )
    m = s.to_metrics()
    assert m["malformed_think_credit/malformed_turn_share"] == pytest.approx(0.1)
    assert m["malformed_think_credit/turn_share/double_close"] == pytest.approx(0.05)
    assert m["malformed_think_credit/turn_share/cut_unclosed"] == pytest.approx(0.1)
    assert m["malformed_think_credit/turn_share/stray_open"] == 0.0


def test_config_parsing():
    assert malformed_think_credit_config(SimpleNamespace()) is None
    assert (
        malformed_think_credit_config(
            SimpleNamespace(malformed_think_credit={"enabled": False})
        )
        is None
    )
    cfg = malformed_think_credit_config(
        SimpleNamespace(malformed_think_credit={"enabled": True})
    )
    assert cfg == MalformedThinkCreditConfig(enabled=True, max_advantage=-2.0)
    cfg = malformed_think_credit_config(
        SimpleNamespace(
            malformed_think_credit=MalformedThinkCreditConfig(
                enabled=True, max_advantage=-1.0
            )
        )
    )
    assert cfg is not None and cfg.max_advantage == -1.0
    with pytest.raises(ValidationError):
        malformed_think_credit_config(
            SimpleNamespace(malformed_think_credit={"enabled": True, "value": -1.0})
        )


def test_resolve_think_tag_ids():
    vocab = {"<think>": 12, "</think>": 13, "<|im_start|>": 10, "<|im_end|>": 11}

    class FakeTokenizer:
        unk_token_id = 0
        eos_token_id = 2

        def __init__(self, vocab):
            self.vocab = vocab

        def convert_tokens_to_ids(self, tag):
            return self.vocab.get(tag, 0)

        def encode(self, tag, add_special_tokens):
            return [self.vocab[tag]] if tag in self.vocab else [3, 4]

    assert resolve_think_tag_ids(FakeTokenizer(vocab)) == IDS
    with pytest.raises(ValueError, match="</think>"):
        resolve_think_tag_ids(
            FakeTokenizer({k: v for k, v in vocab.items() if k != "</think>"})
        )


def test_legacy_trainers_reject_malformed_think_credit():
    from nemo_rl.algorithms.grpo import _raise_if_turn_credit

    _raise_if_turn_credit(SimpleNamespace(malformed_think_credit={"enabled": False}))
    with pytest.raises(NotImplementedError, match="malformed_think_credit"):
        _raise_if_turn_credit(SimpleNamespace(malformed_think_credit={"enabled": True}))
