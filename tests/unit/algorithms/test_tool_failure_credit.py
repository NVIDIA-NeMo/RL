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

from nemo_rl.algorithms.single_controller_utils.tool_failure_credit import (
    ToolFailureCreditConfig,
    ToolFailureCreditStats,
    tool_failure_credit_config,
    zero_positive_credit_on_failed_turns,
)


def test_zeroes_only_positive_trained_tokens_of_failed_turns():
    advantages = torch.tensor(
        [
            [0.0, 1.0, 1.0, 1.0, 1.0],  # rollout with A > 0: failed turn at 1-2
            [
                0.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
            ],  # rollout with A < 0: keeps its negative signal
            [
                0.0,
                2.0,
                2.0,
                2.0,
                0.0,
            ],  # flagged tokens outside the train mask are left alone
        ]
    )
    failed = torch.tensor(
        [
            [False, True, True, False, False],
            [False, True, True, False, False],
            [False, True, True, False, False],
        ]
    )
    train = torch.tensor(
        [[0, 1, 1, 1, 1], [0, 1, 1, 1, 1], [0, 1, 0, 1, 0]], dtype=torch.float32
    )
    out, stats = zero_positive_credit_on_failed_turns(
        advantages, failed_turn_mask=failed, train_mask=train
    )
    expected = advantages.clone()
    expected[0, 1:3] = 0.0
    expected[2, 1] = 0.0
    torch.testing.assert_close(out, expected)
    assert out.dtype == advantages.dtype
    assert (
        stats.trained_tokens == 10
        and stats.flagged_tokens == 5
        and stats.zeroed_tokens == 3
    )
    assert stats.positive_mass == pytest.approx(
        8.0
    ) and stats.removed_mass == pytest.approx(4.0)
    m = stats.to_metrics()
    assert m["tool_failure_credit/zeroed_token_share"] == pytest.approx(0.3)
    assert m["tool_failure_credit/removed_positive_mass_share"] == pytest.approx(0.5)
    assert m["tool_failure_credit/flagged_row_share"] == pytest.approx(1.0)


def test_shape_mismatch_fails_loud():
    with pytest.raises(ValueError, match="one shape"):
        zero_positive_credit_on_failed_turns(
            torch.zeros(2, 4),
            failed_turn_mask=torch.zeros(2, 3, dtype=torch.bool),
            train_mask=torch.ones(2, 4),
        )


def test_stats_merge_and_empty():
    assert ToolFailureCreditStats().to_metrics() == {}
    a = ToolFailureCreditStats(
        trained_tokens=4,
        flagged_tokens=2,
        zeroed_tokens=1,
        positive_mass=2,
        removed_mass=1,
    )
    a.merge(
        ToolFailureCreditStats(
            trained_tokens=4, positive_mass=2, trained_rows=2, flagged_rows=1
        )
    )
    m = a.to_metrics()
    assert m["tool_failure_credit/flagged_token_share"] == 0.25
    assert m["tool_failure_credit/removed_positive_mass_share"] == 0.25
    assert m["tool_failure_credit/flagged_row_share"] == 0.5


def test_config_parsing():
    assert tool_failure_credit_config(SimpleNamespace()) is None
    assert (
        tool_failure_credit_config(
            SimpleNamespace(tool_failure_credit={"enabled": False})
        )
        is None
    )
    cfg = tool_failure_credit_config(
        SimpleNamespace(tool_failure_credit={"enabled": True})
    )
    # Harness denials (permission_denied) and the unvalidated categories are off by default.
    assert cfg == ToolFailureCreditConfig(
        enabled=True, categories=["schema_error", "edit_failed", "read_missing"]
    )
    cfg = tool_failure_credit_config(
        SimpleNamespace(
            tool_failure_credit={
                "enabled": True,
                "categories": ["edit_failed", "schema_error"],
            }
        )
    )
    assert cfg is not None and cfg.categories == ["edit_failed", "schema_error"]


@pytest.mark.parametrize(
    "block",
    [
        {"enabled": True, "categories": ["edit_fail"]},
        {"enabled": True, "categories": []},
        {"enabled": True, "categoreis": ["edit_failed"]},
    ],
)
def test_bad_config_fails_loud(block):
    with pytest.raises(ValidationError):
        tool_failure_credit_config(SimpleNamespace(tool_failure_credit=block))


def _detector(vocab):
    from nemo_rl.algorithms.single_controller_utils.tool_failure_credit import (
        FailedToolTurnDetector,
    )
    from nemo_rl.experience.tool_failures import ToolTagIds

    class FakeTokenizer:
        @staticmethod
        def batch_decode(seqs, skip_special_tokens):
            assert not skip_special_tokens
            return ["".join(vocab[t] for t in seq) for seq in seqs]

    detector = FailedToolTurnDetector(
        {"name": "fake"}, ["edit_failed", "read_missing", "schema_error"]
    )
    detector._tokenizer = FakeTokenizer()
    detector._ids = ToolTagIds(
        tool_call_open=14, tool_response_open=16, tool_response_close=17
    )
    return detector


VOCAB = {
    14: "<tool_call>",
    16: "<tool_response>",
    17: "</tool_response>",
    100: "\n<function=edit>\n",
    101: "Could not find oldString",
    102: "Edit applied successfully.",
    103: "The user has specified a rule which prevents you from using this command",
    104: "\n<function=bash>\n",
    1: "x",
    0: "<pad>",
}


@pytest.mark.parametrize("nested", [False, True])
def test_detector_marks_failed_turns_in_jagged_or_padded_rows(nested):
    rows = [
        [1, 1, 14, 100, 16, 101, 17, 1],  # failed edit: turn 1 marked
        [
            1,
            1,
            14,
            104,
            16,
            103,
            17,
        ],  # permission_denied: counted, not an allowed category
        [
            1,
            1,
            14,
            100,
            16,
            101,
            17,
            1,
        ],  # failed edit, but no trained tokens: not scanned
    ]
    width = 8
    gen = [[0, 1, 1, 1, 0, 0, 0, 1], [0, 1, 1, 1, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0, 1]]
    token_mask = torch.tensor(gen, dtype=torch.float32)
    if nested:
        input_ids = torch.nested.nested_tensor(
            [torch.tensor(r) for r in rows], layout=torch.jagged
        )
    else:
        input_ids = torch.tensor([r + [0] * (width - len(r)) for r in rows])
    failed, stats = _detector(VOCAB).mask(
        input_ids, token_mask, scan_rows=torch.tensor([True, True, False])
    )
    assert failed.shape == (3, width) and failed.dtype == torch.bool
    assert failed[0].tolist() == [False, True, True, True, False, False, False, False]
    assert not failed[1].any() and not failed[2].any()
    assert stats.turns == 3 and stats.failed_turns == 1 and stats.tool_responses == 2
    assert stats.failed_calls == {"edit_failed": 1, "permission_denied": 1}


def test_detection_metrics():
    s = ToolFailureCreditStats(
        trained_tokens=10,
        turns=8,
        failed_turns=2,
        tool_responses=4,
        failed_calls={"edit_failed": 1},
    )
    s.merge(
        ToolFailureCreditStats(
            tool_responses=6, failed_calls={"edit_failed": 1, "read_missing": 2}
        )
    )
    m = s.to_metrics()
    assert m["tool_failure_credit/failed_turn_share"] == 0.25
    assert m[
        "tool_failure_credit/failed_calls_per_100_responses/edit_failed"
    ] == pytest.approx(20.0)
    assert m[
        "tool_failure_credit/failed_calls_per_100_responses/read_missing"
    ] == pytest.approx(20.0)
    assert (
        m["tool_failure_credit/failed_calls_per_100_responses/permission_denied"] == 0.0
    )


def test_legacy_trainers_reject_tool_failure_credit():
    from nemo_rl.algorithms.grpo import _raise_if_turn_credit

    _raise_if_turn_credit(SimpleNamespace())
    _raise_if_turn_credit(SimpleNamespace(tool_failure_credit={"enabled": False}))
    with pytest.raises(NotImplementedError, match="SingleController"):
        _raise_if_turn_credit(SimpleNamespace(tool_failure_credit={"enabled": True}))
