# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from nemo_rl.experience.rollout_reassembler import FinalizedRollout, RolloutReassembler


def _reassembler(cap):
    return RolloutReassembler(
        object(), partition_id="rollouts", staging_partition="staging", pad_token_id=0, max_seq_len=100, max_train_row_tokens=cap
    )


def _row(n, valid=True):
    return FinalizedRollout(
        rollout_id="r", valid=valid, rejection_reason=None if valid else "x", token_ids=list(range(n)),
        token_mask=[1.0] * n, logprobs=[0.0] * n, prompt_len=1, reward=1.0, staging_keys=["r/c1"], model_call_ids=("c1",),
    )


def test_rows_within_cap_or_without_cap_are_untouched():
    row = _row(50)
    assert _reassembler(None)._cap_row_length(row) is row
    assert _reassembler(50)._cap_row_length(row) is row


def test_rows_over_cap_are_rejected_but_keep_staging_evidence():
    capped = _reassembler(40)._cap_row_length(_row(50))
    assert not capped.valid and capped.rejection_reason == "row_too_long:50>40"
    assert capped.token_ids == [] and capped.model_call_ids == () and capped.staging_keys == ["r/c1"]
    assert capped.reward == 1.0


def test_already_rejected_rows_keep_their_reason():
    row = _row(50, valid=False)
    assert _reassembler(10)._cap_row_length(row) is row


def test_cap_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        _reassembler(0)
