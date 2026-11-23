# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Token metrics count generated output rather than stored input deltas."""

from dataclasses import replace

from nemo_rl.experience.reward_penalties import RewardLogContext
from nemo_rl.experience.rollout_reassembler import (
    FinalizedRollout,
    _capture_token_metrics,
)


def test_compaction_lengths_and_invalid_owner() -> None:
    # Three calls before compaction: 1000 input + 500 output, then two
    # 200-token observations + 500 output. Rewritten root: 2400 input.
    def row(prompt: int) -> FinalizedRollout:
        mask = [0.0] * prompt + [1.0] * 500 + ([0.0] * 200 + [1.0] * 500) * 2
        return FinalizedRollout(
            rollout_id="owner",
            valid=True,
            rejection_reason=None,
            token_ids=[1] * len(mask),
            token_mask=mask,
            logprobs=[0.0] * len(mask),
            prompt_len=prompt,
            reward=1.0,
            staging_keys=[],
            generation_lengths=(500, 500, 500),
        )

    first, second = row(1000), row(2400)
    invalid = replace(first, valid=False)
    metrics = _capture_token_metrics(
        [first, second, invalid], [0, 0, 1], [RewardLogContext("agent", None)] * 2
    )
    assert len(first.token_ids) + len(second.token_ids) == 7200
    assert metrics["gen_tokens_per_sample/histogram"] == [3000, 0]
    assert metrics["max_gen_tokens_per_turn/histogram"] == [500, 0]
    assert metrics["total_tokens_per_sample/histogram"] == [4300, 0]
    assert metrics["mean_gen_tokens_per_sample"] == 1500
    assert metrics["agent/gen_tokens_per_sample/histogram"] == [3000, 0]


def test_adjacent_generations_still_have_separate_turn_lengths() -> None:
    row = FinalizedRollout(
        rollout_id="owner",
        valid=True,
        rejection_reason=None,
        token_ids=[10, 11, 20, 21, 22, 23, 24],
        token_mask=[0, 0, 1, 1, 1, 1, 1],
        logprobs=[0.0] * 7,
        prompt_len=2,
        reward=1.0,
        staging_keys=[],
        generation_lengths=(2, 3),
    )
    metrics = _capture_token_metrics([row], [0], [None])
    assert metrics["gen_tokens_per_sample/mean"] == 5
    assert metrics["max_gen_tokens_per_turn/max"] == 3
