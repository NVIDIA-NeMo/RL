# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch

from nemo_rl.algorithms.grpo import RewardScalingConfig, scale_rewards
from nemo_rl.algorithms.reward_functions import (
    GroupLengthPenaltyConfig,
    RewardShapingConfig,
    apply_group_length_penalty,
    apply_group_length_penalty_to_rows,
    apply_reward_shaping,
    group_length_penalty_metrics,
)
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import decompose_message_log
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from tests.unit.algorithms.utils import create_mock_batch_with_responses


def test_reward_scaling_disabled():
    """Test that when reward scaling is disabled, rewards remain unchanged."""
    batch = create_mock_batch_with_responses(
        num_samples=3, response_lengths=[10, 20, 30], initial_rewards=[1.0, 0.5, 0.8]
    )

    original_rewards = batch["total_reward"].clone()
    config = RewardScalingConfig(enabled=False)
    result_batch = scale_rewards(batch, config)
    assert torch.allclose(result_batch["total_reward"], original_rewards)
    assert result_batch is batch  # Should return the same batch object


def test_reward_scaling_base():
    """Test that rewards are linearly scaled from [0.0, 1.0] to [0.0, 0.7]."""
    batch = create_mock_batch_with_responses(
        num_samples=3, response_lengths=[10, 20, 30], initial_rewards=[1.0, 0.5, 0.8]
    )

    config = RewardScalingConfig(
        enabled=True, source_min=0.0, source_max=1.0, target_min=0.0, target_max=0.7
    )

    result_batch = scale_rewards(batch, config)
    # Calculate expected rewards manually
    # Response 0: length=10, initial_reward=1.0, clip_initial_reward=1.0, scaled_reward=0.0 + [(1-0.0)/(1.0-0.0)]*(0.7-0) =  0.7
    # Response 1: length=20, initial_reward=0.5, clip_initial_reward=0.5, scaled_reward=0.0 + [(0.5-0.0)/(1.0-0.0)]*(0.7-0) =  0.35
    # Response 2: length=30, initial_reward=0.8, clip_initial_reward=0.8, scaled_reward=0.0 + [(0.8-0.0)/(1.0-0.0)]*(0.7-0) =  0.56

    expected_rewards = torch.tensor([0.7, 0.35, 0.56])
    assert torch.allclose(result_batch["total_reward"], expected_rewards)
    assert result_batch is batch  # Should return the same batch object


def test_reward_scaling_dapo():
    """Test that verifies binary rewards 0/1 are scaled to -1.0/1.0 respectively used in DAPO algorithm."""
    batch = create_mock_batch_with_responses(
        num_samples=5,
        response_lengths=[10, 20, 30, 40, 50],
        initial_rewards=[1.0, 0.0, 0.0, 1.0, 0.0],
    )

    config = RewardScalingConfig(
        enabled=True, source_min=0.0, source_max=1.0, target_min=-1.0, target_max=1.0
    )

    result_batch = scale_rewards(batch, config)
    expected_rewards = torch.tensor([1.0, -1.0, -1.0, 1.0, -1.0])

    assert torch.allclose(result_batch["total_reward"], expected_rewards)
    assert result_batch is batch  # Should return the same batch object


def test_reward_scaling_clipping():
    """Test that verifies the out-of-range rewards are clipped and scaled to the target range."""
    batch = create_mock_batch_with_responses(
        num_samples=6,
        response_lengths=[10, 20, 30, 40, 50, 60],
        initial_rewards=[-2.8, -0.25, 1.5, 0.5, 2.0, 2.5],
    )

    config = RewardScalingConfig(
        enabled=True, source_min=-2.0, source_max=2.0, target_min=-1.0, target_max=1.0
    )

    result_batch = scale_rewards(batch, config)
    # Calculate expected rewards manually
    # Response 0: initial_reward=-2.8, clip_initial_reward=-2.0, scaled_reward=-1.0 + [(-2.0-(-2.0))/(2.0-(-2.0))]*(1.0-(-1.0)) =  -1.0
    # Response 1: initial_reward=-0.25, clip_initial_reward=-0.25, scaled_reward=-1.0 + [(-0.25-(-2.0))/(2.0-(-2.0))]*(1.0-(-1.0)) =  -0.125
    # Response 2: initial_reward=1.5, clip_initial_reward=1.5, scaled_reward=-1.0 + [(1.5-(-2.0))/(2.0-(-2.0))]*(1.0-(-1.0)) =  0.75
    # Response 3: initial_reward=0.5, clip_initial_reward=0.5, scaled_reward=-1.0 + [(0.5-(-2.0))/(2.0-(-2.0))]*(1.0-(-1.0)) =  0.25
    # Response 4: initial_reward=2.0, clip_initial_reward=2.0, scaled_reward=-1.0 + [(2.0-(-2.0))/(2.0-(-2.0))]*(1.0-(-1.0)) =  1.0
    # Response 5: initial_reward=2.5, clip_initial_reward=2.0, scaled_reward=-1.0 + [(2.0-(-2.0))/(2.0-(-2.0))]*(1.0-(-1.0)) =  1.0

    expected_rewards = torch.tensor([-1.0, -0.125, 0.75, 0.25, 1.0, 1.0])

    assert torch.allclose(result_batch["total_reward"], expected_rewards)
    assert result_batch is batch  # Should return the same batch object


def test_reward_shaping_disabled():
    """Test that when reward shaping is disabled, rewards remain unchanged."""
    # Create batch with various response lengths
    batch = create_mock_batch_with_responses(
        num_samples=3, response_lengths=[10, 20, 30], initial_rewards=[1.0, 0.5, 0.8]
    )

    original_rewards = batch["total_reward"].clone()

    # Disabled reward shaping config
    config = RewardShapingConfig(
        enabled=False,
        overlong_buffer_length=5,
        overlong_buffer_penalty=0.1,
        max_response_length=25,
    )

    # Apply reward shaping
    result_batch = apply_reward_shaping(batch, config)

    # Rewards should remain unchanged
    assert torch.allclose(result_batch["total_reward"], original_rewards)
    assert result_batch is batch  # Should return the same batch object


def test_reward_shaping_no_penalties():
    """Test reward shaping when all responses are within acceptable length."""
    # Create batch where all responses are shorter than expected length
    batch = create_mock_batch_with_responses(
        num_samples=3,
        response_lengths=[10, 15, 18],  # All <= 20 (expected_response_length)
        initial_rewards=[1.0, 0.5, 0.8],
    )

    original_rewards = batch["total_reward"].clone()

    # Config: max_response_length=25, overlong_buffer_length=5 -> expected_response_length=20
    config = RewardShapingConfig(
        enabled=True,
        overlong_buffer_length=5,
        overlong_buffer_penalty=1.0,
        max_response_length=25,
    )

    # Apply reward shaping
    result_batch = apply_reward_shaping(batch, config)

    # Since no responses exceed expected length, rewards should remain unchanged
    assert torch.allclose(result_batch["total_reward"], original_rewards)


def test_reward_shaping_with_penalties():
    """Test reward shaping when responses exceed expected length and receive penalties."""
    # Create batch with responses of varying lengths
    batch = create_mock_batch_with_responses(
        num_samples=4,
        response_lengths=[10, 22, 25, 30],  # expected_response_length = 20
        initial_rewards=[1.0, 0.8, 0.6, 0.4],
    )

    # Config: max_response_length=25, overlong_buffer_length=5 -> expected_response_length=20
    config = RewardShapingConfig(
        enabled=True,
        overlong_buffer_length=5,
        overlong_buffer_penalty=0.5,
        max_response_length=25,
    )

    # Apply reward shaping
    result_batch = apply_reward_shaping(batch, config)

    # Calculate expected rewards manually
    # Response 0: length=10, exceed_length=10-20=-10 (no penalty, reward stays 1.0)
    # Response 1: length=22, exceed_length=22-20=2, penalty=min(-2/5*0.5, 0)=-0.2, reward=0.8-0.2=0.6
    # Response 2: length=25, exceed_length=25-20=5, penalty=min(-5/5*0.5, 0)=-0.5, reward=0.6-0.5=0.1
    # Response 3: length=30, exceed_length=30-20=10, penalty=min(-10/5*0.5, 0)=-1.0, reward=0.4-1.0=-0.6

    expected_rewards = torch.tensor([1.0, 0.6, 0.1, -0.6])
    assert torch.allclose(result_batch["total_reward"], expected_rewards, atol=1e-6)


def test_reward_shaping_preserves_unshaped_reward_overlong():
    """Reward shaping must save the pre-shaping reward so dynamic sampling can
    filter prompt groups on the raw task metric, not on shaped reward whose
    std is corrupted by length-dependent overlong penalties.
    """
    raw_rewards = [0.0, 0.0, 0.0, 0.0]
    batch = create_mock_batch_with_responses(
        num_samples=4,
        response_lengths=[10, 22, 25, 30],
        initial_rewards=raw_rewards,
    )

    config = RewardShapingConfig(
        enabled=True,
        overlong_buffer_length=5,
        overlong_buffer_penalty=0.5,
        max_response_length=25,
    )

    result_batch = apply_reward_shaping(batch, config)

    # Shaped rewards differ across the group due to length-based penalty even
    # though all raw rewards are 0 (i.e. all responses are wrong).
    expected_shaped = torch.tensor([0.0, -0.2, -0.5, -1.0])
    assert torch.allclose(result_batch["total_reward"], expected_shaped, atol=1e-6)

    # The unshaped reward must be retained verbatim.
    assert "unshaped_total_reward" in result_batch
    assert torch.allclose(
        result_batch["unshaped_total_reward"], torch.tensor(raw_rewards)
    )
    # The two tensors must be independent — mutating one must not affect the other.
    assert (
        result_batch["unshaped_total_reward"].data_ptr()
        != result_batch["total_reward"].data_ptr()
    )


def test_reward_shaping_preserves_unshaped_reward_stop_properly():
    """The stop-properly penalty path must also preserve the raw reward."""
    raw_rewards = [1.0, 0.8, 0.6, 0.4]
    batch = create_mock_batch_with_responses(
        num_samples=4,
        response_lengths=[10, 20, 30, 40],
        initial_rewards=raw_rewards,
    )
    batch["truncated"] = torch.tensor([False, True, False, True])

    config = RewardShapingConfig(enabled=True, stop_properly_penalty_coef=0.5)
    result_batch = apply_reward_shaping(batch, config)

    assert "unshaped_total_reward" in result_batch
    assert torch.allclose(
        result_batch["unshaped_total_reward"], torch.tensor(raw_rewards)
    )


def test_reward_shaping_disabled_does_not_save_unshaped_reward():
    """When shaping is disabled, the unshaped_total_reward field should not be added."""
    batch = create_mock_batch_with_responses(
        num_samples=3, response_lengths=[10, 20, 30], initial_rewards=[1.0, 0.5, 0.8]
    )

    config = RewardShapingConfig(
        enabled=False,
        overlong_buffer_length=5,
        overlong_buffer_penalty=0.1,
        max_response_length=25,
    )

    result_batch = apply_reward_shaping(batch, config)
    assert "unshaped_total_reward" not in result_batch


def test_reward_shaping_optional_defaults_are_none():
    config = RewardShapingConfig()

    assert config.overlong_buffer_length is None
    assert config.overlong_buffer_penalty is None
    assert config.max_response_length is None
    assert config.stop_properly_penalty_coef is None


def test_reward_shaping_missing_config_values():
    """Test that missing required DAPO config values raise ValueError."""
    batch = create_mock_batch_with_responses(
        num_samples=1, response_lengths=[20], initial_rewards=[1.0]
    )

    config = RewardShapingConfig(
        enabled=True,
        overlong_buffer_penalty=0.1,
        max_response_length=25,
    )

    with pytest.raises(ValueError, match="DAPO reward shaping is currently supported"):
        apply_reward_shaping(batch, config)

    config.overlong_buffer_length = 5
    config.overlong_buffer_penalty = None

    with pytest.raises(ValueError, match="DAPO reward shaping is currently supported"):
        apply_reward_shaping(batch, config)

    config.overlong_buffer_penalty = 0.1
    config.max_response_length = None

    with pytest.raises(ValueError, match="DAPO reward shaping is currently supported"):
        apply_reward_shaping(batch, config)


def test_reward_shaping_missing_assistant_response():
    """Test that missing assistant response raises assertion error."""
    # Create a batch with only user messages (no assistant responses)
    message_logs = [
        [{"role": "user", "content": "Question", "token_ids": torch.tensor([1, 2, 3])}]
    ]

    batch = BatchedDataDict[DatumSpec](
        {
            "task_name": ["math"],
            "message_log": message_logs,
            "extra_env_info": [{}],
            "loss_multiplier": torch.ones(1),
            "total_reward": torch.tensor([1.0]),
        }
    )

    config = RewardShapingConfig(
        enabled=True,
        overlong_buffer_length=5,
        overlong_buffer_penalty=0.1,
        max_response_length=25,
    )

    with pytest.raises(
        AssertionError, match="Assistant response not found during reward shaping"
    ):
        apply_reward_shaping(batch, config)


@pytest.mark.parametrize("decomposed", [False, True])
def test_reward_shaping_counts_all_assistant_turns(decomposed: bool) -> None:
    """Both input paths count every generated assistant turn."""

    def turn(role: str, n: int) -> dict:
        message = {
            "role": role,
            "content": "",
            "token_ids": torch.zeros(n, dtype=torch.long),
        }
        if role == "assistant":
            message["generation_logprobs"] = torch.zeros(n)
        return message

    message_logs = [
        # Single turn: 15 assistant tokens, within the expected length of 20.
        [turn("user", 5), turn("assistant", 15)],
        # Two turns: 15 + 10 = 25 assistant tokens, 5 over the expected length.
        [
            turn("user", 5),
            turn("assistant", 15),
            turn("environment", 3),
            turn("assistant", 10),
        ],
    ]
    batch = BatchedDataDict(
        {"message_log": message_logs, "total_reward": torch.tensor([1.0, 1.0])}
    )
    if decomposed:
        batch = BatchedDataDict(
            {
                "response_token_lengths": decompose_message_log(message_logs)[
                    "response_token_lengths"
                ],
                "total_reward": torch.tensor([1.0, 1.0]),
            }
        )

    config = RewardShapingConfig(
        enabled=True,
        overlong_buffer_length=10,
        overlong_buffer_penalty=1.0,
        max_response_length=30,
    )
    result = apply_reward_shaping(batch, config)

    assert torch.allclose(result["total_reward"], torch.tensor([1.0, 0.5]))


@pytest.mark.parametrize("decomposed", [False, True])
@pytest.mark.parametrize("history_length", [0, 100])
def test_reward_shaping_excludes_prompt_history(
    decomposed: bool, history_length: int
) -> None:
    """A long history or empty assistant placeholder must not penalize 15 new tokens."""
    batch = create_mock_batch_with_responses(1, [15], [1.0])
    batch["message_log"][0][1:1] = [
        {
            "role": "assistant",
            "content": "prompt history",
            "token_ids": torch.zeros(history_length, dtype=torch.long),
        },
        {
            "role": "user",
            "content": "follow-up",
            "token_ids": torch.ones(5, dtype=torch.long),
        },
    ]
    if decomposed:
        batch = BatchedDataDict(
            {
                "response_token_lengths": decompose_message_log(batch["message_log"])[
                    "response_token_lengths"
                ],
                "total_reward": batch["total_reward"],
            }
        )
    result = apply_reward_shaping(
        batch,
        RewardShapingConfig(
            enabled=True,
            max_response_length=30,
            overlong_buffer_length=10,
            overlong_buffer_penalty=1.0,
        ),
    )
    torch.testing.assert_close(result["total_reward"], torch.tensor([1.0]))


def test_reward_shaping_mismatched_lengths():
    """Test that mismatched message_log and rewards lengths raise assertion error."""
    # Create batch with mismatched lengths
    batch = create_mock_batch_with_responses(
        num_samples=2, response_lengths=[10, 20], initial_rewards=[1.0, 0.5]
    )

    # Manually add an extra reward to create mismatch
    batch["total_reward"] = torch.tensor(
        [1.0, 0.5, 0.3]
    )  # 3 rewards but 2 message_logs

    config = RewardShapingConfig(
        enabled=True,
        overlong_buffer_length=5,
        overlong_buffer_penalty=0.1,
        max_response_length=25,
    )

    with pytest.raises(
        AssertionError,
        match="The number of messages in the batch must match the number of rewards",
    ):
        apply_reward_shaping(batch, config)


def test_stop_properly_penalty(capsys):
    """Test stop_properly_penalty_coef scales rewards for truncated samples."""
    batch = create_mock_batch_with_responses(
        num_samples=4,
        response_lengths=[10, 20, 30, 40],
        initial_rewards=[1.0, 0.8, 0.6, 0.4],
    )
    batch["truncated"] = torch.tensor([False, True, False, True])

    config = RewardShapingConfig(enabled=True, stop_properly_penalty_coef=0.5)
    result_batch = apply_reward_shaping(batch, config)

    assert "[WARN]" not in capsys.readouterr().out
    # Non-truncated unchanged, truncated scaled by 0.5
    expected_rewards = torch.tensor([1.0, 0.4, 0.6, 0.2])
    assert torch.allclose(result_batch["total_reward"], expected_rewards, atol=1e-6)


def test_stop_properly_penalty_boundary_coefs():
    """Test boundary values: coef=0 gives zero reward, coef=1 has no effect."""
    # Test coef=0: truncated samples get zero reward
    batch = create_mock_batch_with_responses(
        num_samples=2, response_lengths=[10, 20], initial_rewards=[1.0, 0.5]
    )
    batch["truncated"] = torch.tensor([True, True])

    config = RewardShapingConfig(enabled=True, stop_properly_penalty_coef=0.0)
    result = apply_reward_shaping(batch, config)
    assert torch.allclose(result["total_reward"], torch.tensor([0.0, 0.0]), atol=1e-6)

    # Test coef=1: no penalty applied
    batch["total_reward"] = torch.tensor([1.0, 0.5])
    config.stop_properly_penalty_coef = 1.0
    result = apply_reward_shaping(batch, config)
    assert torch.allclose(result["total_reward"], torch.tensor([1.0, 0.5]), atol=1e-6)


def test_stop_properly_penalty_error_cases():
    """Test error handling for invalid coef and missing truncated field."""
    batch = create_mock_batch_with_responses(
        num_samples=2, response_lengths=[10, 20], initial_rewards=[1.0, 0.5]
    )

    # Missing truncated field
    config = RewardShapingConfig(enabled=True, stop_properly_penalty_coef=0.5)
    with pytest.raises(AssertionError, match="truncated field not found"):
        apply_reward_shaping(batch, config)

    # Invalid coef values
    batch["truncated"] = torch.tensor([False, True])
    config.stop_properly_penalty_coef = -0.1
    with pytest.raises(AssertionError, match="stop_properly_penalty_coef must be in"):
        apply_reward_shaping(batch, config)

    config.stop_properly_penalty_coef = 1.5
    with pytest.raises(AssertionError, match="stop_properly_penalty_coef must be in"):
        apply_reward_shaping(batch, config)


class TestGroupLengthPenalty:
    """MiMo-V2.6 group-relative length penalty (grpo.group_relative_length_penalty)."""

    @staticmethod
    def _cfg(**overrides) -> GroupLengthPenaltyConfig:
        return GroupLengthPenaltyConfig(enabled=True, **overrides)

    def test_only_long_passing_rollouts_lose_up_to_max_penalty(self):
        # Group of 4 with 3 passing (lengths 100, 100, 200) and a long failing rollout.
        rewards = torch.tensor([1.0, 1.0, 1.0, 0.0])
        lengths = torch.tensor([100.0, 100.0, 200.0, 1000.0])
        penalized, metrics = apply_group_length_penalty(
            rewards, lengths, torch.zeros(4, dtype=torch.long), self._cfg()
        )

        # l* = median(100, 100, 200) = 100; 200 is 2x l*, the full 0.3; failing rollout untouched.
        assert penalized.tolist() == pytest.approx([1.0, 1.0, 0.7, 0.0])
        assert metrics["length_penalty/groups_applied"] == 1.0
        assert metrics["length_penalty/frac_passing_penalized"] == pytest.approx(1 / 3)

    def test_ramp_between_tolerance_and_saturation(self):
        # 160 is 1.6x l*: excess (0.6 - 0.2) / (1.0 - 0.2) = 0.5 -> 0.3 * 0.5**2.
        penalized, _ = apply_group_length_penalty(
            torch.tensor([1.0, 1.0, 1.0]),
            torch.tensor([100.0, 100.0, 160.0]),
            torch.zeros(3, dtype=torch.long),
            self._cfg(),
        )
        assert penalized[2].item() == pytest.approx(1.0 - 0.3 * 0.25)

    def test_groups_at_or_below_min_pass_rate_are_untouched(self):
        rewards = torch.tensor([1.0] + [0.0] * 7)
        penalized, metrics = apply_group_length_penalty(
            rewards,
            torch.arange(8, dtype=torch.float32) * 1000 + 10,
            torch.zeros(8, dtype=torch.long),
            self._cfg(),
        )
        assert torch.equal(penalized, rewards)
        assert metrics["length_penalty/groups_applied"] == 0.0

    def test_groups_above_max_pass_rate_are_untouched(self):
        # Group 0: all 4 pass (rate 1.0 > 0.875), skipped. Group 1: 3 of 4 pass, penalized.
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
        lengths = torch.tensor([100.0, 100.0, 100.0, 300.0, 100.0, 100.0, 200.0, 10.0])
        group_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        penalized, metrics = apply_group_length_penalty(
            rewards, lengths, group_ids, self._cfg(max_pass_rate=0.875)
        )
        assert penalized.tolist() == pytest.approx([1.0] * 6 + [0.7, 0.0])
        assert metrics["length_penalty/groups_applied"] == 1.0
        assert metrics["length_penalty/groups_skipped_above_max_pass_rate"] == 1.0

    def test_groups_need_not_be_contiguous(self):
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
        lengths = torch.tensor([100.0, 100.0, 200.0, 400.0])
        interleaved, _ = apply_group_length_penalty(
            rewards, lengths, torch.tensor([0, 1, 0, 1]), self._cfg(max_pass_rate=1.0)
        )
        # Group 0: (100, 200) -> l* 150; group 1: (100, 400) -> l* 250.
        contiguous, _ = apply_group_length_penalty(
            rewards[[0, 2, 1, 3]],
            lengths[[0, 2, 1, 3]],
            torch.tensor([0, 0, 1, 1]),
            self._cfg(max_pass_rate=1.0),
        )
        assert interleaved[[0, 2, 1, 3]].tolist() == pytest.approx(contiguous.tolist())

    def test_mismatched_shapes_are_rejected(self):
        with pytest.raises(ValueError, match="same shape"):
            apply_group_length_penalty(
                torch.ones(2),
                torch.ones(3),
                torch.zeros(2, dtype=torch.long),
                self._cfg(),
            )

    @pytest.mark.parametrize(
        "overrides, match",
        [
            ({"max_penalty": -0.1}, "max_penalty"),
            ({"tolerance": 1.0, "saturation": 1.0}, "saturation > tolerance"),
            ({"ramp_exponent": 0.5}, "ramp_exponent"),
            (
                {"min_pass_rate": 0.5, "max_pass_rate": 0.5},
                "min_pass_rate < max_pass_rate",
            ),
            ({"reference_percentile": 100}, "reference_percentile"),
        ],
    )
    def test_invalid_parameters_fail_at_config_load(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            GroupLengthPenaltyConfig(**overrides)

    def test_unknown_keys_are_rejected(self):
        with pytest.raises(ValueError):
            GroupLengthPenaltyConfig(max_penalyt=0.1)


def test_rows_helper_counts_only_participating_rows():
    """Non-participating rows (e.g. token-capture placeholders) keep their reward and no vote."""
    # One group of 4: three passing rollouts of 2, 2 and 4 generated tokens, and a passing
    # placeholder (no tokens, not participating).
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
    token_mask = torch.tensor(
        [[0, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 1, 1, 1, 1], [0, 0, 0, 0, 0]],
        dtype=torch.float32,
    )
    participating = torch.tensor([1.0, 1.0, 1.0, 0.0])
    cfg = GroupLengthPenaltyConfig(
        enabled=True, max_pass_rate=1.0, reference_percentile=25
    )

    penalized, partial = apply_group_length_penalty_to_rows(
        rewards, token_mask, torch.zeros((4, 1), dtype=torch.long), participating, cfg
    )
    # l* = 25th percentile of (2, 2, 4) = 2, and 4 is 2x l* -> the full 0.3. Counting the
    # placeholder as a length-0 pass would give 25th pct of (0, 2, 2, 4) = 1.5 and also
    # penalize rows 0 and 1. The placeholder itself is untouched.
    assert penalized.tolist() == pytest.approx([1.0, 1.0, 0.7, 1.0])
    metrics = group_length_penalty_metrics([partial])
    assert metrics["length_penalty/mean_reference_length"] == pytest.approx(2.0)
    assert metrics["length_penalty/frac_passing_penalized"] == pytest.approx(1 / 3)
    assert metrics["reward_with_length_penalty"] == pytest.approx((1.0 + 1.0 + 0.7) / 3)


def test_metrics_add_up_across_calls():
    """Per-call sums reduce to the metrics of one call over the whole batch."""
    cfg = GroupLengthPenaltyConfig(enabled=True, max_pass_rate=1.0)
    rewards = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    lengths = torch.tensor([2, 2, 4, 1, 3, 6, 2, 2])
    token_mask = (torch.arange(8).unsqueeze(0) < lengths.unsqueeze(1)).float()
    group_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    valid = torch.ones(8)

    _, whole = apply_group_length_penalty_to_rows(
        rewards, token_mask, group_ids, valid, cfg
    )
    _, first = apply_group_length_penalty_to_rows(
        rewards[:4], token_mask[:4], group_ids[:4], valid[:4], cfg
    )
    _, second = apply_group_length_penalty_to_rows(
        rewards[4:], token_mask[4:], group_ids[4:], valid[4:], cfg
    )
    assert group_length_penalty_metrics([first, second]) == pytest.approx(
        group_length_penalty_metrics([whole])
    )


def test_legacy_trainers_apply_the_penalty_to_the_estimator_rewards():
    """grpo_train, async_grpo_train and grpo_train_sync share this helper."""
    from types import SimpleNamespace

    from nemo_rl.algorithms.grpo import _group_relative_length_penalty_rewards

    rewards = torch.tensor([1.0, 1.0, 1.0, 0.0])
    token_mask = (
        torch.arange(5).unsqueeze(0) < torch.tensor([[1], [1], [2], [4]])
    ).float()
    group_ids = torch.zeros((4, 1), dtype=torch.long)
    valid = torch.ones(4)

    off = SimpleNamespace(group_relative_length_penalty=GroupLengthPenaltyConfig())
    out, metrics = _group_relative_length_penalty_rewards(
        off, rewards, token_mask, group_ids, valid
    )
    assert out is rewards and metrics == {}

    on = SimpleNamespace(
        group_relative_length_penalty=GroupLengthPenaltyConfig(enabled=True)
    )
    out, metrics = _group_relative_length_penalty_rewards(
        on, rewards, token_mask, group_ids, valid
    )
    # Passing lengths 1, 1, 2: l* = 1 and 2 is 2x l* -> the full 0.3.
    assert out.tolist() == pytest.approx([1.0, 1.0, 0.7, 0.0])
    assert torch.equal(rewards, torch.tensor([1.0, 1.0, 1.0, 0.0])), "input not mutated"
    assert metrics["length_penalty/groups_applied"] == 1.0
