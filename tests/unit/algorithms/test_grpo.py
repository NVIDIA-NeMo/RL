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

from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import ray
import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.advantage_estimator import (
    GDPOAdvantageEstimator,
    GRPOAdvantageEstimator,
    ReinforcePlusPlusAdvantageEstimator,
)
from nemo_rl.algorithms.grpo import (
    AdvEstimatorConfig,
    AsyncGRPOConfig,
    GRPOConfig,
    MasterConfig,
    RewardPenaltyConfig,
    RewardScalingConfig,
    _apply_configured_message_level_advantage_penalties,
    _apply_mask_sample_filter,
    _apply_message_level_advantage_penalties,
    _get_grpo_save_state,
    _initial_grpo_save_state,
    _initial_policy_generation_stale,
    _needs_hf_refit_handshake,
    _raise_if_reward_penalties_enabled_without_nemo_gym,
    _resolve_logprob_skip_flags,
    _resolve_message_level_advantage_penalties,
    _save_async_replay_buffer_checkpoint,
    _validate_multimodal_dedup_capability,
    _validate_use_kl_in_reward_compat,
    aggregate_rollout_metrics,
    async_grpo_train,
    compute_and_apply_seq_logprob_error_masking,
    dynamic_sampling,
    grpo_train,
    refit_policy_generation,
    setup,
    shutdown_environments,
    validate,
)
from nemo_rl.algorithms.grpo_sync import _train_fields_for_step, grpo_train_sync
from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.reward_functions import (
    RewardShapingConfig,
    apply_reward_shaping,
)
from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.nemo_gym import should_use_nemo_gym
from nemo_rl.experience.interfaces import NEXT_NEMO_GYM_TASK_INDEX_KEY
from nemo_rl.experience.rollouts import calculate_rewards
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.dynamo import DynamoConfig
from nemo_rl.models.generation.interfaces import should_use_async_rollouts
from nemo_rl.models.generation.megatron import MegatronGeneration
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from nemo_rl.utils.timer import Timer
from tests.unit.algorithms.utils import (
    create_mock_batch,
)


def _mock_policy_generation() -> MagicMock:
    """Generation-interface stand-in for grpo_train / async_grpo_train tests."""
    policy_generation = MagicMock(spec=MegatronGeneration)
    policy_generation.requires_kv_scale_sync = False
    policy_generation.get_logger_metrics.return_value = {}
    return policy_generation


def test_save_async_replay_buffer_checkpoint(tmp_path):
    replay_buffer = MagicMock()
    replay_buffer.save_to_path.remote.return_value = 7

    with patch("nemo_rl.algorithms.grpo.ray.get", side_effect=lambda value: value):
        count = _save_async_replay_buffer_checkpoint(
            replay_buffer,
            str(tmp_path),
        )

    assert count == 7
    replay_buffer.save_to_path.remote.assert_called_once_with(
        str(tmp_path / "replay_buffer.pt")
    )


@patch("nemo_rl.algorithms.grpo.ray")
def test_refit_policy_generation_forwards_kv_scales_on_colocated_ipc(
    mock_ray: MagicMock,
) -> None:
    mock_ray.get.return_value = [True]
    policy = MagicMock()
    policy_generation = MagicMock()
    # Match VllmGeneration's default; a bare MagicMock would auto-create a truthy
    # weight_synchronizer and refit_policy_generation would delegate to it instead
    # of taking the colocated IPC path under test.
    policy_generation.weight_synchronizer = None
    kv_scales = {"layer.0": 0.5}

    refit_policy_generation(
        policy,
        policy_generation,
        colocated_inference=True,
        _refit_buffer_size_gb=1.0,
        kv_scales=kv_scales,
    )

    policy.stream_weights_via_ipc_zmq.assert_called_once_with(
        buffer_size_bytes=1024**3,
        kv_scales=kv_scales,
    )


class TestMaskSampleFilter:
    def test_masks_env_flagged_samples(self):
        repeated_batch = BatchedDataDict(
            {
                "loss_multiplier": torch.tensor([1.0, 0.5, 1.0]),
                "mask_sample": torch.tensor([False, True, True]),
            }
        )

        num_masked = _apply_mask_sample_filter(repeated_batch)

        assert num_masked == 2
        assert torch.equal(
            repeated_batch["loss_multiplier"], torch.tensor([1.0, 0.0, 0.0])
        )

    def test_masks_list_valued_mask_sample(self):
        repeated_batch = BatchedDataDict(
            {
                "loss_multiplier": torch.tensor([1.0, 0.5, 1.0]),
                "mask_sample": [True, False, True],
            }
        )

        num_masked = _apply_mask_sample_filter(repeated_batch)

        assert num_masked == 2
        assert torch.equal(
            repeated_batch["loss_multiplier"], torch.tensor([0.0, 0.5, 0.0])
        )

    def test_missing_mask_sample_is_noop(self):
        repeated_batch = BatchedDataDict(
            {"loss_multiplier": torch.tensor([1.0, 0.5, 1.0])}
        )

        num_masked = _apply_mask_sample_filter(repeated_batch)

        assert num_masked == 0
        assert torch.equal(
            repeated_batch["loss_multiplier"], torch.tensor([1.0, 0.5, 1.0])
        )


def test_initial_policy_generation_stale() -> None:
    generation = MagicMock()
    generation.weight_synchronizer.is_stale = False

    assert not _initial_policy_generation_stale(generation, completed_steps=0)
    assert _initial_policy_generation_stale(generation, completed_steps=1)

    generation.weight_synchronizer.is_stale = True
    assert _initial_policy_generation_stale(generation, completed_steps=0)


@pytest.fixture
def mock_grpo_components():
    # Create mock components
    policy = MagicMock()
    policy.train.return_value = {
        "loss": torch.tensor(0.5),
        "grad_norm": torch.tensor(1.0),
        "all_mb_metrics": {
            "loss": [0.5],
            "policy_gradient_loss": [0.3],
            "value_loss": [0.2],
            "global_valid_toks": [10],
            "token_mult_prob_error": [
                1.0
            ],  # Must be <= 1.05 to avoid logging extra plots
            "gen_kl_error": [0.0001],
        },
    }
    policy.generate.return_value = {
        "output_ids": torch.randint(0, 100, (2, 20)),
        "generation_lengths": torch.tensor([10, 15]),
        "unpadded_sequence_lengths": torch.tensor([12, 18]),
        "logprobs": torch.randn(2, 20),
    }
    policy.prepare_for_training.return_value = None
    # Mock sharding annotations for async GRPO
    policy.sharding_annotations.get_axis_size.return_value = 1  # data_parallel size

    # Create mock batch with proper structure
    mock_batch = BatchedDataDict[DatumSpec](
        {
            "message_log": [
                [
                    {
                        "role": "user",
                        "content": "test",
                        "token_ids": torch.tensor([1, 2, 3]),
                    },
                ]
            ],
            "task_name": ["math"],
            "extra_env_info": [{}],
            "loss_multiplier": torch.tensor([1.0]),
            "idx": torch.tensor([0]),
            "length": torch.tensor([3]),  # Add length field for GRPO
            "total_reward": torch.tensor(
                [1.0]
            ),  # Add total_reward for rollout processing
        }
    )

    # Create mock dataloader with 10 batches
    train_dataloader = MagicMock(spec=StatefulDataLoader)

    def train_iter(self):
        return iter([mock_batch] * 10)

    train_dataloader.__iter__ = train_iter
    train_dataloader.__len__ = MagicMock(return_value=10)

    val_dataloader = MagicMock(spec=StatefulDataLoader)

    def val_iter(self):
        return iter([mock_batch] * 10)

    val_dataloader.__iter__ = val_iter
    val_dataloader.__len__ = MagicMock(return_value=10)

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0

    loss_config = ClippedPGLossConfig(
        ratio_clip_min=0.8, ratio_clip_max=1.2, ratio_clip_c=1.0
    )
    loss_fn = ClippedPGLossFn(loss_config)
    logger = MagicMock()
    checkpointer = MagicMock()

    # Create mock environment
    task_to_env = {"math": MagicMock()}
    val_task_to_env = {"math": MagicMock()}

    # Mock environment return values
    for env in [task_to_env["math"], val_task_to_env["math"]]:
        env.step.return_value = (
            [{"role": "environment", "content": "correct"}],  # observations
            [{}],  # metadata
            [[]],  # next_stop_strings
            [1.0],  # rewards
            [True],  # terminateds
            [None],  # answers
        )
        env.global_post_process_and_metrics.return_value = (mock_batch, {})

    # Create mock master config
    master_config = MasterConfig.model_construct(
        **{
            "grpo": GRPOConfig.model_construct(
                max_num_steps=5,
                max_num_epochs=2,
                num_prompts_per_step=1,
                num_generations_per_prompt=1,
                max_rollout_turns=1,
                val_period=100,
                val_start_at=-1,
                val_num_generations_per_prompt=1,
                val_batch_size=1,
                val_at_start=False,
                val_at_end=False,
                max_val_samples=10,
                stop_at_validation_metric=None,
                stop_at_validation_threshold=None,
                seed=42,
                advantage_normalization="global",
                use_leave_one_out_baseline=False,
                normalize_rewards=False,
                overlong_filtering=False,
                advantage_clip_low=None,
                advantage_clip_high=None,
                reward_scaling=RewardScalingConfig.model_construct(enabled=False),
                reward_shaping=RewardShapingConfig.model_construct(enabled=False),
                use_dynamic_sampling=False,
                async_grpo=AsyncGRPOConfig.model_construct(
                    enabled=False,
                    max_trajectory_age_steps=1,
                    max_generation_failures=0,
                ),
                seq_logprob_error_threshold=None,
                adv_estimator=AdvEstimatorConfig.model_construct(
                    name="grpo",
                    use_leave_one_out_baseline=False,
                    normalize_rewards=True,
                ),
            ),
            "policy": {
                "precision": "bfloat16",
                "train_global_batch_size": 1,
                "train_micro_batch_size": 1,
                "max_total_sequence_length": 2048,
                "make_sequence_length_divisible_by": 1,
                "generation": {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": None,
                    "val_temperature": 1.0,
                    "val_top_p": 1.0,
                    "val_top_k": None,
                    "backend": "vllm",
                    "colocated": {"enabled": True},
                    "vllm_cfg": {"async_engine": True},  # Support async mode
                },
            },
            "loss_fn": ClippedPGLossConfig(
                use_importance_sampling_correction=True  # Required for async mode
            ),
            "checkpointing": {
                "enabled": False,
                "checkpoint_must_save_by": None,
                "save_period": 10,
            },
            "cluster": {
                "num_nodes": 1,
                "gpus_per_node": 2,
            },
            "logger": {
                "num_val_samples_to_print": 5,
            },
            "data": {
                "use_multiple_dataloader": False,
            },
            "env": {},
        }
    )

    return {
        "policy": policy,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "tokenizer": tokenizer,
        "loss_fn": loss_fn,
        "logger": logger,
        "checkpointer": checkpointer,
        "task_to_env": task_to_env,
        "val_task_to_env": val_task_to_env,
        "master_config": master_config,
    }


def test_get_grpo_save_state_handles_legacy_checkpoint_and_filters_metrics():
    assert _get_grpo_save_state({}) == _initial_grpo_save_state()

    loaded_state = {
        "consumed_samples": 32,
        "current_step": 3,
        "current_epoch": 1,
        "total_steps": 13,
        "val:accuracy": 0.75,
    }

    save_state = _get_grpo_save_state(loaded_state)

    assert vars(save_state) == {
        "consumed_samples": 32,
        "current_step": 3,
        "current_epoch": 1,
        "total_steps": 13,
        "total_valid_tokens": 0,
        "val_reward": -99999999.0,
        # SingleController-only field; None for every other algorithm.
        "sampler_name": None,
    }
    assert "total_valid_tokens" not in loaded_state
    assert not hasattr(save_state, "val:accuracy")


def test_grpo_save_state_checkpoint_round_trip():
    save_state = _initial_grpo_save_state()
    save_state.current_step = 4
    save_state.total_steps = 4
    save_state.total_valid_tokens = 128
    save_state.val_reward = 0.8
    setattr(save_state, "val:accuracy", 0.8)

    restored_state = _get_grpo_save_state(vars(save_state))

    assert restored_state.current_step == 4
    assert restored_state.total_steps == 4
    assert restored_state.total_valid_tokens == 128
    assert restored_state.val_reward == 0.8
    assert not hasattr(restored_state, "val:accuracy")


def test_grpo_config_dynamic_sampling_default_matches_exemplar():
    assert GRPOConfig().dynamic_sampling_max_gen_batches == 10


def test_grpo_config_nested_defaults_are_populated():
    first = GRPOConfig()
    second = GRPOConfig()

    assert isinstance(first.async_grpo, AsyncGRPOConfig)
    assert isinstance(first.adv_estimator, AdvEstimatorConfig)
    assert isinstance(first.reward_shaping, RewardShapingConfig)
    assert isinstance(first.reward_scaling, RewardScalingConfig)
    assert first.async_grpo.enabled is False
    assert first.async_grpo.max_generation_failures == 0
    assert first.adv_estimator.use_leave_one_out_baseline is True
    assert first.adv_estimator.normalize_rewards is True
    assert first.adv_estimator.minus_baseline is True
    assert first.async_grpo is not second.async_grpo
    assert first.adv_estimator is not second.adv_estimator
    assert first.reward_shaping is not second.reward_shaping
    assert first.reward_scaling is not second.reward_scaling


def _mock_seq_logprob_error_result() -> dict[str, object]:
    return {
        "max_seq_mult_prob_error": 0.0,
        "mean_seq_mult_prob_error": 0.0,
        "min_seq_mult_prob_error": 0.0,
        "max_seq_mult_prob_error_after_mask": 0.0,
        "mean_seq_mult_prob_error_after_mask": 0.0,
        "min_seq_mult_prob_error_after_mask": 0.0,
        "num_masked_seqs": 0,
        "masked_correct_pct": 0.0,
    }


def _logged_train_metrics_with_key(logger, key: str):
    for call in logger.log_metrics.call_args_list:
        metrics = call.args[0]
        if call.kwargs.get("prefix") == "train" and key in metrics:
            return metrics
    raise AssertionError(f"No train metrics payload contained {key}")


def test_apply_message_level_advantage_penalties_targets_flagged_message_spans():
    train_data = BatchedDataDict(
        {
            "advantages": torch.tensor(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                ]
            )
        }
    )
    repeated_batch = BatchedDataDict(
        {
            "message_log": [
                [
                    {
                        "role": "user",
                        "token_ids": torch.tensor([1, 2]),
                        "is_invalid_tool_call": True,
                    },
                    {
                        "role": "assistant",
                        "token_ids": torch.tensor([3, 4]),
                        "generation_logprobs": torch.tensor([0.1, 0.2]),
                        "is_invalid_tool_call": True,
                    },
                    {
                        "role": "assistant",
                        "token_ids": torch.tensor([5, 6]),
                        "generation_logprobs": torch.tensor([0.3, 0.4]),
                    },
                ],
                [
                    {
                        "role": "user",
                        "token_ids": torch.tensor([7]),
                    },
                    {
                        "role": "assistant",
                        "token_ids": torch.tensor([8, 9, 10]),
                        "generation_logprobs": torch.tensor([0.5, 0.6, 0.7]),
                        "has_malformed_thinking": True,
                    },
                    {
                        "role": "assistant",
                        "token_ids": torch.tensor([11, 12]),
                        "generation_logprobs": torch.tensor([0.8, 0.9]),
                    },
                ],
            ]
        }
    )
    _apply_message_level_advantage_penalties(
        train_data=train_data,
        message_logs=repeated_batch["message_log"],
        invalid_tool_call_advantage=-5.0,
        malformed_thinking_advantage=-7.0,
    )

    expected = torch.tensor(
        [
            [0.0, 1.0, -5.0, -5.0, 4.0, 5.0],
            [10.0, -7.0, -7.0, -7.0, 14.0, 15.0],
        ]
    )
    torch.testing.assert_close(train_data["advantages"], expected)


def test_apply_message_level_advantage_penalties_materializes_broadcasted_advantages():
    per_sample_advantages = torch.tensor([[1.0], [10.0]])
    train_data = BatchedDataDict({"advantages": per_sample_advantages.expand(2, 6)})
    message_logs = [
        [
            {
                "role": "user",
                "token_ids": torch.tensor([1]),
            },
            {
                "role": "assistant",
                "token_ids": torch.tensor([2, 3, 4]),
                "generation_logprobs": torch.tensor([0.1, 0.2, 0.3]),
                "is_invalid_tool_call": True,
            },
            {
                "role": "assistant",
                "token_ids": torch.tensor([5, 6]),
                "generation_logprobs": torch.tensor([0.4, 0.5]),
            },
        ],
        [
            {
                "role": "user",
                "token_ids": torch.tensor([7, 8]),
            },
            {
                "role": "assistant",
                "token_ids": torch.tensor([9, 10]),
                "generation_logprobs": torch.tensor([0.6, 0.7]),
                "has_malformed_thinking": True,
            },
            {
                "role": "assistant",
                "token_ids": torch.tensor([11, 12]),
                "generation_logprobs": torch.tensor([0.8, 0.9]),
            },
        ],
    ]

    _apply_message_level_advantage_penalties(
        train_data=train_data,
        message_logs=message_logs,
        invalid_tool_call_advantage=-5.0,
        malformed_thinking_advantage=-7.0,
    )

    expected = torch.tensor(
        [
            [1.0, -5.0, -5.0, -5.0, 1.0, 1.0],
            [10.0, 10.0, -7.0, -7.0, 10.0, 10.0],
        ]
    )
    torch.testing.assert_close(train_data["advantages"], expected)
    torch.testing.assert_close(per_sample_advantages, torch.tensor([[1.0], [10.0]]))
    assert train_data["advantages"].stride(-1) != 0


def test_apply_configured_message_level_advantage_penalties_noops_when_disabled(
    mock_grpo_components,
):
    train_data = BatchedDataDict(
        {"advantages": torch.tensor([[1.0, 2.0]], dtype=torch.float32)}
    )
    message_logs = [
        [
            {
                "role": "assistant",
                "token_ids": torch.tensor([1, 2]),
                "generation_logprobs": torch.tensor([0.1, 0.2]),
                "is_invalid_tool_call": True,
            }
        ]
    ]
    master_config = mock_grpo_components["master_config"]

    with patch("nemo_rl.algorithms.grpo.should_use_nemo_gym") as should_use_nemo_gym:
        _apply_configured_message_level_advantage_penalties(
            train_data, message_logs, master_config
        )

    should_use_nemo_gym.assert_not_called()
    torch.testing.assert_close(
        train_data["advantages"], torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    )


def test_apply_configured_message_level_advantage_penalties_uses_config(
    capsys, mock_grpo_components
):
    train_data = BatchedDataDict(
        {"advantages": torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)}
    )
    message_logs = [
        [
            {
                "role": "user",
                "token_ids": torch.tensor([1]),
            },
            {
                "role": "assistant",
                "token_ids": torch.tensor([2, 3]),
                "generation_logprobs": torch.tensor([0.1, 0.2]),
                "is_invalid_tool_call": True,
            },
        ]
    ]
    master_config = mock_grpo_components["master_config"]
    master_config.grpo.invalid_tool_call_advantage = -4.0
    master_config.grpo.malformed_thinking_advantage = -6.0

    with patch(
        "nemo_rl.algorithms.grpo.should_use_nemo_gym", return_value=True
    ) as should_use_nemo_gym:
        _apply_configured_message_level_advantage_penalties(
            train_data, message_logs, master_config, log_config=True
        )

    should_use_nemo_gym.assert_called_once_with(master_config)
    torch.testing.assert_close(
        train_data["advantages"], torch.tensor([[0.0, -4.0, -4.0]])
    )
    captured = capsys.readouterr()
    assert "Invalid tool call advantage: -4.0" in captured.out
    assert "Malformed thinking advantage: -6.0" in captured.out


def test_resolve_message_level_advantage_penalties_requires_nemo_gym(
    mock_grpo_components,
):
    master_config = mock_grpo_components["master_config"]
    master_config.grpo.invalid_tool_call_advantage = -5.0

    with patch("nemo_rl.algorithms.grpo.should_use_nemo_gym", return_value=False):
        with pytest.raises(ValueError, match="NeMo-Gym path"):
            _resolve_message_level_advantage_penalties(master_config)


def test_raise_if_reward_penalties_enabled_without_nemo_gym_noops_when_all_flags_false(
    mock_grpo_components,
):
    master_config = mock_grpo_components["master_config"]
    master_config.reward_penalties = RewardPenaltyConfig()

    _raise_if_reward_penalties_enabled_without_nemo_gym(
        master_config, enable_nemo_gym=False
    )


@pytest.mark.parametrize(
    "penalty_flag",
    [
        "penalize_duplicated_reasoning",
        "penalize_empty_final_answer",
        "penalize_unwanted_tokens",
        "penalize_malformed_think_tag",
    ],
)
def test_raise_if_reward_penalties_enabled_without_nemo_gym_raises(
    mock_grpo_components,
    penalty_flag,
):
    master_config = mock_grpo_components["master_config"]
    penalty_kwargs: dict[str, Any] = {penalty_flag: True}
    if penalty_flag == "penalize_unwanted_tokens":
        penalty_kwargs["token_ids"] = {"unwanted": [2]}
    master_config.reward_penalties = RewardPenaltyConfig(**penalty_kwargs)

    with pytest.raises(ValueError, match="reward_penalties require the NeMo-Gym path"):
        _raise_if_reward_penalties_enabled_without_nemo_gym(
            master_config, enable_nemo_gym=False
        )


def test_raise_if_reward_penalties_enabled_without_nemo_gym_allows_nemo_gym(
    mock_grpo_components,
):
    master_config = mock_grpo_components["master_config"]
    master_config.reward_penalties = RewardPenaltyConfig(
        penalize_empty_final_answer=True
    )

    _raise_if_reward_penalties_enabled_without_nemo_gym(
        master_config, enable_nemo_gym=True
    )


def test_raise_if_message_level_advantage_penalties_enabled_noops_when_unset(
    mock_grpo_components,
):
    from nemo_rl.algorithms.grpo_sync import (
        _raise_if_message_level_advantage_penalties_enabled,
    )

    master_config = mock_grpo_components["master_config"]
    _raise_if_message_level_advantage_penalties_enabled(master_config)


def test_raise_if_message_level_advantage_penalties_enabled_raises_when_set(
    mock_grpo_components,
):
    from nemo_rl.algorithms.grpo_sync import (
        _raise_if_message_level_advantage_penalties_enabled,
    )

    master_config = mock_grpo_components["master_config"]
    master_config.grpo.invalid_tool_call_advantage = -5.0
    with pytest.raises(NotImplementedError, match="data_plane.enabled=true"):
        _raise_if_message_level_advantage_penalties_enabled(master_config)


def test_multimodal_dedup_rejects_unqualified_transfer_paths(
    mock_grpo_components,
):
    master_config = mock_grpo_components["master_config"]
    master_config.grpo.deduplicate_multimodal_data = True
    master_config.policy["generation"]["backend"] = "sglang"

    with pytest.raises(NotImplementedError, match="backend=vllm"):
        _validate_multimodal_dedup_capability(master_config)

    master_config.policy["generation"]["backend"] = "vllm"
    master_config.data_plane = {"enabled": True}
    with pytest.raises(NotImplementedError, match="data_plane.enabled=false"):
        _validate_multimodal_dedup_capability(master_config)

    master_config.data_plane = {"enabled": False}
    _validate_multimodal_dedup_capability(master_config)


def test_grpo_sync_seq_logprob_error_helper_accepts_dict_result(monkeypatch):
    from nemo_rl.algorithms import grpo_sync as grpo_sync_mod

    seq_error_result = {
        "max_seq_mult_prob_error": 2.5,
        "mean_seq_mult_prob_error": 1.5,
        "min_seq_mult_prob_error": 1.0,
        "max_seq_mult_prob_error_after_mask": 1.2,
        "mean_seq_mult_prob_error_after_mask": 1.1,
        "min_seq_mult_prob_error_after_mask": 1.0,
        "num_masked_seqs": 2,
        "masked_correct_pct": 0.5,
    }

    def fake_masking(train_data, rewards, seq_logprob_error_threshold):
        assert seq_logprob_error_threshold == 1.2
        assert rewards.tolist() == [1.0, 0.0]
        train_data["sample_mask"] = torch.tensor([1.0, 0.0])
        return seq_error_result

    monkeypatch.setattr(
        grpo_sync_mod,
        "compute_and_apply_seq_logprob_error_masking",
        fake_masking,
    )

    sample_mask, metrics = grpo_sync_mod._compute_seq_logprob_error_metrics(
        token_mask=torch.ones(2, 3),
        sample_mask=torch.ones(2),
        prev_logprobs=torch.zeros(2, 3),
        generation_logprobs=torch.zeros(2, 3),
        rewards=torch.tensor([1.0, 0.0]),
        seq_logprob_error_threshold=1.2,
    )

    assert torch.equal(sample_mask, torch.tensor([1.0, 0.0]))
    assert metrics["max_seq_mult_prob_error"] == 2.5
    assert metrics["mean_seq_mult_prob_error"] == 1.5
    assert metrics["min_seq_mult_prob_error"] == 1.0
    assert metrics["max_seq_mult_prob_error_after_mask"] == 1.2
    assert metrics["mean_seq_mult_prob_error_after_mask"] == 1.1
    assert metrics["min_seq_mult_prob_error_after_mask"] == 1.0
    assert metrics["num_masked_seqs_by_logprob_error"] == 2
    assert metrics["masked_correct_pct"] == 0.5


# ============================================================================
# Stub classes for async GRPO testing (non-Ray versions for easy mocking)
# ============================================================================


class StubReplayBuffer:
    """Non-Ray stub of ReplayBuffer for unit testing

    Each method returns a MagicMock with a 'remote' attribute that can be called.
    """

    def __init__(self, initial_size=10, mock_batch=None, mock_rollout_metrics=None):
        self._size = initial_size
        self._trajectories = []
        self._mock_batch = mock_batch
        self._mock_rollout_metrics = mock_rollout_metrics or {}

    @property
    def size(self):
        """Return a mock that returns buffer size when .remote() is called"""
        mock = MagicMock()
        mock.remote = MagicMock(return_value=self._size)  # ray.get will extract this
        return mock

    @property
    def sample(self):
        """Return a mock that returns sample result when .remote() is called"""

        def _sample(num_prompt_groups, current_weight_version, max_age_steps):
            # Return proper trajectory structure expected by async GRPO
            trajectories = [
                {
                    "batch": self._mock_batch,
                    "rollout_metrics": self._mock_rollout_metrics,
                }
                for _ in range(num_prompt_groups)
            ]
            return {
                "trajectories": trajectories,
                "avg_trajectory_age": 0.5,
            }

        mock = MagicMock()
        mock.remote = MagicMock(
            side_effect=lambda *args, **kwargs: _sample(*args, **kwargs)
        )
        return mock

    @property
    def get_debug_info(self):
        """Return a mock that returns debug info when .remote() is called"""
        mock = MagicMock()
        mock.remote = MagicMock(
            return_value={
                "total_trajectories": self._size,
                "trajectory_versions": [0],
                "target_weight_versions": [0],
                "max_size": 100,
            }
        )
        return mock

    @property
    def state_dict(self):
        """Return a mock that returns checkpointable buffer state."""
        trajectories = self._trajectories or [
            {
                "batch": self._mock_batch,
                "rollout_metrics": self._mock_rollout_metrics,
            }
            for _ in range(self._size)
        ]
        mock = MagicMock()
        mock.remote = MagicMock(
            return_value={
                "trajectories": list(trajectories),
                "trajectory_versions": [0] * len(trajectories),
                "target_weight_versions": [0] * len(trajectories),
                "last_target_weight_already_generated": 0,
                "max_size": self._size,
            }
        )
        return mock

    @property
    def load_state_dict(self):
        """Return a mock that accepts restored buffer state."""

        def _load_state_dict(state, *args, **kwargs):
            self._trajectories = list(state["trajectories"])
            self._size = len(self._trajectories)

        mock = MagicMock()
        mock.remote = MagicMock(side_effect=_load_state_dict)
        return mock

    @property
    def save_to_path(self):
        """Return a mock that checkpoints state without a driver-sized return."""
        mock = MagicMock()
        mock.remote = MagicMock(return_value=self._size)
        return mock

    @property
    def load_from_path(self):
        """Return compact restore metadata."""
        mock = MagicMock()
        mock.remote = MagicMock(
            return_value={
                "num_trajectories": self._size,
                "next_ng_task_index": 0,
            }
        )
        return mock

    @property
    def get_trajectories_needed(self):
        """Return a mock that reports how many prompt groups are still needed."""
        mock = MagicMock()
        mock.remote = MagicMock(
            side_effect=lambda _target_step, num_prompts_per_step, *_args: max(
                0, num_prompts_per_step - self._size
            )
        )
        return mock

    @property
    def has_complete_batch(self):
        """Return a mock that reports whether the current step can train."""
        mock = MagicMock()
        mock.remote = MagicMock(
            side_effect=lambda _target_step, num_prompts_per_step, *_args: (
                self._size >= num_prompts_per_step
            )
        )
        return mock


class StubAsyncTrajectoryCollector:
    """Non-Ray stub of AsyncTrajectoryCollector for unit testing.

    Actor methods expose MagicMocks with a ``remote`` attribute.
    """

    def __init__(
        self,
        events=None,
        health_side_effect=None,
        remote_error_event=None,
        remote_error_ref=None,
    ):
        self._events = events
        self._remote_error_event = remote_error_event
        self._remote_error_ref = remote_error_ref
        self.check_health = MagicMock()
        self.check_health.remote = MagicMock(
            return_value=None, side_effect=health_side_effect
        )

    def _remote_method(self, event):
        mock = MagicMock()

        def remote(*args, **kwargs):
            if self._events is not None:
                self._events.append(event)
            if event == self._remote_error_event:
                return self._remote_error_ref
            return MagicMock()

        mock.remote = MagicMock(side_effect=remote)
        return mock

    @property
    def start_collection(self):
        """Start collection - returns a remote-callable mock"""
        return self._remote_method("start_collection")

    @property
    def set_weight_version(self):
        """Set weight version - returns a remote-callable mock"""
        return self._remote_method("set_weight_version")

    @property
    def pause(self):
        """Pause collection - returns a remote-callable mock"""
        mock = MagicMock()
        mock.remote = MagicMock(return_value=MagicMock())
        return mock

    @property
    def resume(self):
        """Resume collection - returns a remote-callable mock"""
        mock = MagicMock()
        mock.remote = MagicMock(return_value=MagicMock())
        return mock

    @property
    def prepare_for_refit(self):
        """Prepare for refit - returns a remote-callable mock"""
        mock = MagicMock()
        mock.remote = MagicMock(return_value=MagicMock())
        return mock

    @property
    def resume_after_refit(self):
        """Resume after refit - returns a remote-callable mock"""
        return self._remote_method("resume_after_refit")

    @property
    def stop(self):
        """Stop collection - returns a remote-callable mock"""
        mock = MagicMock()
        mock.remote = MagicMock(return_value=MagicMock())
        return mock

    @property
    def wait_for_stop(self):
        """Wait for stop - returns a remote-callable mock"""
        mock = MagicMock()
        mock.remote = MagicMock(return_value=MagicMock())
        return mock

    @property
    def get_efficiency_metrics(self):
        """Get efficiency metrics - returns a remote-callable mock.

        Returns an empty dict so the driver's efficiency-summary aggregation
        (which iterates ``.items()``) is exercised without real collector timing.
        """
        mock = MagicMock()
        mock.remote = MagicMock(return_value={})
        return mock

    @property
    def get_dataloader_state(self):
        """Return a remote-callable mock yielding a checkpointable dataloader state.

        Exercised by async_grpo_train's checkpoint-save path, which persists the
        collector's dataloader state alongside the replay buffer.
        """
        mock = MagicMock()
        mock.remote = MagicMock(return_value={})
        return mock

    @property
    def get_rollouts_state(self):
        """Return a remote-callable mock yielding collector rollout state."""
        mock = MagicMock()
        mock.remote = MagicMock(return_value={NEXT_NEMO_GYM_TASK_INDEX_KEY: 0})
        return mock


def mock_async_grpo_infrastructure(
    mock_batch,
    mock_rollout_metrics,
    seq_logprob_error_result=None,
    collector_health_side_effect=None,
    collector_events=None,
    refit_side_effect=None,
    collector_remote_error_event=None,
):
    """
    Context manager that mocks all async GRPO infrastructure (Ray actors, venv, etc).

    Returns a dict of patches that can be used as a context manager stack.
    """
    from contextlib import ExitStack

    stack = ExitStack()

    # Create stub instances with mock data
    stub_buffer = StubReplayBuffer(
        initial_size=10,
        mock_batch=mock_batch,
        mock_rollout_metrics=mock_rollout_metrics,
    )
    collector_remote_error_ref = object()
    stub_collector = StubAsyncTrajectoryCollector(
        events=collector_events,
        health_side_effect=collector_health_side_effect,
        remote_error_event=collector_remote_error_event,
        remote_error_ref=collector_remote_error_ref,
    )

    # Patch venv creation
    stack.enter_context(
        patch(
            "nemo_rl.algorithms.grpo.create_local_venv_on_each_node",
            return_value="/fake/venv",
        )
    )
    stack.enter_context(
        patch(
            "nemo_rl.algorithms.grpo.get_actor_python_env", return_value="/fake/python"
        )
    )

    # Patch Ray actor classes to return our stubs
    mock_buffer_cls = MagicMock()
    mock_buffer_cls.options.return_value.remote.return_value = stub_buffer
    stack.enter_context(
        patch("nemo_rl.algorithms.async_utils.ReplayBuffer", mock_buffer_cls)
    )

    mock_collector_cls = MagicMock()
    mock_collector_cls.options.return_value.remote.return_value = stub_collector
    stack.enter_context(
        patch(
            "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector",
            mock_collector_cls,
        )
    )

    # Patch ray.get to return values from our stubs (not remote refs)
    def mock_ray_get(ref, **_kwargs):
        if ref is collector_remote_error_ref:
            raise RuntimeError(f"{collector_remote_error_event} failed")
        # If it's already a plain value (from our stubs), return it
        if isinstance(ref, (int, str, dict, list)):
            return ref
        # If it's a MagicMock, return a default response
        return None

    stack.enter_context(patch("ray.get", side_effect=mock_ray_get))
    stack.enter_context(
        patch("ray.wait", side_effect=lambda refs, **kwargs: (refs, []))
    )
    stack.enter_context(
        patch("ray.kill", return_value=None)
    )  # Mock ray.kill for cleanup

    # Patch the rollout functions used inside async_grpo_train
    stack.enter_context(
        patch(
            "nemo_rl.algorithms.grpo.run_multi_turn_rollout",
            return_value=(mock_batch, mock_rollout_metrics),
        )
    )
    stack.enter_context(
        patch(
            "nemo_rl.algorithms.grpo.run_async_multi_turn_rollout",
            return_value=(mock_batch, mock_rollout_metrics),
        )
    )

    # Patch refit and validate functions
    stack.enter_context(
        patch(
            "nemo_rl.algorithms.grpo.refit_policy_generation",
            side_effect=refit_side_effect,
            return_value=None,
        )
    )
    stack.enter_context(
        patch("nemo_rl.algorithms.grpo.validate", return_value=({}, {}))
    )

    # Mock print_performance_metrics to avoid needing real timing metrics
    stack.enter_context(
        patch("nemo_rl.algorithms.grpo.print_performance_metrics", return_value={})
    )

    # Mock compute_and_apply_seq_logprob_error_masking to avoid needing real logprob data
    seq_logprob_error_result = (
        seq_logprob_error_result or _mock_seq_logprob_error_result()
    )
    stack.enter_context(
        patch(
            "nemo_rl.algorithms.grpo.compute_and_apply_seq_logprob_error_masking",
            return_value=seq_logprob_error_result,
        )
    )

    return stack


def mock_sync_grpo_infrastructure(policy):
    """Context manager that mocks the TQ/data-plane infrastructure of grpo_train_sync.

    Mirrors ``mock_async_grpo_infrastructure``: the Ray rollout actor and the
    TQ round-trips are stubbed so the driver loop runs for real, with small
    real tensors standing in for the per-sample slices the driver computes
    against. ``validate_sync`` is intentionally left unpatched so tests can
    install their own capturing mock.
    """
    stack = ExitStack()

    # Slice returned by the stubbed rollout actor; baseline/std are computed
    # for real on the driver from these fields.
    driver_carry = BatchedDataDict(
        {
            "total_reward": torch.tensor([1.0]),
            "prompt_ids_for_adv": torch.tensor([[1, 2, 3]]),
            "input_lengths": torch.tensor([4]),
            "loss_multiplier": torch.tensor([1.0]),
            "truncated": torch.tensor([False]),
            "length": torch.tensor([3]),
        }
    )
    meta = MagicMock()
    meta.fields = ["input_ids"]
    rollout_metrics = {
        "mean_gen_tokens_per_sample": 10.0,
        "max_gen_tokens": 20,
        "min_gen_tokens": 5,
    }
    rollout_actor = MagicMock()
    rollout_actor.rollout_to_tq.remote.return_value = (
        meta,
        driver_carry,
        rollout_metrics,
        {},
    )
    rollout_actor_cls = MagicMock()
    rollout_actor_cls.options.return_value.remote.return_value = rollout_actor
    stack.enter_context(
        patch("nemo_rl.algorithms.grpo_sync.SyncRolloutActor", rollout_actor_cls)
    )
    stack.enter_context(
        patch("nemo_rl.algorithms.grpo_sync.make_actor_runtime_env", return_value={})
    )
    # The only ray.get on the driver path receives the stub actor's plain tuple.
    stack.enter_context(patch("ray.get", side_effect=lambda ref: ref))

    stack.enter_context(
        patch("nemo_rl.algorithms.grpo_sync.refit_policy_generation", return_value=None)
    )
    stack.enter_context(
        patch(
            "nemo_rl.algorithms.grpo_sync._compute_seq_logprob_error_metrics",
            return_value=(torch.ones(1), _mock_seq_logprob_error_result()),
        )
    )
    adv_estimator = MagicMock()
    adv_estimator.compute_advantage.return_value = torch.zeros(1, 4)
    stack.enter_context(
        patch(
            "nemo_rl.algorithms.grpo_sync._create_advantage_estimator",
            return_value=adv_estimator,
        )
    )
    stack.enter_context(
        patch("nemo_rl.algorithms.grpo_sync.print_performance_metrics", return_value={})
    )

    # TQ-mediated policy methods: per-token slices read back from the data
    # plane, and train results in the same shape as ``policy.train``.
    dp_bank = {
        "generation_logprobs": torch.zeros(1, 4),
        "token_mask": torch.ones(1, 4),
        "prev_logprobs": torch.zeros(1, 4),
        "reference_policy_logprobs": torch.zeros(1, 4),
        "input_ids": torch.ones(1, 4, dtype=torch.long),
    }
    policy.read_from_dataplane.side_effect = lambda meta, select_fields, **kw: (
        BatchedDataDict({k: dp_bank[k].clone() for k in select_fields})
    )
    policy.train_from_meta.return_value = policy.train.return_value
    policy.tq_partition_id = 0

    return stack


def test_async_grpo_propagates_main_loop_collector_failure(mock_grpo_components):
    """A fatal collector health result aborts the trainer and still cleans up."""
    master_config = mock_grpo_components["master_config"]
    master_config.grpo.max_num_steps = 1
    master_config.grpo.val_period = 0
    master_config.grpo.val_at_start = False
    master_config.grpo.val_at_end = False
    master_config.grpo.use_dynamic_sampling = False
    master_config.policy["generation"]["colocated"]["enabled"] = False

    mock_batch = next(iter(mock_grpo_components["train_dataloader"]))
    mock_rollout_metrics = {"mean_gen_tokens_per_sample": 2.0}

    with (
        mock_async_grpo_infrastructure(
            mock_batch,
            mock_rollout_metrics,
            collector_health_side_effect=[
                None,
                RuntimeError("collector health failed"),
            ],
        ),
        pytest.raises(RuntimeError, match="collector health failed"),
    ):
        async_grpo_train(
            mock_grpo_components["policy"],
            _mock_policy_generation(),
            mock_grpo_components["train_dataloader"],
            mock_grpo_components["val_dataloader"],
            mock_grpo_components["tokenizer"],
            mock_grpo_components["loss_fn"],
            mock_grpo_components["task_to_env"],
            mock_grpo_components["val_task_to_env"],
            mock_grpo_components["logger"],
            mock_grpo_components["checkpointer"],
            _initial_grpo_save_state(),
            master_config,
        )

    mock_grpo_components["checkpointer"].shutdown.assert_called_once()
    mock_grpo_components["policy"].shutdown.assert_called_once()


@pytest.mark.parametrize(
    ("generation_config", "expected"),
    [
        ({"backend": "dynamo"}, True),
        ({"backend": "vllm", "vllm_cfg": {"async_engine": False}}, False),
        ({"backend": "vllm", "vllm_cfg": {"async_engine": True}}, True),
        (
            {
                "backend": "vllm",
                "use_async_rollouts": True,
                "vllm_cfg": {"async_engine": False},
            },
            False,
        ),
        ({"backend": "megatron", "mcore_generation_config": {}}, True),
    ],
)
def test_should_use_async_rollouts_selects_backend_specific_config(
    generation_config, expected
):
    assert should_use_async_rollouts(generation_config) is expected


def test_should_use_async_rollouts_rejects_removed_megatron_async_engine():
    with pytest.raises(AssertionError, match="async_engine was removed"):
        should_use_async_rollouts(
            {
                "backend": "megatron",
                "mcore_generation_config": {"async_engine": False},
            }
        )


def test_should_use_nemo_gym_accepts_megatron_always_async():
    master_config = MagicMock()
    master_config.env = {"should_use_nemo_gym": True}
    master_config.policy = {
        "generation": {
            "backend": "megatron",
            "mcore_generation_config": {"expose_http_server": True},
        }
    }

    assert should_use_nemo_gym(master_config)


@pytest.mark.parametrize("backend", ["dynamo", "vllm"])
def test_initial_refit_completes_before_async_collection_starts(
    mock_grpo_components,
    backend,
) -> None:
    master_config = mock_grpo_components["master_config"]
    master_config.policy["generation"]["backend"] = backend
    master_config.policy["generation"]["colocated"]["enabled"] = False
    master_config.grpo.max_num_steps = 1
    master_config.grpo.val_period = 0
    master_config.grpo.val_at_start = False
    master_config.grpo.val_at_end = False
    mock_batch = next(iter(mock_grpo_components["train_dataloader"]))
    rollout_metrics = {"mean_gen_tokens_per_sample": 2.0}
    events = []

    def record_refit(*args, **kwargs):
        events.append("refit")

    with mock_async_grpo_infrastructure(
        mock_batch,
        rollout_metrics,
        collector_events=events,
        refit_side_effect=record_refit,
    ):
        async_grpo_train(
            mock_grpo_components["policy"],
            _mock_policy_generation(),
            mock_grpo_components["train_dataloader"],
            mock_grpo_components["val_dataloader"],
            mock_grpo_components["tokenizer"],
            mock_grpo_components["loss_fn"],
            mock_grpo_components["task_to_env"],
            mock_grpo_components["val_task_to_env"],
            mock_grpo_components["logger"],
            mock_grpo_components["checkpointer"],
            _initial_grpo_save_state(),
            master_config,
        )

    assert events[:3] == ["refit", "set_weight_version", "start_collection"]


def test_async_grpo_awaits_resume_after_refit_failure(mock_grpo_components) -> None:
    master_config = mock_grpo_components["master_config"]
    master_config.policy["generation"]["backend"] = "dynamo"
    master_config.policy["generation"]["colocated"]["enabled"] = False
    master_config.grpo.max_num_steps = 1
    master_config.grpo.val_period = 0
    master_config.grpo.val_at_start = False
    master_config.grpo.val_at_end = False
    mock_batch = next(iter(mock_grpo_components["train_dataloader"]))

    with (
        mock_async_grpo_infrastructure(
            mock_batch,
            {"mean_gen_tokens_per_sample": 2.0},
            collector_remote_error_event="resume_after_refit",
        ),
        pytest.raises(RuntimeError, match="resume_after_refit failed"),
    ):
        async_grpo_train(
            mock_grpo_components["policy"],
            _mock_policy_generation(),
            mock_grpo_components["train_dataloader"],
            mock_grpo_components["val_dataloader"],
            mock_grpo_components["tokenizer"],
            mock_grpo_components["loss_fn"],
            mock_grpo_components["task_to_env"],
            mock_grpo_components["val_task_to_env"],
            mock_grpo_components["logger"],
            mock_grpo_components["checkpointer"],
            _initial_grpo_save_state(),
            master_config,
        )


def test_shutdown_environments_drains_unique_actors_before_kill() -> None:
    shared_environment = MagicMock()
    failing_environment = MagicMock()
    shared_shutdown_ref = object()
    failing_shutdown_ref = object()
    shared_environment.shutdown.remote.return_value = shared_shutdown_ref
    failing_environment.shutdown.remote.return_value = failing_shutdown_ref

    def get_or_fail(ref, timeout=None):
        assert timeout == 10
        if ref is failing_shutdown_ref:
            raise RuntimeError("environment shutdown failed")
        assert ref is shared_shutdown_ref
        return True

    with (
        patch("nemo_rl.algorithms.grpo.ray.get", side_effect=get_or_fail),
        patch("nemo_rl.algorithms.grpo.ray.kill") as ray_kill,
    ):
        shutdown_environments(
            {"train": shared_environment, "failing": failing_environment},
            {"validation": shared_environment},
        )

    shared_environment.shutdown.remote.assert_called_once_with()
    failing_environment.shutdown.remote.assert_called_once_with()
    ray_kill.assert_called_once_with(failing_environment)


def test_should_use_nemo_gym_requires_dynamo_token_wrapper() -> None:
    master_config = MagicMock()
    master_config.env = {"should_use_nemo_gym": True}
    master_config.policy = {
        "generation": {
            "backend": "dynamo",
            "vllm_cfg": {"expose_http_server": False},
        }
    }

    with pytest.raises(AssertionError, match="expose_http_server: true"):
        should_use_nemo_gym(master_config)

    master_config.policy["generation"]["vllm_cfg"]["expose_http_server"] = True
    assert should_use_nemo_gym(master_config) is True


@contextmanager
def _patched_logprob_phase(policy):
    """Provide real tensors for the logprob phase of ``grpo_train``.

    Both PRs #2174 / #2178 (``skip_reference_policy_logprobs_calculation=True``)
    and PR #2177 (``force_on_policy_ratio=True``) use ``torch.zeros_like(...)``
    placeholders inside ``grpo_train``. Those calls fail with
    ``TypeError: zeros_like(): argument 'input' must be Tensor, not MagicMock``
    when the surrounding inputs come straight from the bare ``mock_grpo_components``
    fixture. This helper swaps in real tensors for the duration of the test and
    restores the original mock return values afterwards.
    """
    fake_flat = BatchedDataDict(
        {
            "token_ids": torch.tensor([[1, 2]]),
            "advantages": torch.tensor([[0.5, 0.5]]),
            "generation_logprobs": torch.tensor([[0.0, 0.0]]),
            "token_loss_mask": torch.tensor([[1, 1]]),
            "content": ["ok"],
        }
    )
    fake_lengths = torch.tensor([2])
    saved_lp = policy.get_logprobs.return_value
    saved_rlp = policy.get_reference_policy_logprobs.return_value
    policy.get_logprobs.return_value = {"logprobs": torch.zeros(1, 2)}
    policy.get_reference_policy_logprobs.return_value = {
        "reference_logprobs": torch.zeros(1, 2)
    }
    with patch(
        "nemo_rl.algorithms.grpo.batched_message_log_to_flat_message",
        return_value=(fake_flat, fake_lengths),
    ):
        try:
            yield
        finally:
            policy.get_logprobs.return_value = saved_lp
            policy.get_reference_policy_logprobs.return_value = saved_rlp


@ray.remote(num_cpus=0)
class MockEnvironment(EnvironmentInterface):
    def __init__(self, rewards: list[float]):
        self.rewards = rewards
        self._calls = 0

    def step(
        self, messages: list[LLMMessageLogType], env_info: list[dict]
    ) -> EnvironmentReturn:
        self._calls += 1
        return (
            [{"role": "environment", "content": "observation"}] * len(messages),
            [{}] * len(messages),
            [[]] * len(messages),
            self.rewards,
            [True] * len(messages),
            [None] * len(messages),
        )

    def get_calls(self):
        return self._calls

    def reset_calls(self):
        self._calls = 0
        return True

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        return batch, {}


@pytest.fixture(scope="module")
def mock_env():
    """Create a mock environment for single task tests."""
    env = MockEnvironment.remote(rewards=[1.0, 2.0])
    yield env
    ray.kill(env)


@pytest.fixture(scope="module")
def mock_envs():
    """Create mock environments for multiple task tests."""
    math_env = MockEnvironment.remote(rewards=[1.0, 2.0])
    code_env = MockEnvironment.remote(rewards=[3.0, 4.0])
    yield {"math": math_env, "code": code_env}
    ray.kill(math_env)
    ray.kill(code_env)


@pytest.fixture(autouse=True)
def reset_env_calls(mock_env, mock_envs):
    """Reset call counters before each test."""
    ray.get(mock_env.reset_calls.remote())
    ray.get(mock_envs["math"].reset_calls.remote())
    ray.get(mock_envs["code"].reset_calls.remote())
    yield


def test_calculate_rewards_single_task(mock_env):
    """Test reward calculation with a single task type."""
    task_to_env = {"math": mock_env}

    # Create test data
    task_names = ["math", "math"]
    message_logs = [
        [{"role": "user", "content": "1+1"}, {"role": "assistant", "content": "2"}],
        [{"role": "user", "content": "2+2"}, {"role": "assistant", "content": "4"}],
    ]
    batch = create_mock_batch(2, task_names, message_logs)

    # Calculate rewards
    env_observations, metadata, next_stop_strings, rewards, terminateds, answers = (
        calculate_rewards(batch, task_to_env)
    )

    # Verify results
    assert torch.allclose(rewards, torch.tensor([1.0, 2.0]))
    assert len(env_observations) == 2
    assert len(terminateds) == 2
    assert len(next_stop_strings) == 2
    assert len(metadata) == 2
    assert len(answers) == 2
    assert torch.allclose(rewards, torch.tensor([1.0, 2.0]))
    assert (
        ray.get(mock_env.get_calls.remote()) == 1
    )  # Should only call once for all samples of same task


def test_calculate_rewards_multiple_tasks(mock_envs):
    """Test reward calculation with multiple task types."""
    # Create test data
    task_names = ["math", "math", "code", "code"]
    message_logs = [
        [{"role": "user", "content": "1+1"}, {"role": "assistant", "content": "2"}],
        [{"role": "user", "content": "2+2"}, {"role": "assistant", "content": "4"}],
        [
            {"role": "user", "content": "print('hello')"},
            {"role": "assistant", "content": "hello"},
        ],
        [
            {"role": "user", "content": "print('world')"},
            {"role": "assistant", "content": "world"},
        ],
    ]
    batch = create_mock_batch(4, task_names, message_logs)

    # Calculate rewards
    env_observations, metadata, next_stop_strings, rewards, terminateds, answers = (
        calculate_rewards(batch, mock_envs)
    )

    # Verify results
    assert torch.allclose(rewards, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert len(env_observations) == 4
    assert len(terminateds) == 4
    assert len(next_stop_strings) == 4
    assert len(metadata) == 4
    assert len(answers) == 4
    assert torch.allclose(rewards, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert (
        ray.get(mock_envs["math"].get_calls.remote()) == 1
    )  # One call for all math samples
    assert (
        ray.get(mock_envs["code"].get_calls.remote()) == 1
    )  # One call for all code samples


def test_calculate_rewards_empty_batch(mock_env):
    """Test reward calculation with an empty batch."""
    task_to_env = {"math": mock_env}

    # Create empty test data
    batch = create_mock_batch(0, [], [])

    # Calculate rewards
    env_observations, metadata, next_stop_strings, rewards, terminateds, answers = (
        calculate_rewards(batch, task_to_env)
    )

    # Verify results
    assert len(rewards) == 0
    assert len(env_observations) == 0
    assert len(terminateds) == 0
    assert len(next_stop_strings) == 0
    assert len(metadata) == 0
    assert len(answers) == 0
    assert (
        ray.get(mock_env.get_calls.remote()) == 0
    )  # Should not call environment for empty batch


def test_calculate_rewards_missing_environment():
    """Test reward calculation with a missing environment."""
    # Create test data with unknown task
    task_names = ["unknown_task"]
    message_logs = [[{"role": "user", "content": "test"}]]
    batch = create_mock_batch(1, task_names, message_logs)

    # Try to calculate rewards with missing environment
    task_to_env = {}  # Empty dict means no environments available
    with pytest.raises(
        ValueError, match="No environment found for task type: unknown_task"
    ):
        calculate_rewards(batch, task_to_env)


def test_dapo_dynamic_sampling_filters_nonzero_std(mock_grpo_components):
    """Test that DAPO dynamic sampling only selects prompts with non-zero standard deviation."""
    # Create mock batch data with 6 prompts (2 prompts * 3 generations each)
    batch_size = 6
    message_logs = [
        [
            {"role": "user", "content": f"prompt_{i // 3}"},
            {"role": "assistant", "content": f"response_{i}"},
        ]
        for i in range(batch_size)
    ]
    task_names = ["math"] * batch_size

    # Create batch with some prompts having zero std and others non-zero std
    repeated_batch = create_mock_batch(batch_size, task_names, message_logs)
    repeated_batch["total_reward"] = torch.tensor([1.0, 0.0, 1.0, 0.5, 0.5, 0.0])

    # Mock prompts tensor (2 unique prompts, each repeated 3 times)
    prompts = torch.tensor(
        [
            [1, 2, 3],  # prompt 0
            [1, 2, 3],  # prompt 0
            [1, 2, 3],  # prompt 0
            [4, 5, 6],  # prompt 1
            [4, 5, 6],  # prompt 1
            [4, 5, 6],  # prompt 1
        ]
    )

    # First prompt group has std=0.5 (rewards: 1.0, 0.0, 1.0 -> std ≠ 0)
    # Second prompt group has std=0.25 (rewards: 0.5, 0.5, 0.0 -> std ≠ 0)
    std = torch.tensor(
        [0.5, 0.5, 0.5, 0.25, 0.25, 0.25]
    )  # Both prompts have non-zero std
    baseline = torch.tensor([0.67, 0.67, 0.67, 0.33, 0.33, 0.33])  # Mock baselines

    # Configuration for dynamic sampling
    master_config = mock_grpo_components["master_config"]
    master_config.grpo.use_dynamic_sampling = True
    master_config.grpo.num_prompts_per_step = 2  # Want 2 prompts
    master_config.grpo.num_generations_per_prompt = 3  # Each with 3 generations
    master_config.grpo.dynamic_sampling_max_gen_batches = 5

    timer = Timer()
    dynamic_sampling_num_gen_batches = 1

    # Test dynamic sampling
    result_batch, is_batch_complete, batch_cache, _ = dynamic_sampling(
        repeated_batch,
        std,
        baseline,
        dynamic_sampling_num_gen_batches,
        master_config,
        timer,
    )

    # Since both prompts have non-zero std, all 6 samples should be selected
    assert result_batch.size == 6
    assert is_batch_complete == True
    assert torch.allclose(result_batch["std"], std)
    assert torch.allclose(result_batch["baseline"], baseline)


def test_dapo_dynamic_sampling_filters_zero_std(mock_grpo_components):
    """Test that DAPO dynamic sampling filters out prompts with zero standard deviation."""
    # Create mock batch data
    batch_size = 6
    message_logs = [
        [
            {"role": "user", "content": f"prompt_{i // 3}"},
            {"role": "assistant", "content": f"response_{i}"},
        ]
        for i in range(batch_size)
    ]
    task_names = ["math"] * batch_size

    repeated_batch = create_mock_batch(batch_size, task_names, message_logs)
    repeated_batch["total_reward"] = torch.tensor(
        [1.0, 1.0, 1.0, 0.5, 0.5, 0.0]
    )  # First prompt has same rewards (std=0)

    # Mock prompts tensor
    prompts = torch.tensor(
        [
            [1, 2, 3],  # prompt 0
            [1, 2, 3],  # prompt 0
            [1, 2, 3],  # prompt 0
            [4, 5, 6],  # prompt 1
            [4, 5, 6],  # prompt 1
            [4, 5, 6],  # prompt 1
        ]
    )

    # First prompt has zero std (all rewards are 1.0)
    # Second prompt has non-zero std (rewards: 0.5, 0.5, 0.0)
    std = torch.tensor(
        [0.0, 0.0, 0.0, 0.25, 0.25, 0.25]
    )  # First prompt has zero std, second has non-zero
    baseline = torch.tensor([1.0, 1.0, 1.0, 0.33, 0.33, 0.33])

    master_config = mock_grpo_components["master_config"]
    master_config.grpo.use_dynamic_sampling = True
    master_config.grpo.num_prompts_per_step = 1  # Want 1 prompt only
    master_config.grpo.num_generations_per_prompt = 3
    master_config.grpo.dynamic_sampling_max_gen_batches = 5

    timer = Timer()
    dynamic_sampling_num_gen_batches = 1

    # Test dynamic sampling
    result_batch, is_batch_complete, batch_cache, _ = dynamic_sampling(
        repeated_batch,
        std,
        baseline,
        dynamic_sampling_num_gen_batches,
        master_config,
        timer,
    )

    # Only the second prompt (indices 3,4,5) should be selected since first has zero std
    assert result_batch.size == 3  # Only 3 samples from the second prompt
    assert is_batch_complete == True
    assert torch.allclose(
        result_batch["std"], torch.tensor([0.25, 0.25, 0.25])
    )  # Only non-zero std
    assert torch.allclose(result_batch["baseline"], torch.tensor([0.33, 0.33, 0.33]))

    ## verify that only prompt_1 is selected
    prompts = [
        result_batch["message_log"][i][0]["content"] for i in range(result_batch.size)
    ]
    assert prompts == ["prompt_1", "prompt_1", "prompt_1"]

    # Verify that filtered rewards are correct
    expected_filtered_rewards = torch.tensor(
        [
            0.5,
            0.5,
            0.0,
        ]
    )
    assert torch.allclose(result_batch["filtered_reward"], expected_filtered_rewards)


def test_dapo_dynamic_sampling_preserves_mask_sample_alignment(mock_grpo_components):
    """mask_sample should follow rows through dynamic-sampling filter and slice."""
    batch_size = 9
    message_logs = [
        [
            {"role": "user", "content": f"prompt_{i // 3}"},
            {"role": "assistant", "content": f"response_{i}"},
        ]
        for i in range(batch_size)
    ]
    repeated_batch = create_mock_batch(batch_size, ["math"] * batch_size, message_logs)
    repeated_batch["total_reward"] = torch.tensor(
        [1.0, 0.0, 1.0, 0.5, 0.0, 0.5, 0.2, 0.6, 0.9]
    )
    repeated_batch["mask_sample"] = torch.tensor(
        [False, True, False, True, False, True, False, False, True]
    )

    std = torch.tensor([0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.35, 0.35, 0.35])
    baseline = torch.tensor([0.67, 0.67, 0.67, 0.33, 0.33, 0.33, 0.57, 0.57, 0.57])

    master_config = mock_grpo_components["master_config"]
    master_config.grpo.use_dynamic_sampling = True
    master_config.grpo.num_prompts_per_step = 2
    master_config.grpo.num_generations_per_prompt = 3
    master_config.grpo.dynamic_sampling_max_gen_batches = 5

    result_batch, is_batch_complete, _, _ = dynamic_sampling(
        repeated_batch,
        std,
        baseline,
        dynamic_sampling_num_gen_batches=1,
        master_config=master_config,
        timer=Timer(),
    )

    assert is_batch_complete is True
    assert result_batch.size == 6
    assert torch.equal(
        result_batch["mask_sample"],
        torch.tensor([False, True, False, True, False, True]),
    )


def test_dapo_dynamic_sampling_batch_caching(mock_grpo_components):
    """Test that DAPO dynamic sampling uses batch caching when insufficient non-zero std prompts are found."""
    # Create mock batch with only 1 prompt having non-zero std, but we need 2
    batch_size = 3
    message_logs = [
        [
            {"role": "user", "content": "prompt_0"},
            {"role": "assistant", "content": f"response_{i}"},
        ]
        for i in range(batch_size)
    ]
    task_names = ["math"] * batch_size

    repeated_batch = create_mock_batch(batch_size, task_names, message_logs)
    repeated_batch["total_reward"] = torch.tensor([1.0, 0.0, 0.5])  # Non-zero std

    prompts = torch.tensor(
        [
            [1, 2, 3],  # prompt 0
            [1, 2, 3],  # prompt 0
            [1, 2, 3],  # prompt 0
        ]
    )

    std = torch.tensor([0.4, 0.4, 0.4])  # Only one prompt with non-zero std
    baseline = torch.tensor([0.5, 0.5, 0.5])

    master_config = mock_grpo_components["master_config"]
    master_config.grpo.use_dynamic_sampling = True
    master_config.grpo.num_prompts_per_step = 2  # Need 2 prompts but only have 1
    master_config.grpo.num_generations_per_prompt = 3
    master_config.grpo.dynamic_sampling_max_gen_batches = 5

    timer = Timer()
    dynamic_sampling_num_gen_batches = 1

    # Test dynamic sampling - should indicate batch is not complete
    result_batch, is_batch_complete, batch_cache, _ = dynamic_sampling(
        repeated_batch,
        std,
        baseline,
        dynamic_sampling_num_gen_batches,
        master_config,
        timer,
    )

    # Should have cached the batch but marked as incomplete
    assert (
        result_batch.size == 3
    )  # All samples from the single prompt with non-zero std
    assert is_batch_complete == False  # Not enough prompts, need to continue sampling
    assert batch_cache is not None
    assert batch_cache == result_batch

    # Run dynamic sampling again with the cached batch
    result_batch, is_batch_complete, batch_cache, _ = dynamic_sampling(
        repeated_batch,
        std,
        baseline,
        dynamic_sampling_num_gen_batches,
        master_config,
        timer,
        batch_cache,
    )

    # After running dynamic sampling again, the batch should be complete
    assert (
        result_batch.size == 6
    )  # All samples from the single prompt with non-zero std
    assert is_batch_complete == True
    assert batch_cache is not None


def test_dapo_cache_aligns_deduplicated_media_with_text_only_batch(
    mock_grpo_components,
):
    def make_batch(prompt: str, *, with_media: bool) -> BatchedDataDict:
        message_logs = [
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"response_{i}"},
            ]
            for i in range(3)
        ]
        batch = create_mock_batch(3, ["math"] * 3, message_logs)
        batch["total_reward"] = torch.tensor([1.0, 0.0, 0.5])
        if with_media:
            media = PackedTensor(
                torch.tensor([[1.0]]), dim_to_pack=0
            ).enable_deduplication()
            batch["pixel_values"] = PackedTensor.concat([media] * 3)
        return batch

    master_config = mock_grpo_components["master_config"]
    master_config.grpo.use_dynamic_sampling = True
    master_config.grpo.num_prompts_per_step = 2
    master_config.grpo.num_generations_per_prompt = 3
    master_config.grpo.dynamic_sampling_max_gen_batches = 5
    master_config.grpo.deduplicate_multimodal_data = True
    std = torch.tensor([0.4, 0.4, 0.4])
    baseline = torch.tensor([0.5, 0.5, 0.5])

    _, complete, cache, _ = dynamic_sampling(
        make_batch("visual", with_media=True),
        std,
        baseline,
        dynamic_sampling_num_gen_batches=1,
        master_config=master_config,
        timer=Timer(),
    )
    assert not complete
    assert cache is not None

    result, complete, _, _ = dynamic_sampling(
        make_batch("text", with_media=False),
        std,
        baseline,
        dynamic_sampling_num_gen_batches=2,
        master_config=master_config,
        timer=Timer(),
        batch_cache=cache,
    )

    assert complete
    assert result.size == 6
    assert len(result["pixel_values"]) == 6
    assert len(result["pixel_values"].tensors) == 1
    assert result["pixel_values"].slice([3, 4, 5]).as_tensor() is None


def test_dapo_dynamic_sampling_disabled(mock_grpo_components):
    """Test that when dynamic sampling is disabled, all prompts are kept regardless of std."""
    batch_size = 6
    message_logs = [
        [
            {"role": "user", "content": f"prompt_{i // 3}"},
            {"role": "assistant", "content": f"response_{i}"},
        ]
        for i in range(batch_size)
    ]
    task_names = ["math"] * batch_size

    repeated_batch = create_mock_batch(batch_size, task_names, message_logs)
    repeated_batch["total_reward"] = torch.tensor([1.0, 1.0, 1.0, 0.5, 0.5, 0.0])

    prompts = torch.tensor(
        [
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [4, 5, 6],
            [4, 5, 6],
            [4, 5, 6],
        ]
    )

    # Mix of zero and non-zero std
    std = torch.tensor([0.0, 0.0, 0.0, 0.25, 0.25, 0.25])
    baseline = torch.tensor([1.0, 1.0, 1.0, 0.33, 0.33, 0.33])

    # Disable dynamic sampling
    master_config = mock_grpo_components["master_config"]
    master_config.grpo.use_dynamic_sampling = False
    master_config.grpo.num_prompts_per_step = 2
    master_config.grpo.num_generations_per_prompt = 3
    master_config.grpo.dynamic_sampling_max_gen_batches = 5

    timer = Timer()
    dynamic_sampling_num_gen_batches = 1

    # Test that dynamic sampling is bypassed
    result_batch, is_batch_complete, batch_cache, _ = dynamic_sampling(
        repeated_batch,
        std,
        baseline,
        dynamic_sampling_num_gen_batches,
        master_config,
        timer,
    )

    # All samples should be kept when dynamic sampling is disabled
    assert result_batch.size == 6
    assert is_batch_complete == True
    assert batch_cache is None  # No caching when disabled


def test_dapo_dynamic_sampling_filters_on_raw_metric_after_overlong_shaping(
    mock_grpo_components,
):
    """Regression test for the issue where DAPO dynamic sampling filtered on
    shaped reward std instead of the raw task metric.

    The first prompt group: all responses are raw-wrong (acc=0) but lengths
    differ, so the overlong penalty produces non-zero shaped std. The fix
    must recompute std on the raw (pre-shaping) reward so this group is
    filtered out. The second prompt group has genuinely varied raw rewards
    and must be kept.
    """
    batch_size = 6
    # Vary the assistant response length for the first group so the overlong
    # penalty creates spurious shaped-reward variance.
    response_lengths = [10, 22, 30, 10, 20, 10]
    message_logs = []
    for i, length in enumerate(response_lengths):
        message_logs.append(
            [
                {
                    "role": "user",
                    "content": f"prompt_{i // 3}",
                    "token_ids": torch.tensor([100 + (i // 3), 101, 102]),
                },
                {
                    "role": "assistant",
                    "content": f"response_{i}",
                    "token_ids": torch.arange(length, dtype=torch.long),
                },
            ]
        )
    task_names = ["math"] * batch_size
    repeated_batch = create_mock_batch(batch_size, task_names, message_logs)
    # Group 0: all raw-wrong (acc=0). Group 1: mixed.
    repeated_batch["total_reward"] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 1.0])

    shaping_cfg = RewardShapingConfig(
        enabled=True,
        overlong_buffer_length=5,
        overlong_buffer_penalty=0.5,
        max_response_length=25,
    )
    repeated_batch = apply_reward_shaping(repeated_batch, shaping_cfg)

    # Shaped reward of group 0 is now [0.0, -0.2, -1.0] -> non-zero std.
    shaped_std = repeated_batch["total_reward"][:3].std(unbiased=False)
    assert shaped_std.item() > 0.0

    # Use prompt-only token_ids to identify groups, mirroring the grpo.py
    # call site.
    input_ids = torch.stack([m[0]["token_ids"] for m in repeated_batch["message_log"]])
    rewards = repeated_batch["total_reward"]
    baseline, raw_std = calculate_baseline_and_std_per_prompt(
        input_ids,
        rewards,
        torch.ones_like(rewards),
        leave_one_out_baseline=False,
        std_rewards=repeated_batch["unshaped_total_reward"],
    )

    # Raw std is 0 for the homogeneous group, non-zero for the mixed group.
    assert torch.allclose(raw_std[:3], torch.zeros(3))
    assert (raw_std[3:] > 0).all()

    master_config = mock_grpo_components["master_config"]
    master_config.grpo.use_dynamic_sampling = True
    master_config.grpo.num_prompts_per_step = 1
    master_config.grpo.num_generations_per_prompt = 3
    master_config.grpo.dynamic_sampling_max_gen_batches = 5

    result_batch, is_batch_complete, _, _ = dynamic_sampling(
        repeated_batch,
        raw_std,
        baseline,
        dynamic_sampling_num_gen_batches=1,
        master_config=master_config,
        timer=Timer(),
    )

    # Only the second group should survive — the first group's raw rewards are
    # identical, so filtering on the raw metric drops it.
    assert is_batch_complete is True
    assert result_batch.size == 3
    surviving_prompts = [
        result_batch["message_log"][i][0]["content"] for i in range(result_batch.size)
    ]
    assert surviving_prompts == ["prompt_1", "prompt_1", "prompt_1"]


def test_noncolocated_inference_requires_explicit_gpus_per_node_single_node(
    mock_grpo_components,
):
    """Test that non-colocated inference requires explicit gpus_per_node when policy_nodes=1."""
    from unittest.mock import MagicMock, patch

    from nemo_rl.algorithms.grpo import setup

    master_config = mock_grpo_components["master_config"]
    master_config.policy["generation"]["colocated"] = {
        "enabled": False,  # Non-colocated
        "resources": {
            "gpus_per_node": None,  # This should trigger error
            "num_nodes": None,
        },
    }
    master_config.grpo.val_period = 0
    master_config.grpo.batch_multiplier = 1
    master_config.cluster["num_nodes"] = 1  # Single node, so policy_nodes=1
    master_config.cluster["gpus_per_node"] = 8
    master_config.data["shuffle"] = False
    master_config.data["num_workers"] = 1

    tokenizer = MagicMock()
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=10)

    # Mock everything we don't need to test
    with (
        patch("nemo_rl.algorithms.grpo.Logger") as mock_logger,
        patch("nemo_rl.algorithms.grpo.CheckpointManager") as mock_checkpointer,
        patch("nemo_rl.algorithms.grpo.StatefulDataLoader"),
        pytest.raises(
            AssertionError,
            match="policy.generation.colocated.resources.gpus_per_node must be explicitly set",
        ),
    ):
        # Configure mocks to skip checkpoint loading
        mock_checkpointer.return_value.get_latest_checkpoint_path.return_value = None
        setup(master_config, tokenizer, dataset, None)


def test_dynamo_rejects_colocated_inference_before_setup_side_effects(
    mock_grpo_components,
):
    from nemo_rl.algorithms.grpo import setup

    master_config = mock_grpo_components["master_config"]
    master_config.policy["hf_config_overrides"] = {"rope_theta": 1_000_000.0}
    master_config.policy["generation"] = {
        "backend": "dynamo",
        "dynamo_cfg": {
            "engine": "vllm",
            "startup_timeout_s": 60,
            "request_timeout_s": 60,
            "control_timeout_s": 30,
            "metrics_include_prefixes": None,
            "metrics_exclude_prefixes": None,
            "worker_args": {
                "tool_call_parser": None,
                "reasoning_parser": None,
                "exclude_tools_when_tool_choice_none": True,
                "enable_structural_tag": False,
                "structural_tag_scope": "auto",
                "structural_tag_schema": "auto",
                "custom_jinja_template": None,
                "endpoint_types": ["chat", "completions"],
                "extra_cli_args": [],
            },
            "frontend_args": {
                "tokenizer": "default",
                "tokenizer_cache": False,
                "tokenizer_cache_bytes": 1024,
                "router_mode": "kv",
                "router_reset_states": True,
                "extra_cli_args": [],
            },
        },
        "vllm_cfg": {
            "async_engine": True,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "gpu_memory_utilization": 0.6,
            "max_model_len": 512,
            "kv_cache_dtype": "auto",
            "load_format": "auto",
            "precision": "bfloat16",
            "enforce_eager": True,
            "expose_http_server": False,
            "enable_vllm_metrics_logger": True,
            "vllm_metrics_logger_interval": 1.0,
            "env_vars": {},
        },
        "vllm_kwargs": {},
        "colocated": {
            "enabled": True,
            "resources": {"gpus_per_node": None, "num_nodes": None},
        },
    }

    master_config.grpo.async_grpo.in_flight_weight_updates = True
    with (
        patch("nemo_rl.algorithms.grpo.Logger") as mock_logger,
        pytest.raises(ValueError, match="in_flight_weight_updates must be false"),
    ):
        setup(
            master_config,
            tokenizer=MagicMock(),
            dataset=MagicMock(),
            val_dataset=None,
        )
    mock_logger.assert_not_called()

    master_config.grpo.async_grpo.in_flight_weight_updates = False

    with (
        patch("nemo_rl.algorithms.grpo.Logger") as mock_logger,
        pytest.raises(
            ValueError,
            match="must be false",
        ),
    ):
        setup(
            master_config,
            tokenizer=MagicMock(),
            dataset=MagicMock(),
            val_dataset=None,
        )

    mock_logger.assert_not_called()
    assert (
        master_config.policy["generation"]["vllm_kwargs"]["hf_overrides"]
        == (master_config.policy["hf_config_overrides"])
    )

    del master_config.policy["generation"]["vllm_kwargs"]
    del master_config.policy["hf_config_overrides"]
    with pytest.raises(ValueError, match="must be false"):
        setup(
            master_config,
            tokenizer=MagicMock(),
            dataset=MagicMock(),
            val_dataset=None,
        )
    assert master_config.policy["generation"]["vllm_kwargs"]["hf_overrides"] == {}


def test_setup_initializes_noncolocated_dynamo_with_nemo_gym(monkeypatch) -> None:
    from nemo_rl.algorithms import grpo as grpo_mod

    repo_root = Path(__file__).resolve().parents[3]
    register_omegaconf_resolvers()
    config = OmegaConf.to_container(
        load_config(repo_root / "examples/configs/grpo_math_1B_dynamo.yaml"),
        resolve=True,
    )
    config["cluster"].update({"num_nodes": 2, "gpus_per_node": 4, "segment_size": 1})
    generation = config["policy"]["generation"]
    generation["vllm_cfg"].update(
        {
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 2,
            "expert_parallel_size": 2,
            "expose_http_server": True,
        }
    )
    generation["colocated"]["resources"] = {
        "gpus_per_node": 4,
        "num_nodes": 1,
    }
    config["env"]["should_use_nemo_gym"] = True
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    config["policy"]["generation"] = configure_generation_config(generation, tokenizer)
    master_config = MasterConfig.model_validate(config)

    cluster_instances = []

    class DummyCluster:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.num_gpus_per_node = kwargs["num_gpus_per_node"]
            self.get_placement_groups = MagicMock(return_value=[object()])
            cluster_instances.append(self)

    class DummyLoader:
        def __init__(self, *_args, **_kwargs):
            pass

        def __len__(self):
            return 1

    class DummyCheckpointer:
        def get_latest_checkpoint_path(self):
            return None

        def load_training_info(self, _path):
            return None

        def get_resume_paths(self, _path):
            return None, None

    class DummyPolicy:
        def print_node_ip_and_gpu_id(self):
            pass

    dynamo_init = MagicMock()

    class DummyDynamoGeneration:
        weight_synchronizer = None
        dp_openai_server_base_urls = ["http://dynamo-wrapper.example/v1"]
        frontend_url = "http://dynamo-frontend.example/v1"

        def __init__(self, *, cluster, config, tokenizer, tokenizer_config):
            dynamo_init(
                cluster=cluster,
                config=config,
                tokenizer=tokenizer,
                tokenizer_config=tokenizer_config,
            )

    synchronizer = MagicMock()
    nemo_gym_actor = object()
    spinup_nemo_gym_actor = MagicMock(return_value=nemo_gym_actor)
    monkeypatch.setattr(grpo_mod, "Logger", lambda *_args, **_kwargs: MagicMock())
    monkeypatch.setattr(
        grpo_mod, "CheckpointManager", lambda *_args, **_kwargs: DummyCheckpointer()
    )
    monkeypatch.setattr(
        grpo_mod, "ClippedPGLossFn", lambda *_args, **_kwargs: MagicMock()
    )
    monkeypatch.setattr(grpo_mod, "StatefulDataLoader", DummyLoader)
    monkeypatch.setattr(
        grpo_mod,
        "get_ray_cluster_topology",
        lambda: {
            "train-node": ("nvlink_domain_train", 0),
            "inference-node": ("nvlink_domain_inference", 1),
        },
    )
    monkeypatch.setattr(grpo_mod, "RayVirtualCluster", DummyCluster)
    monkeypatch.setattr(grpo_mod, "Policy", lambda *_args, **_kwargs: DummyPolicy())
    monkeypatch.setattr(grpo_mod, "DynamoGeneration", DummyDynamoGeneration)
    monkeypatch.setattr(
        grpo_mod, "create_weight_synchronizer", lambda **_kwargs: synchronizer
    )
    monkeypatch.setattr(grpo_mod, "spinup_nemo_gym_actor", spinup_nemo_gym_actor)

    dataset = MagicMock()
    dataset.__len__.return_value = 2
    result = setup(master_config, tokenizer, dataset, None)

    train_cluster, inference_cluster = cluster_instances
    assert train_cluster.kwargs["bundle_ct_per_node_list"] == [4]
    assert inference_cluster.kwargs["bundle_ct_per_node_list"] == [4]
    assert train_cluster.kwargs["node_resource_constraints"] == [
        {"nvlink_domain_train": 0.001}
    ]
    assert inference_cluster.kwargs["node_resource_constraints"] is None
    assert result[1].dp_openai_server_base_urls == ["http://dynamo-wrapper.example/v1"]
    assert result[2] is nemo_gym_actor
    dynamo_config = dynamo_init.call_args.kwargs["config"]
    assert dynamo_init.call_args.kwargs["cluster"] is inference_cluster
    assert DynamoConfig.model_validate(dynamo_config).engine_world_size == 4
    synchronizer.init_communicator.assert_called_once_with()
    spinup_nemo_gym_actor.assert_called_once_with(
        env_configs=master_config.env,
        base_urls=["http://dynamo-wrapper.example/v1"],
        model_name=master_config.policy["model_name"],
        tokenizer=tokenizer,
        enable_router_replay=False,
        routed_experts_dtype="int16",
        use_fastokens=False,
    )


def test_noncolocated_inference_requires_explicit_gpus_per_node_multi_node(
    mock_grpo_components,
):
    """Test that non-colocated inference requires explicit gpus_per_node when policy_nodes>1."""
    from unittest.mock import MagicMock, patch

    from nemo_rl.algorithms.grpo import setup

    master_config = mock_grpo_components["master_config"]
    master_config.policy["generation"]["colocated"] = {
        "enabled": False,  # Non-colocated
        "resources": {
            "gpus_per_node": None,  # This should trigger error
            "num_nodes": 1,  # Use 1 node for inference
        },
    }
    master_config.grpo.val_period = 0
    master_config.grpo.batch_multiplier = 1
    # Multi-node, so policy_nodes=1 after subtracting inference
    master_config.cluster["num_nodes"] = 2
    master_config.cluster["gpus_per_node"] = 8
    master_config.data["shuffle"] = False
    master_config.data["num_workers"] = 1

    tokenizer = MagicMock()
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=10)

    # Mock everything we don't need to test
    with (
        patch("nemo_rl.algorithms.grpo.Logger") as mock_logger,
        patch("nemo_rl.algorithms.grpo.CheckpointManager") as mock_checkpointer,
        patch("nemo_rl.algorithms.grpo.StatefulDataLoader"),
        pytest.raises(
            AssertionError,
            match="policy.generation.colocated.resources.gpus_per_node must be explicitly set",
        ),
    ):
        # Configure mocks to skip checkpoint loading
        mock_checkpointer.return_value.get_latest_checkpoint_path.return_value = None
        setup(master_config, tokenizer, dataset, None)


def test_noncolocated_opd_teacher_must_fit_on_one_cluster_node(
    mock_grpo_components,
):
    """Reject teacher placement groups wider than a physical cluster node."""
    from unittest.mock import MagicMock, patch

    from nemo_rl.algorithms.grpo import setup
    from nemo_rl.algorithms.opd import OnPolicyDistillationConfig

    master_config = mock_grpo_components["master_config"]
    master_config.cluster["num_nodes"] = 3
    master_config.cluster["gpus_per_node"] = 4
    master_config.grpo.val_period = 0
    master_config.grpo.batch_multiplier = 1
    master_config.on_policy_distillation = OnPolicyDistillationConfig.model_validate(
        {
            "enabled": True,
            "teacher_model_by_agent_name": {
                "default_teacher": "/checkpoints/default_teacher"
            },
            "non_colocated_teachers": {
                "enabled": True,
                "default_teacher_cfg": {
                    "num_nodes": 1,
                },
            },
        }
    )
    master_config.data["shuffle"] = False
    master_config.data["num_workers"] = 1

    tokenizer = MagicMock()
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=10)

    with (
        patch("nemo_rl.algorithms.grpo.Logger"),
        patch("nemo_rl.algorithms.grpo.CheckpointManager") as mock_checkpointer,
        patch("nemo_rl.algorithms.grpo.StatefulDataLoader"),
        patch(
            "nemo_rl.algorithms.grpo.opd_module.reserve_teacher_clusters"
        ) as mock_reserve_teacher_clusters,
        pytest.raises(
            AssertionError,
            match=(
                "OPD teacher 'default_teacher' requests gpus_per_node=8 > "
                "cluster.gpus_per_node=4"
            ),
        ),
    ):
        mock_checkpointer.return_value.get_latest_checkpoint_path.return_value = None
        setup(master_config, tokenizer, dataset, None)

    mock_reserve_teacher_clusters.assert_not_called()


@pytest.mark.parametrize(
    "initial_skip_flag",
    [None, False],
)
def test_setup_auto_enables_skip_reference_logprobs_with_legacy_policy_factory(
    monkeypatch, mock_grpo_components, initial_skip_flag
):
    from nemo_rl.algorithms import grpo as grpo_mod

    class DummyLogger:
        def log_hyperparams(self, *_args, **_kwargs):
            pass

        def log_metrics(self, *_args, **_kwargs):
            pass

    class DummyCheckpointer:
        def get_latest_checkpoint_path(self):
            return None

        def load_training_info(self, _path):
            return None

        def get_resume_paths(self, _path):
            return None, None

    class DummyLoader:
        def __init__(self, *_args, **_kwargs):
            pass

        def __len__(self):
            return 1

        def load_state_dict(self, _state):
            pass

    class DummyCluster:
        def __init__(self, *_args, **_kwargs):
            pass

        def world_size(self):
            return 1

        def get_master_address_and_port(self):
            return "127.0.0.1", 1234

    class DummyPolicy:
        def print_node_ip_and_gpu_id(self):
            pass

        def init_collective(self, *_args, **_kwargs):
            return []

        def prepare_refit_info(self):
            return {}

        def set_rollout_num_gpus_per_engine(self, _num_gpus_per_engine):
            pass

    def legacy_policy_factory(
        *,
        cluster,
        config,
        tokenizer,
        processor,
        weights_path,
        optimizer_path,
        init_optimizer,
        init_reference_model,
    ):
        del (
            cluster,
            config,
            tokenizer,
            processor,
            weights_path,
            optimizer_path,
            init_optimizer,
            init_reference_model,
        )
        return DummyPolicy()

    class DummySGLangGeneration:
        num_gpus_per_engine = 1

        def finish_generation(self):
            pass

        def prepare_refit_info(self, _state):
            pass

        def init_collective(self, *_args, **_kwargs):
            return []

    monkeypatch.setattr(grpo_mod, "Logger", lambda *_args, **_kwargs: DummyLogger())
    monkeypatch.setattr(
        grpo_mod, "CheckpointManager", lambda *_args, **_kwargs: DummyCheckpointer()
    )
    monkeypatch.setattr(
        grpo_mod, "ClippedPGLossFn", lambda *_args, **_kwargs: MagicMock()
    )
    monkeypatch.setattr(grpo_mod, "StatefulDataLoader", DummyLoader)
    monkeypatch.setattr(grpo_mod, "RayVirtualCluster", DummyCluster)
    monkeypatch.setattr(
        grpo_mod,
        "SGLangGeneration",
        lambda *_args, **_kwargs: DummySGLangGeneration(),
    )
    monkeypatch.setattr(grpo_mod.ray, "get", lambda x: x)

    master_config = mock_grpo_components["master_config"]
    master_config.policy["model_name"] = "fake-model"
    master_config.policy["dtensor_cfg"] = {"enabled": False}
    master_config.policy["megatron_cfg"] = {
        "enabled": False,
        "pipeline_model_parallel_size": 1,
    }
    master_config.policy["generation"]["backend"] = "sglang"
    master_config.policy["generation"]["colocated"] = {
        "enabled": True,
        "resources": {"gpus_per_node": None, "num_nodes": None},
    }
    master_config.policy["generation"]["sglang_cfg"] = {
        "gpus_per_server": 1,
        "dp_size": 1,
        "pp_size": 1,
        "ep_size": 1,
    }
    master_config.loss_fn = ClippedPGLossConfig(reference_policy_kl_penalty=0.0)
    master_config.grpo.val_period = 0
    master_config.grpo.batch_multiplier = 1
    if initial_skip_flag is not None:
        master_config.grpo.skip_reference_policy_logprobs_calculation = (
            initial_skip_flag
        )
    master_config.cluster["gpus_per_node"] = 4
    master_config.data["shuffle"] = False
    master_config.data["num_workers"] = 0

    tokenizer = MagicMock()
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=1)

    grpo_mod.setup(
        master_config,
        tokenizer,
        dataset,
        None,
        policy_factory=legacy_policy_factory,
    )

    assert master_config.grpo.skip_reference_policy_logprobs_calculation is True


def test_setup_starts_nemo_gym_for_trtllm(monkeypatch, mock_grpo_components):
    """Guard the TRT-LLM NeMo-Gym startup path in shared GRPO setup."""
    from nemo_rl.algorithms import grpo as grpo_mod

    class DummyLogger:
        def log_hyperparams(self, *_args, **_kwargs):
            pass

        def log_metrics(self, *_args, **_kwargs):
            pass

    class DummyCheckpointer:
        def get_latest_checkpoint_path(self):
            return None

        def load_training_info(self, _path):
            return None

        def get_resume_paths(self, _path):
            return None, None

    class DummyLoader:
        def __init__(self, *_args, **_kwargs):
            pass

        def __len__(self):
            return 1

    class DummyCluster:
        def __init__(self, *_args, **_kwargs):
            pass

    class DummyPolicy:
        def print_node_ip_and_gpu_id(self):
            pass

        def prepare_refit_info(self):
            return {}

    class DummyTrtllmGeneration:
        dp_openai_server_base_urls = ["http://trtllm.example/v1"]
        weight_synchronizer = None

        def finish_generation(self):
            pass

        def prepare_refit_info(self, _state):
            pass

    nemo_gym_actor = object()
    spinup_nemo_gym_actor = MagicMock(return_value=nemo_gym_actor)
    monkeypatch.setattr(grpo_mod, "Logger", lambda *_args, **_kwargs: DummyLogger())
    monkeypatch.setattr(
        grpo_mod, "CheckpointManager", lambda *_args, **_kwargs: DummyCheckpointer()
    )
    monkeypatch.setattr(
        grpo_mod, "ClippedPGLossFn", lambda *_args, **_kwargs: MagicMock()
    )
    monkeypatch.setattr(grpo_mod, "StatefulDataLoader", DummyLoader)
    monkeypatch.setattr(grpo_mod, "RayVirtualCluster", DummyCluster)
    monkeypatch.setattr(grpo_mod, "Policy", lambda *_args, **_kwargs: DummyPolicy())
    monkeypatch.setattr(
        grpo_mod,
        "TrtllmGeneration",
        lambda *_args, **_kwargs: DummyTrtllmGeneration(),
    )
    monkeypatch.setattr(grpo_mod, "spinup_nemo_gym_actor", spinup_nemo_gym_actor)

    master_config = mock_grpo_components["master_config"]
    master_config.policy["model_name"] = "test-model"
    master_config.policy["tokenizer"] = {"use_fastokens": False}
    master_config.policy["dtensor_cfg"] = {"enabled": False}
    master_config.policy["megatron_cfg"] = {
        "enabled": False,
        "pipeline_model_parallel_size": 1,
    }
    master_config.policy["generation"] = {
        "backend": "trtllm",
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "val_temperature": 1.0,
        "val_top_p": 1.0,
        "val_top_k": None,
        "colocated": {
            "enabled": True,
            "resources": {"gpus_per_node": None, "num_nodes": None},
        },
        "trtllm_cfg": {
            "tensor_parallel_size": 1,
            "async_engine": True,
            "expose_http_server": True,
        },
    }
    master_config.env = {"should_use_nemo_gym": True}
    master_config.loss_fn = ClippedPGLossConfig(reference_policy_kl_penalty=0.0)
    master_config.grpo.val_period = 0
    master_config.grpo.batch_multiplier = 1
    master_config.cluster["gpus_per_node"] = 1
    master_config.data["shuffle"] = False
    master_config.data["num_workers"] = 0

    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=1)
    tokenizer = MagicMock()
    result = grpo_mod.setup(master_config, tokenizer, dataset, None)

    assert result[2] is nemo_gym_actor
    spinup_nemo_gym_actor.assert_called_once_with(
        env_configs=master_config.env,
        base_urls=["http://trtllm.example/v1"],
        model_name="test-model",
        tokenizer=tokenizer,
        enable_router_replay=False,
        routed_experts_dtype="int16",
        use_fastokens=False,
    )


def test_grpo_train_collects_generation_logger_and_seq_metrics(
    monkeypatch, mock_grpo_components
):
    from nemo_rl.algorithms import grpo as grpo_mod

    generation_logger_metrics = {
        "inflight_batch_sizes": {0: [0, 1, 2]},
        "num_pending_samples": {0: [3, 2, 0]},
    }
    seq_logprob_error_result = {
        "max_seq_mult_prob_error": 2.5,
        "mean_seq_mult_prob_error": 1.5,
        "min_seq_mult_prob_error": 1.0,
        "max_seq_mult_prob_error_after_mask": 1.2,
        "mean_seq_mult_prob_error_after_mask": 1.1,
        "min_seq_mult_prob_error_after_mask": 1.0,
        "num_masked_seqs": 2,
        "masked_correct_pct": 0.5,
    }
    policy_generation = MagicMock()
    policy_generation.clear_logger_metrics = MagicMock()
    policy_generation.get_logger_metrics = MagicMock(
        return_value=generation_logger_metrics
    )
    policy_generation.prepare_for_generation = MagicMock()
    policy_generation.finish_generation = MagicMock()

    mock_batch = next(iter(mock_grpo_components["train_dataloader"]))
    mock_rollout_metrics = {"gen_kl_error": 0.0, "mean_gen_tokens_per_sample": 2.0}

    def fake_batched_message_log_to_flat_message(*_args, **_kwargs):
        flat = BatchedDataDict(
            {
                "token_ids": torch.tensor([[1, 2]]),
                "advantages": torch.tensor([[0.5, 0.5]]),
                "generation_logprobs": torch.tensor([[0.0, 0.0]]),
                "token_loss_mask": torch.tensor([[1, 1]]),
                "content": ["ok"],
            }
        )
        return flat, torch.tensor([2])

    monkeypatch.setattr(
        grpo_mod,
        "batched_message_log_to_flat_message",
        fake_batched_message_log_to_flat_message,
    )
    monkeypatch.setattr(
        grpo_mod, "should_use_async_rollouts", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(
        grpo_mod,
        "run_async_multi_turn_rollout",
        lambda *_args, **_kwargs: (mock_batch, mock_rollout_metrics),
    )
    monkeypatch.setattr(
        grpo_mod,
        "run_multi_turn_rollout",
        lambda *_args, **_kwargs: (mock_batch, mock_rollout_metrics),
    )
    monkeypatch.setattr(
        grpo_mod,
        "calculate_baseline_and_std_per_prompt",
        lambda *_args, **_kwargs: (torch.tensor([0.1]), torch.tensor([1.0])),
    )
    monkeypatch.setattr(
        grpo_mod,
        "refit_policy_generation",
        lambda *_args, **_kwargs: {"delta/changed_pct": 4.0},
    )
    monkeypatch.setattr(
        grpo_mod, "print_performance_metrics", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        grpo_mod, "maybe_gpu_profile_step", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        grpo_mod,
        "compute_and_apply_seq_logprob_error_masking",
        lambda *_args, **_kwargs: seq_logprob_error_result,
    )

    master_config = mock_grpo_components["master_config"]
    master_config.grpo.max_num_steps = 1
    master_config.grpo.max_num_epochs = 1
    master_config.grpo.val_period = 0
    master_config.grpo.val_at_start = False
    master_config.grpo.use_dynamic_sampling = False

    grpo_mod.grpo_train(
        mock_grpo_components["policy"],
        policy_generation,
        mock_grpo_components["train_dataloader"],
        mock_grpo_components["val_dataloader"],
        mock_grpo_components["tokenizer"],
        mock_grpo_components["loss_fn"],
        mock_grpo_components["task_to_env"],
        mock_grpo_components["val_task_to_env"],
        mock_grpo_components["logger"],
        mock_grpo_components["checkpointer"],
        _initial_grpo_save_state(),
        master_config,
    )

    assert policy_generation.clear_logger_metrics.called
    assert policy_generation.get_logger_metrics.called
    train_metrics = _logged_train_metrics_with_key(
        mock_grpo_components["logger"], "generation_logger_metrics"
    )
    assert train_metrics["generation_logger_metrics"] == generation_logger_metrics
    assert train_metrics["max_seq_mult_prob_error"] == 2.5
    assert train_metrics["mean_seq_mult_prob_error"] == 1.5
    assert train_metrics["min_seq_mult_prob_error"] == 1.0
    assert train_metrics["max_seq_mult_prob_error_after_mask"] == 1.2
    assert train_metrics["mean_seq_mult_prob_error_after_mask"] == 1.1
    assert train_metrics["min_seq_mult_prob_error_after_mask"] == 1.0
    assert train_metrics["num_masked_seqs_by_logprob_error"] == 2
    assert train_metrics["masked_correct_pct"] == 0.5
    assert any(
        call.args[0] == {"delta/changed_pct": 4.0}
        and call.kwargs.get("prefix") == "refit"
        for call in mock_grpo_components["logger"].log_metrics.call_args_list
    )


def test_grpo_train_shutdown_on_epoch_completion(mock_grpo_components, tmp_path):
    """Regression test for epoch-bounded runs losing the final async checkpoint.

    When training exits because max_num_epochs is reached (not max_num_steps or
    timeout), the last step may start background checkpoint finalization via
    begin_finalization() but never hit the inline shutdown() early returns.
    grpo_train must flush pending finalization in a finally block.
    """
    from nemo_rl.algorithms import grpo as grpo_mod

    mock_batch = next(iter(mock_grpo_components["train_dataloader"]))
    mock_rollout_metrics = {"mean_gen_tokens_per_sample": 2.0}
    policy = mock_grpo_components["policy"]
    checkpointer = mock_grpo_components["checkpointer"]

    master_config = mock_grpo_components["master_config"]
    master_config.grpo.max_num_epochs = 1
    master_config.grpo.max_num_steps = 100
    master_config.grpo.val_period = 0
    master_config.grpo.val_at_start = False
    master_config.grpo.val_at_end = False
    master_config.grpo.use_dynamic_sampling = False
    master_config.checkpointing["enabled"] = True
    master_config.checkpointing["save_period"] = 1000
    master_config.checkpointing["metric_name"] = None

    single_batch_dataloader = MagicMock(spec=StatefulDataLoader)
    single_batch_dataloader.__iter__ = lambda self: iter([mock_batch])
    single_batch_dataloader.__len__ = MagicMock(return_value=1)
    single_batch_dataloader.state_dict = MagicMock(return_value={})

    checkpointer.init_tmp_checkpoint.return_value = "/tmp/checkpoint"
    # grpo_train writes latest_checkpoint_status.json under checkpoint_dir; give
    # the mocked checkpointer a real directory so that write succeeds.
    checkpointer.checkpoint_dir = tmp_path

    with (
        _patched_logprob_phase(policy),
        patch(
            "nemo_rl.algorithms.grpo.run_multi_turn_rollout",
            return_value=(mock_batch, mock_rollout_metrics),
        ),
        patch(
            "nemo_rl.algorithms.grpo.run_async_multi_turn_rollout",
            return_value=(mock_batch, mock_rollout_metrics),
        ),
        patch(
            "nemo_rl.algorithms.grpo.compute_and_apply_seq_logprob_error_masking",
            return_value=_mock_seq_logprob_error_result(),
        ),
        patch("nemo_rl.algorithms.grpo.torch.save"),
    ):
        grpo_mod.grpo_train(
            policy,
            _mock_policy_generation(),
            single_batch_dataloader,
            mock_grpo_components["val_dataloader"],
            mock_grpo_components["tokenizer"],
            mock_grpo_components["loss_fn"],
            mock_grpo_components["task_to_env"],
            mock_grpo_components["val_task_to_env"],
            mock_grpo_components["logger"],
            checkpointer,
            _initial_grpo_save_state(),
            master_config,
        )

    checkpointer.begin_finalization.assert_called_once()
    checkpointer.shutdown.assert_called_once()


@pytest.mark.parametrize("train_func", [grpo_train, async_grpo_train])
def test_grpo_ft_save_period_triggers_periodic_saves(
    mock_grpo_components, train_func, tmp_path
):
    """ft_save_period triggers checkpoint saves independent of save_period.

    Covers both GRPO training paths (grpo_train and async_grpo_train). With
    save_period large enough to only fire on the final step, ft_save_period=2
    must add saves at steps 2 and 4, and the last step (5) is saved as usual.
    """
    mock_batch = next(iter(mock_grpo_components["train_dataloader"]))
    mock_rollout_metrics = {"mean_gen_tokens_per_sample": 2.0}
    policy = mock_grpo_components["policy"]
    checkpointer = mock_grpo_components["checkpointer"]

    master_config = mock_grpo_components["master_config"]
    master_config.grpo.max_num_steps = 5
    master_config.grpo.max_num_epochs = 1
    master_config.grpo.val_period = 0
    master_config.grpo.val_at_start = False
    master_config.grpo.val_at_end = False
    master_config.grpo.use_dynamic_sampling = False
    master_config.checkpointing["enabled"] = True
    master_config.checkpointing["save_period"] = 100  # only the final step saves
    master_config.checkpointing["ft_save_period"] = 2
    master_config.checkpointing["metric_name"] = None

    checkpointer.init_tmp_checkpoint.return_value = "/tmp/checkpoint"
    # Both paths write latest_checkpoint_status.json under checkpoint_dir; give
    # the mocked checkpointer a real directory so that write succeeds.
    checkpointer.checkpoint_dir = tmp_path

    if train_func == async_grpo_train:
        master_config.policy["generation"]["colocated"]["enabled"] = False
        with (
            mock_async_grpo_infrastructure(mock_batch, mock_rollout_metrics),
            _patched_logprob_phase(policy),
            patch("nemo_rl.algorithms.grpo.torch.save"),
        ):
            train_func(
                policy,
                _mock_policy_generation(),
                mock_grpo_components["train_dataloader"],
                mock_grpo_components["val_dataloader"],
                mock_grpo_components["tokenizer"],
                mock_grpo_components["loss_fn"],
                mock_grpo_components["task_to_env"],
                mock_grpo_components["val_task_to_env"],
                mock_grpo_components["logger"],
                checkpointer,
                _initial_grpo_save_state(),
                master_config,
            )
    else:
        with (
            _patched_logprob_phase(policy),
            patch(
                "nemo_rl.algorithms.grpo.run_multi_turn_rollout",
                return_value=(mock_batch, mock_rollout_metrics),
            ),
            patch(
                "nemo_rl.algorithms.grpo.run_async_multi_turn_rollout",
                return_value=(mock_batch, mock_rollout_metrics),
            ),
            patch(
                "nemo_rl.algorithms.grpo.compute_and_apply_seq_logprob_error_masking",
                return_value=_mock_seq_logprob_error_result(),
            ),
            patch("nemo_rl.algorithms.grpo.torch.save"),
        ):
            train_func(
                policy,
                _mock_policy_generation(),
                mock_grpo_components["train_dataloader"],
                mock_grpo_components["val_dataloader"],
                mock_grpo_components["tokenizer"],
                mock_grpo_components["loss_fn"],
                mock_grpo_components["task_to_env"],
                mock_grpo_components["val_task_to_env"],
                mock_grpo_components["logger"],
                checkpointer,
                _initial_grpo_save_state(),
                master_config,
            )

    # ft_save_period=2 -> steps 2, 4; save_period=100 contributes only the last
    # step (5). Each save calls init_tmp_checkpoint(step, ...).
    saved_steps = [c.args[0] for c in checkpointer.init_tmp_checkpoint.call_args_list]
    assert saved_steps == [2, 4, 5]


@pytest.mark.parametrize("train_func", [grpo_train, async_grpo_train])
def test_grpo_train_skips_reference_policy_logprobs(mock_grpo_components, train_func):
    """Regression test for issue #1968 (Bug 1) and PRs #2174 / #2178.

    When skip_reference_policy_logprobs_calculation=True, both grpo_train
    and async_grpo_train MUST NOT call policy.get_reference_policy_logprobs.
    Without the skip guards in grpo.py, training would crash inside
    use_reference_model() because the reference model state was never loaded.
    """
    master_config = mock_grpo_components["master_config"]
    master_config.loss_fn.reference_policy_kl_penalty = 0
    master_config.grpo.skip_reference_policy_logprobs_calculation = True
    master_config.grpo.max_num_steps = 1
    master_config.grpo.max_num_epochs = 1
    master_config.grpo.val_period = 0
    master_config.grpo.val_at_start = False
    master_config.grpo.use_dynamic_sampling = False

    if train_func == async_grpo_train:
        master_config.policy["generation"]["colocated"]["enabled"] = False

    grpo_save_state = _initial_grpo_save_state()
    mock_rollout_metrics = {
        "mean_gen_tokens_per_sample": 10.0,
        "max_gen_tokens": 20,
        "min_gen_tokens": 5,
    }
    mock_batch = next(iter(mock_grpo_components["train_dataloader"]))
    policy = mock_grpo_components["policy"]

    if train_func == async_grpo_train:
        with (
            mock_async_grpo_infrastructure(mock_batch, mock_rollout_metrics),
            _patched_logprob_phase(policy),
        ):
            train_func(
                policy,
                _mock_policy_generation(),
                mock_grpo_components["train_dataloader"],
                mock_grpo_components["val_dataloader"],
                mock_grpo_components["tokenizer"],
                mock_grpo_components["loss_fn"],
                mock_grpo_components["task_to_env"],
                mock_grpo_components["val_task_to_env"],
                mock_grpo_components["logger"],
                mock_grpo_components["checkpointer"],
                grpo_save_state,
                master_config,
            )
    else:
        with (
            _patched_logprob_phase(policy),
            patch(
                "nemo_rl.algorithms.grpo.run_multi_turn_rollout",
                return_value=(mock_batch, mock_rollout_metrics),
            ),
            patch(
                "nemo_rl.algorithms.grpo.run_async_multi_turn_rollout",
                return_value=(mock_batch, mock_rollout_metrics),
            ),
            patch(
                "nemo_rl.algorithms.grpo.compute_and_apply_seq_logprob_error_masking",
                return_value=_mock_seq_logprob_error_result(),
            ),
        ):
            train_func(
                policy,
                _mock_policy_generation(),
                mock_grpo_components["train_dataloader"],
                mock_grpo_components["val_dataloader"],
                mock_grpo_components["tokenizer"],
                mock_grpo_components["loss_fn"],
                mock_grpo_components["task_to_env"],
                mock_grpo_components["val_task_to_env"],
                mock_grpo_components["logger"],
                mock_grpo_components["checkpointer"],
                grpo_save_state,
                master_config,
            )

    assert not policy.get_reference_policy_logprobs.called, (
        "policy.get_reference_policy_logprobs was called even though "
        "skip_reference_policy_logprobs_calculation=True. "
        "This indicates a regression of issue #1968 / PRs #2174, #2178."
    )


def _run_single_grpo_train_step(mock_grpo_components, train_func, monkeypatch):
    """Run one GRPO training step with rollout/logprob infrastructure mocked."""
    mock_batch = next(iter(mock_grpo_components["train_dataloader"]))
    mock_rollout_metrics = {"mean_gen_tokens_per_sample": 2.0}
    policy = mock_grpo_components["policy"]
    master_config = mock_grpo_components["master_config"]
    master_config.grpo.max_num_steps = 1
    master_config.grpo.max_num_epochs = 1
    master_config.grpo.val_period = 0
    master_config.grpo.val_at_start = False
    master_config.grpo.use_dynamic_sampling = False

    if train_func == async_grpo_train:
        master_config.policy["generation"]["colocated"]["enabled"] = False
        with (
            mock_async_grpo_infrastructure(mock_batch, mock_rollout_metrics),
            _patched_logprob_phase(policy),
        ):
            train_func(
                policy,
                _mock_policy_generation(),
                mock_grpo_components["train_dataloader"],
                mock_grpo_components["val_dataloader"],
                mock_grpo_components["tokenizer"],
                mock_grpo_components["loss_fn"],
                mock_grpo_components["task_to_env"],
                mock_grpo_components["val_task_to_env"],
                mock_grpo_components["logger"],
                mock_grpo_components["checkpointer"],
                _initial_grpo_save_state(),
                master_config,
            )
    else:
        with (
            _patched_logprob_phase(policy),
            patch(
                "nemo_rl.algorithms.grpo.run_multi_turn_rollout",
                return_value=(mock_batch, mock_rollout_metrics),
            ),
            patch(
                "nemo_rl.algorithms.grpo.run_async_multi_turn_rollout",
                return_value=(mock_batch, mock_rollout_metrics),
            ),
            patch(
                "nemo_rl.algorithms.grpo.compute_and_apply_seq_logprob_error_masking",
                return_value=_mock_seq_logprob_error_result(),
            ),
        ):
            train_func(
                policy,
                _mock_policy_generation(),
                mock_grpo_components["train_dataloader"],
                mock_grpo_components["val_dataloader"],
                mock_grpo_components["tokenizer"],
                mock_grpo_components["loss_fn"],
                mock_grpo_components["task_to_env"],
                mock_grpo_components["val_task_to_env"],
                mock_grpo_components["logger"],
                mock_grpo_components["checkpointer"],
                _initial_grpo_save_state(),
                master_config,
            )


@pytest.mark.parametrize("train_func", [grpo_train, async_grpo_train])
def test_grpo_train_clips_advantages_when_configured(
    mock_grpo_components, train_func, monkeypatch
):
    """Advantages passed to policy.train are clamped when clip bounds are set."""
    extreme_advantages = torch.tensor([[-10.0, 15.0]])
    mock_adv_estimator = MagicMock()
    mock_adv_estimator.compute_advantage.return_value = extreme_advantages.clone()
    monkeypatch.setattr(
        "nemo_rl.algorithms.grpo._create_advantage_estimator",
        lambda _cfg: mock_adv_estimator,
    )

    master_config = mock_grpo_components["master_config"]
    master_config.grpo.advantage_clip_low = -2.0
    master_config.grpo.advantage_clip_high = 3.0

    _run_single_grpo_train_step(mock_grpo_components, train_func, monkeypatch)

    policy = mock_grpo_components["policy"]
    policy.train.assert_called_once()
    clipped = policy.train.call_args[0][0]["advantages"]
    assert clipped.min().item() == -2.0
    assert clipped.max().item() == 3.0


@pytest.mark.parametrize("train_func", [grpo_train, async_grpo_train])
def test_grpo_train_preserves_advantages_when_clipping_disabled(
    mock_grpo_components, train_func, monkeypatch
):
    """Advantages are unchanged when advantage_clip_low/high are null."""
    extreme_advantages = torch.tensor([[-10.0, 15.0]])
    mock_adv_estimator = MagicMock()
    mock_adv_estimator.compute_advantage.return_value = extreme_advantages.clone()
    monkeypatch.setattr(
        "nemo_rl.algorithms.grpo._create_advantage_estimator",
        lambda _cfg: mock_adv_estimator,
    )

    master_config = mock_grpo_components["master_config"]
    master_config.grpo.advantage_clip_low = None
    master_config.grpo.advantage_clip_high = None

    _run_single_grpo_train_step(mock_grpo_components, train_func, monkeypatch)

    policy = mock_grpo_components["policy"]
    policy.train.assert_called_once()
    advantages = policy.train.call_args[0][0]["advantages"]
    assert torch.equal(advantages, extreme_advantages)


def test_clip_grpo_advantages_respects_config_bounds():
    """Shared clip helper clamps only when bounds are configured."""
    from nemo_rl.algorithms.grpo import _clip_grpo_advantages

    extreme_advantages = torch.tensor([[-10.0, 15.0], [0.0, 5.0]])

    clipped = _clip_grpo_advantages(
        extreme_advantages.clone(),
        GRPOConfig.model_construct(advantage_clip_low=-2.0, advantage_clip_high=3.0),
    )
    assert clipped.min().item() == -2.0
    assert clipped.max().item() == 3.0

    unclipped = _clip_grpo_advantages(
        extreme_advantages.clone(),
        GRPOConfig.model_construct(advantage_clip_low=None, advantage_clip_high=None),
    )
    assert torch.equal(unclipped, extreme_advantages)


@pytest.mark.parametrize("train_func", [grpo_train, async_grpo_train])
def test_grpo_train_skips_prev_logprobs_when_force_on_policy_ratio(
    mock_grpo_components, train_func
):
    """Regression test for PR #2177.

    When ``loss_fn.force_on_policy_ratio=True``, both ``grpo_train`` and
    ``async_grpo_train`` MUST NOT call ``policy.get_logprobs`` to compute
    ``prev_logprobs`` -- the importance-sampling ratio is forced to 1.0 so
    the prev-policy forward pass would be wasted compute.
    """
    master_config = mock_grpo_components["master_config"]
    master_config.loss_fn.force_on_policy_ratio = True
    master_config.grpo.seq_logprob_error_threshold = None
    master_config.grpo.max_num_steps = 1
    master_config.grpo.max_num_epochs = 1
    master_config.grpo.val_period = 0
    master_config.grpo.val_at_start = False
    master_config.grpo.use_dynamic_sampling = False

    if train_func == async_grpo_train:
        master_config.policy["generation"]["colocated"]["enabled"] = False

    grpo_save_state = _initial_grpo_save_state()
    mock_rollout_metrics = {
        "mean_gen_tokens_per_sample": 10.0,
        "max_gen_tokens": 20,
        "min_gen_tokens": 5,
    }
    mock_batch = next(iter(mock_grpo_components["train_dataloader"]))
    policy = mock_grpo_components["policy"]

    if train_func == async_grpo_train:
        with (
            mock_async_grpo_infrastructure(mock_batch, mock_rollout_metrics),
            _patched_logprob_phase(policy),
        ):
            train_func(
                policy,
                _mock_policy_generation(),
                mock_grpo_components["train_dataloader"],
                mock_grpo_components["val_dataloader"],
                mock_grpo_components["tokenizer"],
                mock_grpo_components["loss_fn"],
                mock_grpo_components["task_to_env"],
                mock_grpo_components["val_task_to_env"],
                mock_grpo_components["logger"],
                mock_grpo_components["checkpointer"],
                grpo_save_state,
                master_config,
            )
    else:
        with (
            _patched_logprob_phase(policy),
            patch(
                "nemo_rl.algorithms.grpo.run_multi_turn_rollout",
                return_value=(mock_batch, mock_rollout_metrics),
            ),
            patch(
                "nemo_rl.algorithms.grpo.run_async_multi_turn_rollout",
                return_value=(mock_batch, mock_rollout_metrics),
            ),
            patch(
                "nemo_rl.algorithms.grpo.compute_and_apply_seq_logprob_error_masking",
                return_value=_mock_seq_logprob_error_result(),
            ),
        ):
            train_func(
                policy,
                _mock_policy_generation(),
                mock_grpo_components["train_dataloader"],
                mock_grpo_components["val_dataloader"],
                mock_grpo_components["tokenizer"],
                mock_grpo_components["loss_fn"],
                mock_grpo_components["task_to_env"],
                mock_grpo_components["val_task_to_env"],
                mock_grpo_components["logger"],
                mock_grpo_components["checkpointer"],
                grpo_save_state,
                master_config,
            )

    assert not policy.get_logprobs.called, (
        "policy.get_logprobs was called even though force_on_policy_ratio=True. "
        "This indicates a regression of PR #2177."
    )


@pytest.mark.parametrize("train_func", [grpo_train, async_grpo_train, grpo_train_sync])
@pytest.mark.parametrize(
    ("val_at_end", "expected_validation_steps"),
    [(False, [4]), (True, [4, 5])],
)
def test_periodic_validation_starts_at_configured_step(
    mock_grpo_components, train_func, val_at_end, expected_validation_steps
):
    """All three trainers preserve cadence while honoring the validation lower bound."""
    master_config = mock_grpo_components["master_config"]
    master_config.grpo.max_num_steps = 5
    master_config.grpo.val_period = 2
    master_config.grpo.val_start_at = 3
    master_config.grpo.val_at_end = val_at_end
    mock_batch = next(iter(mock_grpo_components["train_dataloader"]))
    mock_rollout_metrics = {
        "mean_gen_tokens_per_sample": 10.0,
        "max_gen_tokens": 20,
        "min_gen_tokens": 5,
    }

    with ExitStack() as stack:
        validate_target = "nemo_rl.algorithms.grpo.validate"
        if train_func is grpo_train_sync:
            master_config.data_plane = {"enabled": True}
            stack.enter_context(
                mock_sync_grpo_infrastructure(mock_grpo_components["policy"])
            )
            validate_target = "nemo_rl.algorithms.grpo_sync.validate_sync"
        elif train_func is async_grpo_train:
            master_config.policy["generation"]["colocated"]["enabled"] = False
            stack.enter_context(
                mock_async_grpo_infrastructure(mock_batch, mock_rollout_metrics)
            )
        else:
            stack.enter_context(
                patch(
                    "nemo_rl.algorithms.grpo.run_multi_turn_rollout",
                    return_value=(mock_batch, mock_rollout_metrics),
                )
            )
            stack.enter_context(
                patch(
                    "nemo_rl.algorithms.grpo.run_async_multi_turn_rollout",
                    return_value=(mock_batch, mock_rollout_metrics),
                )
            )
            stack.enter_context(
                patch(
                    "nemo_rl.algorithms.grpo.compute_and_apply_seq_logprob_error_masking",
                    return_value=_mock_seq_logprob_error_result(),
                )
            )

        mock_validate = stack.enter_context(
            patch(validate_target, return_value=({}, {}))
        )
        train_func(
            mock_grpo_components["policy"],
            _mock_policy_generation(),
            mock_grpo_components["train_dataloader"],
            mock_grpo_components["val_dataloader"],
            mock_grpo_components["tokenizer"],
            mock_grpo_components["loss_fn"],
            mock_grpo_components["task_to_env"],
            mock_grpo_components["val_task_to_env"],
            mock_grpo_components["logger"],
            mock_grpo_components["checkpointer"],
            _initial_grpo_save_state(),
            master_config,
        )

    assert [call.kwargs["step"] for call in mock_validate.call_args_list] == (
        expected_validation_steps
    )


def _enter_stop_test_mocks(
    stack,
    train_func,
    master_config,
    mock_grpo_components,
    mock_batch,
    mock_rollout_metrics,
):
    """Enter per-trainer infrastructure mocks; returns the validate patch target."""
    if train_func is grpo_train_sync:
        master_config.data_plane = {"enabled": True}
        stack.enter_context(
            mock_sync_grpo_infrastructure(mock_grpo_components["policy"])
        )
        return "nemo_rl.algorithms.grpo_sync.validate_sync"
    if train_func is async_grpo_train:
        master_config.policy["generation"]["colocated"]["enabled"] = False
        stack.enter_context(
            mock_async_grpo_infrastructure(mock_batch, mock_rollout_metrics)
        )
        return "nemo_rl.algorithms.grpo.validate"
    stack.enter_context(
        patch(
            "nemo_rl.algorithms.grpo.run_multi_turn_rollout",
            return_value=(mock_batch, mock_rollout_metrics),
        )
    )
    stack.enter_context(
        patch(
            "nemo_rl.algorithms.grpo.run_async_multi_turn_rollout",
            return_value=(mock_batch, mock_rollout_metrics),
        )
    )
    stack.enter_context(
        patch(
            "nemo_rl.algorithms.grpo.compute_and_apply_seq_logprob_error_masking",
            return_value=_mock_seq_logprob_error_result(),
        )
    )
    return "nemo_rl.algorithms.grpo.validate"


@pytest.mark.parametrize("train_func", [grpo_train, async_grpo_train, grpo_train_sync])
def test_training_stops_at_validation_threshold(mock_grpo_components, train_func):
    """All three trainers stop early once the stop metric reaches the threshold."""
    master_config = mock_grpo_components["master_config"]
    master_config.grpo.max_num_steps = 5
    master_config.grpo.val_period = 2
    master_config.grpo.stop_at_validation_metric = "accuracy"
    master_config.grpo.stop_at_validation_threshold = 0.5
    master_config.grpo.val_at_end = False
    mock_batch = next(iter(mock_grpo_components["train_dataloader"]))
    mock_rollout_metrics = {
        "mean_gen_tokens_per_sample": 10.0,
        "max_gen_tokens": 20,
        "min_gen_tokens": 5,
    }

    with ExitStack() as stack:
        validate_target = _enter_stop_test_mocks(
            stack,
            train_func,
            master_config,
            mock_grpo_components,
            mock_batch,
            mock_rollout_metrics,
        )
        mock_validate = stack.enter_context(
            patch(validate_target, return_value=({"accuracy": 0.75}, {}))
        )
        train_func(
            mock_grpo_components["policy"],
            _mock_policy_generation(),
            mock_grpo_components["train_dataloader"],
            mock_grpo_components["val_dataloader"],
            mock_grpo_components["tokenizer"],
            mock_grpo_components["loss_fn"],
            mock_grpo_components["task_to_env"],
            mock_grpo_components["val_task_to_env"],
            mock_grpo_components["logger"],
            mock_grpo_components["checkpointer"],
            _initial_grpo_save_state(),
            master_config,
        )

    # Validation fires at step 2 with accuracy above the threshold, so
    # training stops before the step-4 validation ever runs.
    assert [call.kwargs["step"] for call in mock_validate.call_args_list] == [2]


@pytest.mark.parametrize("train_func", [grpo_train, async_grpo_train, grpo_train_sync])
def test_training_stops_at_initial_validation(mock_grpo_components, train_func):
    """A val_at_start result meeting the threshold stops before any training."""
    master_config = mock_grpo_components["master_config"]
    master_config.grpo.max_num_steps = 5
    master_config.grpo.val_period = 2
    master_config.grpo.val_at_start = True
    master_config.grpo.stop_at_validation_metric = "accuracy"
    master_config.grpo.stop_at_validation_threshold = 0.5
    master_config.grpo.val_at_end = False
    mock_batch = next(iter(mock_grpo_components["train_dataloader"]))
    mock_rollout_metrics = {
        "mean_gen_tokens_per_sample": 10.0,
        "max_gen_tokens": 20,
        "min_gen_tokens": 5,
    }

    with ExitStack() as stack:
        validate_target = _enter_stop_test_mocks(
            stack,
            train_func,
            master_config,
            mock_grpo_components,
            mock_batch,
            mock_rollout_metrics,
        )
        mock_validate = stack.enter_context(
            patch(validate_target, return_value=({"accuracy": 0.75}, {}))
        )
        train_func(
            mock_grpo_components["policy"],
            _mock_policy_generation(),
            mock_grpo_components["train_dataloader"],
            mock_grpo_components["val_dataloader"],
            mock_grpo_components["tokenizer"],
            mock_grpo_components["loss_fn"],
            mock_grpo_components["task_to_env"],
            mock_grpo_components["val_task_to_env"],
            mock_grpo_components["logger"],
            mock_grpo_components["checkpointer"],
            _initial_grpo_save_state(),
            master_config,
        )

    # The initial validation already meets the threshold, so training exits
    # before the periodic step-2/step-4 validations ever run.
    assert [call.kwargs["step"] for call in mock_validate.call_args_list] == [0]


@pytest.mark.parametrize("train_func", [grpo_train, async_grpo_train, grpo_train_sync])
def test_early_stop_saves_final_checkpoint(mock_grpo_components, train_func, tmp_path):
    """The early-stop step is checkpointed before training exits."""
    master_config = mock_grpo_components["master_config"]
    master_config.grpo.max_num_steps = 5
    master_config.grpo.val_period = 2
    master_config.grpo.stop_at_validation_metric = "accuracy"
    master_config.grpo.stop_at_validation_threshold = 0.5
    master_config.grpo.val_at_end = False
    master_config.checkpointing["enabled"] = True
    # save_period alone can never fire, so only the early stop saves.
    master_config.checkpointing["save_period"] = 1000
    master_config.checkpointing["metric_name"] = None
    checkpointer = mock_grpo_components["checkpointer"]
    checkpointer.init_tmp_checkpoint.return_value = str(tmp_path)
    checkpointer.checkpoint_dir = tmp_path

    mock_batch = next(iter(mock_grpo_components["train_dataloader"]))
    mock_rollout_metrics = {
        "mean_gen_tokens_per_sample": 10.0,
        "max_gen_tokens": 20,
        "min_gen_tokens": 5,
    }

    with ExitStack() as stack:
        validate_target = _enter_stop_test_mocks(
            stack,
            train_func,
            master_config,
            mock_grpo_components,
            mock_batch,
            mock_rollout_metrics,
        )
        stack.enter_context(patch("nemo_rl.algorithms.grpo.torch.save"))
        stack.enter_context(patch("nemo_rl.algorithms.grpo_sync.torch.save"))
        mock_validate = stack.enter_context(
            patch(validate_target, return_value=({"accuracy": 0.75}, {}))
        )
        train_func(
            mock_grpo_components["policy"],
            _mock_policy_generation(),
            mock_grpo_components["train_dataloader"],
            mock_grpo_components["val_dataloader"],
            mock_grpo_components["tokenizer"],
            mock_grpo_components["loss_fn"],
            mock_grpo_components["task_to_env"],
            mock_grpo_components["val_task_to_env"],
            mock_grpo_components["logger"],
            checkpointer,
            _initial_grpo_save_state(),
            master_config,
        )

    # Training stopped after the step-2 validation...
    assert [call.kwargs["step"] for call in mock_validate.call_args_list] == [2]
    # ...but only after checkpointing that step with its validation metrics.
    checkpointer.init_tmp_checkpoint.assert_called_once()
    assert checkpointer.init_tmp_checkpoint.call_args.args[0] == 2
    assert checkpointer.init_tmp_checkpoint.call_args.args[1]["val_reward"] == 0.75
    mock_grpo_components["policy"].save_checkpoint.assert_called_once()
    assert checkpointer.shutdown.called


def test_training_stops_on_configured_pass_k_metric(mock_grpo_components):
    """grpo.stop_at_validation_metric=pass_k stops on pass_k, not accuracy."""
    master_config = mock_grpo_components["master_config"]
    master_config.grpo.max_num_steps = 5
    master_config.grpo.val_period = 2
    master_config.grpo.stop_at_validation_threshold = 0.69
    master_config.grpo.stop_at_validation_metric = "pass_k"
    master_config.grpo.val_at_end = False
    mock_batch = next(iter(mock_grpo_components["train_dataloader"]))
    mock_rollout_metrics = {
        "mean_gen_tokens_per_sample": 10.0,
        "max_gen_tokens": 20,
        "min_gen_tokens": 5,
    }

    with (
        patch(
            "nemo_rl.algorithms.grpo.run_multi_turn_rollout",
            return_value=(mock_batch, mock_rollout_metrics),
        ),
        patch(
            "nemo_rl.algorithms.grpo.run_async_multi_turn_rollout",
            return_value=(mock_batch, mock_rollout_metrics),
        ),
        patch(
            "nemo_rl.algorithms.grpo.compute_and_apply_seq_logprob_error_masking",
            return_value=_mock_seq_logprob_error_result(),
        ),
        patch(
            "nemo_rl.algorithms.grpo.validate",
            # accuracy stays below the threshold; only pass_k crosses it.
            return_value=({"accuracy": 0.63, "pass_k": 0.74}, {}),
        ) as mock_validate,
    ):
        grpo_train(
            mock_grpo_components["policy"],
            _mock_policy_generation(),
            mock_grpo_components["train_dataloader"],
            mock_grpo_components["val_dataloader"],
            mock_grpo_components["tokenizer"],
            mock_grpo_components["loss_fn"],
            mock_grpo_components["task_to_env"],
            mock_grpo_components["val_task_to_env"],
            mock_grpo_components["logger"],
            mock_grpo_components["checkpointer"],
            _initial_grpo_save_state(),
            master_config,
        )

    # pass_k (0.74) crosses 0.69 at the first validation (step 2).
    assert [call.kwargs["step"] for call in mock_validate.call_args_list] == [2]


def test_stop_metric_missing_from_validation_fails_loudly(mock_grpo_components):
    """A stop metric that validation does not report raises, not skips."""
    master_config = mock_grpo_components["master_config"]
    master_config.grpo.max_num_steps = 5
    master_config.grpo.val_period = 2
    master_config.grpo.stop_at_validation_threshold = 0.69
    master_config.grpo.stop_at_validation_metric = "pass_k"
    master_config.grpo.val_at_end = False
    mock_batch = next(iter(mock_grpo_components["train_dataloader"]))
    mock_rollout_metrics = {
        "mean_gen_tokens_per_sample": 10.0,
        "max_gen_tokens": 20,
        "min_gen_tokens": 5,
    }

    with (
        patch(
            "nemo_rl.algorithms.grpo.run_multi_turn_rollout",
            return_value=(mock_batch, mock_rollout_metrics),
        ),
        patch(
            "nemo_rl.algorithms.grpo.run_async_multi_turn_rollout",
            return_value=(mock_batch, mock_rollout_metrics),
        ),
        patch(
            "nemo_rl.algorithms.grpo.compute_and_apply_seq_logprob_error_masking",
            return_value=_mock_seq_logprob_error_result(),
        ),
        patch(
            "nemo_rl.algorithms.grpo.validate",
            return_value=({"accuracy": 0.99}, {}),
        ),
        pytest.raises(AssertionError, match="stop_at_validation_metric"),
    ):
        grpo_train(
            mock_grpo_components["policy"],
            _mock_policy_generation(),
            mock_grpo_components["train_dataloader"],
            mock_grpo_components["val_dataloader"],
            mock_grpo_components["tokenizer"],
            mock_grpo_components["loss_fn"],
            mock_grpo_components["task_to_env"],
            mock_grpo_components["val_task_to_env"],
            mock_grpo_components["logger"],
            mock_grpo_components["checkpointer"],
            _initial_grpo_save_state(),
            master_config,
        )


@pytest.mark.parametrize("train_func", [grpo_train, async_grpo_train])
def test_grpo_exit_on_max_steps(mock_grpo_components, train_func):
    """Test that GRPO training loop exits when max_num_steps is reached"""
    # Set max steps to 12
    master_config = mock_grpo_components["master_config"]
    master_config.grpo.max_num_steps = 12

    grpo_save_state = _initial_grpo_save_state()

    # Async GRPO requires non-colocated inference
    if train_func == async_grpo_train:
        master_config.policy["generation"]["colocated"]["enabled"] = False

    # Prepare mock data
    mock_rollout_metrics = {
        "mean_gen_tokens_per_sample": 10.0,
        "max_gen_tokens": 20,
        "min_gen_tokens": 5,
    }
    mock_batch = next(iter(mock_grpo_components["train_dataloader"]))

    # Use our helper to mock async infrastructure if needed
    if train_func == async_grpo_train:
        with mock_async_grpo_infrastructure(mock_batch, mock_rollout_metrics):
            train_func(
                mock_grpo_components["policy"],
                _mock_policy_generation(),
                mock_grpo_components["train_dataloader"],
                mock_grpo_components["val_dataloader"],
                mock_grpo_components["tokenizer"],
                mock_grpo_components["loss_fn"],
                mock_grpo_components["task_to_env"],
                mock_grpo_components["val_task_to_env"],
                mock_grpo_components["logger"],
                mock_grpo_components["checkpointer"],
                grpo_save_state,
                master_config,
            )
    else:
        # For sync grpo_train, just mock the rollout functions
        with patch(
            "nemo_rl.algorithms.grpo.run_multi_turn_rollout",
            return_value=(mock_batch, mock_rollout_metrics),
        ):
            with patch(
                "nemo_rl.algorithms.grpo.run_async_multi_turn_rollout",
                return_value=(mock_batch, mock_rollout_metrics),
            ):
                with patch(
                    "nemo_rl.algorithms.grpo.compute_and_apply_seq_logprob_error_masking",
                    return_value=_mock_seq_logprob_error_result(),
                ):
                    train_func(
                        mock_grpo_components["policy"],
                        _mock_policy_generation(),
                        mock_grpo_components["train_dataloader"],
                        mock_grpo_components["val_dataloader"],
                        mock_grpo_components["tokenizer"],
                        mock_grpo_components["loss_fn"],
                        mock_grpo_components["task_to_env"],
                        mock_grpo_components["val_task_to_env"],
                        mock_grpo_components["logger"],
                        mock_grpo_components["checkpointer"],
                        grpo_save_state,
                        master_config,
                    )

    # Verify we trained for exactly 12 steps
    assert mock_grpo_components["policy"].train.call_count == 12


@pytest.mark.parametrize(
    "train_func", [grpo_train]
)  # Only test sync version for epochs (async uses steps)
def test_grpo_exit_on_max_epochs(mock_grpo_components, train_func):
    """Test that GRPO training loop exits when max_num_epochs is reached"""
    # Set max epochs to 2 and max steps to a large number
    master_config = mock_grpo_components["master_config"]
    master_config.grpo.max_num_epochs = 2
    master_config.grpo.max_num_steps = 100

    grpo_save_state = _initial_grpo_save_state()

    # Mock rollout functions to return proper metrics
    mock_rollout_metrics = {
        "mean_gen_tokens_per_sample": 10.0,
        "max_gen_tokens": 20,
        "min_gen_tokens": 5,
    }

    # Get a mock batch to return
    mock_batch = next(iter(mock_grpo_components["train_dataloader"]))

    with patch("nemo_rl.algorithms.grpo.run_multi_turn_rollout") as mock_rollout:
        mock_rollout.return_value = (mock_batch, mock_rollout_metrics)

        with patch(
            "nemo_rl.algorithms.grpo.run_async_multi_turn_rollout"
        ) as mock_async_rollout:
            mock_async_rollout.return_value = (mock_batch, mock_rollout_metrics)

            with patch(
                "nemo_rl.algorithms.grpo.compute_and_apply_seq_logprob_error_masking",
                return_value=_mock_seq_logprob_error_result(),
            ):
                # Run training
                train_func(
                    mock_grpo_components["policy"],
                    _mock_policy_generation(),
                    mock_grpo_components["train_dataloader"],
                    mock_grpo_components["val_dataloader"],
                    mock_grpo_components["tokenizer"],
                    mock_grpo_components["loss_fn"],
                    mock_grpo_components["task_to_env"],
                    mock_grpo_components["val_task_to_env"],
                    mock_grpo_components["logger"],
                    mock_grpo_components["checkpointer"],
                    grpo_save_state,
                    master_config,
                )

    # Verify we trained for exactly two epochs (20 batches)
    assert mock_grpo_components["policy"].train.call_count == 20


@pytest.mark.parametrize("train_func", [grpo_train, async_grpo_train])
def test_grpo_exit_on_timeout(mock_grpo_components, train_func, capsys):
    """Test that GRPO training loop exits when timeout is reached"""
    # Set max steps and epochs to large numbers
    master_config = mock_grpo_components["master_config"]
    master_config.grpo.max_num_steps = 100
    master_config.grpo.max_num_epochs = 10

    grpo_save_state = _initial_grpo_save_state()

    # Async GRPO requires non-colocated inference
    if train_func == async_grpo_train:
        master_config.policy["generation"]["colocated"]["enabled"] = False

    # Prepare mock data
    mock_rollout_metrics = {
        "mean_gen_tokens_per_sample": 10.0,
        "max_gen_tokens": 20,
        "min_gen_tokens": 5,
    }
    mock_batch = next(iter(mock_grpo_components["train_dataloader"]))

    # Mock TimeoutChecker to return False for first 7 checks, then True (timeout)
    with patch("nemo_rl.algorithms.grpo.TimeoutChecker") as mock_timeout_class:
        mock_timeout_instance = MagicMock()
        check_results = [False] * 7 + [True]
        mock_timeout_instance.check_save.side_effect = check_results
        mock_timeout_class.return_value = mock_timeout_instance

        # Use our helper for async, or simple mocking for sync
        if train_func == async_grpo_train:
            with mock_async_grpo_infrastructure(mock_batch, mock_rollout_metrics):
                train_func(
                    mock_grpo_components["policy"],
                    _mock_policy_generation(),
                    mock_grpo_components["train_dataloader"],
                    mock_grpo_components["val_dataloader"],
                    mock_grpo_components["tokenizer"],
                    mock_grpo_components["loss_fn"],
                    mock_grpo_components["task_to_env"],
                    mock_grpo_components["val_task_to_env"],
                    mock_grpo_components["logger"],
                    mock_grpo_components["checkpointer"],
                    grpo_save_state,
                    master_config,
                )
        else:
            with patch(
                "nemo_rl.algorithms.grpo.run_multi_turn_rollout",
                return_value=(mock_batch, mock_rollout_metrics),
            ):
                with patch(
                    "nemo_rl.algorithms.grpo.run_async_multi_turn_rollout",
                    return_value=(mock_batch, mock_rollout_metrics),
                ):
                    with patch(
                        "nemo_rl.algorithms.grpo.compute_and_apply_seq_logprob_error_masking",
                        return_value=_mock_seq_logprob_error_result(),
                    ):
                        train_func(
                            mock_grpo_components["policy"],
                            _mock_policy_generation(),
                            mock_grpo_components["train_dataloader"],
                            mock_grpo_components["val_dataloader"],
                            mock_grpo_components["tokenizer"],
                            mock_grpo_components["loss_fn"],
                            mock_grpo_components["task_to_env"],
                            mock_grpo_components["val_task_to_env"],
                            mock_grpo_components["logger"],
                            mock_grpo_components["checkpointer"],
                            grpo_save_state,
                            master_config,
                        )

        # Verify training stopped at 8 steps (when check_save returned True)
        assert mock_grpo_components["policy"].train.call_count == 8

        # Verify the timeout message was printed and training actually stopped
        captured = capsys.readouterr()
        output_lines = captured.out.strip().split("\n")

        # Find the timeout message
        timeout_line_idx = None
        for i, line in enumerate(output_lines):
            if "Timeout has been reached, stopping training early" in line:
                timeout_line_idx = i
                break

        assert timeout_line_idx is not None, "Timeout message not found in output"

        # Check what comes after the timeout message
        remaining_lines = output_lines[timeout_line_idx + 1 :]

        # For async_grpo_train, we expect cleanup messages in the finally block
        if train_func.__name__ == "async_grpo_train":
            cleanup_found = any(
                "Stopping trajectory collection" in line
                or "Async GRPO training complete" in line
                for line in remaining_lines
            )
            assert cleanup_found, (
                "Expected cleanup messages after timeout in async mode"
            )

        # Verify no new epoch/step started after timeout
        for line in remaining_lines:
            assert "Epoch" not in line or "Epoch 1/10" in line, (
                f"Training continued to next epoch after timeout: {line}"
            )
            assert not (line.startswith("Step ") and "Step 9" in line), (
                f"Training continued to next step after timeout: {line}"
            )


# ============================================================================
# Tests for GRPOAdvantageEstimator class
# ============================================================================


def test_grpo_advantage_estimator_zero_std():
    """Test GRPOAdvantageEstimator when std contains zeros (all rewards same for a prompt).

    This test verifies that:
    1. When std=0 (all rewards identical for a prompt), normalization is skipped and advantage=0
    2. When std>0, advantages are properly normalized by std
    """
    estimator_config = AdvEstimatorConfig.model_construct(
        use_leave_one_out_baseline=False,
        normalize_rewards=True,
    )
    loss_config = ClippedPGLossConfig()
    estimator = GRPOAdvantageEstimator(estimator_config, loss_config)

    # prompt 0: all same rewards -> std=0; prompt 1: different rewards -> std>0
    prompt_ids = torch.tensor(
        [[0], [0], [1], [1]]
    )  # Shape (4, 1) for unique prompt matching
    rewards = torch.tensor(
        [2.0, 2.0, 1.0, 3.0]
    )  # prompt 0: std=0; prompt 1: std=sqrt(2)
    mask = torch.ones(4, 5)

    result = estimator.compute_advantage(
        prompt_ids=prompt_ids,
        rewards=rewards,
        mask=mask,
    )

    # prompt 0: std=0 -> skip normalization, advantage=0 (reward - mean = 0)
    # prompt 1: With Bessel correction for 2 samples, std = sqrt(2), normalized = ±1/sqrt(2) ≈ ±0.7071
    expected_prompt_0 = torch.zeros(2, 5)  # advantage=0 for all same rewards
    sqrt2_inv = 1.0 / (2.0**0.5)
    expected_prompt_1 = torch.tensor([-sqrt2_inv, sqrt2_inv]).unsqueeze(-1).expand(2, 5)

    assert torch.allclose(result[:2], expected_prompt_0, rtol=1e-5)
    assert torch.allclose(result[2:], expected_prompt_1, rtol=1e-4)


def test_grpo_advantage_estimator_tensor_shapes():
    """Test GRPOAdvantageEstimator with different tensor shapes.

    This test verifies that the estimator works correctly with:
    1. Small batch size (batch=2, single prompt)
    2. Larger batch size (batch=10, single prompt)
    """
    estimator_config = AdvEstimatorConfig.model_construct(
        use_leave_one_out_baseline=False,
        normalize_rewards=True,
    )
    loss_config = ClippedPGLossConfig()
    estimator = GRPOAdvantageEstimator(estimator_config, loss_config)

    # Test with batch size 2
    prompt_ids = torch.tensor([[0], [0]])
    rewards = torch.tensor([1.0, 3.0])  # mean=2, std=sqrt(2) with Bessel
    mask = torch.ones(2, 3)

    result = estimator.compute_advantage(
        prompt_ids=prompt_ids,
        rewards=rewards,
        mask=mask,
    )
    assert result.shape == (2, 3)

    # Verify normalized values: (reward - mean) / std
    # With Bessel correction for 2 samples: std = sqrt(2)
    sqrt2_inv = 1.0 / (2.0**0.5)
    expected = torch.tensor([[-sqrt2_inv], [sqrt2_inv]]).expand(2, 3)
    assert torch.allclose(result, expected, rtol=1e-4)

    # Test with larger batch (10 samples, single prompt)
    prompt_ids = torch.tensor([[0]] * 10)
    rewards = torch.arange(10, dtype=torch.float32)  # 0, 1, 2, ..., 9
    mask = torch.ones(10, 5)

    result = estimator.compute_advantage(
        prompt_ids=prompt_ids,
        rewards=rewards,
        mask=mask,
    )
    assert result.shape == (10, 5)

    # After normalization, mean should be ~0
    result_mean = result.mean()
    assert torch.abs(result_mean) < 1e-5


def test_grpo_advantage_estimator_negative_advantages():
    """Test GRPOAdvantageEstimator with rewards that produce negative advantages.

    This test verifies that negative advantages are handled correctly.
    """
    estimator_config = AdvEstimatorConfig.model_construct(
        use_leave_one_out_baseline=False,
        normalize_rewards=True,
    )
    loss_config = ClippedPGLossConfig()
    estimator = GRPOAdvantageEstimator(estimator_config, loss_config)

    # Rewards with values below and above mean
    prompt_ids = torch.tensor([[0], [0], [0]])
    rewards = torch.tensor([0.0, 2.0, 4.0])  # mean=2, deviations: -2, 0, +2
    mask = torch.ones(3, 4)

    result = estimator.compute_advantage(
        prompt_ids=prompt_ids,
        rewards=rewards,
        mask=mask,
    )

    # Verify ordering: first should be negative, middle ~0, last positive
    assert result[0, 0] < 0  # below mean -> negative advantage
    assert torch.abs(result[1, 0]) < 1e-5  # at mean -> ~0 advantage
    assert result[2, 0] > 0  # above mean -> positive advantage

    # Verify symmetry
    assert torch.allclose(result[0], -result[2], rtol=1e-5)


def test_grpo_advantage_estimator_zero_std_and_zero_advantage():
    """Test GRPOAdvantageEstimator when all rewards are identical (std=0, advantage=0).

    This test verifies that when all rewards for a prompt are the same:
    1. The advantages are all zero (since reward - mean = 0)
    2. No division by zero occurs (normalization is skipped when std=0)
    """
    estimator_config = AdvEstimatorConfig.model_construct(
        use_leave_one_out_baseline=False,
        normalize_rewards=True,
    )
    loss_config = ClippedPGLossConfig()
    estimator = GRPOAdvantageEstimator(estimator_config, loss_config)

    # All rewards identical -> std=0, all advantages=0
    prompt_ids = torch.tensor([[0], [0], [0], [0]])
    rewards = torch.tensor([5.0, 5.0, 5.0, 5.0])  # all same
    mask = torch.ones(4, 3)

    result = estimator.compute_advantage(
        prompt_ids=prompt_ids,
        rewards=rewards,
        mask=mask,
    )

    # All advantages should be exactly 0
    expected = torch.zeros(4, 3)
    assert torch.allclose(result, expected, rtol=1e-5)


def test_grpo_advantage_estimator_small_nonzero_std():
    """Test GRPOAdvantageEstimator with small but non-zero std values.

    This test verifies that small but non-zero std values are still normalized
    (no arbitrary threshold that would skip normalization).
    """
    estimator_config = AdvEstimatorConfig.model_construct(
        use_leave_one_out_baseline=False,
        normalize_rewards=True,
    )
    loss_config = ClippedPGLossConfig()
    estimator = GRPOAdvantageEstimator(estimator_config, loss_config)

    # Small reward differences -> small std but non-zero
    # Use larger difference to avoid floating point precision issues in std calculation
    prompt_ids = torch.tensor([[0], [0]])
    rewards = torch.tensor([1.0, 1.01])  # small but detectable difference
    mask = torch.ones(2, 3)

    result = estimator.compute_advantage(
        prompt_ids=prompt_ids,
        rewards=rewards,
        mask=mask,
    )

    # Even with small std, normalization should still happen
    # After normalization, the values should be ±1/sqrt(2) (for 2 samples with Bessel)
    sqrt2_inv = 1.0 / (2.0**0.5)
    assert torch.allclose(torch.abs(result[0, 0]), torch.tensor(sqrt2_inv), rtol=1e-3)
    assert torch.allclose(torch.abs(result[1, 0]), torch.tensor(sqrt2_inv), rtol=1e-3)

    # Verify opposite signs
    assert result[0, 0] * result[1, 0] < 0


# ============================================================================
# Tests for ReinforcePlusPlusAdvantageEstimator class
# ============================================================================


def test_gdpo_advantage_estimator_multiple_rewards():
    """Test GDPOAdvantageEstimator with multiple rewards."""
    estimator_config = AdvEstimatorConfig.model_construct(
        use_leave_one_out_baseline=False,
        normalize_rewards=True,
    )
    loss_config = ClippedPGLossConfig()
    estimator = GDPOAdvantageEstimator(estimator_config, loss_config)

    prompt_ids = torch.tensor([[0], [0]])
    mask = torch.ones(2, 3)
    repeated_batch = {
        "reward/correctness": torch.tensor([1.0, 1.0]),
        "reward/integer": torch.tensor([1.0, -1.0]),
        "reward/format": torch.tensor([1.0, 0.0]),
    }

    result = estimator.compute_advantage(prompt_ids, None, mask, repeated_batch)
    assert result.shape == (2, 3)
    assert torch.allclose(result[0, 0], torch.tensor(0.7071))
    assert torch.allclose(result[1, 0], torch.tensor(-0.7071))


def test_gdpo_advantage_estimator_single_reward():
    """Test GDPOAdvantageEstimator with multiple rewards."""
    estimator_config = AdvEstimatorConfig.model_construct(
        use_leave_one_out_baseline=False,
        normalize_rewards=True,
    )
    loss_config = ClippedPGLossConfig()
    estimator = GDPOAdvantageEstimator(estimator_config, loss_config)

    prompt_ids = torch.tensor([[0], [0]])
    mask = torch.ones(2, 3)
    repeated_batch = {"reward/correctness": torch.tensor([1.0, 3.0])}

    with pytest.raises(ValueError):
        estimator.compute_advantage(prompt_ids, None, mask, repeated_batch)


def test_gdpo_advantage_estimator_reward_weights():
    """GDPO per-reward weights: uniform weights match the default; non-uniform differ; wrong length raises."""
    loss_config = ClippedPGLossConfig()
    prompt_ids = torch.tensor([[0], [0], [0]])
    mask = torch.ones(3, 2)
    repeated_batch = {
        "reward/correctness": torch.tensor([1.0, 0.0, 1.0]),
        "reward/format": torch.tensor([1.0, 1.0, 0.0]),
    }

    def run(weights):
        config = AdvEstimatorConfig(
            use_leave_one_out_baseline=False,
            normalize_rewards=True,
            reward_weights=weights,
        )
        estimator = GDPOAdvantageEstimator(config, loss_config)
        return estimator.compute_advantage(prompt_ids, None, mask, dict(repeated_batch))

    default = run(None)

    # Any positive uniform scaling is invariant after the final per-batch normalization.
    assert torch.allclose(default, run([2.0, 2.0]), atol=1e-5)

    # Non-uniform weights change the advantages.
    assert not torch.allclose(default, run([1.0, 0.25]), atol=1e-3)

    # Wrong number of weights -> ValueError.
    with pytest.raises(ValueError):
        run([1.0])


# ============================================================================
# Tests for ReinforcePlusPlusAdvantageEstimator class
# ============================================================================


def test_reinforce_plus_plus_global_normalization():
    """Test that ReinforcePlusPlusAdvantageEstimator applies global normalization.

    This test verifies that:
    1. After global normalization, the mean of advantages is approximately 0
    2. The advantages are properly scaled by the global std
    """
    estimator_config = AdvEstimatorConfig.model_construct(
        minus_baseline=True,
    )
    loss_config = ClippedPGLossConfig(
        use_kl_in_reward=False,
        reference_policy_kl_penalty=0.0001,
        reference_policy_kl_type="k2",
    )
    estimator = ReinforcePlusPlusAdvantageEstimator(estimator_config, loss_config)

    prompt_ids = torch.tensor(
        [[0], [0], [0], [0]]
    )  # Shape (4, 1) for unique prompt matching
    rewards = torch.tensor([0.0, 1.0, 2.0, 3.0])  # mean=1.5
    mask = torch.ones(4, 5)

    result = estimator.compute_advantage(
        prompt_ids=prompt_ids,
        rewards=rewards,
        mask=mask,
    )

    # After global normalization, mean should be ~0
    result_mean = (result * mask).sum() / mask.sum()
    assert torch.abs(result_mean) < 1e-5

    # Check the normalized advantages have correct relative ordering
    # Lower rewards should have negative advantages, higher should have positive
    assert result[0, 0] < result[1, 0] < result[2, 0] < result[3, 0]


# ============================================================================
# Tests for validate function
# ============================================================================


class TestValidateFunction:
    """Tests for the validate() function."""

    def test_validate_logs_data_when_logger_provided(self, mock_grpo_components):
        """Test that validation data is logged to JSONL when logger is provided."""

        # Create mock components
        mock_policy_gen = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        # Create mock batch
        mock_batch = BatchedDataDict[DatumSpec](
            {
                "message_log": [
                    [
                        {
                            "role": "user",
                            "content": "test1",
                            "token_ids": torch.tensor([1, 2, 3]),
                        },
                        {
                            "role": "assistant",
                            "content": "response1",
                            "token_ids": torch.tensor([4, 5, 6]),
                        },
                    ],
                    [
                        {
                            "role": "user",
                            "content": "test2",
                            "token_ids": torch.tensor([7, 8, 9]),
                        },
                        {
                            "role": "assistant",
                            "content": "response2",
                            "token_ids": torch.tensor([10, 11, 12]),
                        },
                    ],
                ],
                "task_name": ["math", "math"],
                "extra_env_info": [{}, {}],
                "loss_multiplier": torch.tensor([1.0, 1.0]),
                "idx": torch.tensor([0, 1]),
                "length": torch.tensor([6, 6]),
                "total_reward": torch.tensor([1.0, 0.5]),
            }
        )

        # Create mock dataloader that yields mock_batch
        mock_dataloader = MagicMock(spec=StatefulDataLoader)
        mock_dataloader.__iter__ = MagicMock(return_value=iter([mock_batch]))

        # Create mock environment
        mock_env = MagicMock(spec=EnvironmentInterface)
        mock_env.global_post_process_and_metrics.return_value = (mock_batch, {})

        # Create mock logger that captures calls
        mock_logger = MagicMock()
        logged_data = {}

        def capture_log(data, filename):
            logged_data["data"] = data
            logged_data["filename"] = filename

        mock_logger.log_batched_dict_as_jsonl = MagicMock(side_effect=capture_log)

        # Mock config
        mock_config = mock_grpo_components["master_config"]
        mock_config.grpo.val_batch_size = 2
        mock_config.logger["num_val_samples_to_print"] = 2

        mock_rollout_metrics = {"mean_gen_tokens_per_sample": 10.0}

        with patch("nemo_rl.algorithms.grpo.run_multi_turn_rollout") as mock_rollout:
            mock_rollout.return_value = (mock_batch, mock_rollout_metrics)
            with patch(
                "nemo_rl.algorithms.grpo.should_use_nemo_gym", return_value=False
            ):
                with patch(
                    "nemo_rl.algorithms.grpo.should_use_async_rollouts",
                    return_value=False,
                ):
                    with patch("nemo_rl.algorithms.grpo.print_message_log_samples"):
                        val_metrics, timing = validate(
                            mock_policy_gen,
                            mock_dataloader,
                            mock_tokenizer,
                            {"math": mock_env},
                            step=5,
                            master_config=mock_config,
                            logger=mock_logger,
                        )

        # Verify log_batched_dict_as_jsonl was called
        mock_logger.log_batched_dict_as_jsonl.assert_called_once()

        # Verify the filename
        assert logged_data["filename"] == "val_data_step5.jsonl"

        # Verify the data structure
        assert "content" in logged_data["data"]
        assert "rewards" in logged_data["data"]

    def test_validate_works_without_logger(self, mock_grpo_components):
        """Test that validation works when logger is None (backward compat)."""
        # Create mock components
        mock_policy_gen = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        # Create mock batch
        mock_batch = BatchedDataDict[DatumSpec](
            {
                "message_log": [
                    [
                        {
                            "role": "user",
                            "content": "test1",
                            "token_ids": torch.tensor([1, 2, 3]),
                        },
                        {
                            "role": "assistant",
                            "content": "response1",
                            "token_ids": torch.tensor([4, 5, 6]),
                        },
                    ],
                ],
                "task_name": ["math"],
                "extra_env_info": [{}],
                "loss_multiplier": torch.tensor([1.0]),
                "idx": torch.tensor([0]),
                "length": torch.tensor([6]),
                "total_reward": torch.tensor([1.0]),
            }
        )

        # Create mock dataloader
        mock_dataloader = MagicMock(spec=StatefulDataLoader)
        mock_dataloader.__iter__ = MagicMock(return_value=iter([mock_batch]))

        # Create mock environment
        mock_env = MagicMock(spec=EnvironmentInterface)
        mock_env.global_post_process_and_metrics.return_value = (mock_batch, {})

        # Mock config
        mock_config = mock_grpo_components["master_config"]
        mock_config.logger["num_val_samples_to_print"] = 1

        mock_rollout_metrics = {"mean_gen_tokens_per_sample": 10.0}

        with patch("nemo_rl.algorithms.grpo.run_multi_turn_rollout") as mock_rollout:
            mock_rollout.return_value = (mock_batch, mock_rollout_metrics)
            with patch(
                "nemo_rl.algorithms.grpo.should_use_nemo_gym", return_value=False
            ):
                with patch(
                    "nemo_rl.algorithms.grpo.should_use_async_rollouts",
                    return_value=False,
                ):
                    with patch("nemo_rl.algorithms.grpo.print_message_log_samples"):
                        # Call validate without logger (should not raise exception)
                        val_metrics, timing = validate(
                            mock_policy_gen,
                            mock_dataloader,
                            mock_tokenizer,
                            {"math": mock_env},
                            step=5,
                            master_config=mock_config,
                            logger=None,
                        )

        # Verify metrics are returned correctly
        assert "accuracy" in val_metrics
        assert "avg_length" in val_metrics

    def test_grouped_validation_reports_pass_k(self, mock_grpo_components):
        mock_batch = BatchedDataDict[DatumSpec](
            {
                "message_log": [
                    [{"role": "user", "content": "a", "token_ids": torch.tensor([1])}],
                    [{"role": "user", "content": "b", "token_ids": torch.tensor([2])}],
                ],
                "task_name": ["math", "math"],
                "extra_env_info": [{}, {}],
                "loss_multiplier": torch.tensor([1.0, 1.0]),
                "idx": torch.tensor([0, 1]),
                "length": torch.tensor([1, 1]),
                "total_reward": torch.tensor([0.0, 0.0]),
            }
        )
        mock_dataloader = MagicMock(spec=StatefulDataLoader)
        mock_dataloader.__iter__ = MagicMock(return_value=iter([mock_batch]))
        mock_config = mock_grpo_components["master_config"]
        mock_config.grpo.max_val_samples = 2
        mock_config.grpo.val_batch_size = 2
        mock_config.grpo.val_num_generations_per_prompt = 4

        def run_rollout(_policy, repeated_batch, *_args, **_kwargs):
            # Each prompt is repeated k=4 times, contiguously.
            assert repeated_batch["idx"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
            # Prompt 0 passes once out of 4; prompt 1 never passes.
            repeated_batch["total_reward"] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            )
            return repeated_batch, {"mean_gen_tokens_per_sample": 1.0}

        with (
            patch(
                "nemo_rl.algorithms.grpo.run_multi_turn_rollout",
                side_effect=run_rollout,
            ),
            patch("nemo_rl.algorithms.grpo.should_use_nemo_gym", return_value=False),
            patch(
                "nemo_rl.algorithms.grpo.should_use_async_rollouts",
                return_value=False,
            ),
            patch("nemo_rl.algorithms.grpo.print_message_log_samples"),
        ):
            val_metrics, _ = validate(
                MagicMock(),
                mock_dataloader,
                MagicMock(),
                {"math": MagicMock(spec=EnvironmentInterface)},
                step=0,
                master_config=mock_config,
            )

        # accuracy stays the plain mean over all 8 rollouts; pass@4 counts
        # prompts with at least one passing rollout (1 of 2).
        assert val_metrics["accuracy"] == pytest.approx(0.125)
        assert val_metrics["pass_k"] == pytest.approx(0.5)

    def test_validation_uses_val_sampling_params_on_gym_path(
        self, mock_grpo_components
    ):
        mock_batch = BatchedDataDict[DatumSpec](
            {
                "message_log": [
                    [{"role": "user", "content": "a", "token_ids": torch.tensor([1])}],
                    [{"role": "user", "content": "b", "token_ids": torch.tensor([2])}],
                ],
                "task_name": ["math", "math"],
                "extra_env_info": [{}, {}],
                "loss_multiplier": torch.tensor([1.0, 1.0]),
                "idx": torch.tensor([0, 1]),
                "length": torch.tensor([1, 1]),
                "total_reward": torch.tensor([0.0, 0.0]),
            }
        )
        mock_dataloader = MagicMock(spec=StatefulDataLoader)
        mock_dataloader.__iter__ = MagicMock(return_value=iter([mock_batch]))
        mock_config = mock_grpo_components["master_config"]
        mock_config.grpo.max_val_samples = 2
        mock_config.grpo.val_batch_size = 2
        mock_config.grpo.val_num_generations_per_prompt = 2
        # Train samples at 1.0/1.0; validation runs near-greedy.
        mock_config.policy["generation"].update(
            {"val_temperature": 0.1, "val_top_p": 0.9, "val_top_k": None}
        )
        mock_config.logger.update({"wandb_enabled": False, "wandb": {}})
        mock_config.env = {}

        def run_gym_rollout(**kwargs):
            repeated_batch = kwargs["input_batch"]
            # 2 prompts x k=2 validation rollouts, contiguous per prompt.
            assert repeated_batch["idx"].tolist() == [0, 0, 1, 1]
            repeated_batch["total_reward"] = torch.tensor([1.0, 0.0, 0.0, 0.0])
            return MagicMock(
                final_batch=repeated_batch,
                rollout_metrics={"mean_gen_tokens_per_sample": 1.0},
            )

        with (
            patch(
                "nemo_rl.algorithms.grpo.run_nemo_gym_rollout_sync",
                side_effect=run_gym_rollout,
            ) as mock_rollout,
            patch("nemo_rl.algorithms.grpo.should_use_nemo_gym", return_value=True),
            patch("nemo_rl.algorithms.grpo.print_message_log_samples"),
        ):
            val_metrics, _ = validate(
                MagicMock(),
                mock_dataloader,
                MagicMock(),
                {"math": MagicMock(spec=EnvironmentInterface)},
                step=0,
                master_config=mock_config,
            )

        sampling_params = mock_rollout.call_args.kwargs["sampling_params"]
        assert sampling_params.temperature == pytest.approx(0.1)
        assert sampling_params.top_p == pytest.approx(0.9)
        assert sampling_params.top_k is None
        assert val_metrics["accuracy"] == pytest.approx(0.25)
        assert val_metrics["pass_k"] == pytest.approx(0.5)

    def test_setup_rejects_val_sampling_outside_gym_vllm_path(
        self, mock_grpo_components
    ):
        master_config = mock_grpo_components["master_config"]
        # Non-gym rollouts (env has no nemo_gym) with validation sampling
        # different from training must be rejected at setup time.
        master_config.policy["generation"].update(
            {"backend": "megatron", "val_temperature": 0.1}
        )
        master_config.env = {}

        with pytest.raises(AssertionError, match="only supported for vLLM NeMo-Gym"):
            setup(master_config, MagicMock(), MagicMock(), None)

    def test_validate_returns_empty_when_no_dataloader(self, mock_grpo_components):
        """Test that validate returns empty dicts when no dataloader is provided."""
        mock_policy_gen = MagicMock()
        mock_tokenizer = MagicMock()

        mock_config = mock_grpo_components["master_config"]
        mock_config.grpo.val_period = 0  # Required for the assertion

        val_metrics, timing = validate(
            mock_policy_gen,
            None,  # No dataloader
            mock_tokenizer,
            None,
            step=0,
            master_config=mock_config,
            logger=None,
        )

        assert val_metrics == {}
        assert timing == {}


# ============================================================================
# Tests for compute_and_apply_seq_logprob_error_masking function
# ============================================================================


class TestComputeAndApplySeqLogprobErrorMasking:
    """Tests for the compute_and_apply_seq_logprob_error_masking function."""

    def _create_train_data(
        self,
        batch_size: int,
        seq_length: int,
        prev_logprobs: torch.Tensor,
        generation_logprobs: torch.Tensor,
        token_mask: torch.Tensor = None,
        sample_mask: torch.Tensor = None,
    ) -> BatchedDataDict:
        """Helper to create mock train_data for testing."""
        if token_mask is None:
            token_mask = torch.ones(batch_size, seq_length)
        if sample_mask is None:
            sample_mask = torch.ones(batch_size)

        return BatchedDataDict(
            {
                "token_mask": token_mask,
                "sample_mask": sample_mask,
                "prev_logprobs": prev_logprobs,
                "generation_logprobs": generation_logprobs,
            }
        )

    def test_no_threshold_only_computes_metrics(self):
        """Test that when threshold is None, only metrics are computed (no masking)."""
        batch_size, seq_length = 4, 10

        # Create logprobs with varying errors
        prev_logprobs = torch.zeros(batch_size, seq_length)
        generation_logprobs = torch.zeros(batch_size, seq_length)
        # Add small errors to sequences
        generation_logprobs[0, 1:5] = 0.1  # Small error
        generation_logprobs[1, 1:5] = 0.5  # Medium error
        generation_logprobs[2, 1:5] = 1.0  # Large error
        generation_logprobs[3, 1:5] = 2.0  # Very large error

        train_data = self._create_train_data(
            batch_size, seq_length, prev_logprobs, generation_logprobs
        )
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
        original_sample_mask = train_data["sample_mask"].clone()

        result = compute_and_apply_seq_logprob_error_masking(
            train_data, rewards, seq_logprob_error_threshold=None
        )

        # Verify metrics are computed
        assert result["max_seq_mult_prob_error"] > 0.0, "Should compute max error"
        assert result["num_masked_seqs"] == 0, (
            "Should not mask any sequences when threshold is None"
        )
        assert result["masked_correct_pct"] == 0.0, "Should have 0% masked"
        # Verify sample_mask is unchanged
        assert torch.equal(train_data["sample_mask"], original_sample_mask)

    def test_masking_with_threshold(self):
        """Test that sequences exceeding threshold are masked."""
        batch_size, seq_length = 4, 10

        # Create logprobs with specific errors
        # Note: The metric is averaged over all tokens, so errors get diluted.
        # Formula: seq_mult_prob_error = sum(exp(error) * mask) / sum(mask)
        # With seq_length=10 and slicing [:, 1:], we have 9 tokens per sequence.
        prev_logprobs = torch.zeros(batch_size, seq_length)
        generation_logprobs = torch.zeros(batch_size, seq_length)
        # Sequence 0: small error -> avg ≈ 1.047 (below threshold 1.2)
        generation_logprobs[0, 1:5] = 0.1
        # Sequence 1: small error -> avg ≈ 1.047 (below threshold 1.2)
        generation_logprobs[1, 1:5] = 0.1
        # Sequence 2: medium error -> avg ≈ 1.288 (above threshold 1.2)
        # 4 tokens with exp(0.5)≈1.649, 5 tokens with exp(0)=1 -> (4*1.649+5)/9≈1.288
        generation_logprobs[2, 1:5] = 0.5
        # Sequence 3: large error -> avg ≈ 1.764 (above threshold 1.2)
        # 4 tokens with exp(1.0)≈2.718, 5 tokens with exp(0)=1 -> (4*2.718+5)/9≈1.764
        generation_logprobs[3, 1:5] = 1.0

        train_data = self._create_train_data(
            batch_size, seq_length, prev_logprobs, generation_logprobs
        )
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])

        # Use threshold 1.2 which should mask sequences 2 and 3
        result = compute_and_apply_seq_logprob_error_masking(
            train_data, rewards, seq_logprob_error_threshold=1.2
        )

        # Verify masking occurred
        assert result["num_masked_seqs"] == 2, (
            "Should mask 2 sequences (indices 2 and 3)"
        )
        # Sequence 2 had reward=1, sequence 3 had reward=0, so 50% correct
        assert result["masked_correct_pct"] == 0.5, (
            "50% of masked sequences should be correct"
        )

        # Verify sample_mask is updated correctly
        expected_mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
        assert torch.allclose(train_data["sample_mask"], expected_mask), (
            "Should mask sequences 2 and 3"
        )

    def test_no_sequences_masked_when_all_below_threshold(self):
        """Test that no sequences are masked when all are below threshold."""
        batch_size, seq_length = 3, 8

        # Create logprobs with small errors (all below threshold)
        prev_logprobs = torch.zeros(batch_size, seq_length)
        generation_logprobs = torch.zeros(batch_size, seq_length)
        generation_logprobs[:, 1:4] = 0.05  # Very small error for all

        train_data = self._create_train_data(
            batch_size, seq_length, prev_logprobs, generation_logprobs
        )
        rewards = torch.tensor([1.0, 1.0, 1.0])
        original_sample_mask = train_data["sample_mask"].clone()

        result = compute_and_apply_seq_logprob_error_masking(
            train_data, rewards, seq_logprob_error_threshold=2.0
        )

        # Verify no masking occurred
        assert result["num_masked_seqs"] == 0, "Should not mask any sequences"
        assert result["masked_correct_pct"] == 0.0
        # All sequences should remain in sample_mask
        assert torch.equal(train_data["sample_mask"], original_sample_mask)

    def test_all_sequences_masked_when_all_above_threshold(self):
        """Test that all sequences are masked when all exceed threshold."""
        batch_size, seq_length = 3, 8

        # Create logprobs with large errors (all above threshold)
        prev_logprobs = torch.zeros(batch_size, seq_length)
        generation_logprobs = torch.zeros(batch_size, seq_length)
        generation_logprobs[:, 1:4] = 1.0  # Large error for all (exp(1) ~ 2.7)

        train_data = self._create_train_data(
            batch_size, seq_length, prev_logprobs, generation_logprobs
        )
        rewards = torch.tensor([1.0, 0.0, 1.0])  # 2 correct, 1 incorrect

        result = compute_and_apply_seq_logprob_error_masking(
            train_data, rewards, seq_logprob_error_threshold=1.0
        )

        # Verify all sequences are masked
        assert result["num_masked_seqs"] == 3, "Should mask all 3 sequences"
        assert result["masked_correct_pct"] == pytest.approx(2 / 3, rel=1e-5), (
            "2/3 of masked should be correct"
        )
        # All sequences should be zeroed in sample_mask
        assert torch.equal(train_data["sample_mask"], torch.zeros(batch_size))

    def test_respects_existing_sample_mask(self):
        """Test that masking respects already-masked sequences in sample_mask."""
        batch_size, seq_length = 4, 8

        # Create logprobs with large errors
        prev_logprobs = torch.zeros(batch_size, seq_length)
        generation_logprobs = torch.zeros(batch_size, seq_length)
        generation_logprobs[:, 1:4] = 1.0  # Large error for all

        # Pre-mask sequence 1 (it's already excluded)
        sample_mask = torch.tensor([1.0, 0.0, 1.0, 1.0])

        train_data = self._create_train_data(
            batch_size,
            seq_length,
            prev_logprobs,
            generation_logprobs,
            sample_mask=sample_mask,
        )
        rewards = torch.tensor([1.0, 1.0, 0.0, 1.0])

        result = compute_and_apply_seq_logprob_error_masking(
            train_data, rewards, seq_logprob_error_threshold=1.0
        )

        # Only 3 sequences were originally unmasked, all should be masked now
        assert result["num_masked_seqs"] == 3, (
            "Should mask 3 sequences (indices 0, 2, 3)"
        )
        # Sequences 0 and 3 had reward=1, sequence 2 had reward=0
        assert result["masked_correct_pct"] == pytest.approx(2 / 3, rel=1e-5), (
            "2/3 of newly masked should be correct"
        )
        # All should be zeroed (including already-masked seq 1)
        assert torch.equal(train_data["sample_mask"], torch.zeros(batch_size))

    def test_masked_correct_pct_calculation(self):
        """Test that masked_correct_pct is calculated correctly."""
        batch_size, seq_length = 5, 8

        prev_logprobs = torch.zeros(batch_size, seq_length)
        generation_logprobs = torch.zeros(batch_size, seq_length)
        # Make sequences 2, 3, 4 have high error (will be masked)
        generation_logprobs[2:5, 1:4] = 1.5

        train_data = self._create_train_data(
            batch_size, seq_length, prev_logprobs, generation_logprobs
        )
        # Rewards: seq 2 correct, seq 3 incorrect, seq 4 correct
        rewards = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0])

        result = compute_and_apply_seq_logprob_error_masking(
            train_data, rewards, seq_logprob_error_threshold=1.5
        )

        assert result["num_masked_seqs"] == 3, "Should mask 3 sequences"
        # 2 out of 3 masked sequences were correct (reward=1)
        assert result["masked_correct_pct"] == pytest.approx(2 / 3, rel=1e-5), (
            "2/3 of masked should be correct"
        )

    def test_token_mask_is_respected(self):
        """Test that token_mask affects the error calculation correctly."""
        batch_size, seq_length = 2, 8

        prev_logprobs = torch.zeros(batch_size, seq_length)
        generation_logprobs = torch.zeros(batch_size, seq_length)
        # Add large error to both sequences at positions 1:6
        generation_logprobs[:, 1:6] = 1.0

        # But mask out tokens 3-5 for sequence 0 (reducing its effective error)
        # After slicing [:, 1:], this affects positions 2-4 in the 7-token sequence
        token_mask = torch.ones(batch_size, seq_length)
        token_mask[0, 3:6] = 0.0  # Mask out high-error tokens for seq 0

        # After slicing [:, 1:] and accounting for token_mask:
        # Seq 0: 4 valid tokens (positions 0,1,5,6), 2 have error -> avg ≈ 1.859
        # Seq 1: 7 valid tokens, 5 have error -> avg ≈ 2.227
        # Use threshold 2.0 so seq 0 passes but seq 1 fails

        train_data = self._create_train_data(
            batch_size,
            seq_length,
            prev_logprobs,
            generation_logprobs,
            token_mask=token_mask,
        )
        rewards = torch.tensor([1.0, 0.0])

        # Sequence 0 should have lower error due to masked tokens
        # Sequence 1 should have higher error
        result = compute_and_apply_seq_logprob_error_masking(
            train_data, rewards, seq_logprob_error_threshold=2.0
        )

        # Only sequence 1 should be masked (seq 0 has reduced error due to token_mask)
        assert result["num_masked_seqs"] == 1, "Should mask only sequence 1"
        assert result["masked_correct_pct"] == 0.0, "Masked sequence had reward=0"
        assert train_data["sample_mask"][0] == 1.0, "Sequence 0 should remain unmasked"
        assert train_data["sample_mask"][1] == 0.0, "Sequence 1 should be masked"

    def test_fully_token_masked_sequences_are_excluded_from_metrics(self):
        """Fully token-masked sequences should not drag reported error metrics to 0."""
        batch_size, seq_length = 3, 3

        prev_logprobs = torch.zeros(batch_size, seq_length)
        generation_logprobs = torch.zeros(batch_size, seq_length)
        generation_logprobs[0, 1] = torch.log(torch.tensor(2.0))
        generation_logprobs[1, 1] = torch.log(torch.tensor(4.0))
        generation_logprobs[2, 1] = torch.log(torch.tensor(16.0))

        token_mask = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        train_data = self._create_train_data(
            batch_size,
            seq_length,
            prev_logprobs,
            generation_logprobs,
            token_mask=token_mask,
        )
        rewards = torch.tensor([0.0, 1.0, 1.0])

        result = compute_and_apply_seq_logprob_error_masking(
            train_data, rewards, seq_logprob_error_threshold=3.0
        )

        assert result["max_seq_mult_prob_error"] == pytest.approx(4.0)
        assert result["mean_seq_mult_prob_error"] == pytest.approx(3.0)
        assert result["min_seq_mult_prob_error"] == pytest.approx(2.0)
        assert result["num_masked_seqs"] == 1
        assert result["masked_correct_pct"] == 1.0
        assert result["max_seq_mult_prob_error_after_mask"] == pytest.approx(2.0)
        assert result["mean_seq_mult_prob_error_after_mask"] == pytest.approx(2.0)
        assert result["min_seq_mult_prob_error_after_mask"] == pytest.approx(2.0)
        assert torch.equal(train_data["sample_mask"], torch.tensor([1.0, 0.0, 1.0]))

    def test_empty_batch_returns_zero_metrics(self):
        """Test handling of edge case with empty batch."""
        # Create empty train_data
        train_data = BatchedDataDict(
            {
                "token_mask": torch.zeros(0, 8),
                "sample_mask": torch.zeros(0),
                "prev_logprobs": torch.zeros(0, 8),
                "generation_logprobs": torch.zeros(0, 8),
            }
        )
        rewards = torch.zeros(0)

        result = compute_and_apply_seq_logprob_error_masking(
            train_data, rewards, seq_logprob_error_threshold=1.5
        )

        assert result["max_seq_mult_prob_error"] == 0.0, (
            "Empty batch should have max_error=0"
        )
        assert result["num_masked_seqs"] == 0, (
            "Empty batch should have no masked sequences"
        )
        assert result["masked_correct_pct"] == 0.0, "Empty batch should have 0% masked"

    def test_threshold_boundary_values(self):
        """Test behavior at exact threshold boundary."""
        batch_size, seq_length = 3, 8

        # Create logprobs where error is exactly at threshold
        prev_logprobs = torch.zeros(batch_size, seq_length)
        generation_logprobs = torch.zeros(batch_size, seq_length)

        # Set up specific errors: sequence-level mult_prob_error will be approximately:
        # exp(error * 1) * 1 (for 1 token with error)
        # So if error=0.4, mult_prob_error ~ exp(0.4) ~ 1.49
        # If error=0.41, mult_prob_error ~ exp(0.41) ~ 1.51
        generation_logprobs[0, 1] = 0.4  # Below threshold 1.5
        generation_logprobs[1, 1] = 0.405  # Very close to threshold
        generation_logprobs[2, 1] = 0.41  # Just above threshold 1.5

        # Only consider position 1 as valid token
        token_mask = torch.zeros(batch_size, seq_length)
        token_mask[:, 1] = 1.0

        train_data = self._create_train_data(
            batch_size,
            seq_length,
            prev_logprobs,
            generation_logprobs,
            token_mask=token_mask,
        )
        rewards = torch.tensor([1.0, 1.0, 1.0])

        # Threshold of 1.5 should mask sequence 2 (exp(0.41) > 1.5)
        result = compute_and_apply_seq_logprob_error_masking(
            train_data, rewards, seq_logprob_error_threshold=1.5
        )

        # At least sequence 2 should be masked
        assert result["num_masked_seqs"] >= 1, "At least one sequence should be masked"
        assert train_data["sample_mask"][0] == 1.0, "Sequence 0 should be kept"


class TestAggregateRolloutMetrics:
    """Tests for aggregate_rollout_metrics which aggregates per-group metrics by semantic type."""

    def test_min_metrics_take_minimum(self):
        metrics = {
            "gen_tokens/min": [10, 5, 8],
            "min_reward": [0.3, 0.1, 0.5],
        }
        result = aggregate_rollout_metrics(metrics)
        assert result["gen_tokens/min"] == 5
        assert result["min_reward"] == 0.1

    def test_max_metrics_take_maximum(self):
        metrics = {
            "gen_tokens/max": [10, 50, 30],
            "max_reward": [0.3, 0.9, 0.5],
        }
        result = aggregate_rollout_metrics(metrics)
        assert result["gen_tokens/max"] == 50
        assert result["max_reward"] == 0.9

    def test_rate_suffix_excluded_from_min_max(self):
        """min_*_rate and max_*_rate should be averaged, not min/maxed."""
        metrics = {
            "min_completion_rate": [0.2, 0.8, 0.5],
            "max_completion_rate": [0.3, 0.9, 0.6],
        }
        result = aggregate_rollout_metrics(metrics)
        assert result["min_completion_rate"] == pytest.approx(0.5)
        assert result["max_completion_rate"] == pytest.approx(0.6)

    def test_total_turns_summed(self):
        metrics = {"total_turns": [10, 20, 30]}
        result = aggregate_rollout_metrics(metrics)
        assert result["total_turns"] == 60

    def test_mean_metrics_averaged(self):
        metrics = {
            "mean_gen_tokens_per_sample": [100, 200, 300],
            "reward/mean": [0.5, 0.7, 0.9],
        }
        result = aggregate_rollout_metrics(metrics)
        assert result["mean_gen_tokens_per_sample"] == pytest.approx(200.0)
        assert result["reward/mean"] == pytest.approx(0.7)

    def test_non_numeric_passed_through(self):
        metrics = {"some_list_metric": [["a", "b"], ["c", "d"]]}
        result = aggregate_rollout_metrics(metrics)
        assert result["some_list_metric"] == [["a", "b"], ["c", "d"]]

    def test_mixed_metrics(self):
        """Full integration test with a realistic mix of metric types."""
        metrics = {
            "gen_tokens/min": [5, 3, 7],
            "gen_tokens/max": [100, 200, 150],
            "gen_tokens/mean": [50, 60, 70],
            "min_reward": [0.1, 0.2, 0.05],
            "max_reward": [0.9, 0.8, 0.95],
            "total_turns": [10, 15, 20],
            "accuracy": [0.8, 0.9, 0.7],
            "min_accuracy_rate": [0.1, 0.2, 0.3],
        }
        result = aggregate_rollout_metrics(metrics)
        assert result["gen_tokens/min"] == 3
        assert result["gen_tokens/max"] == 200
        assert result["gen_tokens/mean"] == pytest.approx(60.0)
        assert result["min_reward"] == 0.05
        assert result["max_reward"] == 0.95
        assert result["total_turns"] == 45
        assert result["accuracy"] == pytest.approx(0.8)
        assert result["min_accuracy_rate"] == pytest.approx(0.2)


def _cfg(
    *, force=False, threshold=None, skip_ref=None, kl_reward=False, kl_penalty=0.01
):
    return MasterConfig.model_construct(
        loss_fn=ClippedPGLossConfig(
            force_on_policy_ratio=force,
            use_kl_in_reward=kl_reward,
            reference_policy_kl_penalty=kl_penalty,
        ),
        grpo=GRPOConfig.model_construct(
            seq_logprob_error_threshold=threshold,
            skip_reference_policy_logprobs_calculation=skip_ref,
        ),
    )


@pytest.mark.parametrize(
    "kw, expected",
    [
        ({}, (False, None)),
        ({"force": True}, (True, None)),
        ({"force": True, "threshold": 1.5}, (False, None)),  # threshold overrides skip
        ({"skip_ref": True}, (False, True)),
    ],
    ids=["default", "force_on_policy", "force_plus_threshold", "skip_ref"],
)
def test_resolve_logprob_skip_flags(kw, expected):
    if kw.get("force") and kw.get("threshold") is not None:
        with pytest.warns(UserWarning, match="seq_logprob_error_threshold is set"):
            assert _resolve_logprob_skip_flags(_cfg(**kw)) == expected
    else:
        assert _resolve_logprob_skip_flags(_cfg(**kw)) == expected


def test_validate_use_kl_in_reward_rejects_force_on_policy_ratio():
    with pytest.raises(AssertionError, match="use_kl_in_reward"):
        _validate_use_kl_in_reward_compat(_cfg(force=True, kl_reward=True))


def test_validate_use_kl_in_reward_allows_zero_kl_penalty():
    # kl_coef=0 zeros the KL term regardless, so a zero-placeholder
    # prev_logprobs can't corrupt the advantage.
    _validate_use_kl_in_reward_compat(_cfg(force=True, kl_reward=True, kl_penalty=0.0))


@pytest.mark.parametrize(
    "skip_prev_logprobs, expect_prev",
    [(False, True), (True, False)],
    ids=["keep_prev_logprobs", "skip_prev_logprobs"],
)
def test_train_fields_for_step(skip_prev_logprobs, expect_prev):
    fields = _train_fields_for_step(skip_prev_logprobs)
    assert ("prev_logprobs" in fields) is expect_prev


@pytest.mark.parametrize(
    "backend, nccl_reshard, colocated, expected",
    [
        # MInf refits through mcore's swap_model_weights and never touches HF
        # names; a revert here is silent (setup time + peak memory only), so
        # every megatron combination must stay False.
        ("megatron", False, True, False),
        ("megatron", False, False, False),
        ("megatron", True, False, False),
        ("megatron", True, True, False),
        # vLLM keeps the handshake, except NCCL-reshard non-colocated, which
        # builds its own refit info.
        ("vllm", False, True, True),
        ("vllm", False, False, True),
        ("vllm", True, False, False),
        ("vllm", True, True, True),
    ],
)
def test_needs_hf_refit_handshake(backend, nccl_reshard, colocated, expected):
    assert _needs_hf_refit_handshake(backend, nccl_reshard, colocated) is expected
