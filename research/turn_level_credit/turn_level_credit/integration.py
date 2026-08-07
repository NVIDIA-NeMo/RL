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

"""Scoped integration with the legacy synchronous NeMo-RL GRPO path."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

from nemo_rl.algorithms import grpo as grpo_module
from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.data.interfaces import DatumSpec, TokenizerType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.experience import rollouts as rollout_module
from nemo_rl.models.generation.interfaces import GenerationInterface
from turn_level_credit.advantage import TurnLevelGRPOAdvantageEstimator
from turn_level_credit.config import TurnCreditConfig
from turn_level_credit.trace import (
    attach_turn_batch,
    compute_environment_credit,
    record_environment_turn,
    remove_turn_annotations,
    tensorize_turn_traces,
    validate_raw_reward_sums,
    validate_turn_count,
)
from turn_level_credit.validation import validate_supported_path


@contextmanager
def install_turn_credit_runtime(
    turn_credit_config: TurnCreditConfig,
) -> Iterator[None]:
    """Install and later restore the research-only rollout/estimator hooks."""
    if not turn_credit_config.enabled:
        yield
        return

    original_calculate_rewards = rollout_module.calculate_rewards
    original_rollout = grpo_module.run_multi_turn_rollout
    original_estimator_factory = grpo_module._create_advantage_estimator

    def calculate_rewards_with_turn_recording(
        batch: BatchedDataDict[DatumSpec],
        task_to_env: dict[str, EnvironmentInterface],
    ) -> EnvironmentReturn:
        environment_return = original_calculate_rewards(batch, task_to_env)
        record_environment_turn(batch["message_log"], environment_return)
        return environment_return

    def rollout_with_turn_tensors(
        policy_generation: GenerationInterface,
        input_batch: BatchedDataDict[DatumSpec],
        tokenizer: TokenizerType,
        task_to_env: dict[str, EnvironmentInterface],
        max_seq_len: int,
        max_rollout_turns: int = 999999,
        greedy: bool = False,
    ) -> tuple[BatchedDataDict[DatumSpec], dict[str, Any]]:
        final_batch, metrics = original_rollout(
            policy_generation=policy_generation,
            input_batch=input_batch,
            tokenizer=tokenizer,
            task_to_env=task_to_env,
            max_seq_len=max_seq_len,
            max_rollout_turns=max_rollout_turns,
            greedy=greedy,
        )
        turn_batch = tensorize_turn_traces(final_batch["message_log"])
        if turn_batch.max_turns == 0:
            raise ValueError("Enabled turn credit captured no environment transitions")
        if "total_turns" not in metrics:
            raise ValueError("Rollout metrics are missing required total_turns")
        validate_turn_count(turn_batch, metrics["total_turns"])
        validate_raw_reward_sums(
            turn_batch,
            final_batch["total_reward"],
            atol=turn_credit_config.raw_reward_atol,
        )
        attach_turn_batch(final_batch, turn_batch)
        remove_turn_annotations(final_batch["message_log"])

        turns_per_sample = turn_batch.mask.sum(dim=1)
        trainable_turns_per_sample = turn_batch.trainable_mask.sum(dim=1)
        observed_rewards = turn_batch.rewards[turn_batch.mask]
        credit = compute_environment_credit(
            turn_batch,
            mode=turn_credit_config.environment_mode,
            discount=turn_credit_config.discount,
        )
        observed_credit = credit[turn_batch.mask]
        metrics.update(
            {
                "turn_credit/turns_per_sample/mean": float(
                    turns_per_sample.float().mean().item()
                ),
                "turn_credit/turns_per_sample/max": int(turns_per_sample.max().item()),
                "turn_credit/trainable_turns_per_sample/mean": float(
                    trainable_turns_per_sample.float().mean().item()
                ),
                "turn_credit/trainable_turns_per_sample/max": int(
                    trainable_turns_per_sample.max().item()
                ),
                "turn_credit/environment_reward/mean": float(
                    observed_rewards.mean().item()
                ),
                "turn_credit/environment_reward/std": float(
                    observed_rewards.std(unbiased=False).item()
                ),
                "turn_credit/credit/mean": float(observed_credit.mean().item()),
                "turn_credit/credit/std": float(
                    observed_credit.std(unbiased=False).item()
                ),
            }
        )
        return final_batch, metrics

    def create_turn_credit_estimator(
        master_config: MasterConfig,
    ) -> TurnLevelGRPOAdvantageEstimator:
        validate_supported_path(master_config, turn_credit_config)
        base_estimator = original_estimator_factory(master_config)
        print("  ✓ Adding native turn-level credit to GRPO advantages")
        return TurnLevelGRPOAdvantageEstimator(
            base_estimator=base_estimator,
            config=turn_credit_config,
        )

    rollout_module.calculate_rewards = calculate_rewards_with_turn_recording
    grpo_module.run_multi_turn_rollout = rollout_with_turn_tensors
    grpo_module._create_advantage_estimator = cast(Any, create_turn_credit_estimator)
    try:
        yield
    finally:
        rollout_module.calculate_rewards = original_calculate_rewards
        grpo_module.run_multi_turn_rollout = original_rollout
        grpo_module._create_advantage_estimator = original_estimator_factory
