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
import math
from typing import Any, TypeVar

import torch
from pydantic import BaseModel, Field

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

Tensor = TypeVar("Tensor", bound=torch.Tensor)


class ContextCostShapingConfig(BaseModel, extra="allow"):
    """Multiplicative context-cost shaping for correct rollouts in a prompt group.

    Multi-turn agent rollouts pay for context (prompt, every assistant turn and
    every tool result), and the reward alone does not distinguish a correct
    rollout that read one sheet from one that read the whole data room. This
    shaping scales the reward of each *correct* rollout by

        d + (1 - d) * exp(-beta * max(0, ctx - ctx_min_correct) / ref_tokens)

    where ``ctx`` is the rollout's total token count, ``ctx_min_correct`` the
    smallest such count among the correct rollouts of the same prompt group and
    ``d`` the group's failure rate. Correctness is judged on the raw task reward
    (``unshaped_total_reward`` when an earlier shaping step saved it), incorrect
    rollouts keep their reward, and a shaped correct rollout is never pushed
    below the best incorrect rollout of its group, so correctness always
    dominates; prompts the policy mostly fails (``d -> 1``) are barely shaped,
    prompts it already solves reliably (``d -> 0``) get the full efficiency
    pressure. Follows the efficiency-shaping idea of OTC-PO
    (https://arxiv.org/abs/2504.14870) and the EAPO family of efficiency-aware
    policy optimization methods, applied to context tokens rather than
    tool-call counts. GRPO rollout loops only: the shaping needs
    ``message_log`` in the batch.
    """

    enabled: bool = False
    # A rollout counts as correct when its raw reward is at least this value.
    correct_reward_threshold: float = 1.0
    # Decay strength per ``ref_tokens`` of context above the group minimum.
    beta: float = 0.5
    # Token scale of the decay.
    ref_tokens: int = 32768


class RewardShapingConfig(BaseModel, extra="allow"):
    """Configuration for reward function processing.

    This configuration enables custom reward shaping, currently supporting DAPO-style
    penalties for responses that exceed the maximum response length threshold, and
    context-cost shaping for multi-turn rollouts (see ``context_cost``). The two are
    independent: ``context_cost`` applies whenever its own ``enabled`` is set.
    """

    enabled: bool = False

    # The length of the buffer to penalize responses that exceed the maximum response length threshold.
    # Responses of length greater than overlong_buffer_length + max_response_length will
    # receive the maximum penalty.
    overlong_buffer_length: int | None = None

    # The penalty for responses that exceed the maximum response length threshold.
    overlong_buffer_penalty: float | None = None

    # The maximum response length threshold. Responses exceeding this length will be penalized.
    max_response_length: int | None = None

    # Stop properly penalty: scale factor for rewards of truncated responses (0-1).
    # When set to 0, truncated responses get zero reward.
    # When set to 1, no penalty is applied (default behavior).
    stop_properly_penalty_coef: float | None = None

    # Context-cost shaping for multi-turn rollouts (independent of ``enabled``).
    context_cost: ContextCostShapingConfig = Field(
        default_factory=ContextCostShapingConfig
    )


def apply_reward_shaping(
    batch: BatchedDataDict, cfg: RewardShapingConfig
) -> BatchedDataDict:
    """Process rewards by applying penalties for responses exceeding max_response_length. Currently, this function only supports DAPO reward shaping as illustrated in the DAPO paper : https://arxiv.org/pdf/2503.14476.

    Nonetheless, it can be potentially extended to support any custom reward logic.
    """
    rewards = batch["total_reward"]
    if not cfg.enabled:
        return batch

    # Preserve the pre-shaping reward so downstream consumers (e.g. DAPO
    # dynamic sampling) can filter prompt groups on the raw task metric
    # rather than on length-dependent shaped rewards.
    batch["unshaped_total_reward"] = rewards.clone()

    # Apply stop properly penalty if configured
    if cfg.stop_properly_penalty_coef is not None:
        stop_properly_penalty_coef = cfg.stop_properly_penalty_coef
        assert 0 <= stop_properly_penalty_coef <= 1, (
            f"stop_properly_penalty_coef must be in [0, 1], got {stop_properly_penalty_coef}"
        )
        # Warn user that DAPO overlong parameters are ignored when stop_properly_penalty_coef is set
        ignored_params = []
        if cfg.overlong_buffer_length is not None:
            ignored_params.append("overlong_buffer_length")
        if cfg.overlong_buffer_penalty is not None:
            ignored_params.append("overlong_buffer_penalty")
        if cfg.max_response_length is not None:
            ignored_params.append("max_response_length")
        if ignored_params:
            print(
                f"[WARN] stop_properly_penalty_coef is set, so the following DAPO overlong "
                f"parameters are ignored: {', '.join(ignored_params)}. "
                f"Set stop_properly_penalty_coef=null to use DAPO overlong reward shaping instead.",
                flush=True,
            )
        truncated = batch.get("truncated")
        assert truncated is not None, "truncated field not found in batch"
        if isinstance(truncated, list):
            truncated = torch.tensor(truncated, dtype=torch.bool, device=rewards.device)
        else:
            truncated = truncated.to(device=rewards.device)

        num_truncated = truncated.sum().item()
        if num_truncated > 0:
            original_rewards = rewards.clone()
            # For truncated samples, scale the reward by stop_properly_penalty_coef
            rewards = torch.where(
                truncated, rewards * stop_properly_penalty_coef, rewards
            )
            batch["total_reward"] = rewards
            print(
                f"[INFO] stop properly penalty applied: {num_truncated}/{len(truncated)} samples truncated, "
                f"coef={stop_properly_penalty_coef}, "
                f"original_reward_mean={original_rewards[truncated].mean().item():.4f}, "
                f"shaped_reward_mean={rewards[truncated].mean().item():.4f}",
                flush=True,
            )
        else:
            print(
                "[INFO] stop properly penalty: no truncated samples (truncation_rate=0)",
                flush=True,
            )

        return batch

    # DAPO reward shaping requires overlong_buffer_length, overlong_buffer_penalty, and max_response_length to be set.
    overlong_buffer_length = cfg.overlong_buffer_length
    overlong_buffer_penalty = cfg.overlong_buffer_penalty
    max_response_length = cfg.max_response_length
    if (
        overlong_buffer_length is None
        or overlong_buffer_penalty is None
        or max_response_length is None
    ):
        raise ValueError(
            "Reward function is enabled but only DAPO reward shaping is currently supported. Please ensure overlong_buffer_length, overlong_buffer_penalty, and max_response_length are properly configured."
        )

    assert overlong_buffer_penalty >= 0, f"{overlong_buffer_penalty=} must be >=0"
    # Calculate the expected response length
    expected_response_length = max_response_length - overlong_buffer_length

    # Prefer slim per-sample tensor (data-plane path: message_log lives in
    # TQ, slice carries response_token_lengths). Fall back to scanning
    # message_log for the legacy non-data-plane caller.
    response_token_lengths = batch.get("response_token_lengths")
    if response_token_lengths is not None:
        if isinstance(response_token_lengths, torch.Tensor):
            response_lengths = response_token_lengths.tolist()
        else:
            response_lengths = list(response_token_lengths)
    else:
        response_lengths = []
        for message_log in batch["message_log"]:
            length = None
            for message in message_log:
                if message["role"] == "assistant":
                    length = message["token_ids"].shape[0]
                    break
            assert length is not None, (
                "Assistant response not found during reward shaping"
            )
            response_lengths.append(length)

    assert len(response_lengths) == len(rewards), (
        "The number of messages in the batch must match the number of rewards"
    )

    updated_rewards = torch.zeros_like(rewards)
    for i, message_response_length in enumerate(response_lengths):
        # Calculate the exceed length and the corresponding reward penalty
        exceed_length = message_response_length - expected_response_length
        overlong_reward = min(
            -exceed_length / overlong_buffer_length * overlong_buffer_penalty, 0
        )
        updated_rewards[i] = rewards[i] + overlong_reward

    # Update the rewards in the batch
    batch["total_reward"] = updated_rewards

    return batch


def _message_log_token_count(message_log: list[dict[str, Any]]) -> int:
    """Total tokens a rollout put through the model: prompt, turns and tool results."""
    total = 0
    for message in message_log:
        token_ids = message.get("token_ids")
        if token_ids is None:
            continue
        total += (
            int(token_ids.numel()) if hasattr(token_ids, "numel") else len(token_ids)
        )
    return total


def apply_context_cost_shaping(
    batch: BatchedDataDict, cfg: ContextCostShapingConfig, num_generations: int
) -> BatchedDataDict:
    """Scale the rewards of correct rollouts by their context cost within each prompt group.

    ``batch`` is ordered by prompt group: every ``num_generations`` consecutive
    samples share a prompt, as produced by both the synchronous rollout path
    (before dynamic sampling) and the async replay buffer, which concatenates
    whole groups. GRPO's own baseline groups by identical prompt tokens, so two
    sampled groups with the same prompt are one group there and two here.
    Context is measured from ``message_log`` as the total number of tokens
    across all messages. Correctness is judged on the raw task reward:
    ``unshaped_total_reward`` when an earlier shaping step saved it, otherwise
    ``total_reward``, which this function then saves as
    ``unshaped_total_reward`` so dynamic-sampling filters can keep using the raw
    task metric. A shaped correct rollout is floored at the best incorrect
    rollout of its group so correctness still dominates for non-binary rewards.

    Raises:
        ValueError: On a bad ``num_generations`` / ``ref_tokens``, a batch that
            is not a whole number of groups, or a batch without ``message_log``
            (e.g. the driver-carry batch of the data-plane loop, which this
            shaping does not support).
    """
    if not cfg.enabled:
        return batch
    if num_generations <= 0:
        raise ValueError(f"num_generations must be positive, got {num_generations}")
    if cfg.ref_tokens <= 0:
        raise ValueError(
            f"context_cost.ref_tokens must be positive, got {cfg.ref_tokens}"
        )

    if "message_log" not in batch:
        raise ValueError(
            "reward_shaping.context_cost needs `message_log` in the batch to "
            "measure context; it is only supported in the GRPO rollout loops."
        )

    rewards = batch["total_reward"]
    message_logs = batch["message_log"]
    num_samples = len(message_logs)
    if num_samples % num_generations != 0:
        raise ValueError(
            f"batch size {num_samples} is not a multiple of num_generations={num_generations}"
        )

    if "unshaped_total_reward" not in batch:
        batch["unshaped_total_reward"] = rewards.clone()
    raw_rewards = batch["unshaped_total_reward"]

    context = [_message_log_token_count(log) for log in message_logs]
    shaped = rewards.detach().clone().to(torch.float32)
    factors: list[float] = []
    num_groups = num_samples // num_generations
    for group in range(num_groups):
        indices = range(group * num_generations, (group + 1) * num_generations)
        correct = [
            i for i in indices if float(raw_rewards[i]) >= cfg.correct_reward_threshold
        ]
        if not correct:
            continue
        # Correctness must dominate: never push a correct rollout below the best
        # incorrect one of its group (a no-op for binary rewards, where it is 0).
        floor = max(
            (float(rewards[i]) for i in indices if i not in correct),
            default=-math.inf,
        )
        failure_rate = 1.0 - len(correct) / num_generations
        min_context = min(context[i] for i in correct)
        for i in correct:
            factor = failure_rate + (1.0 - failure_rate) * math.exp(
                -cfg.beta * max(0, context[i] - min_context) / cfg.ref_tokens
            )
            shaped[i] = max(float(rewards[i]) * factor, floor)
            factors.append(factor)

    batch["total_reward"] = shaped.to(dtype=rewards.dtype, device=rewards.device)
    sorted_context = sorted(context)
    p90 = sorted_context[int(0.9 * (num_samples - 1))] if num_samples else 0
    print(
        f"[INFO] context-cost shaping: groups={num_groups} samples={num_samples} "
        f"correct={len(factors)} factor_mean={sum(factors) / len(factors) if factors else 1.0:.3f} "
        f"factor_min={min(factors) if factors else 1.0:.3f} "
        f"ctx_mean={sum(context) / max(num_samples, 1):.0f} ctx_p90={p90} "
        f"ctx_max={max(context) if context else 0} beta={cfg.beta} ref_tokens={cfg.ref_tokens}",
        flush=True,
    )
    return batch
