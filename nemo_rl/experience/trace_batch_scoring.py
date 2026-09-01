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

"""Prepare exact physical trace rows for policy/reference logprob scoring."""

from __future__ import annotations

import math
from typing import Any, Mapping, MutableMapping, Protocol, TypedDict

import torch

from nemo_rl.algorithms.advantage_estimator import (
    GRPOAdvantageEstimator,
    ReinforceBaselineAdvantageEstimator,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.rollout_traces import (
    TraceBatchPlan,
    build_trace_batch_plan,
)
from nemo_rl.experience.trace_batch_materialization import (
    TraceBatchMaterialization,
    materialize_trace_batch_plan,
)


class TraceScoringPreparation(TypedDict):
    """Logical GRPO ownership plus exact pre-scoring physical rows."""

    rollout_advantages: dict[str, float]
    plan: TraceBatchPlan
    materialization: TraceBatchMaterialization
    logprob_data: BatchedDataDict[Any]


class TraceScoringResult(TypedDict):
    """Exact physical rows after policy/reference logprob attachment."""

    preparation: TraceScoringPreparation
    train_data: BatchedDataDict[Any]


class _TraceLogprobPolicy(Protocol):
    def get_logprobs(
        self,
        data: BatchedDataDict[Any],
        timer: Any | None = None,
    ) -> Mapping[str, Any]: ...

    def get_reference_policy_logprobs(
        self,
        data: BatchedDataDict[Any],
        timer: Any | None = None,
    ) -> Mapping[str, Any]: ...


def _require_rollout_aligned_sequence(
    rollout_batch: Mapping[str, Any],
    key: str,
    *,
    rollout_count: int,
) -> list[Any]:
    value = rollout_batch.get(key)
    if not isinstance(value, list) or len(value) != rollout_count:
        raise ValueError(
            f"Trace-aware rollout batch field {key!r} must contain exactly "
            f"{rollout_count} rollout-aligned values"
        )
    return value


def _require_unmasked_logical_rollouts(
    rollout_batch: Mapping[str, Any],
    *,
    rollout_count: int,
) -> None:
    """Reject masking semantics not yet represented by TraceBatchPlan."""
    for key in ("loss_multiplier", "mask_sample", "truncated"):
        value = rollout_batch.get(key)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if value.ndim != 1 or value.shape[0] != rollout_count:
                raise ValueError(
                    f"Trace-aware rollout batch field {key!r} is not rollout-aligned"
                )
            values = value.tolist()
        elif isinstance(value, list) and len(value) == rollout_count:
            values = value
        else:
            raise ValueError(
                f"Trace-aware rollout batch field {key!r} is not rollout-aligned"
            )
        if key == "loss_multiplier":
            rejected = any(float(item) != 1.0 for item in values)
        else:
            rejected = any(bool(item) for item in values)
        if rejected:
            raise ValueError(
                "Trace-aware GRPO currently rejects masked or truncated logical "
                f"rollouts; unsupported field={key!r}"
            )


def _validate_prompt_group_partition(
    prompt_ids: torch.Tensor,
    bundles: list[Mapping[str, Any]],
) -> None:
    """Prove prompt-token grouping and declared comparison groups agree."""
    for left in range(len(bundles)):
        for right in range(left + 1, len(bundles)):
            same_prompt = torch.equal(prompt_ids[left], prompt_ids[right])
            same_group = bundles[left].get("group_id") == bundles[right].get("group_id")
            if same_prompt != same_group:
                raise ValueError(
                    "Prompt-token equality and rollout comparison-group ownership "
                    f"disagree for rows {left} and {right}"
                )


def _compute_rollout_advantages(
    advantage_estimator: GRPOAdvantageEstimator | ReinforceBaselineAdvantageEstimator,
    *,
    bundles: list[Mapping[str, Any]],
    prompt_ids: torch.Tensor,
    rewards: torch.Tensor,
) -> dict[str, float]:
    supported_estimators = (
        GRPOAdvantageEstimator,
        ReinforceBaselineAdvantageEstimator,
    )
    if not isinstance(advantage_estimator, supported_estimators):
        raise TypeError(
            "Trace-aware scoring preparation currently supports only the "
            "GRPOAdvantageEstimator and ReinforceBaselineAdvantageEstimator"
        )
    rollout_count = len(bundles)
    if (
        prompt_ids.ndim != 2
        or prompt_ids.shape[0] != rollout_count
        or rewards.ndim != 1
        or rewards.shape[0] != rollout_count
    ):
        raise ValueError(
            "Trace-aware prompt IDs and rewards must be logical-rollout aligned"
        )
    if not torch.isfinite(rewards).all():
        raise ValueError("Trace-aware rollout rewards must be finite")
    _validate_prompt_group_partition(prompt_ids, bundles)

    if isinstance(advantage_estimator, ReinforceBaselineAdvantageEstimator):
        advantages = advantage_estimator.compute_rollout_advantages(
            prompt_ids,
            rewards,
        )
        action_token_counts = torch.tensor(
            [
                sum(
                    int(token_is_eligible)
                    for trace in bundle.get("physical_traces", [])
                    for segment in trace.get("segments", [])
                    for token_is_eligible in segment.get("loss_mask", [])
                )
                for bundle in bundles
            ],
            dtype=torch.float32,
            device=rewards.device,
        )
        advantages = advantage_estimator.whiten_rollout_advantages(
            advantages,
            action_token_counts,
        ).unsqueeze(-1)
    else:
        scalar_mask = torch.ones(
            (rollout_count, 1),
            dtype=rewards.dtype,
            device=rewards.device,
        )
        advantages = advantage_estimator.compute_advantage(
            prompt_ids=prompt_ids,
            rewards=rewards,
            mask=scalar_mask,
        )
    expected_shape = (rollout_count, 1)
    if (
        not isinstance(advantages, torch.Tensor)
        or advantages.shape != expected_shape
        or not torch.isfinite(advantages).all()
    ):
        raise ValueError(
            "Advantage estimator did not produce one finite scalar per logical rollout"
        )

    result: dict[str, float] = {}
    for index, bundle in enumerate(bundles):
        rollout_id = bundle.get("rollout_id")
        if not isinstance(rollout_id, str) or not rollout_id:
            raise ValueError(f"Trace bundle {index} has no rollout identity")
        if rollout_id in result:
            raise ValueError(f"Duplicate logical rollout ID {rollout_id!r}")
        bundle_reward = bundle.get("reward")
        if (
            isinstance(bundle_reward, bool)
            or not isinstance(bundle_reward, (int, float))
            or not math.isclose(
                float(bundle_reward),
                float(rewards[index].item()),
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        ):
            raise ValueError(
                f"Trace bundle {rollout_id!r} reward disagrees with GRPO reward"
            )
        result[rollout_id] = float(advantages[index, 0].item())
    return result


def _build_logprob_data(
    materialization: TraceBatchMaterialization,
) -> BatchedDataDict[Any]:
    train_data = materialization["train_data"]
    logprob_data = BatchedDataDict(
        {
            "input_ids": train_data["input_ids"],
            "input_lengths": train_data["input_lengths"],
            "token_mask": train_data["token_mask"],
            "sample_mask": train_data["sample_mask"],
        }
    )
    logprob_data.update(train_data.get_multimodal_dict(as_tensors=False))
    # This non-tensor identity stays row-aligned for audit/debugging; the
    # model-facing image_cache_keys PackedTensor is copied above.
    logprob_data["ordered_media_ids"] = train_data["ordered_media_ids"]
    if "routed_experts" in train_data:
        logprob_data["routed_experts"] = train_data["routed_experts"]
    return logprob_data


def prepare_trace_batch_for_scoring(
    rollout_batch: Mapping[str, Any],
    *,
    prompt_ids: torch.Tensor,
    advantage_estimator: GRPOAdvantageEstimator | ReinforceBaselineAdvantageEstimator,
    expected_rollouts_per_group: int,
    batch_quantum: int,
    optimizer_step_id: str,
    pad_token_id: int,
    make_sequence_length_divisible_by: int = 1,
    training_admission: bool = False,
) -> TraceScoringPreparation:
    """Compute logical GRPO advantages, then expand exact physical rows.

    This function deliberately stops before calling a policy/reference worker.
    It also rejects rollout masking, truncation, reward rewriting, and
    non-standard advantage estimators until their multi-trace semantics are
    explicitly implemented.
    """
    raw_bundles = rollout_batch.get("rollout_trace_bundle")
    if not isinstance(raw_bundles, list) or not raw_bundles:
        raise ValueError(
            "Trace-aware scoring requires rollout_trace_bundle for every rollout"
        )
    if any(not isinstance(bundle, Mapping) for bundle in raw_bundles):
        raise TypeError("rollout_trace_bundle values must be mappings")
    bundles = list(raw_bundles)
    rollout_count = len(bundles)
    physical_message_logs = _require_rollout_aligned_sequence(
        rollout_batch,
        "physical_message_logs",
        rollout_count=rollout_count,
    )
    _require_unmasked_logical_rollouts(
        rollout_batch,
        rollout_count=rollout_count,
    )

    rewards = rollout_batch.get("total_reward")
    if not isinstance(rewards, torch.Tensor):
        raise TypeError("Trace-aware scoring requires tensor total_reward")
    rollout_advantages = _compute_rollout_advantages(
        advantage_estimator,
        bundles=bundles,
        prompt_ids=prompt_ids,
        rewards=rewards,
    )
    advantage_estimator_name = (
        "reinforce_baseline"
        if isinstance(advantage_estimator, ReinforceBaselineAdvantageEstimator)
        else "grpo"
    )

    plan = build_trace_batch_plan(
        bundles,
        rollout_advantages=rollout_advantages,
        expected_rollouts_per_group=expected_rollouts_per_group,
        batch_quantum=batch_quantum,
        optimizer_step_id=optimizer_step_id,
        training_admission=training_admission,
        advantage_estimator_name=advantage_estimator_name,
        sequence_level_ratios_enabled=False,
        sequence_level_clipping_enabled=False,
    )
    physical_message_logs_by_rollout = {
        str(bundle["rollout_id"]): logs
        for bundle, logs in zip(bundles, physical_message_logs)
    }
    if len(physical_message_logs_by_rollout) != rollout_count:
        raise ValueError("Duplicate rollout identity changed trace-log ownership")
    materialization = materialize_trace_batch_plan(
        plan,
        bundles=bundles,
        physical_message_logs_by_rollout=physical_message_logs_by_rollout,
        pad_token_id=pad_token_id,
        make_sequence_length_divisible_by=make_sequence_length_divisible_by,
    )
    # This is an ownership-transfer boundary: train_data now owns the packed
    # multimodal inputs needed by workers, while materialization retained only
    # compact text for metrics. Drop the rollout-side graph promptly so raw
    # images and per-message tensors are not kept alive by both representations.
    if isinstance(rollout_batch, MutableMapping):
        rollout_batch.pop("physical_message_logs", None)
    del physical_message_logs_by_rollout
    del physical_message_logs
    return {
        "rollout_advantages": rollout_advantages,
        "plan": plan,
        "materialization": materialization,
        "logprob_data": _build_logprob_data(materialization),
    }


def _validated_logprobs(
    output: Mapping[str, Any],
    *,
    key: str,
    expected_shape: torch.Size,
    effective_token_mask: torch.Tensor,
) -> torch.Tensor:
    value = output.get(key)
    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or value.shape != expected_shape
    ):
        raise ValueError(
            f"Trace-aware worker output {key!r} must be a floating tensor with "
            f"shape {tuple(expected_shape)}"
        )
    if value.device != effective_token_mask.device:
        raise ValueError(
            f"Trace-aware worker output {key!r} is on {value.device}, expected "
            f"{effective_token_mask.device}"
        )
    if not torch.isfinite(value[effective_token_mask]).all():
        raise ValueError(
            f"Trace-aware worker output {key!r} is non-finite on an eligible token"
        )
    # Prompt and padding positions are outside the supported token-level
    # objective. Canonicalize them to zero so masked NaN/Inf values cannot leak
    # through a later multiplication.
    return torch.where(effective_token_mask, value, torch.zeros_like(value))


def score_prepared_trace_batch(
    preparation: TraceScoringPreparation,
    *,
    policy: _TraceLogprobPolicy,
    timer: Any | None = None,
    skip_policy_logprobs: bool = False,
    skip_reference_logprobs: bool = False,
) -> TraceScoringResult:
    """Call logprob workers on exact rows and validate their returned alignment.

    The caller remains responsible for worker mode transitions. This function
    does not compute ratios, loss, gradients, or optimizer/scheduler state.
    """
    materialization = preparation["materialization"]
    train_data = materialization["train_data"]
    logprob_data = preparation["logprob_data"]
    expected_shape = train_data["input_ids"].shape
    effective_token_mask = train_data["token_mask"].bool() & (
        train_data["sample_mask"].bool().unsqueeze(-1)
    )
    if (
        effective_token_mask.shape != expected_shape
        or torch.count_nonzero(effective_token_mask).item()
        != preparation["plan"]["eligible_action_token_count"]
    ):
        raise ValueError(
            "Trace-aware scoring mask disagrees with the physical trace plan"
        )

    if skip_policy_logprobs:
        prev_logprobs = torch.zeros_like(train_data["generation_logprobs"])
    else:
        output = policy.get_logprobs(logprob_data, timer=timer)
        if not isinstance(output, Mapping):
            raise TypeError("Policy logprob worker output must be a mapping")
        prev_logprobs = _validated_logprobs(
            output,
            key="logprobs",
            expected_shape=expected_shape,
            effective_token_mask=effective_token_mask,
        )

    if skip_reference_logprobs:
        reference_logprobs = torch.zeros_like(prev_logprobs)
    else:
        output = policy.get_reference_policy_logprobs(logprob_data, timer=timer)
        if not isinstance(output, Mapping):
            raise TypeError("Reference logprob worker output must be a mapping")
        reference_logprobs = _validated_logprobs(
            output,
            key="reference_logprobs",
            expected_shape=expected_shape,
            effective_token_mask=effective_token_mask,
        )

    train_data["prev_logprobs"] = prev_logprobs
    train_data["reference_policy_logprobs"] = reference_logprobs
    return {
        "preparation": preparation,
        "train_data": train_data,
    }
