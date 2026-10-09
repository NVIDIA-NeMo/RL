# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Logical rollout identities for variable-length compaction trace batches."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from nemo_rl.environments.session_traces import (
    session_training_semantics,
    validate_tagged_session_rollout,
)


def validate_multitrace_resume_progress(
    *,
    recovery_enabled: bool,
    current_step: int,
    total_steps: int,
    trainer_version: int | None,
    consumed_groups: int,
    groups_per_update: int,
    max_updates: int,
) -> None:
    """Reject unqualified recovery or a reset/inconsistent successful-update count."""
    if not recovery_enabled:
        raise ValueError(
            "Multi-trace replay/checkpoint resume requires explicit "
            "async_rl.multi_trace_recovery_enabled=true and native qualification"
        )
    values = (
        current_step,
        total_steps,
        consumed_groups,
        groups_per_update,
        max_updates,
    )
    if (
        any(type(value) is not int or value < 0 for value in values)
        or groups_per_update == 0
    ):
        raise ValueError("Multi-trace resume has invalid progress counters")
    restored_version = (
        current_step
        if trainer_version is None and current_step == 0
        else trainer_version
    )
    if (
        type(restored_version) is not int
        or restored_version != current_step
        or total_steps != current_step
        or current_step > max_updates
    ):
        raise ValueError(
            "Multi-trace resume optimizer progress and trainer version disagree"
        )
    if consumed_groups != current_step * groups_per_update:
        raise ValueError(
            "Multi-trace resume consumed group count disagrees with successful updates"
        )


def validate_multitrace_resume_config(
    saved: Mapping[str, Any],
    current: Mapping[str, Any],
) -> None:
    """Keep task supply, loss, optimizer schedule and logical batch fixed on resume.

    Operational paths/stop hooks may change across allocations. The launcher also
    binds the immutable arm/runtime contract, including compaction settings.
    """
    paths = (
        ("data", "train"),
        ("data", "shuffle"),
        ("grpo", "seed"),
        ("grpo", "max_num_steps"),
        ("grpo", "num_prompts_per_step"),
        ("grpo", "num_generations_per_prompt"),
        ("grpo", "adv_estimator"),
        ("grpo", "overlong_filtering"),
        ("grpo", "seq_logprob_error_threshold"),
        ("grpo", "invalid_tool_call_advantage"),
        ("grpo", "malformed_thinking_advantage"),
        ("loss_fn",),
        ("policy", "model_name"),
        ("policy", "tokenizer"),
        ("policy", "train_global_batch_size"),
        ("policy", "megatron_cfg", "optimizer"),
        ("policy", "megatron_cfg", "scheduler"),
        ("async_rl", "sampler"),
        ("token_capture", "enabled"),
    )

    def value_at(config: Mapping[str, Any], path: tuple[str, ...]) -> Any:
        value: Any = config
        for key in path:
            if not isinstance(value, Mapping):
                return None
            value = value.get(key)
        return value

    changed = [
        ".".join(path)
        for path in paths
        if value_at(saved, path) != value_at(current, path)
    ]
    if changed:
        raise ValueError(f"Multi-trace resume changed training semantics: {changed}")
    if session_training_semantics(saved.get("env", {})) != session_training_semantics(
        current.get("env", {})
    ):
        raise ValueError("Multi-trace resume changed session training mode or limits")


@dataclass
class LogicalRolloutBatch:
    """One reward/baseline vote per original sibling, with trace broadcast map."""

    prompt_ids: torch.Tensor
    rewards: torch.Tensor
    valid_mask: torch.Tensor
    trace_to_rollout: torch.Tensor


def logical_rollout_batch(
    tags: list[dict[str, Any]],
    *,
    rewards: torch.Tensor,
    sample_mask: torch.Tensor,
    token_mask: torch.Tensor,
    expected_generations: int,
) -> LogicalRolloutBatch:
    """Validate complete ownership and collapse traces into original siblings.

    A scoring/environment error already masks every trace of its rollout. Local
    token/logprob filtering can retain some traces: a rollout votes once while it
    still has a trainable causal target, and never votes if all traces are masked.
    The original group ID determines sibling identity, independent of context.
    """
    if len(tags) != len(rewards):
        raise ValueError("Multi-trace tags must identify every training row")
    groups: dict[str, dict[int, list[int]]] = {}
    for row, tag in enumerate(tags):
        group_id = tag.get("group_id")
        rollout_index = tag.get("rollout_index")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("Multi-trace row is missing its original group_id")
        if (
            not isinstance(rollout_index, int)
            or isinstance(rollout_index, bool)
            or not 0 <= rollout_index < expected_generations
        ):
            raise ValueError("Multi-trace row has an invalid logical rollout index")
        if tag.get("num_logical_rollouts") != expected_generations:
            raise ValueError("Trace count cannot replace the configured sibling count")
        groups.setdefault(group_id, {}).setdefault(rollout_index, []).append(row)

    reward_rows = []
    validity = []
    group_indices = []
    trace_to_rollout = torch.empty(len(tags), dtype=torch.long, device=rewards.device)
    has_targets = token_mask[:, 1:].bool().any(dim=1)
    for group_index, siblings in enumerate(groups.values()):
        if set(siblings) != set(range(expected_generations)):
            raise ValueError(
                "Multi-trace advantage stage requires complete sibling groups"
            )
        for rows in siblings.values():
            first = rows[0]
            validate_tagged_session_rollout([tags[row] for row in rows])
            expected_traces = tags[first].get("num_traces")
            if (
                expected_traces != len(rows)
                or any(tags[row].get("num_traces") != expected_traces for row in rows)
                or {tags[row].get("trace_index") for row in rows}
                != set(range(len(rows)))
            ):
                raise ValueError(
                    "Missing or duplicate trace ownership in a logical rollout"
                )
            if not torch.equal(rewards[rows], rewards[first].expand(len(rows))):
                raise ValueError("All traces must share their original rollout reward")
            trace_to_rollout[rows] = len(reward_rows)
            reward_rows.append(first)
            validity.append((sample_mask[rows] * has_targets[rows]).max())
            group_indices.append(group_index)
    if not reward_rows:
        raise ValueError("Cannot train an empty multi-trace batch")
    return LogicalRolloutBatch(
        prompt_ids=torch.tensor(group_indices, device=rewards.device).unsqueeze(-1),
        rewards=rewards[reward_rows],
        valid_mask=torch.stack(validity),
        trace_to_rollout=trace_to_rollout,
    )


def trace_training_evidence(
    tags: list[dict[str, Any]],
    *,
    sample_ids: list[str],
    token_mask: torch.Tensor,
    effective_mask: torch.Tensor,
    advantages: torch.Tensor,
) -> list[dict[str, Any]]:
    """Describe causal targets after all filtering and advantage processing."""
    generated = token_mask[:, 1:].bool()
    retained = effective_mask[:, 1:].bool()
    nonzero = retained & (advantages[:, 1:] != 0)
    result = []
    for row, tag in enumerate(tags):
        evidence = {
            **tag,
            "sample_id": sample_ids[row],
            "generated_targets": int(generated[row].sum()),
            "retained_targets": int(retained[row].sum()),
            "nonzero_advantage_targets": int(nonzero[row].sum()),
            "rollout_advantage": float(advantages[row, 0]),
        }
        if tag.get("training_mode") == "subagent":
            spans = tag.get("generation_spans")
            if not isinstance(spans, list):
                raise ValueError(
                    "Subagent training evidence requires owned generation spans"
                )
            owned = torch.zeros_like(generated[row])
            span_evidence = []
            previous_end = 0
            for span in spans:
                start, end = span.get("target_start"), span.get("target_end")
                if (
                    type(start) is not int
                    or type(end) is not int
                    or start < previous_end
                    or end <= start
                    or end > token_mask.shape[1]
                ):
                    raise ValueError(
                        "Subagent generation spans overlap or exceed their sequence"
                    )
                previous_end = end
                # The first full-sequence position has no causal prediction.
                positions = slice(max(start, 1) - 1, end - 1)
                owned[positions] = True
                span_evidence.append(
                    {
                        **span,
                        "generated_targets": int(generated[row, positions].sum()),
                        "retained_targets": int(retained[row, positions].sum()),
                        "nonzero_advantage_targets": int(nonzero[row, positions].sum()),
                    }
                )
            context_generated = int(generated[row, ~owned].sum())
            context_retained = int(retained[row, ~owned].sum())
            if context_generated or context_retained:
                raise ValueError("Session context contains unowned training targets")
            evidence.update(
                generation_spans=span_evidence,
                context_generated_targets=context_generated,
                context_retained_targets=context_retained,
            )
        result.append(evidence)
    return result


def validate_nonempty_optimizer_metrics(metrics: dict[str, Any]) -> None:
    """Reject incomplete/nonfinite optimizer results before advancing progress."""
    for field in ("loss", "grad_norm", "global_valid_toks"):
        value = metrics.get(field)
        if value is None or not math.isfinite(float(value)):
            raise RuntimeError(
                f"Multi-trace optimizer result has invalid {field}: {value!r}"
            )
    if float(metrics["global_valid_toks"]) <= 0:
        raise RuntimeError("Multi-trace optimizer result has no valid training tokens")
