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

"""Tensor reductions for opt-in batch attribution; no Ray or model imports."""

import torch

from nemo_rl.utils.batch_attribution import GroupTrainingStats, task_metric_prefix


@torch.no_grad()
def summarize_training_groups(
    *,
    group_ids: list[str],
    task_names: list[str],
    rewards: torch.Tensor,
    training_mask: torch.Tensor,
    sample_mask: torch.Tensor,
) -> tuple[GroupTrainingStats, ...]:
    """Reduce the right-shifted, post-filter loss mask to three scalars per row."""
    if len(group_ids) != len(task_names) or len(group_ids) != training_mask.shape[0]:
        raise ValueError("attribution identities do not match training rows")
    row_tokens = training_mask.sum(dim=-1)
    weighted_rewards = torch.where(sample_mask != 0, rewards * sample_mask, 0)
    row_scalars = (
        torch.stack((sample_mask, row_tokens, weighted_rewards), dim=-1)
        .detach()
        .double()
        .cpu()
        .tolist()
    )
    grouped: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        grouped.setdefault(group_id, []).append(index)
    stats = []
    for group_id, indices in grouped.items():
        names = {task_names[i] for i in indices}
        if len(names) != 1:
            raise ValueError(f"attribution group {group_id} mixes task names")
        weights = sum(row_scalars[i][0] for i in indices)
        stats.append(
            GroupTrainingStats(
                group_id=group_id,
                task_name=names.pop(),
                selected_samples=len(indices),
                valid_sequence_weight=weights,
                valid_tokens=sum(row_scalars[i][1] for i in indices),
                reward_sum=sum(row_scalars[i][2] for i in indices),
                reward_weight=weights,
                response_lengths=tuple(
                    round(row_scalars[i][1] / row_scalars[i][0])
                    for i in indices
                    if row_scalars[i][0] > 0
                ),
            )
        )
    return tuple(stats)


@torch.no_grad()
def task_entropy_metrics(
    *,
    curr_logprobs: torch.Tensor,
    generation_logprobs: torch.Tensor,
    mask: torch.Tensor,
    task_names: list[str],
    global_valid_toks: torch.Tensor,
    entropy_values: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Partition the ordinary entropy numerator with the same final loss mask.

    Contributions and token fractions share the global token denominator.
    Sum both across ranks/chunks before dividing, never average task means.
    """
    if len(task_names) != mask.shape[0]:
        raise ValueError("attribution task labels do not match loss rows")
    if entropy_values is None:
        entropy_values = torch.exp(curr_logprobs - generation_logprobs) * curr_logprobs
    row_entropy = -(entropy_values * mask).sum(dim=-1)
    row_tokens = mask.sum(dim=-1)
    metrics = {}
    for task_name in dict.fromkeys(task_names):
        rows = [i for i, name in enumerate(task_names) if name == task_name]
        prefix = task_metric_prefix(task_name)
        metrics[prefix + "entropy_contribution"] = row_entropy[rows].sum() / (
            global_valid_toks + 1e-8
        )
        metrics[prefix + "entropy_token_fraction"] = row_tokens[rows].sum() / (
            global_valid_toks + 1e-8
        )
    return metrics
