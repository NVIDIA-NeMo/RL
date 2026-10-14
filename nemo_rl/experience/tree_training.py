# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rollout-level reward accounting for a dense per-call training view."""

from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch

TREE_ID = "rollout_tree_id"
TREE_ROWS = "rollout_tree_rows"


def tree_sample_ids(root: str, count: int) -> list[str]:
    """Stable call-row identities, also used to clean uncertain publications."""
    return [f"{root}/call{index}" for index in range(max(1, count))]


def validate_tree_tags(tags: list[dict[str, Any]]) -> list[str] | None:
    """Reject incomplete/mixed trees; group sharding must retain entire trees."""
    if not any(TREE_ID in tag for tag in tags):
        return None
    if not all(isinstance(tag.get(TREE_ID), str) and tag[TREE_ID] for tag in tags):
        raise ValueError("mixed or missing rollout tree identities")
    roots = [tag[TREE_ID] for tag in tags]
    counts = Counter(roots)
    if any(tag.get(TREE_ROWS) != counts[root] for tag, root in zip(tags, roots)):
        raise ValueError("incomplete rollout tree rows")
    return roots


@dataclass(frozen=True)
class TreeProjection:
    first: torch.Tensor
    inverse: torch.Tensor
    rewards: torch.Tensor
    prompts: torch.Tensor
    valid: torch.Tensor

    def advantages(self, estimator: Any, shape: torch.Size) -> torch.Tensor:
        # A single GRPO sample per rollout, including leave-one-out and std.
        root_advantages = estimator.compute_advantage(
            prompt_ids=self.prompts,
            rewards=self.rewards,
            mask=self.rewards.new_ones((len(self.first), 1)),
            valid_mask=self.valid,
        )
        return root_advantages[self.inverse].expand(shape)


def project_trees(
    tags: list[dict[str, Any]] | None,
    rewards: torch.Tensor,
    prompts: torch.Tensor,
    valid: torch.Tensor,
) -> TreeProjection | None:
    roots = validate_tree_tags(tags) if tags is not None else None
    if roots is None:
        return None
    members: dict[str, list[int]] = {}
    for i, root in enumerate(roots):
        members.setdefault(root, []).append(i)
    first = []
    inverse = torch.empty(len(roots), dtype=torch.long, device=rewards.device)
    root_valid = []
    for ordinal, indices in enumerate(members.values()):
        first.append(indices[0])
        inverse[indices] = ordinal
        if not torch.equal(rewards[indices], rewards[indices[0]].expand(len(indices))):
            raise ValueError("one rollout tree has inconsistent rewards")
        if not torch.equal(
            prompts[indices], prompts[indices[0]].expand(len(indices), -1)
        ):
            raise ValueError("one rollout tree spans prompt groups")
        # If one call is filtered, mask the whole tree. Partial trees must not
        # quietly become a different training objective or baseline sample.
        root_valid.append(valid[indices].amin())
    first_t = torch.tensor(first, dtype=torch.long, device=rewards.device)
    return TreeProjection(
        first_t, inverse, rewards[first_t], prompts[first_t], torch.stack(root_valid)
    )
