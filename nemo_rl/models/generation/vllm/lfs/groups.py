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

"""Stable cross-DP group identity for async rollouts."""

from __future__ import annotations

from typing import Any

def build_async_cross_dp_group_ids(
    batch: Any, *, num_generations: int
) -> list[str]:
    """Build one stable cross-DP group key per generated trajectory.

    Dataset ``idx`` values are stable across async dataloader batches, whereas
    a prompt's position inside a batch is not. Reusing batch-local positions
    would therefore merge unrelated prompts in ``history_lfs``.
    """

    if num_generations <= 0:
        raise ValueError("num_generations must be positive")
    if "idx" in batch:
        raw_group_ids = batch["idx"]
        if len(raw_group_ids) != batch.size:
            raise ValueError(
                "batch idx count does not match prompt count: "
                f"idx={len(raw_group_ids)} prompts={batch.size}"
            )
        prompt_group_ids = [
            str(value.item() if hasattr(value, "item") else value)
            for value in raw_group_ids
        ]
    else:
        # Retain the pre-existing fallback for custom datasets without ``idx``.
        # Such datasets cannot provide cross-batch history identity.
        prompt_group_ids = [str(prompt_idx) for prompt_idx in range(batch.size)]
    return [
        group_id
        for group_id in prompt_group_ids
        for _ in range(num_generations)
    ]
