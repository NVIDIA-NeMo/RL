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

"""Prompt-group selection over a TQReplayBuffer."""

from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.data_plane import KVBatchMeta


class StalenessSampler:
    """Pick complete prompt groups from a TQReplayBuffer.

    Args:
        buffer: Shared TQReplayBuffer holding the candidate slots.
        max_staleness_versions: Max weight-version gap a sample may have from the trainer.
        sample_freshest_first: Prefer smallest lag when picking from the in-window set.
        require_order: Take only from the oldest in-window weight_version and wait for its batch to fill.
        force_in_order: Match each slot's target_step against current_train_weight, ignoring the window; mirrors legacy async_grpo target_weight semantics.
    """

    def __init__(
        self,
        buffer: TQReplayBuffer,
        max_staleness_versions: int,
        sample_freshest_first: bool = False,
        require_order: bool = False,
        force_in_order: bool = False,
    ) -> None:
        if max_staleness_versions < 0:
            raise ValueError(
                f"max_staleness_versions must be non-negative, got "
                f"{max_staleness_versions}"
            )
        if require_order and sample_freshest_first:
            raise ValueError(
                "require_order and sample_freshest_first are mutually exclusive"
            )
        self._buffer = buffer
        self.max_staleness_versions = max_staleness_versions
        self.sample_freshest_first = sample_freshest_first
        self.require_order = require_order
        self.force_in_order = force_in_order

    async def select(
        self,
        *,
        current_train_weight: int,
        min_prompt_groups: int,
    ) -> tuple[KVBatchMeta | None, int]:
        """Concat the first min_prompt_groups eligible groups and drop them from the buffer.

        Eligibility = ready and weight in
        [current_train_weight - max_staleness_versions, current_train_weight].
        DataPlane rows survive the local drop; caller clears them at step boundary.

        Args:
            current_train_weight: Current trainer weight version.
            min_prompt_groups: Minimum groups required; returns (None, 0) below this.

        Returns:
            meta: Concatenated KVBatchMeta, or None if not enough groups.
            num_groups: Number of prompt groups in meta; 0 when meta is None.
        """
        if min_prompt_groups < 1:
            raise ValueError(f"min_prompt_groups must be >= 1, got {min_prompt_groups}")

        if self.force_in_order:
            # target_step exact match; staleness window ignored.
            valid_idxs = [
                i
                for i, target in enumerate(self._buffer.target_step_list)
                if target == current_train_weight and self._buffer.ready_list[i]
            ]
        else:
            min_valid_version = max(
                0, current_train_weight - self.max_staleness_versions
            )
            if self.require_order:
                in_window = [
                    weight
                    for weight in self._buffer.start_weight_list
                    if min_valid_version <= weight <= current_train_weight
                ]
                if not in_window:
                    return None, 0
                target_version = min(in_window)
                valid_idxs = [
                    i
                    for i, weight in enumerate(self._buffer.start_weight_list)
                    if weight == target_version and self._buffer.ready_list[i]
                ]
            else:
                valid_idxs = [
                    i
                    for i, weight in enumerate(self._buffer.start_weight_list)
                    if min_valid_version <= weight <= current_train_weight
                    and self._buffer.ready_list[i]
                ]

        if len(valid_idxs) < min_prompt_groups:
            return None, 0

        if self.sample_freshest_first:
            valid_idxs.sort(
                key=lambda i: (
                    current_train_weight - self._buffer.start_weight_list[i],
                    i,
                )
            )

        selected_idxs = valid_idxs[:min_prompt_groups]
        selected_metas = [self._buffer.meta_list[i] for i in selected_idxs]

        await self._buffer.remove(selected_idxs, remove_in_dp=False)

        return (
            selected_metas[0].concat(*selected_metas[1:]),  # type: ignore
            len(selected_idxs),
        )

    async def evict(self, *, current_train_weight: int) -> int:
        """Drop groups whose weight falls below the staleness window.

        Future entries (weight > current_train_weight) are left alone.

        Args:
            current_train_weight: Current trainer weight version; groups with
                weight < current_train_weight - max_staleness_versions are dropped.

        Returns:
            Number of group entries removed from the buffer.
        """
        min_valid_version = max(0, current_train_weight - self.max_staleness_versions)
        stale_idxs = [
            i
            for i, weight in enumerate(self._buffer.start_weight_list)
            if weight < min_valid_version
        ]
        if not stale_idxs:
            return 0
        return await self._buffer.remove(stale_idxs, remove_in_dp=True)
