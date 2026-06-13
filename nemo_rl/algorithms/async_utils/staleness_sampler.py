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
    """Pick complete prompt groups inside a version staleness window.

    Defaults to FIFO (sample_freshest_first=False); pass True to prefer smallest lag.
    """

    def __init__(
        self,
        buffer: TQReplayBuffer,
        max_staleness_versions: int,
        sample_freshest_first: bool = False,
    ) -> None:
        if max_staleness_versions < 0:
            raise ValueError(
                f"max_staleness_versions must be non-negative, got "
                f"{max_staleness_versions}"
            )
        self._buffer = buffer
        self.max_staleness_versions = max_staleness_versions
        self.sample_freshest_first = sample_freshest_first

    async def select(
        self,
        *,
        current_train_weight: int,
        min_prompt_groups: int,
    ) -> tuple[KVBatchMeta | None, int]:
        """Return a concat of the first min_prompt_groups eligible groups, or None.

        Freshest-first (smallest lag, ties by insertion order) when
        sample_freshest_first is set, else insertion-order FIFO.
        Selected entries are dropped from the buffer locally; DataPlane rows survive
        for the trainer and are cleared by the caller at step boundary.

        Args:
            current_train_weight: Current trainer weight version. Eligibility window is
                [current_train_weight - max_staleness_versions, current_train_weight].
            min_prompt_groups: Minimum groups required; returns (None, 0) below this.

        Returns:
            meta: Concatenated KVBatchMeta covering num_groups groups, or None.
            num_groups: Number of prompt groups in meta; 0 when meta is None.
        """
        if min_prompt_groups < 1:
            raise ValueError(f"min_prompt_groups must be >= 1, got {min_prompt_groups}")

        min_valid_version = max(0, current_train_weight - self.max_staleness_versions)
        valid_idxs = [
            i
            for i, weight in enumerate(self._buffer.weight_list)
            if min_valid_version <= weight <= current_train_weight
        ]
        if len(valid_idxs) < min_prompt_groups:
            return None, 0

        if self.sample_freshest_first:
            valid_idxs.sort(
                key=lambda i: (
                    current_train_weight - self._buffer.weight_list[i],
                    i,
                )
            )

        selected_idxs = valid_idxs[:min_prompt_groups]
        selected_metas = [self._buffer.meta_list[i] for i in selected_idxs]

        await self._buffer.remove(selected_idxs, remove_in_dp=False)

        return (
            selected_metas[0].concat(*selected_metas[1:]),
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
            for i, weight in enumerate(self._buffer.weight_list)
            if weight < min_valid_version
        ]
        if not stale_idxs:
            return 0
        return await self._buffer.remove(stale_idxs, remove_in_dp=True)
