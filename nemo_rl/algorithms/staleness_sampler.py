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

"""Prompt-group selection over a TQReplayBuffer.

The sampler is pure filter logic: it reads ``meta_list`` / ``weight_list``
on the buffer it was constructed with and drives :meth:`TQReplayBuffer.remove`
when groups fall outside the staleness window. It never touches the data
plane directly.
"""

from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.data_plane import KVBatchMeta


class StalenessSampler:
    """Pick complete prompt groups inside a version staleness window."""

    def __init__(
        self,
        buffer: TQReplayBuffer,
        max_staleness_versions: int,
        sample_freshest_first: bool = True,
    ) -> None:
        if max_staleness_versions < 0:
            raise ValueError(
                f"max_staleness_versions must be non-negative, got "
                f"{max_staleness_versions}"
            )
        self._buffer = buffer
        self.max_staleness_versions = max_staleness_versions
        self.sample_freshest_first = sample_freshest_first

    def select(
        self,
        *,
        current_train_weight: int,
        min_prompt_groups: int,
    ) -> KVBatchMeta | None:
        """Return a concat of the first ``min_prompt_groups`` eligible groups.

        Eligible = lag in ``[0, max_staleness_versions]``. Order is
        freshest-first (smallest lag, ties broken by insertion order)
        when ``sample_freshest_first`` is set, else strict insertion-order
        FIFO. Returns ``None`` if fewer than ``min_prompt_groups`` groups
        are eligible.
        """
        if min_prompt_groups < 1:
            raise ValueError(
                f"min_prompt_groups must be >= 1, got {min_prompt_groups}"
            )

        eligible: list[tuple[int, int, KVBatchMeta]] = []
        for idx, (meta, weight) in enumerate(
            zip(self._buffer.meta_list, self._buffer.weight_list)
        ):
            lag = current_train_weight - weight
            if lag < 0 or lag > self.max_staleness_versions:
                continue
            eligible.append((lag, idx, meta))

        if len(eligible) < min_prompt_groups:
            return None

        if self.sample_freshest_first:
            eligible.sort(key=lambda item: (item[0], item[1]))
        else:
            eligible.sort(key=lambda item: item[1])

        chosen = [meta for _, _, meta in eligible[:min_prompt_groups]]
        return chosen[0].concat(*chosen[1:])

    async def evict(self, *, current_train_weight: int) -> int:
        """Drop groups whose lag exceeds the staleness window.

        Future entries (``lag < 0``) are left alone — they can't be
        produced under normal flow and the safer behavior is to surface
        them later rather than silently delete.

        Returns:
            Number of group entries removed from the buffer.
        """
        stale_sample_ids: list[str] = []
        for meta, weight in zip(self._buffer.meta_list, self._buffer.weight_list):
            lag = current_train_weight - weight
            if lag > self.max_staleness_versions:
                stale_sample_ids.extend(meta.sample_ids)
        if not stale_sample_ids:
            return 0
        return await self._buffer.remove(stale_sample_ids)
