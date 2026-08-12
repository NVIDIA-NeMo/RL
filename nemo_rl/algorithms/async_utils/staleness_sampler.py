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


def _emit_group_log(
    tag: str,
    buffer: TQReplayBuffer,
    idx: int,
    current_train_weight: int,
    *,
    ready: bool | None = None,
) -> int | None:
    """Print one ``[tag]`` line describing the group at ``buffer`` index ``idx``.

    Shared by evict (``[EVICT]``) and select (``[SELECT]``) so their per-group
    logs share one format for downstream parsing. Fields:
      group_id / instance_id / prompt_idx — identity
      start_v / end_v / target_step / train_v / lag — timing
      ready — commit-before-eviction flag (evict only; omitted when None)
      num_rows / resolved_count — group-level pass/fail summary
      per_row_reasons / per_row_resolved — per-rollout attribution
      per_row_tokens / total_tokens — per-row token lengths (ready groups only)

    Args:
        tag: Log marker, "EVICT" or "SELECT".
        buffer: The replay buffer holding the group's metadata.
        idx: Index of the group within the buffer's parallel lists.
        current_train_weight: Current trainer weight version (for lag).
        ready: When not None, adds a ``ready=`` field. Evict passes the slot's
            ready flag; select omits it since every selected group is ready.

    Returns:
        The group's total token count (sum of per-row lengths) when known, else
        None — evict uses it to accumulate evicted-token totals.
    """
    meta = buffer.meta_list[idx]
    start_v = buffer.start_weight_list[idx]
    end_v = buffer.end_weight_list[idx]
    target_step = buffer.target_step_list[idx]
    group_id = buffer._group_ids[idx]
    instance_id = (
        buffer.instance_id_list[idx]
        if getattr(buffer, "instance_id_list", None)
        else "unknown"
    )
    prompt_idx = (
        buffer.prompt_idx_list[idx]
        if getattr(buffer, "prompt_idx_list", None)
        else -1
    )
    lag = current_train_weight - start_v
    if meta and meta.tags:
        per_row_reasons = [t.get("failure_reason", "?") for t in meta.tags]
        per_row_resolved = [int(bool(t.get("resolved", False))) for t in meta.tags]
    else:
        per_row_reasons = None
        per_row_resolved = None
    per_row_tokens = (
        list(meta.sequence_lengths) if meta and meta.sequence_lengths else None
    )
    total_tokens = sum(per_row_tokens) if per_row_tokens else None
    num_rows = len(meta.sample_ids) if meta else None
    resolved_count = sum(per_row_resolved) if per_row_resolved else None
    ready_field = f"ready={ready} " if ready is not None else ""
    print(
        f"[{tag}] "
        f"group_id={group_id} "
        f"instance_id={instance_id} "
        f"prompt_idx={prompt_idx} "
        f"start_v={start_v} "
        f"end_v={end_v} "
        f"target_step={target_step} "
        f"train_v={current_train_weight} "
        f"lag={lag} "
        f"{ready_field}"
        f"num_rows={num_rows} "
        f"resolved_count={resolved_count}/{num_rows if num_rows is not None else '?'} "
        f"per_row_reasons={per_row_reasons} "
        f"per_row_resolved={per_row_resolved} "
        f"per_row_tokens={per_row_tokens} "
        f"total_tokens={total_tokens}",
        flush=True,
    )
    return total_tokens


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
        max_prompt_groups: int,
    ) -> tuple[KVBatchMeta | None, int]:
        """Concat up to max_prompt_groups eligible groups and drop them from the buffer.

        Eligibility = ready and weight in
        [current_train_weight - max_staleness_versions, current_train_weight].
        DataPlane rows survive the local drop; caller clears them at step boundary.

        Args:
            current_train_weight: Current trainer weight version.
            min_prompt_groups: Minimum groups required; returns (None, 0) below this.
            max_prompt_groups: Cap on groups returned when the threshold is met.

        Returns:
            meta: Concatenated KVBatchMeta, or None if not enough groups.
            num_groups: Number of prompt groups in meta; 0 when meta is None.
        """
        if min_prompt_groups < 1:
            raise ValueError(f"min_prompt_groups must be >= 1, got {min_prompt_groups}")
        if max_prompt_groups < min_prompt_groups:
            raise ValueError(
                f"max_prompt_groups ({max_prompt_groups}) must be >= "
                f"min_prompt_groups ({min_prompt_groups})"
            )

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

        requested_groups = min(len(valid_idxs), max_prompt_groups)
        selected_idxs = valid_idxs[:requested_groups]
        selected_metas = [self._buffer.meta_list[i] for i in selected_idxs]

        # Per-group [SELECT] log (same format as [EVICT], no ready field since all
        # selected groups are ready) so downstream analysis can compare the token
        # lengths of ACCEPTED rollouts against evicted ones. Emitted before the
        # drop, while the buffer indices are still valid.
        for i in selected_idxs:
            _emit_group_log("SELECT", self._buffer, i, current_train_weight)

        await self._buffer.remove(selected_idxs, remove_in_dp=False)

        return (
            selected_metas[0].concat(*selected_metas[1:]),  # type: ignore
            len(selected_idxs),
        )

    async def evict(self, *, current_train_weight: int) -> tuple[int, int]:
        """Drop groups whose weight falls below the staleness window.

        Future entries (weight > current_train_weight) are left alone.

        Args:
            current_train_weight: Current trainer weight version; groups with
                weight < current_train_weight - max_staleness_versions are dropped.

        Returns:
            (num_groups_removed, total_evicted_tokens). The token total counts
            only ready (committed) groups, whose ``meta.sequence_lengths`` is
            populated; groups evicted before their rollout committed have no
            token metadata and contribute 0.
        """
        min_valid_version = max(0, current_train_weight - self.max_staleness_versions)
        stale_idxs = [
            i
            for i, weight in enumerate(self._buffer.start_weight_list)
            if weight < min_valid_version
        ]
        if not stale_idxs:
            return 0, 0

        # Per-group [EVICT] log; token totals accumulate over ready slots only
        # (unready slots were evicted before commit and carry no token metadata).
        evicted_tokens = 0
        for i in stale_idxs:
            total_tokens = _emit_group_log(
                "EVICT",
                self._buffer,
                i,
                current_train_weight,
                ready=self._buffer.ready_list[i],
            )
            if total_tokens:
                evicted_tokens += total_tokens

        num_removed = await self._buffer.remove(stale_idxs, remove_in_dp=True)
        return num_removed, evicted_tokens
