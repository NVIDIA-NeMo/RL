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

"""Compute async GRPO prev-policy logprobs while a step waits for its groups."""

from typing import Any, Callable, Optional

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.utils.timer import Timer


def _token_hash(token_ids: torch.Tensor) -> int:
    """Return a hash of a row's token ids.

    Python's built-in hash is enough here: a match is confirmed by comparing
    the tokens, and the table lives in one process.
    """
    return hash(token_ids.cpu().numpy().tobytes())


class OverlappedLogprobs:
    """Prev-policy logprobs for a step's rows, computed while the step waits.

    Used by ``grpo.async_grpo.overlap_logprobs``. The policy weights change only
    in training, so while step K waits for its last groups the policy already
    holds the weights its prev-policy logprobs need. ``compute_arrived()`` runs
    in the wait loop and computes logprobs for the step's groups that have
    arrived; ``get_logprobs()`` then returns logprobs for the whole training
    batch, computing only the rows that were not done during the wait.

    Rows are matched to the batch by their tokens, not their position, so they
    are reused whatever order the batch takes the groups in. A row that matches
    nothing is recomputed.
    """

    def __init__(
        self,
        policy: ColocatablePolicyInterface,
        replay_buffer: Any,
        build_batch: Callable[[BatchedDataDict], BatchedDataDict],
    ) -> None:
        """Initialize the overlap.

        Args:
            policy: Policy that computes the logprobs.
            replay_buffer: The async GRPO replay buffer actor.
            build_batch: Turns arrived rollout rows into a logprob input batch,
                built the same way as the step's training batch (with
                ``input_ids`` and ``input_lengths``). It may modify the rows'
                message logs in place.
        """
        self._policy = policy
        self._replay_buffer = replay_buffer
        self._build_batch = build_batch
        self._dp_size = policy.sharding_annotations.get_axis_size("data_parallel")
        # Per-step state, cleared by _reset() when a new step starts waiting.
        self._weight_version = -1  # the step being waited for
        self._num_groups = 0  # groups fetched from the buffer so far
        self._num_rows_done = 0  # rows with logprobs computed so far
        self._pending: Optional[BatchedDataDict] = None  # fetched, not computed
        # Computed rows keyed by a hash of their tokens: (token ids, logprobs).
        self._computed: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def _reset(self, weight_version: int) -> None:
        self._weight_version = weight_version
        self._num_groups = 0
        self._num_rows_done = 0
        self._pending = None
        self._computed = {}

    def compute_arrived(self, weight_version: int) -> None:
        """Compute logprobs for the step's rows that arrived since the last call.

        Computes a multiple of the data-parallel size and holds the remaining
        rows for a later call.

        Args:
            weight_version: The training step being waited for.
        """
        if weight_version != self._weight_version:
            self._reset(weight_version)
        self._fetch_arrived_rows()
        if self._pending is None:
            return

        num_rows = self._pending.size - self._pending.size % self._dp_size
        if num_rows == 0:
            return
        rows = self._pending.slice(0, num_rows)
        self._pending = (
            self._pending.slice(num_rows, self._pending.size)
            if num_rows < self._pending.size
            else None
        )
        print(
            f"⏩ Step {weight_version}: computing logprobs for {num_rows} rows while "
            f"waiting ({self._num_rows_done} done, "
            f"{0 if self._pending is None else self._pending.size} held)",
            flush=True,
        )
        if self._num_rows_done == 0:
            self._policy.prepare_for_lp_inference()
        data = self._build_batch(rows)
        logprobs = self._policy.get_logprobs(data)["logprobs"]

        # Hash the rows now, while the step waits anyway.
        for ids, length, row_logprobs in zip(
            data["input_ids"], data["input_lengths"].tolist(), logprobs
        ):
            self._computed[_token_hash(ids[:length])] = (
                ids[:length],
                row_logprobs[:length],
            )
        self._num_rows_done += num_rows

    def get_logprobs(
        self, train_data: BatchedDataDict, timer: Timer
    ) -> tuple[torch.Tensor, int]:
        """Return prev-policy logprobs for the step's batch, computing the rest.

        Args:
            train_data: The step's training batch.
            timer: Timer passed to the policy for the rows computed here.

        Returns:
            The logprobs, and how many of the batch's rows were computed
            during the wait.
        """
        logprobs = torch.zeros_like(train_data["generation_logprobs"])
        missing: list[int] = []
        for i, (ids, length) in enumerate(
            zip(train_data["input_ids"], train_data["input_lengths"].tolist())
        ):
            match = self._computed.get(_token_hash(ids[:length]))
            # Compare the tokens too, so a hash collision cannot reuse wrong logprobs.
            if match is not None and torch.equal(match[0], ids[:length]):
                logprobs[i, :length] = match[1]
            else:
                missing.append(i)

        num_reused = train_data.size - len(missing)
        print(
            f"  Logprobs computed during the wait: {num_reused}/{train_data.size} rows"
        )
        if missing:
            # Round the call up to a multiple of the data-parallel size with
            # already-reused rows; the batch size is a multiple of it.
            missing_set = set(missing)
            reused = [i for i in range(train_data.size) if i not in missing_set]
            rows = missing + reused[: -len(missing) % self._dp_size]
            logprobs[rows] = self._policy.get_logprobs(
                train_data.select_indices(rows), timer=timer
            )["logprobs"]
        self._reset(weight_version=-1)
        return logprobs, num_reused

    def _fetch_arrived_rows(self) -> None:
        """Add the rows of the step's groups that arrived since the last fetch."""
        groups = ray.get(
            self._replay_buffer.peek.remote(
                self._weight_version, start=self._num_groups
            )
        )
        if not groups:
            return
        self._num_groups += len(groups)
        batches = [group["batch"] for group in groups]
        if self._pending is not None:
            batches.insert(0, self._pending)
        self._pending = BatchedDataDict.from_batches(batches)
