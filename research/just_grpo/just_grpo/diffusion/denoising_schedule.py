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
"""Backend-independent denoising schedules and diffusion score aggregation."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class DenoisingSchedule(ABC):
    """Describe model inputs and harvested-token alignment for each reveal level.

    Trajectories carry clean input_ids, target_ids, position_ids, token_mask,
    and masked_indices as [N,S] tensors. The model owns noisy/clean
    concatenation and block attention. original_positions maps [N,S] back to
    rollout coordinates; sample_indices maps [N] to original batch rows.
    Only token_mask positions contribute to the loss.
    """

    def __init__(self, *, num_steps: int, num_samples: int) -> None:
        self.num_samples = num_samples
        self._configure_steps(num_steps)

    def _configure_steps(self, num_steps: int) -> None:
        if num_steps < 1:
            raise ValueError("A denoising schedule needs at least one step")
        self.num_steps = num_steps

    @abstractmethod
    def generate_single_trajectory(self, time_step: int) -> BatchedDataDict[Any]:
        """Return one reveal level; do not retain all expanded levels."""

    def iter_levels(
        self, data: BatchedDataDict[Any] | None = None
    ) -> Iterator[BatchedDataDict[Any]]:
        """Yield one level at a time, optionally aligning rollout loss fields.

        This is the authoritative traversal order for scoring and training.
        Backend adapters consume it without choosing or enumerating levels.
        """
        for step in range(self.num_steps):
            yield (
                self.generate_single_trajectory(step)
                if data is None
                else self.prepare_trajectory(step, data)
            )

    def return_schedule(self, microbatch_size: int) -> Iterator[BatchedDataDict[Any]]:
        """Yield all levels for gradient accumulation before one optimizer step."""
        if microbatch_size < 1:
            raise ValueError("microbatch_size must be positive")
        for trajectory in self.iter_levels():
            for start in range(0, self.num_samples, microbatch_size):
                yield BatchedDataDict(
                    {
                        key: value[start : start + microbatch_size]
                        for key, value in trajectory.items()
                    }
                )

    def prepare_trajectory(
        self, time_step: int, data: BatchedDataDict[Any]
    ) -> BatchedDataDict[Any]:
        """Align rollout fields to one level without retaining other levels."""
        trajectory = self.generate_single_trajectory(time_step)
        indices, positions = (
            trajectory["sample_indices"],
            trajectory["original_positions"],
        )
        trajectory["sample_mask"] = data["sample_mask"][indices]
        trajectory["input_lengths"] = torch.full_like(
            indices, trajectory["input_ids"].shape[1]
        )
        for key in (
            "advantages",
            "prev_logprobs",
            "generation_logprobs",
            "reference_policy_logprobs",
        ):
            if key in data:
                trajectory[key] = data[key][indices].gather(1, positions)
        return trajectory


@torch.no_grad()
def aggregate_diffusion_logprobs(
    trajectories: Iterable[BatchedDataDict[Any]],
    score_fn: Callable[[BatchedDataDict[Any]], torch.Tensor],
    *,
    original_shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Score one view at a time and collect selected scores in rollout coordinates.

    The caller supplies views at its backend's batch granularity and manages
    model/reference weights. score_fn returns same-position [batch, sequence]
    scores. Empty views are passed through: distributed scorers may need their
    collectives even when this rank has no selected tokens.
    """
    result = torch.zeros(original_shape, device=device, dtype=torch.float32)
    for trajectory in trajectories:
        trajectory = trajectory.to(device)
        values = score_fn(trajectory).to(device=device, dtype=result.dtype)
        active = trajectory["token_mask"].bool()
        rows = trajectory["sample_indices"][:, None].expand_as(values)
        result.index_put_(
            (rows[active], trajectory["original_positions"][active]),
            values[active],
            accumulate=True,
        )
        del trajectory, values
    return result
