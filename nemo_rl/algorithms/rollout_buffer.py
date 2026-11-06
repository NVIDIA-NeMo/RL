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

"""Bounded, synchronous rollout buffering; no policy-dependent training state."""

import copy
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import torch
from torch.utils._pytree import tree_map

from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

BUFFER_FILENAME = "rollout_buffer.pt"
BUFFER_VERSION = 1


@dataclass
class RolloutBuffer:
    """Own completed CPU rollouts until each whole optimizer batch is consumed.

    A dataloader checkpoint points *past* the collected prompts. This sidecar
    must therefore be committed with the policy/optimizer and dataloader state,
    including when empty. Derived logprobs/advantages/coefficients never enter
    the buffer. ``peek`` isolates mutable reward/message processing from it.
    """

    prompts_per_update: int
    responses_per_prompt: int
    multiplier: int
    batch: BatchedDataDict[DatumSpec] | None = None
    input_ids: torch.Tensor | None = None
    collection_step: int | None = None
    cursor: int = 0
    epoch_exhausted: bool = False

    @property
    def update_size(self) -> int:
        return self.prompts_per_update * self.responses_per_prompt

    @property
    def remaining_updates(self) -> int:
        return (
            0
            if self.batch is None
            else (self.batch.size - self.cursor) // self.update_size
        )

    def prompt_batches(
        self, dataloader: Iterable[BatchedDataDict[DatumSpec]], remaining_steps: int
    ) -> Iterator[BatchedDataDict[DatumSpec] | None]:
        """Drain restored rollouts first; coalesce only within the current epoch.

        None means the caller must use the pending rollouts, without generation
        or dataloader advancement. Keep the dataloader's drop-last semantics;
        a final collection can contain fewer than ``multiplier`` full batches.
        """
        iterator: Iterator[BatchedDataDict[DatumSpec]] | None = None
        for step in range(remaining_steps):
            if self.remaining_updates:
                yield None
                continue
            if self.epoch_exhausted:
                self.epoch_exhausted = False
                return
            if iterator is None:
                iterator = iter(dataloader)
            requested = min(self.multiplier, remaining_steps - step)
            batches = list(islice(iterator, requested))
            self.epoch_exhausted = len(batches) < requested
            if not batches:
                self.epoch_exhausted = False
                return
            if any(batch.size != self.prompts_per_update for batch in batches):
                raise ValueError(
                    "Buffered rollouts require full per-update prompt batches (drop_last=True)"
                )
            yield BatchedDataDict.from_batches(batches)

    def fill(
        self, batch: BatchedDataDict[DatumSpec], input_ids: torch.Tensor, *, step: int
    ) -> None:
        if self.remaining_updates:
            raise RuntimeError("Cannot replace unconsumed rollouts")
        if (
            not 0 < batch.size <= self.multiplier * self.update_size
            or batch.size % self.update_size
        ):
            raise ValueError("Rollout collection must contain whole optimizer batches")
        if input_ids.shape[0] != batch.size:
            raise ValueError(
                "Rollouts and original prompt IDs have different row counts"
            )
        groups = input_ids.reshape(-1, self.responses_per_prompt, input_ids.shape[-1])
        if not torch.equal(groups, groups[:, :1].expand_as(groups)):
            raise ValueError(
                "Buffered rollout rows must retain complete contiguous prompt groups"
            )
        self.batch = BatchedDataDict(
            tree_map(
                lambda value: value.detach().cpu()
                if isinstance(value, torch.Tensor)
                else value,
                dict(batch),
            )
        )
        self.input_ids = input_ids.detach().cpu()
        self.collection_step = step
        self.cursor = 0

    def peek(self) -> tuple[BatchedDataDict[DatumSpec], torch.Tensor]:
        if not self.remaining_updates or self.input_ids is None:
            raise RuntimeError("No buffered optimizer batch available")
        assert self.batch is not None
        end = self.cursor + self.update_size
        return (
            copy.deepcopy(self.batch.slice(self.cursor, end)),
            self.input_ids[self.cursor : end].clone(),
        )

    def commit(self) -> None:
        """Advance only after the optimizer call succeeds."""
        if not self.remaining_updates:
            raise RuntimeError("No buffered optimizer batch to commit")
        self.cursor += self.update_size
        if not self.remaining_updates:
            self.batch = None
            self.input_ids = None
            self.collection_step = None
            self.cursor = 0

    def save(self, checkpoint_dir: str | Path, *, completed_steps: int) -> None:
        # Save only unconsumed records, with no tensor views retaining old rows.
        torch.save(
            {
                "version": BUFFER_VERSION,
                "completed_steps": completed_steps,
                "prompts_per_update": self.prompts_per_update,
                "responses_per_prompt": self.responses_per_prompt,
                "multiplier": self.multiplier,
                "collection_step": self.collection_step,
                "epoch_exhausted": self.epoch_exhausted,
                "batch": copy.deepcopy(
                    dict(self.batch.slice(self.cursor, self.batch.size))
                )
                if self.batch is not None
                else None,
                "input_ids": self.input_ids[self.cursor :].clone()
                if self.input_ids is not None
                else None,
            },
            Path(checkpoint_dir) / BUFFER_FILENAME,
        )

    def restore(self, checkpoint_dir: str | Path, *, completed_steps: int) -> None:
        # Native checkpoints are trusted local artifacts, as are dataloader
        # checkpoints; message logs may contain non-tensor Python objects.
        state = torch.load(
            Path(checkpoint_dir) / BUFFER_FILENAME,
            map_location="cpu",
            weights_only=False,
        )
        for key, expected in {
            "version": BUFFER_VERSION,
            "completed_steps": completed_steps,
            "prompts_per_update": self.prompts_per_update,
            "responses_per_prompt": self.responses_per_prompt,
            "multiplier": self.multiplier,
        }.items():
            if state[key] != expected:
                raise ValueError(
                    f"Incompatible rollout buffer {key}: {state[key]!r}, expected {expected!r}"
                )
        if not isinstance(state["epoch_exhausted"], bool):
            raise ValueError("Invalid dataloader exhaustion flag")
        self.epoch_exhausted = state["epoch_exhausted"]
        if state["batch"] is None:
            if state["input_ids"] is not None or state["collection_step"] is not None:
                raise ValueError("Invalid empty rollout buffer checkpoint")
            return
        step = state["collection_step"]
        if not isinstance(step, int) or not 0 <= step < completed_steps:
            raise ValueError("Invalid rollout collection step in checkpoint")
        self.fill(BatchedDataDict(state["batch"]), state["input_ids"], step=step)
        if completed_steps - step + self.remaining_updates > self.multiplier:
            raise ValueError("Rollout checkpoint cursor exceeds its collection size")
