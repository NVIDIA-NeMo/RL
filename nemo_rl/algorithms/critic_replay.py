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
"""Bounded CPU replay for text-only, Monte Carlo PPO critics."""

import random
from collections.abc import Mapping
from pathlib import Path

import torch
from pydantic import BaseModel, Field, model_validator


class CriticReplayConfig(BaseModel, extra="forbid"):
    """Capacity and batch size are measured in trajectories, not tokens."""

    enabled: bool = False
    capacity: int = Field(default=512, ge=1)
    batch_size: int = Field(default=128, ge=1)
    seed: int = 0
    # Extra inference on the same fresh batch after each update; 0 disables it.
    probe_every_n_steps: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_capacity(self) -> "CriticReplayConfig":
        if self.capacity < self.batch_size:
            raise ValueError("critic replay capacity must be >= batch_size")
        return self


class CriticReplayBuffer:
    """FIFO eviction with uniform sampling without replacement per update.

    The caller trains the fresh batch once, then samples subsequent updates.
    Only critic inputs and fixed MC targets are retained; actor data is never
    replayed. Masks are preserved, including rows excluded by rollout filters.
    """

    filename = "critic_replay.pt"
    sequence_keys = ("input_ids", "token_mask", "returns")
    scalar_keys = ("input_lengths", "sample_mask")

    def __init__(self, config: CriticReplayConfig) -> None:
        self.config = config
        self.rows: list[dict[str, torch.Tensor]] = []
        self.steps: list[int] = []
        self.rng = random.Random(config.seed)
        self.update_counts: list[int] = []
        self.total_inserted = 0
        self.total_evicted = 0
        self.total_updates = 0
        self.restored_size = 0
        self.pending_indices: list[int] = []
        self.used_this_step: set[int] = set()
        self.updates_this_step = 0
        self.fresh_updates_this_step = 0

    def append(self, data: Mapping[str, torch.Tensor], *, step: int) -> None:
        """Copy unpadded CPU rows, then evict the oldest trajectories."""
        batch_size = data["input_ids"].shape[0]
        if batch_size != self.config.batch_size:
            raise ValueError(
                f"critic replay requires a fresh batch of {self.config.batch_size} "
                f"trajectories, received {batch_size}"
            )
        rows = []
        for index in range(batch_size):
            length = int(data["input_lengths"][index].item())
            if not 0 < length <= data["input_ids"].shape[1]:
                raise ValueError("Invalid critic replay input length")
            row = {
                key: data[key][index, :length].detach().cpu().clone()
                for key in self.sequence_keys
            }
            row.update(
                {
                    key: data[key][index].detach().cpu().clone()
                    for key in self.scalar_keys
                }
            )
            rows.append(row)
        self.total_evicted += max(0, len(self.rows) + batch_size - self.config.capacity)
        self.total_inserted += batch_size
        self.update_counts = (self.update_counts + [0] * batch_size)[
            -self.config.capacity :
        ]
        self.rows = (self.rows + rows)[-self.config.capacity :]
        self.steps = (self.steps + [step] * batch_size)[-self.config.capacity :]
        self.pending_indices = []
        self.used_this_step.clear()
        self.updates_this_step = 0
        self.fresh_updates_this_step = 0

    def sample(self, *, step: int) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Return one uniformly sampled batch, right-padded with masked zeros."""
        indices = self.rng.sample(range(len(self.rows)), self.config.batch_size)
        self.pending_indices = indices
        data = {
            key: torch.nn.utils.rnn.pad_sequence(
                [self.rows[index][key] for index in indices], batch_first=True
            )
            for key in self.sequence_keys
        }
        data.update(
            {
                key: torch.stack([self.rows[index][key] for index in indices])
                for key in self.scalar_keys
            }
        )
        ages = [step - self.steps[index] for index in indices]
        valid = data["token_mask"].bool() & (data["sample_mask"][:, None] > 0)
        return data, {
            "size": float(len(self.rows)),
            "sample_age_mean": sum(ages) / len(ages),
            "sample_age_max": float(max(ages)),
            "sample_fresh_fraction": sum(age == 0 for age in ages) / len(ages),
            "sample_valid_sequences": float(valid.any(1).sum()),
            "sample_valid_tokens": float(valid.sum()),
            "sample_positive_token_fraction": (
                float((data["returns"][valid] >= 0.5).float().mean())
                if valid.any()
                else float("nan")
            ),
            "sample_previous_updates_mean": sum(self.update_counts[i] for i in indices)
            / len(indices),
        }

    def record_update(self, *, fresh: bool, step: int) -> None:
        """Count exposures only after a train call succeeds, retaining masks."""
        indices = (
            [i for i, age_step in enumerate(self.steps) if age_step == step]
            if fresh
            else self.pending_indices
        )
        if len(indices) != self.config.batch_size:
            raise ValueError("Critic replay update must account for exactly one batch")
        for index in indices:
            self.update_counts[index] += 1
        self.used_this_step.update(indices)
        self.pending_indices = []
        self.updates_this_step += 1
        self.fresh_updates_this_step += int(fresh)
        self.total_updates += 1

    def metrics(self, *, step: int) -> dict[str, float]:
        """Report FIFO occupancy, coverage and completed training exposures."""
        size = len(self.rows)
        fresh_counts = [
            count
            for count, collected in zip(self.update_counts, self.steps)
            if collected == step
        ]
        eligible = [
            bool(row["sample_mask"] > 0) and bool(row["token_mask"].any())
            for row in self.rows
        ]
        return {
            "size": float(size),
            "capacity": float(self.config.capacity),
            "fill_fraction": size / self.config.capacity,
            "inserted_total": float(self.total_inserted),
            "evicted_total": float(self.total_evicted),
            "restored_size": float(self.restored_size),
            "updates_completed": float(self.updates_this_step),
            "fresh_updates_completed": float(self.fresh_updates_this_step),
            "replay_updates_completed": float(
                self.updates_this_step - self.fresh_updates_this_step
            ),
            "updates_total": float(self.total_updates),
            "trajectory_exposures": float(
                self.updates_this_step * self.config.batch_size
            ),
            "unique_trajectories_used": float(len(self.used_this_step)),
            "valid_unique_trajectories_used": float(
                sum(eligible[i] for i in self.used_this_step)
            ),
            "resident_valid_sequences": float(sum(eligible)),
            "fresh_valid_sequences": float(
                sum(
                    ok and collected == step
                    for ok, collected in zip(eligible, self.steps)
                )
            ),
            "buffer_coverage": len(self.used_this_step) / size if size else 0.0,
            "fresh_coverage": sum(count > 0 for count in fresh_counts)
            / len(fresh_counts)
            if fresh_counts
            else 0.0,
            "resident_updates_mean": sum(self.update_counts) / size if size else 0.0,
            "resident_updates_min": float(min(self.update_counts)) if size else 0.0,
            "oldest_age": float(step - min(self.steps)) if size else 0.0,
            "storage_bytes": float(
                sum(
                    t.numel() * t.element_size()
                    for row in self.rows
                    for t in row.values()
                )
            ),
        }

    def save(self, checkpoint: str | Path) -> None:
        """Save plain tensors and RNG state alongside the model checkpoint."""
        torch.save(
            {
                "version": 2,
                "config": self.config.model_dump(),
                "rows": self.rows,
                "steps": self.steps,
                "rng": self.rng.getstate(),
                "update_counts": self.update_counts,
                "total_inserted": self.total_inserted,
                "total_evicted": self.total_evicted,
                "total_updates": self.total_updates,
            },
            Path(checkpoint) / self.filename,
        )

    def restore(self, checkpoint: str | Path | None) -> None:
        """Resume exactly, or explicitly cold-start a legacy model checkpoint."""
        if checkpoint is None:
            return
        path = Path(checkpoint) / self.filename
        if not path.exists():
            print(
                "Critic replay: legacy checkpoint has no buffer; starting empty.",
                flush=True,
            )
            return
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["version"] != 2 or state["config"] != self.config.model_dump():
            raise ValueError("Critic replay checkpoint configuration mismatch")
        if (
            len(state["rows"]) != len(state["steps"])
            or len(state["rows"]) > self.config.capacity
            or len(state["update_counts"]) != len(state["rows"])
        ):
            raise ValueError("Invalid critic replay checkpoint size")
        self.rows = state["rows"]
        self.steps = state["steps"]
        self.rng.setstate(state["rng"])
        self.update_counts = state["update_counts"]
        self.total_inserted = state["total_inserted"]
        self.total_evicted = state["total_evicted"]
        self.total_updates = state["total_updates"]
        self.restored_size = len(self.rows)
        print(
            f"Critic replay restored: {self.restored_size} trajectories, {self.total_updates} completed updates",
            flush=True,
        )
