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

"""Lazy Ray-backed routed-expert payloads for legacy async GRPO.

R3 adds a ``[tokens, layers, topk]`` expert-id tensor to every rollout.  A
full async step can contain hundreds of GiB of these ids.  Returning all
trajectory groups from the replay actor used to deserialize the whole payload
on the driver and then pad it once more before DP sharding.

The classes below keep the tensor bytes in Ray's object store.  The replay
actor, driver, and ``BatchedDataDict`` carry only object references; a policy
worker resolves and pads the rows after DP sharding and microbatch slicing.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

import ray
import torch


@dataclass(frozen=True)
class RoutedExpertsTensorRef:
    """Reference to one message's ``[tokens, layers, topk]`` route tensor."""

    object_ref: ray.ObjectRef | None
    shape: tuple[int, ...]
    dtype: torch.dtype
    fill_value: int = -1

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "RoutedExpertsTensorRef":
        if tensor.ndim != 3:
            raise ValueError(
                "routed_experts must have shape [tokens, layers, topk], "
                f"got {tuple(tensor.shape)}"
            )
        tensor = tensor.detach().to(device="cpu").contiguous()
        return cls(
            object_ref=ray.put(tensor),
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
        )

    @classmethod
    def filled_like(
        cls,
        *,
        num_tokens: int,
        template: "RoutedExpertsTensorRef",
        fill_value: int = -1,
    ) -> "RoutedExpertsTensorRef":
        return cls(
            object_ref=None,
            shape=(num_tokens, *template.shape[1:]),
            dtype=template.dtype,
            fill_value=fill_value,
        )

    def materialize(self) -> torch.Tensor:
        if self.object_ref is None:
            return torch.full(self.shape, self.fill_value, dtype=self.dtype)
        tensor = ray.get(self.object_ref)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "Routed-expert object reference resolved to "
                f"{type(tensor).__name__}, expected torch.Tensor"
            )
        if tuple(tensor.shape) != self.shape or tensor.dtype != self.dtype:
            raise ValueError(
                "Routed-expert object changed shape/dtype: expected "
                f"{self.shape}/{self.dtype}, got {tuple(tensor.shape)}/{tensor.dtype}"
            )
        return tensor


class RoutedExpertsBatch:
    """Logical route rows backed by per-message Ray object references."""

    def __init__(self, rows: Sequence[Sequence[RoutedExpertsTensorRef]]) -> None:
        self.rows = [list(row) for row in rows]
        for row in self.rows:
            if not row:
                raise ValueError("Every routed-expert row must contain a segment")
            tail = row[0].shape[1:]
            dtype = row[0].dtype
            if any(ref.shape[1:] != tail or ref.dtype != dtype for ref in row):
                raise ValueError(
                    "All routed-expert segments in a row must share layers/topk/dtype"
                )

    def __len__(self) -> int:
        return len(self.rows)

    @classmethod
    def from_message_segments(
        cls, segments: Sequence[RoutedExpertsTensorRef]
    ) -> "RoutedExpertsBatch":
        return cls([segments])

    @classmethod
    def concat(cls, batches: Sequence["RoutedExpertsBatch"]) -> "RoutedExpertsBatch":
        return cls([row for batch in batches for row in batch.rows])

    def slice(self, indices: list[int] | torch.Tensor) -> "RoutedExpertsBatch":
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return RoutedExpertsBatch([self.rows[index] for index in indices])

    def to(self, _device: str | torch.device) -> "RoutedExpertsBatch":
        # Bytes remain in Ray until a policy worker owns a DP-local shard.
        return self

    def materialize(
        self, *, pad_value: int = -1, pad_to_tokens: int | None = None
    ) -> torch.Tensor:
        """Resolve this DP-local shard and right-pad rows to one dense tensor."""
        refs = [ref for row in self.rows for ref in row if ref.object_ref is not None]
        resolved = ray.get([ref.object_ref for ref in refs]) if refs else []
        resolved_by_id = {
            ref.object_ref: tensor for ref, tensor in zip(refs, resolved)
        }

        rows: list[torch.Tensor] = []
        for segments in self.rows:
            tensors: list[torch.Tensor] = []
            for ref in segments:
                if ref.object_ref is None:
                    tensor = torch.full(ref.shape, ref.fill_value, dtype=ref.dtype)
                else:
                    tensor = resolved_by_id[ref.object_ref]
                    if tuple(tensor.shape) != ref.shape or tensor.dtype != ref.dtype:
                        raise ValueError(
                            "Routed-expert object changed shape/dtype while materializing"
                        )
                tensors.append(tensor)
            rows.append(torch.cat(tensors, dim=0))

        max_tokens = max(row.shape[0] for row in rows)
        if pad_to_tokens is not None:
            if pad_to_tokens < max_tokens:
                raise ValueError(
                    "Cannot materialize routed experts into fewer tokens than a row: "
                    f"pad_to_tokens={pad_to_tokens}, max_row_tokens={max_tokens}"
                )
            max_tokens = pad_to_tokens
        padded = [
            torch.nn.functional.pad(
                row,
                (0, 0, 0, 0, 0, max_tokens - row.shape[0]),
                value=pad_value,
            )
            for row in rows
        ]
        return torch.stack(padded, dim=0)


def offload_routed_experts_inplace(value: Any) -> int:
    """Replace route tensors nested in a trajectory with Ray references.

    Returns the number of tensor bytes moved out of the replay actor heap.
    """
    moved_bytes = 0
    # Production async trajectories store ``message_log`` inside a
    # ``BatchedDataDict`` (a UserDict/MutableMapping), not necessarily a
    # built-in dict.  Traverse the mapping protocol so the replay boundary
    # actually reaches those route tensors.
    if isinstance(value, MutableMapping):
        for key, item in list(value.items()):
            if key == "routed_experts" and isinstance(item, torch.Tensor):
                moved_bytes += item.numel() * item.element_size()
                value[key] = RoutedExpertsTensorRef.from_tensor(item)
            else:
                moved_bytes += offload_routed_experts_inplace(item)
    elif isinstance(value, list):
        for item in value:
            moved_bytes += offload_routed_experts_inplace(item)
    return moved_bytes


def materialize_routed_experts_inplace(data: Any) -> None:
    """Resolve an already-microbatched route payload in place.

    The caller must invoke this only after both DP sharding and policy
    microbatch slicing.  Padding to the longest *valid* row, rather than the
    parent batch's rectangular input width, keeps short packed sequences from
    expanding back to the global 65K-token width.
    """
    routed_experts = data.get("routed_experts")
    if isinstance(routed_experts, RoutedExpertsBatch):
        input_lengths = data.get("input_lengths")
        if isinstance(input_lengths, torch.Tensor) and input_lengths.numel() > 0:
            pad_to_tokens = int(input_lengths.max().item())
        else:
            input_ids = data.get("input_ids")
            pad_to_tokens = (
                int(input_ids.shape[1])
                if isinstance(input_ids, torch.Tensor)
                else None
            )
        data["routed_experts"] = routed_experts.materialize(
            pad_to_tokens=pad_to_tokens
        )
