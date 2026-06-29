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

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TensorMeta:
    name: str
    shape: torch.Size
    dtype: torch.dtype
    chunk_offset: int
    chunk_size: int
    offset: int | None

    @property
    def nbytes(self) -> int:
        return self.shape.numel() * self.dtype.itemsize


class CheckpointEngine(ABC):
    cleanup_after_load: bool = True

    @abstractmethod
    def prepare(self) -> Any:
        """Allocate or register backend resources and return serializable metadata."""
        raise NotImplementedError

    @abstractmethod
    def init_policy_process_group(
        self,
        *,
        worker_rank: int,
        train_world_size: int,
        rollout_world_size: int,
        metadata: list[Any],
    ) -> None:
        """Connect a policy worker to its transfer peer."""
        raise NotImplementedError

    @abstractmethod
    def init_rollout_process_group(
        self,
        *,
        rollout_rank: int,
        train_world_size: int,
        rollout_world_size: int,
        metadata: list[Any],
    ) -> None:
        """Connect a rollout worker to its transfer peer."""
        raise NotImplementedError

    def finalize(self) -> None:
        """Release per-refit backend state."""
        pass

    @abstractmethod
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
    ) -> None:
        """Send ``(name, tensor)`` weights from the policy side."""
        raise NotImplementedError

    @abstractmethod
    async def receive_weight_batches(
        self,
    ) -> AsyncGenerator[list[tuple[str, torch.Tensor]], None]:
        """Yield ``(name, tensor)`` batches on the generation side."""
        raise NotImplementedError


def create_checkpoint_engine(
    backend: str, *, bucket_size_bytes: int, engine_kwargs: dict[str, Any]
) -> CheckpointEngine:
    if backend == "nixl":
        backend = "nemo_rl.utils.checkpoint_engines.nixl:NIXLCheckpointEngine"
    module_name, class_name = backend.split(":", 1)
    engine_cls = getattr(importlib.import_module(module_name), class_name)
    return engine_cls(bucket_size=bucket_size_bytes, **engine_kwargs)


def split_weight_chunks(
    weights: Generator[tuple[str, torch.Tensor], None, None], bucket_size: int
) -> Generator[tuple[TensorMeta, torch.Tensor], None, None]:
    for name, weight in weights:
        buffer = weight.contiguous().view(-1).view(torch.uint8)
        for chunk_offset in range(0, weight.nbytes, bucket_size):
            chunk_size = min(bucket_size, weight.nbytes - chunk_offset)
            yield (
                TensorMeta(
                    name, weight.shape, weight.dtype, chunk_offset, chunk_size, None
                ),
                buffer[chunk_offset : chunk_offset + chunk_size],
            )


async def merge_weight_chunk_batches(
    chunk_batches: AsyncGenerator[list[tuple[TensorMeta, torch.Tensor]], None],
) -> AsyncGenerator[list[tuple[str, torch.Tensor]], None]:
    merge_weight: torch.Tensor | None = None
    merge_offset = 0
    async for chunk_batch in chunk_batches:
        weight_batch: list[tuple[str, torch.Tensor]] = []
        for meta, chunk in chunk_batch:
            if meta.chunk_offset == 0 and meta.chunk_size == meta.nbytes:
                weight_batch.append(
                    (meta.name, chunk.view(meta.dtype).view(meta.shape))
                )
                continue
            if merge_weight is None:
                merge_weight = torch.empty(
                    meta.shape, dtype=meta.dtype, device=chunk.device
                )
                merge_offset = 0
            merge_weight.view(-1).view(torch.uint8)[
                meta.chunk_offset : meta.chunk_offset + meta.chunk_size
            ] = chunk
            merge_offset += meta.chunk_size
            if merge_offset == meta.nbytes:
                weight_batch.append((meta.name, merge_weight))
                merge_weight = None
                merge_offset = 0
        if weight_batch:
            yield weight_batch
