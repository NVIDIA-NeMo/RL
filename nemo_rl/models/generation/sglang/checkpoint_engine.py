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

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from nemo_rl.utils.checkpoint_engines.base import CheckpointEngine


def _group_batch_by_dtype(
    batch: list[tuple[str, torch.Tensor]],
) -> dict[torch.dtype, list[tuple[str, torch.Tensor]]]:
    grouped: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = defaultdict(list)
    for name, tensor in batch:
        grouped[tensor.dtype].append((name, tensor))
    return dict(grouped)


def _validate_rank_batches(
    batches: list[list[tuple[str, torch.Tensor]]],
) -> list[dict[torch.dtype, list[tuple[str, torch.Tensor]]]]:
    """Validate the rank-local batches expected by SGLang's TP loader."""
    grouped = [_group_batch_by_dtype(batch) for batch in batches]
    expected_dtypes = list(grouped[0])
    expected_dtype_set = set(expected_dtypes)
    for rank, rank_groups in enumerate(grouped[1:], start=1):
        if set(rank_groups) != expected_dtype_set:
            raise RuntimeError(
                "Checkpoint-engine batches diverged across SGLang ranks: "
                f"rank 0 has dtypes {expected_dtypes}, rank {rank} has "
                f"{list(rank_groups)}."
            )
        for dtype in expected_dtypes:
            expected_names = [name for name, _tensor in grouped[0][dtype]]
            names = [name for name, _tensor in rank_groups[dtype]]
            if names != expected_names:
                raise RuntimeError(
                    "Checkpoint-engine batches diverged across SGLang ranks for "
                    f"dtype {dtype}: rank 0 has {expected_names}, rank {rank} has "
                    f"{names}."
                )
    return grouped


async def _aligned_checkpoint_engine_batches(
    checkpoint_engines: list["CheckpointEngine"],
) -> AsyncGenerator[list[list[tuple[str, torch.Tensor]]], None]:
    """Align rank-local streams by weight name, independent of bucket boundaries."""
    if not checkpoint_engines:
        raise RuntimeError("SGLang checkpoint-engine refit has no rank receivers.")
    generators = [
        checkpoint_engine.receive_weight_batches().__aiter__()
        for checkpoint_engine in checkpoint_engines
    ]
    pending: list[deque[tuple[str, torch.Tensor]]] = [
        deque() for _generator in generators
    ]
    finished = [False] * len(generators)
    exhausted = object()

    while True:
        empty_ranks = [
            rank
            for rank, queue in enumerate(pending)
            if not queue and not finished[rank]
        ]
        if empty_ranks:
            received = await asyncio.gather(
                *(anext(generators[rank], exhausted) for rank in empty_ranks)
            )
            for rank, batch in zip(empty_ranks, received, strict=True):
                if batch is exhausted:
                    finished[rank] = True
                    continue
                if not isinstance(batch, list):
                    raise TypeError(
                        "Checkpoint-engine receiver returned a non-list batch."
                    )
                if not batch:
                    raise RuntimeError(
                        f"Checkpoint-engine receiver for SGLang rank {rank} "
                        "returned an empty batch."
                    )
                pending[rank].extend(batch)

        if all(finished[rank] and not queue for rank, queue in enumerate(pending)):
            break
        if any(finished[rank] and not queue for rank, queue in enumerate(pending)):
            raise RuntimeError(
                "Checkpoint-engine streams ended with different weights across "
                "SGLang ranks."
            )
        if any(not queue for queue in pending):
            continue

        aligned = [[] for _queue in pending]
        while all(pending):
            expected_name, expected_tensor = pending[0][0]
            for rank, queue in enumerate(pending[1:], start=1):
                name, tensor = queue[0]
                if name != expected_name or tensor.dtype != expected_tensor.dtype:
                    raise RuntimeError(
                        "Checkpoint-engine weight order diverged across SGLang "
                        f"ranks: rank 0 has {expected_name!r}/{expected_tensor.dtype}, "
                        f"rank {rank} has {name!r}/{tensor.dtype}."
                    )
            for rank, queue in enumerate(pending):
                aligned[rank].append(queue.popleft())
        yield aligned


class SGLangCheckpointEngineMixin:
    """Receive checkpoint-engine batches and hand them to SGLang via CUDA IPC."""

    checkpoint_engines: list["CheckpointEngine"]
    base_gpu_id: int | None
    gpus_per_node: int
    node_rank: int
    num_gpus_per_engine: int | None
    _checkpoint_engine_rollout_rank_start: int
    _checkpoint_engine_target_devices: list[torch.device]
    _checkpoint_engine_weight_version: int

    def _checkpoint_engine_devices(self) -> list[torch.device]:
        if self.base_gpu_id is None or self.num_gpus_per_engine is None:
            raise RuntimeError(
                "SGLang worker GPU topology must be initialized before "
                "checkpoint-engine refit."
            )
        if self.num_gpus_per_engine > self.gpus_per_node:
            raise NotImplementedError(
                "SGLang checkpoint-engine refit currently requires each logical "
                "engine to fit on one node."
            )
        base_gpu_id = self._to_local_gpu_id(self.base_gpu_id)
        return [
            torch.device("cuda", base_gpu_id + local_rank)
            for local_rank in range(self.num_gpus_per_engine)
        ]

    def checkpoint_engine_total_memory_bytes(self) -> list[int]:
        return [
            torch.cuda.get_device_properties(device).total_memory
            for device in self._checkpoint_engine_devices()
        ]

    def init_checkpoint_engine(
        self,
        backend: str,
        bucket_size_bytes: int,
        engine_kwargs: dict[str, Any],
        rollout_rank_start: int,
    ) -> None:
        if engine_kwargs.get("shard_expert_weights", False):
            raise NotImplementedError(
                "SGLang checkpoint-engine refit does not support "
                "shard_expert_weights=true; use full-weight MoE refit instead."
            )
        if getattr(self, "checkpoint_engines", None) is not None:
            return

        from nemo_rl.models.generation.sglang.utils.train_utils import (
            monkey_patch_torch_reductions,
        )
        from nemo_rl.utils.checkpoint_engines.base import create_checkpoint_engine

        monkey_patch_torch_reductions()
        self._checkpoint_engine_rollout_rank_start = rollout_rank_start
        self._checkpoint_engine_target_devices = self._checkpoint_engine_devices()
        self.checkpoint_engines = []
        for device in self._checkpoint_engine_target_devices:
            rank_engine_kwargs = dict(engine_kwargs)
            configured_device = torch.device(rank_engine_kwargs.get("device", "cuda"))
            if configured_device.type == "cuda":
                rank_engine_kwargs["device"] = str(device)
            with torch.cuda.device(device):
                self.checkpoint_engines.append(
                    create_checkpoint_engine(
                        backend,
                        bucket_size_bytes=bucket_size_bytes,
                        engine_kwargs=rank_engine_kwargs,
                    )
                )

    def prepare_checkpoint_engine(self) -> list[Any]:
        metadata = []
        for local_rank, checkpoint_engine in enumerate(self.checkpoint_engines):
            rank_metadata = checkpoint_engine.prepare()
            if isinstance(rank_metadata, dict):
                rank_metadata = {
                    **rank_metadata,
                    "rank": self._checkpoint_engine_rollout_rank_start + local_rank,
                }
            metadata.append(rank_metadata)
        return metadata

    def init_checkpoint_engine_process_group(
        self,
        train_world_size: int,
        rollout_world_size: int,
        metadata: list[Any],
    ) -> None:
        for local_rank, checkpoint_engine in enumerate(self.checkpoint_engines):
            checkpoint_engine.init_rollout_process_group(
                rollout_rank=self._checkpoint_engine_rollout_rank_start + local_rank,
                train_world_size=train_world_size,
                rollout_world_size=rollout_world_size,
                metadata=metadata,
            )

    def _serialize_checkpoint_engine_batches(
        self,
        batches: list[list[tuple[str, torch.Tensor]]],
    ) -> tuple[list[list[str]], list[Any]]:
        from nemo_rl.models.generation.sglang.utils.train_utils import (
            FlattenedTensorBucket,
            MultiprocessingSerializer,
        )

        grouped = _validate_rank_batches(batches)
        serialized_by_dtype: list[list[str]] = []
        keepalive_buckets = []
        for dtype in grouped[0]:
            rank_payloads = []
            for device, rank_groups in zip(
                self._checkpoint_engine_target_devices, grouped, strict=True
            ):
                named_tensors = [
                    (name, tensor.to(device, non_blocking=True))
                    for name, tensor in rank_groups[dtype]
                ]
                bucket = FlattenedTensorBucket(named_tensors=named_tensors)
                keepalive_buckets.append(bucket)
                payload = {
                    "flattened_tensor": bucket.get_flattened_tensor(),
                    "metadata": bucket.get_metadata(),
                }
                rank_payloads.append(
                    MultiprocessingSerializer.serialize(payload, output_str=True)
                )
            serialized_by_dtype.append(rank_payloads)
        return serialized_by_dtype, keepalive_buckets

    async def _update_weights_from_checkpoint_engine_async(self) -> bool:
        loaded_batches = 0
        loaded_tensors = 0
        loaded_bytes = 0
        start_time = time.time()

        async for typed_batches in _aligned_checkpoint_engine_batches(
            self.checkpoint_engines
        ):
            serialized_by_dtype, keepalive_buckets = (
                self._serialize_checkpoint_engine_batches(typed_batches)
            )
            for serialized_named_tensors in serialized_by_dtype:
                result = self.update_weights_from_tensor(
                    serialized_named_tensors=serialized_named_tensors,
                    load_format="flattened_bucket",
                    flush_cache=False,
                    weight_version=str(
                        getattr(self, "_checkpoint_engine_weight_version", 0) + 1
                    ),
                )
                if result is not None and not result.get("success", True):
                    error = result.get("error_message") or result.get(
                        "message", "unknown error"
                    )
                    raise RuntimeError(
                        f"SGLang checkpoint-engine refit failed: {error}"
                    )

            loaded_batches += 1
            loaded_tensors += sum(len(batch) for batch in typed_batches)
            loaded_bytes += sum(
                tensor.nbytes for batch in typed_batches for _name, tensor in batch
            )
            del typed_batches, serialized_by_dtype, keepalive_buckets

        self._checkpoint_engine_weight_version = (
            getattr(self, "_checkpoint_engine_weight_version", 0) + 1
        )
        self.invalidate_kv_cache()
        total_time = time.time() - start_time
        print(
            "[SGLang refit] Loaded "
            f"{loaded_tensors} rank-local tensors in {loaded_batches} batches via "
            f"checkpoint engine; bytes={loaded_bytes / 1024**3:.2f}GiB "
            f"total={total_time:.2f}s"
        )
        return True

    def update_weights_from_checkpoint_engine(self) -> bool:
        return asyncio.run(self._update_weights_from_checkpoint_engine_async())

    def finalize_checkpoint_engine(self) -> None:
        for checkpoint_engine in getattr(self, "checkpoint_engines", []):
            checkpoint_engine.finalize()
