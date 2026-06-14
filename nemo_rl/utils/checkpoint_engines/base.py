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
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Generator

import torch

logger = logging.getLogger(__name__)


@dataclass
class TensorMeta:
    """Metadata for one tensor chunk inside a checkpoint-engine transfer bucket."""

    name: str
    shape: torch.Size
    dtype: torch.dtype
    chunk_offset: int
    chunk_size: int
    offset: int | None
    children: list["TensorMeta"] | None = None

    @property
    def nbytes(self) -> int:
        return self.shape.numel() * self.dtype.itemsize


class CheckpointEngine(ABC):
    """Transfer model weights from policy workers to generation workers.

    Backend plugins should subclass this interface and either register with
    ``CheckpointEngineRegistry.register("backend_name")`` or be referenced by
    class path in config, for example ``"my_pkg.my_backend:MyEngine"``.
    """

    cleanup_after_load: bool = True

    @abstractmethod
    def prepare(self) -> Any:
        """Allocate/register transfer buffers and return worker metadata."""
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
        """Initialize the checkpoint-engine topology for a policy worker."""
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
        """Initialize the checkpoint-engine topology for a generation worker."""
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> None:
        """Release per-refit communication state."""
        raise NotImplementedError

    @abstractmethod
    async def send_weights(
        self, weights: Generator[tuple[str, torch.Tensor], None, None]
    ) -> None:
        """Send model weights to the generation side."""
        raise NotImplementedError

    @abstractmethod
    def receive_weight_batches(
        self,
    ) -> AsyncGenerator[list[tuple[str, torch.Tensor]], None]:
        """Receive model weights in batches that share a transfer buffer.

        Backends that yield CUDA tensors backed by reusable transfer buffers may
        return a list subclass with ``record_cuda_load_complete()``. vLLM calls
        that hook after queuing its load operation so the backend can fence
        buffer reuse with a CUDA event instead of forcing a device-wide sync.
        """
        raise NotImplementedError


class CheckpointEngineRegistry:
    """Registry for refit checkpoint-engine transfer backends."""

    _registry: dict[str, type[CheckpointEngine]] = {}
    _builtin_modules = {
        "nixl": "nemo_rl.utils.checkpoint_engines.nixl",
    }

    @classmethod
    def register(cls, backend: str):
        """Register a checkpoint-engine backend class."""

        def wrapper(engine_cls: type[CheckpointEngine]):
            cls._validate_engine_class(backend, engine_cls)
            cls._registry[backend] = engine_cls
            return engine_cls

        return wrapper

    @classmethod
    def get(cls, backend: str) -> type[CheckpointEngine]:
        """Return a registered backend or resolve a plugin class path."""
        if backend in cls._registry:
            return cls._registry[backend]

        cls._import_builtin_backend(backend)
        if backend in cls._registry:
            return cls._registry[backend]

        if _looks_like_class_path(backend):
            cls._registry[backend] = cls._load_backend_class(backend)
            return cls._registry[backend]

        available_backends = ", ".join(cls.available_backends())
        raise ValueError(
            f"Checkpoint engine backend {backend!r} is not registered. "
            f"Available backends: {available_backends}"
        )

    @classmethod
    def new(cls, backend: str, *args: Any, **kwargs: Any) -> CheckpointEngine:
        """Instantiate a checkpoint-engine backend."""
        return cls.get(backend)(*args, **kwargs)

    @classmethod
    def available_backends(cls) -> list[str]:
        """Return registered and built-in backend names."""
        return sorted(set(cls._registry) | set(cls._builtin_modules))

    @classmethod
    def _import_builtin_backend(cls, backend: str) -> None:
        module_name = cls._builtin_modules.get(backend)
        if module_name is not None:
            importlib.import_module(module_name)

    @classmethod
    def _load_backend_class(cls, backend: str) -> type[CheckpointEngine]:
        module_name, class_name = _split_class_path(backend)
        module = importlib.import_module(module_name)
        try:
            engine_cls = getattr(module, class_name)
        except AttributeError as exc:
            raise ValueError(
                f"Checkpoint engine backend class {class_name!r} was not found "
                f"in module {module_name!r}."
            ) from exc
        cls._validate_engine_class(backend, engine_cls)
        return engine_cls

    @staticmethod
    def _validate_engine_class(backend: str, engine_cls: Any) -> None:
        if not isinstance(engine_cls, type) or not issubclass(
            engine_cls, CheckpointEngine
        ):
            raise TypeError(
                f"Checkpoint engine backend {backend!r} must resolve to a "
                f"{CheckpointEngine.__name__} subclass."
            )


def create_checkpoint_engine(
    backend: str,
    *,
    bucket_size_bytes: int,
    engine_kwargs: dict[str, Any],
) -> CheckpointEngine:
    """Create a checkpoint engine from backend-specific configuration."""
    return CheckpointEngineRegistry.new(
        backend,
        bucket_size=bucket_size_bytes,
        **engine_kwargs,
    )


def _looks_like_class_path(backend: str) -> bool:
    return ":" in backend or "." in backend


def _split_class_path(backend: str) -> tuple[str, str]:
    if ":" in backend:
        module_name, class_name = backend.split(":", 1)
    else:
        module_name, class_name = backend.rsplit(".", 1)
    if not module_name or not class_name:
        raise ValueError(
            "Checkpoint engine plugin backends must be formatted as "
            "'module.submodule:ClassName' or 'module.submodule.ClassName'."
        )
    return module_name, class_name


async def split_weight_chunks(
    weights: Generator[tuple[str, torch.Tensor], None, None],
    bucket_size: int,
) -> AsyncGenerator[tuple[TensorMeta, torch.Tensor], None]:
    """Split tensors into byte chunks no larger than bucket_size."""
    for name, weight in weights:
        buffer = weight.contiguous().view(-1).view(torch.uint8)
        chunk_offset = 0
        while chunk_offset < weight.nbytes:
            chunk_size = min(bucket_size, weight.nbytes - chunk_offset)
            yield (
                TensorMeta(
                    name=name,
                    shape=weight.shape,
                    dtype=weight.dtype,
                    chunk_offset=chunk_offset,
                    chunk_size=chunk_size,
                    offset=None,
                ),
                buffer[chunk_offset : chunk_offset + chunk_size],
            )
            chunk_offset += chunk_size


async def merge_weight_chunk_batches(
    chunk_batches: AsyncGenerator[list[tuple[TensorMeta, torch.Tensor]], None],
    bucket_size: int,
) -> AsyncGenerator[list[tuple[str, torch.Tensor]], None]:
    """Merge received tensor chunks while preserving transfer-bucket boundaries."""
    merge_name: str | None = None
    merge_weight: torch.Tensor | None = None
    merge_offset = 0
    chunk_batch_count = 0
    chunk_count = 0
    weight_count = 0
    logical_bytes = 0
    alloc_time = 0.0
    copy_time = 0.0
    view_time = 0.0

    async for chunk_batch in chunk_batches:
        chunk_batch_count += 1
        chunk_count += len(chunk_batch)
        weight_batch: list[tuple[str, torch.Tensor]] = []
        for tensor_meta, chunk in chunk_batch:
            if chunk.dtype != torch.uint8:
                raise TypeError(
                    f"Checkpoint-engine chunks must be uint8, got {chunk.dtype}"
                )

            if tensor_meta.children:
                for child_meta in tensor_meta.children:
                    if child_meta.offset is None:
                        raise RuntimeError(
                            f"Missing packed offset for {child_meta.name}."
                        )
                    weight = chunk[
                        child_meta.offset : child_meta.offset + child_meta.chunk_size
                    ]
                    view_start = time.perf_counter()
                    weight_batch.append(
                        (
                            child_meta.name,
                            weight.view(child_meta.dtype).view(child_meta.shape),
                        )
                    )
                    view_time += time.perf_counter() - view_start
                    weight_count += 1
                    logical_bytes += child_meta.nbytes
                continue

            if tensor_meta.nbytes <= bucket_size:
                if merge_weight is not None:
                    raise RuntimeError(f"Unexpected open merge for {merge_name}.")
                view_start = time.perf_counter()
                weight_batch.append(
                    (
                        tensor_meta.name,
                        chunk.view(tensor_meta.dtype).view(tensor_meta.shape),
                    )
                )
                view_time += time.perf_counter() - view_start
                weight_count += 1
                logical_bytes += tensor_meta.nbytes
                continue

            if merge_weight is None:
                if tensor_meta.chunk_offset != 0:
                    raise RuntimeError(
                        f"First chunk for {tensor_meta.name} starts at "
                        f"{tensor_meta.chunk_offset}, expected 0."
                    )
                merge_name = tensor_meta.name
                alloc_start = time.perf_counter()
                merge_weight = torch.empty(
                    tensor_meta.shape,
                    dtype=tensor_meta.dtype,
                    device=chunk.device,
                )
                alloc_time += time.perf_counter() - alloc_start
                merge_offset = 0

            if tensor_meta.name != merge_name:
                raise RuntimeError(
                    f"Expected chunk for {merge_name}, got {tensor_meta.name}."
                )
            if merge_offset != tensor_meta.chunk_offset:
                raise RuntimeError(
                    f"Expected chunk offset {merge_offset}, got {tensor_meta.chunk_offset}."
                )

            copy_start = time.perf_counter()
            merge_weight.view(-1).view(torch.uint8)[
                tensor_meta.chunk_offset : tensor_meta.chunk_offset
                + tensor_meta.chunk_size
            ] = chunk
            copy_time += time.perf_counter() - copy_start
            merge_offset += tensor_meta.chunk_size

            if tensor_meta.chunk_offset + tensor_meta.chunk_size == tensor_meta.nbytes:
                if merge_name is None:
                    raise RuntimeError("Missing tensor name for completed merge.")
                weight_batch.append((merge_name, merge_weight))
                weight_count += 1
                logical_bytes += tensor_meta.nbytes
                merge_name = None
                merge_weight = None
                merge_offset = 0

        if weight_batch:
            yield weight_batch

    if merge_weight is not None:
        raise RuntimeError(f"Unfinished tensor merge for {merge_name}.")
    if chunk_batch_count > 0:
        logger.info(
            "Checkpoint-engine merge completed: batches=%d chunks=%d "
            "weights=%d bytes=%.2fGiB alloc=%.2fs copy=%.2fs view=%.2fs",
            chunk_batch_count,
            chunk_count,
            weight_count,
            logical_bytes / (1024 * 1024 * 1024),
            alloc_time,
            copy_time,
            view_time,
        )
