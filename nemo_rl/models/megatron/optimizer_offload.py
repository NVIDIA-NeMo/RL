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
"""Move a Megatron optimizer's state between GPU and host memory.

Colocated workers park the optimizer state on the host while the inference
engine uses the GPU, and bring it back before the optimizer step.
"""

import warnings
import weakref
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import torch
from megatron.core.optimizer import ChainedOptimizer, MegatronOptimizer

_CHUNK_BYTES = 1 << 30
_ALIGN_BYTES = 512


def _iter_state_items(
    optimizer: MegatronOptimizer,
) -> Iterator[tuple[dict[str, Any], str, Any]]:
    if isinstance(optimizer, ChainedOptimizer):
        optimizer_state = optimizer.state
    else:
        optimizer_state = optimizer._get_state()
    for _, state in optimizer_state.items():
        for key, value in state.items():
            yield state, key, value


def _storage_use_count(tensor: torch.Tensor) -> int:
    """Reference count of the tensor's StorageImpl; every tensor sharing it adds one."""
    return torch._C._storage_Use_Count(tensor.untyped_storage()._cdata)


def _owns_its_storage(tensor: torch.Tensor, *, sole_owner_count: int) -> bool:
    """Whether the tensor is a plain dense tensor that alone holds its whole storage."""
    return (
        type(tensor) is torch.Tensor
        and tensor.storage_offset() == 0
        and tensor.is_contiguous()
        and tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
        and _storage_use_count(tensor) == sole_owner_count
    )


def _cudart() -> Any:
    cudart = torch.cuda.cudart()
    assert cudart is not None
    return cudart


def _storage_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """A flat uint8 tensor over the whole storage of ``tensor``."""
    return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
        tensor.untyped_storage()
    )


class _HostChunk:
    """A host buffer that stages one or more state tensors."""

    def __init__(self, nbytes: int, *, persistent: bool):
        self.nbytes = nbytes
        self.persistent = persistent
        self.buffer: Optional[torch.Tensor] = None
        self._registered_ptr: Optional[int] = None
        if persistent:
            self.buffer = torch.empty(nbytes, dtype=torch.uint8)
            self._pin()

    def _pin(self) -> None:
        assert self.buffer is not None
        ptr = self.buffer.data_ptr()
        try:
            torch.cuda.check_error(_cudart().cudaHostRegister(ptr, self.nbytes, 0))
        except torch.cuda.CudaError as error:
            warnings.warn(
                f"cudaHostRegister of a {self.nbytes / 2**30:.2f} GiB optimizer "
                f"offload buffer failed ({error}); it stays pageable, so offloads "
                "through it are slower. Lower "
                "megatron_cfg.optimizer_offload_pinned_buffer_gb to avoid pinning.",
                stacklevel=2,
            )
            return
        self._registered_ptr = ptr

    def acquire(self) -> torch.Tensor:
        if self.buffer is None:
            self.buffer = torch.empty(self.nbytes, dtype=torch.uint8)
        return self.buffer

    def release_transient(self) -> None:
        if not self.persistent:
            self.buffer = None

    def __del__(self) -> None:
        # Unregister before the buffer's memory goes back to the host allocator.
        if self._registered_ptr is not None:
            _cudart().cudaHostUnregister(self._registered_ptr)
            self._registered_ptr = None


@dataclass
class _Slot:
    tensor_ref: weakref.ref
    nbytes: int
    chunk: _HostChunk
    offset: int


class OptimizerStateOffloader:
    """Offloads optimizer state to host memory and reloads it, keeping tensor identity.

    A state tensor that alone holds its storage stays in the optimizer state: offload
    copies its bytes into a host buffer and frees the CUDA storage with
    ``untyped_storage().resize_(0)``; reload re-allocates the storage and copies
    the bytes back. All copies are issued non-blocking and synchronized once.
    Host buffers within ``pinned_buffer_gb`` are chunks of about 1 GiB,
    allocated once, page-locked with ``cudaHostRegister`` and reused by every
    offload. Tensors past the cap get exact-size pageable buffers that each
    offload allocates and the reload frees.

    Freeing the storage would break any other tensor on it, so a state tensor
    whose storage another tensor also uses (a view or alias anywhere in the
    process, detected through the storage's reference count), a tensor listed
    twice in the state, or a tensor subclass is instead replaced in the state by
    a CPU copy on offload and by a CUDA copy on reload. The plan is rebuilt when
    the set of resident state tensors changes.

    Args:
        pinned_buffer_gb: Cap, in GiB, on the host memory kept pinned across
            offloads, or None for no cap.
    """

    def __init__(self, *, pinned_buffer_gb: Optional[float]):
        if pinned_buffer_gb is not None and pinned_buffer_gb < 0:
            raise ValueError(
                "megatron_cfg.optimizer_offload_pinned_buffer_gb must be "
                f"non-negative or null, got {pinned_buffer_gb}"
            )
        self._pinned_buffer_bytes = (
            None if pinned_buffer_gb is None else int(pinned_buffer_gb * 2**30)
        )
        self._slots: list[_Slot] = []
        self._offloaded = False
        # The count for a tensor that is the only user of its storage.
        self._sole_owner_count = _storage_use_count(torch.empty(1))

    def _plan_matches(self, tensors: list[torch.Tensor]) -> bool:
        return len(tensors) == len(self._slots) and all(
            slot.tensor_ref() is tensor
            and slot.nbytes == tensor.untyped_storage().nbytes()
            for slot, tensor in zip(self._slots, tensors)
        )

    def _build_plan(self, tensors: list[torch.Tensor]) -> None:
        self._slots = []
        cap = self._pinned_buffer_bytes
        # Per tensor: (index of its pinned chunk, offset), or None past the cap.
        placements: list[Optional[tuple[int, int]]] = []
        chunk_sizes: list[int] = []
        pinned_bytes = 0
        for tensor in tensors:
            nbytes = tensor.untyped_storage().nbytes()
            used = chunk_sizes[-1] if chunk_sizes else 0
            offset = -(-used // _ALIGN_BYTES) * _ALIGN_BYTES
            new_chunk = not chunk_sizes or offset + nbytes > _CHUNK_BYTES
            if new_chunk:
                offset = 0
            added = offset + nbytes - (0 if new_chunk else used)
            if cap is not None and pinned_bytes + added > cap:
                placements.append(None)
                continue
            if new_chunk:
                chunk_sizes.append(0)
            chunk_sizes[-1] = offset + nbytes
            pinned_bytes += added
            placements.append((len(chunk_sizes) - 1, offset))

        chunks = [_HostChunk(size, persistent=True) for size in chunk_sizes]
        for tensor, placement in zip(tensors, placements):
            nbytes = tensor.untyped_storage().nbytes()
            if placement is None:
                # Exact-size pageable buffer, like a plain .to("cpu").
                slot = _Slot(
                    weakref.ref(tensor), nbytes, _HostChunk(nbytes, persistent=False), 0
                )
            else:
                chunk_index, offset = placement
                slot = _Slot(weakref.ref(tensor), nbytes, chunks[chunk_index], offset)
            self._slots.append(slot)

    @torch.no_grad()
    def offload(self, optimizer: MegatronOptimizer) -> None:
        """Move the CUDA-resident optimizer state to host memory."""
        in_place: list[torch.Tensor] = []
        replaced: list[tuple[dict[str, Any], str, torch.Tensor]] = []
        storage_users: dict[int, int] = {}
        resident = []
        for state, key, value in _iter_state_items(optimizer):
            # An offloaded tensor keeps is_cuda with an empty storage.
            if torch.is_tensor(value) and value.is_cuda:
                if value.untyped_storage().nbytes() > 0:
                    resident.append((state, key, value))
                    ptr = value.untyped_storage().data_ptr()
                    storage_users[ptr] = storage_users.get(ptr, 0) + 1
        for state, key, value in resident:
            if (
                not self._offloaded
                and _owns_its_storage(value, sole_owner_count=self._sole_owner_count)
                and storage_users[value.untyped_storage().data_ptr()] == 1
            ):
                in_place.append(value)
            else:
                replaced.append((state, key, value))

        if in_place:
            if not self._plan_matches(in_place):
                self._build_plan(in_place)
            for slot, tensor in zip(self._slots, in_place):
                host = slot.chunk.acquire()[slot.offset : slot.offset + slot.nbytes]
                host.copy_(_storage_bytes(tensor), non_blocking=True)
        for state, key, value in replaced:
            state[key] = value.to("cpu")
        torch.cuda.synchronize()
        for tensor in in_place:
            tensor.untyped_storage().resize_(0)
        if in_place:
            self._offloaded = True

    @torch.no_grad()
    def reload(self, optimizer: MegatronOptimizer) -> None:
        """Move the optimizer state back to the GPU."""
        if self._offloaded:
            for slot in self._slots:
                tensor = slot.tensor_ref()
                if tensor is None:
                    continue
                tensor.untyped_storage().resize_(slot.nbytes)
                host = slot.chunk.acquire()[slot.offset : slot.offset + slot.nbytes]
                _storage_bytes(tensor).copy_(host, non_blocking=True)
        for state, key, value in _iter_state_items(optimizer):
            if torch.is_tensor(value) and not value.is_cuda:
                state[key] = value.to("cuda")
        torch.cuda.synchronize()
        if self._offloaded:
            for slot in self._slots:
                slot.chunk.release_transient()
            self._offloaded = False
