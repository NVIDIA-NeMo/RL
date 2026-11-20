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
"""Keep the activations autograd saves during a forward pass in pinned host memory."""

import weakref
from typing import Any

import torch
from torch import nn
from torch.autograd.graph import saved_tensors_hooks
from torch.distributed.tensor import DTensor

# Below this size a host round trip costs more than the GPU memory it frees.
MIN_OFFLOAD_BYTES = 1 << 20

_OffloadedTensor = tuple[torch.device, torch.Tensor]
# A weak reference to a saved tensor, its version when it was copied, and the copy.
_CachedCopy = tuple[weakref.ref[torch.Tensor], int, _OffloadedTensor]


def _storage_key(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage()._cdata


def _is_offloadable(tensor: torch.Tensor) -> bool:
    """Return whether ``tensor`` is a plain CUDA tensor worth moving to the host.

    Parameters and views of them, and tensor subclasses such as DTensor, are skipped.
    """
    base = tensor._base if tensor._base is not None else tensor
    return (
        type(tensor) is torch.Tensor
        and type(base) is torch.Tensor
        and tensor.is_cuda
        and tensor.layout == torch.strided
        and tensor.numel() * tensor.element_size() >= MIN_OFFLOAD_BYTES
    )


def _copy_to_host(tensor: torch.Tensor) -> _OffloadedTensor:
    host = torch.empty_like(tensor, device="cpu", pin_memory=True)
    stream = torch.cuda.current_stream(tensor.device)
    host.copy_(tensor, non_blocking=True)
    # Autograd no longer holds the source, so it can be freed while the copy is queued;
    # keep the caching allocator from reusing it on another stream before then.
    tensor.record_stream(stream)
    return tensor.device, host


def _unpack(packed: torch.Tensor | _OffloadedTensor) -> torch.Tensor:
    if isinstance(packed, torch.Tensor):
        return packed
    device, host = packed
    return host.to(device, non_blocking=True)


class ActivationCpuOffload(saved_tensors_hooks):
    """Context manager that moves the activations saved for backward to pinned host memory.

    Inside the context, every plain CUDA tensor of at least ``MIN_OFFLOAD_BYTES`` that
    autograd saves is copied to pinned host memory, and the GPU copy is released.
    Backward copies it back when it needs it. Both copies run on the current stream.

    Tensors that share storage with a parameter of ``model`` stay on the GPU: FSDP
    re-gathers parameters for backward, so a host copy would only add traffic. This
    covers the unsharded parameters FSDP registers during forward and the plain local
    tensors of DTensor parameters that compiled code or ``detach()`` hands to autograd.

    Use it with activation checkpointing: autograd then saves only each checkpointed
    block's input outside the checkpoint frames, and recomputation in backward runs
    outside this context, so the host holds one hidden-state tensor per block.

    When several operations save the same tensor, for example a residual stream that
    feeds both a norm and an add, the host keeps one copy for all of them.

    Each host copy lives as long as the autograd graph that saved it, so one instance
    can be reused for every microbatch.

    Example:
        >>> offload = ActivationCpuOffload(model)
        >>> with offload:
        ...     logits = model(input_ids)
        >>> loss_fn(logits).backward()
        >>> offload.empty_host_cache()
    """

    def __init__(self, model: nn.Module) -> None:
        """Register the pack and unpack hooks for the forward passes of ``model``.

        Args:
            model: Model whose parameters must not be offloaded.
        """
        # FSDP swaps sharded and unsharded parameters in place under these names, so
        # they are looked up again on every pack.
        self._parameter_slots = [
            (module, name)
            for module in model.modules()
            for name, _ in module.named_parameters(recurse=False)
        ]
        # id(parameter) -> (weak reference to it, its storage key).
        self._parameter_storage_keys: dict[
            int, tuple[weakref.ref[torch.Tensor], int]
        ] = {}
        # id(tensor) -> cached copy, for the current forward only. The weak reference
        # keeps a recycled id from matching.
        self._copies: dict[int, _CachedCopy] = {}
        super().__init__(self._pack, _unpack)

    def __enter__(self) -> None:
        """Start a forward pass with no host copies to reuse."""
        self._copies.clear()
        super().__enter__()

    def __exit__(self, *args: Any) -> None:
        """Drop the references to this forward's host copies; autograd keeps its own."""
        super().__exit__(*args)
        self._copies.clear()

    def empty_host_cache(self) -> None:
        """Return the process's cached, unused pinned host memory to the system.

        The pinned-memory allocator keeps freed blocks for reuse, rounded up to a power
        of two, so without this the host copies of the last training step stay pinned
        through generation. The next step pins its memory again.
        """
        # A cached block is released only once the copies that used it have finished.
        torch.cuda.synchronize()
        torch.accelerator.empty_host_cache()

    def _parameter_storages(self) -> set[int]:
        keys = set()
        for module, name in self._parameter_slots:
            param = getattr(module, name)
            if param is None:
                continue
            cached = self._parameter_storage_keys.get(id(param))
            if cached is None or cached[0]() is not param:
                # Without grad mode to_local() returns the local tensor itself.
                with torch.no_grad():
                    local = param.to_local() if isinstance(param, DTensor) else param
                cached = (weakref.ref(param), _storage_key(local))
                self._parameter_storage_keys[id(param)] = cached
            keys.add(cached[1])
        return keys

    def _pack(self, tensor: torch.Tensor) -> torch.Tensor | _OffloadedTensor:
        """Copy ``tensor`` to the host once, unless it stays on the GPU."""
        cached = self._copies.get(id(tensor))
        if cached is not None:
            tensor_ref, version, packed = cached
            if tensor_ref() is tensor and version == tensor._version:
                return packed
        if (
            not _is_offloadable(tensor)
            or _storage_key(tensor) in self._parameter_storages()
        ):
            return tensor
        packed = _copy_to_host(tensor)
        self._copies[id(tensor)] = (weakref.ref(tensor), tensor._version, packed)
        return packed
