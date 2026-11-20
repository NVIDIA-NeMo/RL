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
from torch.autograd.graph import saved_tensors_hooks

# Below this size a host round trip costs more than the GPU memory it frees.
MIN_OFFLOAD_BYTES = 1 << 20

_OffloadedTensor = tuple[torch.device, torch.Tensor]
# A weak reference to a saved tensor, its version when it was copied, and the copy.
_CachedCopy = tuple[weakref.ref[torch.Tensor], int, _OffloadedTensor]


def _is_offloadable(tensor: torch.Tensor) -> bool:
    """Return whether ``tensor`` is a plain CUDA activation worth moving to the host.

    Parameters (FSDP's unsharded parameters included) and views of them stay on the
    GPU: FSDP re-gathers them for backward, so a host copy would only add traffic.
    Tensor subclasses such as DTensor keep their own storage layout and are skipped.
    """
    base = tensor._base if tensor._base is not None else tensor
    return (
        type(tensor) is torch.Tensor
        and type(base) is torch.Tensor
        and tensor.is_cuda
        and tensor.layout == torch.strided
        and tensor.numel() * tensor.element_size() >= MIN_OFFLOAD_BYTES
    )


def _pack(tensor: torch.Tensor) -> torch.Tensor | _OffloadedTensor:
    if not _is_offloadable(tensor):
        return tensor
    host = torch.empty_like(tensor, device="cpu", pin_memory=True)
    # Ordered on the current stream, so the source cannot be reused before the copy ends.
    host.copy_(tensor, non_blocking=True)
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

    Use it with activation checkpointing: autograd then saves only each checkpointed
    block's input outside the checkpoint frames, and recomputation in backward runs
    outside this context, so the host holds one hidden-state tensor per block.

    When several operations save the same tensor, for example a residual stream that
    feeds both a norm and an add, the host keeps one copy for all of them.

    Each host copy lives as long as the autograd graph that saved it, so one instance
    can be reused for every microbatch.

    Example:
        >>> offload = ActivationCpuOffload()
        >>> with offload:
        ...     logits = model(input_ids)
        >>> loss_fn(logits).backward()
    """

    def __init__(self) -> None:
        """Register the pack and unpack hooks."""
        # id(tensor) -> cached copy, for the current forward only. The weak reference
        # keeps a recycled id from matching.
        self._copies: dict[int, _CachedCopy] = {}
        super().__init__(self._pack_once, _unpack)

    def __enter__(self) -> None:
        """Start a forward pass with no host copies to reuse."""
        self._copies.clear()
        super().__enter__()

    def __exit__(self, *args: Any) -> None:
        """Drop the references to this forward's host copies; autograd keeps its own."""
        super().__exit__(*args)
        self._copies.clear()

    def _pack_once(self, tensor: torch.Tensor) -> torch.Tensor | _OffloadedTensor:
        """Pack ``tensor``, reusing its host copy if it was saved before and not modified since."""
        cached = self._copies.get(id(tensor))
        if cached is not None:
            tensor_ref, version, packed = cached
            if tensor_ref() is tensor and version == tensor._version:
                return packed
        packed = _pack(tensor)
        if isinstance(packed, tuple):
            self._copies[id(tensor)] = (weakref.ref(tensor), tensor._version, packed)
        return packed
