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

from typing import Any

import torch


OptimizerCpuBufferCache = dict[tuple[Any, Any], torch.Tensor]


def copy_to_reusable_cpu_buffer(
    source: torch.Tensor,
    destination: torch.Tensor | None,
) -> torch.Tensor:
    """Copy a tensor into compatible pageable CPU storage."""
    can_reuse = (
        destination is not None
        and destination.device.type == "cpu"
        and not destination.is_pinned()
        and destination.layout == torch.strided
        and source.layout == torch.strided
        and destination.shape == source.shape
        and destination.dtype == source.dtype
        and destination.stride() == source.stride()
    )
    if not can_reuse:
        if source.layout != torch.strided:
            return source.to("cpu")
        destination = torch.empty_strided(
            source.shape,
            source.stride(),
            dtype=source.dtype,
            device="cpu",
        )
    destination.copy_(source)
    return destination


def move_optimizer_state(
    optimizer_state: Any,
    *,
    device: str,
    reuse_cpu_buffers: bool,
    cpu_buffer_cache: OptimizerCpuBufferCache,
) -> None:
    """Move optimizer tensors while optionally reusing pageable CPU storage."""
    if device not in {"cpu", "cuda"}:
        raise ValueError(
            f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
        )

    active_cache_keys: set[tuple[Any, Any]] = set()
    for state_key, state in optimizer_state.items():
        for field, value in state.items():
            if not torch.is_tensor(value):
                continue
            if device == "cpu" and value.is_cuda:
                if reuse_cpu_buffers:
                    cache_key = (state_key, field)
                    state[field] = copy_to_reusable_cpu_buffer(
                        value,
                        cpu_buffer_cache.get(cache_key),
                    )
                    cpu_buffer_cache[cache_key] = state[field]
                    active_cache_keys.add(cache_key)
                else:
                    state[field] = value.to("cpu")
            elif device == "cuda" and not value.is_cuda:
                state[field] = value.to("cuda")

    if device == "cpu" and reuse_cpu_buffers:
        stale_cache_keys = cpu_buffer_cache.keys() - active_cache_keys
        for cache_key in stale_cache_keys:
            del cpu_buffer_cache[cache_key]
