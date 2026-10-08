# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Compatibility exports for :mod:`megatron.rl.shared_prefix_tensors`.

The implementation is owned by Megatron. Imports stay lazy so ordinary dense
NeMo RL backends do not require the optional Megatron installation.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from megatron.rl.shared_prefix_tensors import (
        SharedPrefixTensorIndices,
        SharedPrefixTensorBin,
        SharedPrefixTensorPlan,
        SharedPrefixContextParallelShard,
        resolve_shared_prefix_parallel_topology,
        get_shared_prefix_physical_alignment,
        resolve_shared_prefix_physical_padding_multiple,
        get_shared_prefix_context_parallel_indices,
        shard_shared_prefix_tensor_bin_for_context_parallel,
        build_star_attention_allow_mask,
        materialize_shared_prefix_layout,
        materialize_shared_prefix_token_aligned_tensor,
        build_shared_prefix_tensor_plan,
    )

__all__ = [
    "SharedPrefixTensorIndices",
    "SharedPrefixTensorBin",
    "SharedPrefixTensorPlan",
    "SharedPrefixContextParallelShard",
    "resolve_shared_prefix_parallel_topology",
    "get_shared_prefix_physical_alignment",
    "resolve_shared_prefix_physical_padding_multiple",
    "get_shared_prefix_context_parallel_indices",
    "shard_shared_prefix_tensor_bin_for_context_parallel",
    "build_star_attention_allow_mask",
    "materialize_shared_prefix_layout",
    "materialize_shared_prefix_token_aligned_tensor",
    "build_shared_prefix_tensor_plan",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module("megatron.rl.shared_prefix_tensors"), name)
    globals()[name] = value
    return value
