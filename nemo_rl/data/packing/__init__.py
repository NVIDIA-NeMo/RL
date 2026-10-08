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

from nemo_rl.data.packing.algorithms import (
    BalancedGreedyKnapsackPacker,
    ConcatenativePacker,
    FirstFitDecreasingPacker,
    FirstFitShufflePacker,
    GreedyKnapsackPacker,
    ModifiedFirstFitDecreasingPacker,
    PackingAlgorithm,
    SequencePacker,
    get_packer,
)
from nemo_rl.data.packing.metrics import PackingMetrics


__all__ = [
    "BalancedGreedyKnapsackPacker",
    "PackingAlgorithm",
    "SequencePacker",
    "ConcatenativePacker",
    "FirstFitDecreasingPacker",
    "FirstFitShufflePacker",
    "GreedyKnapsackPacker",
    "FixedExecutionSlotPlan",
    "GroupCoherentShardPlan",
    "ModifiedFirstFitDecreasingPacker",
    "get_packer",
    "PackingMetrics",
    "SharedPrefixFallback",
    "SharedPrefixFallbackReason",
    "SharedPrefixContextParallelShard",
    "SharedPrefixForestLayout",
    "SharedPrefixLayout",
    "SharedPrefixPlan",
    "SharedPrefixRow",
    "SharedPrefixTensorBin",
    "SharedPrefixTensorIndices",
    "SharedPrefixTensorPlan",
    "SHARED_PREFIX_EXECUTION_SLOT",
    "build_shared_prefix_layout",
    "build_shared_prefix_tensor_plan",
    "build_star_attention_allow_mask",
    "get_shared_prefix_context_parallel_indices",
    "get_shared_prefix_physical_alignment",
    "materialize_shared_prefix_layout",
    "materialize_shared_prefix_token_aligned_tensor",
    "plan_fixed_execution_slots",
    "plan_group_coherent_shards",
    "plan_shared_prefix_bins",
    "resolve_shared_prefix_parallel_topology",
    "resolve_shared_prefix_physical_padding_multiple",
    "shard_shared_prefix_tensor_bin_for_context_parallel",
]


_SHARED_PREFIX_EXPORTS = {
    "SharedPrefixFallback": "nemo_rl.data.packing.shared_prefix",
    "SharedPrefixFallbackReason": "nemo_rl.data.packing.shared_prefix",
    "SharedPrefixForestLayout": "nemo_rl.data.packing.shared_prefix",
    "SharedPrefixLayout": "nemo_rl.data.packing.shared_prefix",
    "SharedPrefixPlan": "nemo_rl.data.packing.shared_prefix",
    "SharedPrefixRow": "nemo_rl.data.packing.shared_prefix",
    "build_shared_prefix_layout": "nemo_rl.data.packing.shared_prefix",
    "plan_shared_prefix_bins": "nemo_rl.data.packing.shared_prefix",
    "SHARED_PREFIX_EXECUTION_SLOT": "nemo_rl.data.packing.shared_prefix_metadata",
    "FixedExecutionSlotPlan": "nemo_rl.data.packing.shared_prefix_metadata",
    "GroupCoherentShardPlan": "nemo_rl.data.packing.shared_prefix_metadata",
    "plan_fixed_execution_slots": "nemo_rl.data.packing.shared_prefix_metadata",
    "plan_group_coherent_shards": "nemo_rl.data.packing.shared_prefix_metadata",
    "SharedPrefixContextParallelShard": "nemo_rl.data.packing.shared_prefix_tensors",
    "SharedPrefixTensorBin": "nemo_rl.data.packing.shared_prefix_tensors",
    "SharedPrefixTensorIndices": "nemo_rl.data.packing.shared_prefix_tensors",
    "SharedPrefixTensorPlan": "nemo_rl.data.packing.shared_prefix_tensors",
    "build_shared_prefix_tensor_plan": "nemo_rl.data.packing.shared_prefix_tensors",
    "build_star_attention_allow_mask": "nemo_rl.data.packing.shared_prefix_tensors",
    "get_shared_prefix_context_parallel_indices": "nemo_rl.data.packing.shared_prefix_tensors",
    "get_shared_prefix_physical_alignment": "nemo_rl.data.packing.shared_prefix_tensors",
    "materialize_shared_prefix_layout": "nemo_rl.data.packing.shared_prefix_tensors",
    "materialize_shared_prefix_token_aligned_tensor": "nemo_rl.data.packing.shared_prefix_tensors",
    "resolve_shared_prefix_parallel_topology": "nemo_rl.data.packing.shared_prefix_tensors",
    "resolve_shared_prefix_physical_padding_multiple": "nemo_rl.data.packing.shared_prefix_tensors",
    "shard_shared_prefix_tensor_bin_for_context_parallel": "nemo_rl.data.packing.shared_prefix_tensors",
}


def __getattr__(name: str):
    if name not in _SHARED_PREFIX_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(_SHARED_PREFIX_EXPORTS[name]), name)
    globals()[name] = value
    return value
