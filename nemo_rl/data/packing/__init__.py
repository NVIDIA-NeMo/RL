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
from nemo_rl.data.packing.shared_prefix import (
    SharedPrefixFallback,
    SharedPrefixFallbackReason,
    SharedPrefixForestLayout,
    SharedPrefixLayout,
    SharedPrefixPlan,
    SharedPrefixRow,
    build_shared_prefix_layout,
    plan_shared_prefix_bins,
)
from nemo_rl.data.packing.shared_prefix_metadata import (
    SHARED_PREFIX_EXECUTION_SLOT,
    FixedExecutionSlotPlan,
    GroupCoherentShardPlan,
    plan_fixed_execution_slots,
    plan_group_coherent_shards,
)
from nemo_rl.data.packing.shared_prefix_tensors import (
    SharedPrefixContextParallelShard,
    SharedPrefixTensorBin,
    SharedPrefixTensorIndices,
    SharedPrefixTensorPlan,
    build_shared_prefix_tensor_plan,
    build_star_attention_allow_mask,
    get_shared_prefix_context_parallel_indices,
    get_shared_prefix_physical_alignment,
    materialize_shared_prefix_layout,
    materialize_shared_prefix_token_aligned_tensor,
    resolve_shared_prefix_parallel_topology,
    resolve_shared_prefix_physical_padding_multiple,
    shard_shared_prefix_tensor_bin_for_context_parallel,
)

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
