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
"""Shared constants and type aliases for the data-plane meta contract."""

from typing import Literal

# Materialization layout for `codec.materialize` / `read_columns` / worker fetch.
Layout = Literal["padded", "jagged"]

# Per-shard packing metadata keys in `KVBatchMeta.extra_info`.
META_MICRO_BATCH_INDICES = "micro_batch_indices"
META_MICRO_BATCH_LENGTHS = "micro_batch_lengths"
META_ELEM_COUNTS_PER_GB = "elem_counts_per_gb"

# Preshard-internal column that rides through
# `BatchedDataDict.shard_by_batch_size` so each rank can recover its
# meta-index mapping.
META_IDX = "meta_idx"

# Tensor fields in the train partition. Rollout writes the input
# subset on first put; later stages add prev_logprobs /
# reference_policy_logprobs (workers) and advantages (driver).
DP_TRAIN_FIELDS = (
    "input_ids",
    "input_lengths",
    "generation_logprobs",
    "prev_logprobs",
    "reference_policy_logprobs",
    "advantages",
    "token_mask",
    "sample_mask",
)

# Subset fetched by logprob / ref-logprob workers.
LP_SEED_FIELDS = (
    "input_ids",
    "input_lengths",
    "token_mask",
    "sample_mask",
)
