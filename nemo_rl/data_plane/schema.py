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

from typing import Literal, Sequence

# Materialization layout for `codec.materialize` / `read_columns` / worker fetch.
Layout = Literal["padded", "jagged"]

# Per-shard packing metadata keys in `KVBatchMeta.extra_info`.
MICRO_BATCH_INDICES = "micro_batch_indices"
MICRO_BATCH_LENGTHS = "micro_batch_lengths"
ELEM_COUNTS_PER_GB = "elem_counts_per_gb"

# Skeleton field names from `shard_meta_for_dp`.
INPUT_IDS = "input_ids"
INPUT_LENGTHS = "input_lengths"
SAMPLE_MASK = "sample_mask"
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

# Subset written by the rollout actor's first `kv_batch_put`.
ROLLOUT_SEED_FIELDS = (
    "input_ids",
    "input_lengths",
    "generation_logprobs",
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

ROUTED_EXPERTS_FIELD = "routed_experts"


def fields_with_optional_routed_experts(
    fields: Sequence[str],
    *,
    enabled: bool,
) -> list[str]:
    """Return `fields` plus routed experts when router replay is enabled."""
    out = list(fields)
    if enabled and ROUTED_EXPERTS_FIELD not in out:
        out.append(ROUTED_EXPERTS_FIELD)
    return out
