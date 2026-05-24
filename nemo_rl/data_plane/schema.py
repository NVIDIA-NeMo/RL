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
MICRO_BATCH_INDICES = "micro_batch_indices"
MICRO_BATCH_LENGTHS = "micro_batch_lengths"
ELEM_COUNTS_PER_GB = "elem_counts_per_gb"
GLOBAL_FORWARD_PAD_SEQLEN = "global_forward_pad_seqlen"

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

# Subset fetched by logprob / ref-logprob workers.
LP_SEED_FIELDS = (
    "input_ids",
    "input_lengths",
    "token_mask",
    "sample_mask",
)

# Fields requested for KV-scale calibration. Positive include-list:
# calibration only handles seq-dim tensor inputs, so we name them
# explicitly. Train-side deltas (logprobs/advantages/masks) and
# wire-only message-log bulk fields are skipped by virtue of not being
# in this list. ``multi_modal_inputs`` covers VLM extras (pixel values,
# grid metadata, etc.) when present; it's harmlessly absent for
# text-only models so the filter skips it on those.
DP_CALIB_INPUT_FIELDS = (INPUT_IDS, INPUT_LENGTHS, "multi_modal_inputs")
