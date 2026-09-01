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

from typing import Literal, Optional, Sequence

# Materialization layout for `codec.materialize` / `read_columns` / worker fetch.
Layout = Literal["padded", "jagged"]

# Per-shard packing metadata keys in `KVBatchMeta.extra_info`.
MICRO_BATCH_INDICES = "micro_batch_indices"
MICRO_BATCH_LENGTHS = "micro_batch_lengths"
ELEM_COUNTS_PER_GB = "elem_counts_per_gb"
GLOBAL_FORWARD_PAD_SEQLEN = "global_forward_pad_seqlen"

# Per-prompt-group rollout metrics: a list of one metrics dict per group.
# Unlike the packing keys above, this is not copied to each shard: the train
# pump pops it off the meta before dispatch. sync_rollout_actor.py writes the
# same string with a flat-dict shape, so this constant is not a drop-in there.
ROLLOUT_METRICS = "rollout_metrics"

# Skeleton field names from `shard_meta_for_dp`.
INPUT_IDS = "input_ids"
INPUT_LENGTHS = "input_lengths"
SAMPLE_MASK = "sample_mask"
MASK_SAMPLE = "mask_sample"
TRUNCATED = "truncated"
META_IDX = "meta_idx"

# Token-aligned message-violation fields consumed by SingleController advantages.
INVALID_TOOL_CALL_MASK = "invalid_tool_call_mask"
MALFORMED_THINKING_MASK = "malformed_thinking_mask"

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

# Full-vocabulary MOPD teacher payload columns. Exactly one is written per run,
# selected by ``on_policy_distillation.full.teacher_payload``. Both are per-token
# ``[N, S, D]`` columns, so they are jagged-packed like the other token-aligned
# fields; ``D`` is the teacher hidden size or the vocabulary size.
OPD_FULL_HIDDEN_STATES_FIELD = "teacher_full_hidden_states"
OPD_FULL_LOGITS_FIELD = "teacher_full_logits"
OPD_FULL_FIELDS = (OPD_FULL_HIDDEN_STATES_FIELD, OPD_FULL_LOGITS_FIELD)


# Full known tensor schema for SingleController's long-lived rollout partition.
# The initial rollout put writes the first seven payload fields; later stages add
# student/reference logprobs, advantages, PPO critic columns, and the MOPD teacher
# column. Registering their names once before concurrent producers start avoids
# TransferQueue's lazy field-name registration race.
SC_ROLLOUT_SCHEMA_FIELDS = (
    *DP_TRAIN_FIELDS,
    MASK_SAMPLE,
    TRUNCATED,
    "prompt_ids_for_adv",
    "total_reward",
    "values",
    "returns",
    "teacher_reference_logprobs",
    *OPD_FULL_FIELDS,
    INVALID_TOOL_CALL_MASK,
    MALFORMED_THINKING_MASK,
)

# Subset fetched by logprob / ref-logprob workers.
LP_SEED_FIELDS = (
    "input_ids",
    "input_lengths",
    "token_mask",
    "sample_mask",
)

# Text-only inputs fetched by frozen MOPD teachers for logprob inference.
TEACHER_LP_FIELDS = (INPUT_IDS, INPUT_LENGTHS)

# Kept out of DP_TRAIN_FIELDS: a GRPO run writes neither, and a worker fetching
# a column nobody wrote errors out rather than reading zeros.
PPO_VALUE_FIELDS = (
    "values",
    "returns",
)

DP_VALUE_TRAIN_FIELDS = (
    "input_ids",
    "input_lengths",
    "token_mask",
    "sample_mask",
    *PPO_VALUE_FIELDS,
)

VALUE_SEED_FIELDS = LP_SEED_FIELDS

# Fields requested for KV-scale calibration. Positive include-list:
# calibration only handles seq-dim tensor inputs, so we name them
# explicitly. Train-side deltas (logprobs/advantages/masks) and
# wire-only message-log bulk fields are skipped by virtue of not being
# in this list. ``multi_modal_inputs`` covers VLM extras (pixel values,
# grid metadata, etc.) when present; it's harmlessly absent for
# text-only models so the filter skips it on those.
DP_CALIB_INPUT_FIELDS = (INPUT_IDS, INPUT_LENGTHS, "multi_modal_inputs")

ROUTED_EXPERTS_FIELD = "routed_experts"

# Per-sample 1D scalar fields. The TQ adapter promotes these to ``(N, 1)``
# on write to work around TQ v0.1.9's KVStorageManager schema/data mismatch on
# the Mooncake backend, and squeezes them back to ``(N,)`` on read. This is the
# authoritative user-level schema; no per-row shape metadata is carried.
#
# Fields listed here must be dense ``(N,)`` tensors when written through the
# Mooncake adapter. Dense 1D fields not listed here are rejected on that path so
# a new field cannot silently reintroduce the upstream shape mismatch.
#
# Delete this set and the corresponding adapter transforms when upstream TQ
# fixes 1D field schema extraction.
PROMOTE_1D_FIELDS: frozenset[str] = frozenset(
    {
        INPUT_LENGTHS,
        MASK_SAMPLE,
        "total_reward",
        SAMPLE_MASK,
        TRUNCATED,
    }
)


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


def fields_with_optional_opd_full(
    fields: Sequence[str],
    *,
    field: Optional[str],
) -> list[str]:
    """Return `fields` plus the full-vocabulary MOPD teacher payload column.

    The column is added only when the run configures one. A GRPO run that
    requested a column nobody wrote would error on read, which is why this is
    never folded unconditionally into ``DP_TRAIN_FIELDS``.

    Args:
        fields: Base field list.
        field: Payload column name, or ``None`` when opd_full is off.

    Returns:
        The field list, with the payload column appended when applicable.
    """
    out = list(fields)
    if field is not None and field not in out:
        out.append(field)
    return out
