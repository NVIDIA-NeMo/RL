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
"""Driver-side balanced packing + per-rank fan-out helpers.

Extracted from the ``grpo_sync`` inline block (commit a085559c) so the same
two operations can be reused across both sync and async data-plane trainers.

These helpers operate on full ``BatchedDataDict``s and rely on
``shard_by_batch_size``'s ``bin_count_multiple=DP_world`` behavior to keep
per-rank microbatch counts uniform — without that, sequence packing /
dynamic batching produce variable per-rank bin counts and Megatron
deadlocks at the first cross-DP collective.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
from tensordict import TensorDict

from nemo_rl.data_plane.interfaces import DataPlaneClient, KVBatchMeta
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

# Tensor fields the ``train`` partition schema declares. The rollout
# actor's first ``kv_batch_put`` writes the input-side subset
# (input_ids, input_lengths, generation_logprobs, token_mask,
# sample_mask) plus any multimodal extras present in the rollout
# output; later stages add ``prev_logprobs`` /
# ``reference_policy_logprobs`` (worker write-back) and ``advantages``
# (driver delta-write). Consumers (``train_presharded`` workers) fetch
# the union via ``select_fields``.
DP_SEED_FIELDS = (
    "input_ids",
    "input_lengths",
    "generation_logprobs",
    "prev_logprobs",
    "reference_policy_logprobs",
    "advantages",
    "token_mask",
    "sample_mask",
)

# Subset used by ``get_logprobs_from_meta`` / ``get_reference_policy_logprobs_from_meta``
# — logprob workers only need the input + masks, not the full train fields.
LP_SEED_FIELDS = (
    "input_ids",
    "input_lengths",
    "token_mask",
    "sample_mask",
)


def select_meta_indices(
    meta: KVBatchMeta,
    indices: Sequence[int],
) -> KVBatchMeta:
    """Return a new KVBatchMeta with keys/sequence_lengths sub-selected.

    Pure metadata operation — no I/O. Use to filter a meta after a
    driver-side selection (e.g. dynamic_sampling's non-zero-std mask).
    The dropped uids' TQ payload is the caller's responsibility to
    ``kv_clear``; this helper only updates the meta.
    """
    keys = [meta.keys[i] for i in indices]
    seq_lens: Optional[list[int]] = None
    if meta.sequence_lengths is not None:
        seq_lens = [meta.sequence_lengths[i] for i in indices]
    return KVBatchMeta(
        partition_id=meta.partition_id,
        task_name=meta.task_name,
        keys=keys,
        fields=meta.fields,
        sequence_lengths=seq_lens,
        extra_info=dict(meta.extra_info or {}),
    )


def concat_metas(metas: Sequence[KVBatchMeta]) -> KVBatchMeta:
    """Concatenate multiple metas into one (same partition_id required).

    Use after dynamic_sampling cache merge: each iteration produces its
    own meta of survivors; concatenating them gives the meta for the
    fully-accumulated training batch. Pure metadata; no I/O.
    """
    if not metas:
        raise ValueError("concat_metas: empty input")
    pid = metas[0].partition_id
    if any(m.partition_id != pid for m in metas):
        raise ValueError("concat_metas: partition_ids must match")
    keys: list[str] = []
    seq_lens: Optional[list[int]] = []
    for m in metas:
        keys.extend(m.keys)
        if m.sequence_lengths is None:
            seq_lens = None
            break
        seq_lens.extend(m.sequence_lengths)
    if seq_lens is None:
        seq_lens = None
    return KVBatchMeta(
        partition_id=pid,
        task_name=metas[0].task_name,
        keys=keys,
        fields=metas[0].fields,
        sequence_lengths=seq_lens,
        extra_info=dict(metas[0].extra_info or {}),
    )


def slice_meta(meta: KVBatchMeta, start: int, stop: int) -> KVBatchMeta:
    """Slice a meta's keys/sequence_lengths to ``[start:stop)``.

    Use to trim an over-full cache to ``train_prompts_size`` after
    dynamic_sampling overflow. Caller is responsible for ``kv_clear``ing
    the discarded keys; this helper only updates the meta.
    """
    seq_lens: Optional[list[int]] = None
    if meta.sequence_lengths is not None:
        seq_lens = list(meta.sequence_lengths[start:stop])
    return KVBatchMeta(
        partition_id=meta.partition_id,
        task_name=meta.task_name,
        keys=list(meta.keys[start:stop]),
        fields=meta.fields,
        sequence_lengths=seq_lens,
        extra_info=dict(meta.extra_info or {}),
    )


def shard_meta_for_dp(
    meta: KVBatchMeta,
    *,
    dp_world: int,
    batch_size: Optional[int] = None,
    sequence_packing_args: Optional[dict[str, Any]] = None,
    dynamic_batching_args: Optional[dict[str, Any]] = None,
) -> tuple[list[KVBatchMeta], Optional[list[int]]]:
    """Pure key-list split: assign ``meta.keys`` to ``dp_world`` ranks.

    Mirrors verl's ``BatchData.chunk(KVBatchMeta)`` (verl/protocol.py:1271-1289)
    with NeMo-RL's seq-len-aware packing on top. **No I/O, no key minting.**
    Returned per-rank metas reference subsets of the input ``meta.keys``
    under the same ``partition_id``; workers fetch their slice via the
    existing ``*_presharded`` flow.

    Use this for every dispatch *after* rollout (logprob, ref-logprob, train).
    The rollout actor's first write is a flat ``kv_batch_put`` (see
    :func:`nemo_rl.algorithms.sync_utils.rollout_to_tq`) — no fan-out.

    Per-rank packing metadata (``micro_batch_indices`` /
    ``micro_batch_lengths`` / ``elem_counts_per_gb``) lands in each shard's
    ``extra_info`` so the ``*_presharded`` worker can reattach packing exactly
    as it does today via the legacy fan-out path.

    Args:
        meta: input KVBatchMeta covering the full step batch. Must have
            ``sequence_lengths`` populated (per-key seq lens).
        dp_world: number of data-parallel ranks.
        batch_size: total samples — passed to ``shard_by_batch_size``.
            Use ``None`` for the logprob path (matches ``_shard_for_logprob``);
            use the GBS for the train path (matches ``_shard_for_train``).
        sequence_packing_args / dynamic_batching_args: packing config —
            same dicts passed to ``BatchedDataDict.shard_by_batch_size``.
            Mutually exclusive. Both ``None`` → unpacked interleave-split.

    Returns:
        ``(per_rank_metas, unsorted_indices)``. ``per_rank_metas`` is the
        list of ``dp_world`` ``KVBatchMeta`` slices. ``unsorted_indices``
        is the inverse permutation that maps aggregated DP-rank-order
        outputs back to original ``meta.keys`` order — pass it to
        ``BatchedDataDict.reorder_data`` after worker results are
        aggregated. ``None`` when no reorder occurred (rare; even the
        unpacked path interleaves via ``shard_by_batch_size``).
    """
    n = len(meta.keys)
    if dp_world <= 0:
        raise ValueError(f"dp_world must be positive, got {dp_world}")
    if meta.sequence_lengths is None or len(meta.sequence_lengths) != n:
        raise ValueError(
            "shard_meta_for_dp requires meta.sequence_lengths populated and "
            f"of length {n} (got {meta.sequence_lengths!r}). The rollout "
            "actor's fan-out should populate this from input_lengths."
        )
    if sequence_packing_args is not None and dynamic_batching_args is not None:
        raise ValueError(
            "Pass at most one of sequence_packing_args / dynamic_batching_args."
        )

    seq_lens = list(meta.sequence_lengths)
    # Skeleton BatchedDataDict — `shard_by_batch_size` only needs
    # input_ids (placeholder), input_lengths (real), sample_mask (ones).
    # ``_meta_idx`` lets us recover which original meta index each shard row
    # corresponds to, so we can slice ``meta.keys`` per rank.
    skeleton = BatchedDataDict(
        {
            "input_ids": torch.zeros(n, 1, dtype=torch.int64),
            "input_lengths": torch.tensor(seq_lens, dtype=torch.int64),
            "sample_mask": torch.ones(n, dtype=torch.float32),
            "_meta_idx": torch.arange(n, dtype=torch.int64),
        }
    )

    if dynamic_batching_args is not None:
        sharded, _ = skeleton.shard_by_batch_size(
            dp_world,
            batch_size=batch_size,
            dynamic_batching_args=dynamic_batching_args,
        )
    elif sequence_packing_args is not None:
        sharded, _ = skeleton.shard_by_batch_size(
            dp_world,
            batch_size=batch_size,
            sequence_packing_args=sequence_packing_args,
        )
    else:
        sharded = skeleton.shard_by_batch_size(dp_world, batch_size=batch_size)

    base_extra: dict[str, Any] = dict(meta.extra_info or {})
    out: list[KVBatchMeta] = []
    flat_idx: list[int] = []
    for shard in sharded:
        idx_list: list[int] = shard["_meta_idx"].tolist()
        flat_idx.extend(idx_list)
        rank_keys = [meta.keys[i] for i in idx_list]
        rank_seqlens = [seq_lens[i] for i in idx_list]
        rank_extra = dict(base_extra)
        # Per-shard packing metadata — set by ``shard_by_batch_size`` when
        # sequence_packing/dynamic_batching is enabled. Workers' *_presharded
        # paths look these up off ``meta.extra_info``.
        for attr in ("micro_batch_indices", "micro_batch_lengths", "elem_counts_per_gb"):
            val = getattr(shard, attr, None)
            if val is not None:
                rank_extra[attr] = val
        out.append(
            KVBatchMeta(
                partition_id=meta.partition_id,
                task_name=meta.task_name,
                keys=rank_keys,
                fields=meta.fields,
                sequence_lengths=rank_seqlens,
                extra_info=rank_extra,
            )
        )

    # Build inverse permutation: unsorted[orig_idx] = position_in_aggregated.
    # When workers' results are concatenated in DP-rank order, row `j` of
    # the aggregate corresponds to original index `flat_idx[j]`. To restore
    # original meta.keys order, the caller does aggregated.reorder_data(
    # unsorted_indices) — same contract as `_shard_for_logprob`.
    unsorted: Optional[list[int]] = None
    if flat_idx != list(range(n)):
        unsorted = [0] * n
        for new_pos, old_idx in enumerate(flat_idx):
            unsorted[old_idx] = new_pos
    return out, unsorted
