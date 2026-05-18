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

Shared by sync and async data-plane trainers. Operates on full
``BatchedDataDict``s and relies on ``shard_by_batch_size``'s
``bin_count_multiple=DP_world`` behavior to keep per-rank microbatch
counts uniform — without that, sequence packing / dynamic batching
produce variable per-rank bin counts and Megatron deadlocks at the
first cross-DP collective.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from nemo_rl.data_plane.interfaces import KVBatchMeta
from nemo_rl.data_plane.schema import (
    ELEM_COUNTS_PER_GB,
    INPUT_IDS,
    INPUT_LENGTHS,
    META_IDX,
    MICRO_BATCH_INDICES,
    MICRO_BATCH_LENGTHS,
    SAMPLE_MASK,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def shard_meta_for_dp(
    meta: KVBatchMeta,
    *,
    dp_world: int,
    batch_size: Optional[int] = None,
    sequence_packing_args: Optional[dict[str, Any]] = None,
    dynamic_batching_args: Optional[dict[str, Any]] = None,
) -> tuple[list[KVBatchMeta], Optional[list[int]]]:
    """Pure key-list split: assign ``meta.keys`` to ``dp_world`` ranks.

    Seq-len-aware on top of ``shard_by_batch_size``. No I/O, no key
    minting. Used for every dispatch after rollout (logprob, ref-logprob,
    train); the rollout actor's first write goes through
    :func:`nemo_rl.experience.sync_rollout_actor.kv_first_write` directly.

    Per-rank packing metadata (``micro_batch_indices`` /
    ``micro_batch_lengths`` / ``elem_counts_per_gb``) is set in each
    shard's ``extra_info`` so the ``*_presharded`` worker can reattach
    packing as it does on the legacy fan-out path.

    Args:
        meta: Full-batch ``KVBatchMeta`` with ``sequence_lengths`` populated.
        dp_world: Number of DP ranks.
        batch_size: Total samples; ``None`` for the logprob path, GBS for train.
        sequence_packing_args: Packing config dict for ``shard_by_batch_size``.
        dynamic_batching_args: Dynamic-batching config dict; mutually exclusive with the above.

    Returns:
        ``(per_rank_metas, unsorted_indices)``. ``unsorted_indices`` is
        the inverse permutation that maps DP-rank-order outputs back to
        original ``meta.keys`` order (feed to
        ``BatchedDataDict.reorder_data`` post-aggregation); ``None`` if
        no reorder occurred.
    """
    n = len(meta.keys)
    if n == 0:
        raise ValueError("shard_meta_for_dp: empty meta — nothing to shard")
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
    # ``meta_idx`` lets us recover which original meta index each shard row
    # corresponds to, so we can slice ``meta.keys`` per rank.
    #
    # ``INPUT_IDS`` seq dim sizing: the dynamic-batching microbatch planner
    # in ``BatchedDataDict.shard_by_batch_size`` reads ``input_ids.shape[1]``
    # as an ``unpadded_seqlen`` cap (``min(padded_seqlen, unpadded_seqlen)``).
    # A trivial ``(n, 1)`` shape made the cap clamp every microbatch length
    # to 1, producing bogus ``micro_batch_lengths`` that, when consumed by
    # workers, truncated real sequences to 1 token → zero grad_norm. Size
    # the placeholder to ``max_tokens_per_microbatch`` (the largest seqlen
    # the planner can ever request, per its own assertion) so the cap is
    # never the binding factor. Memory cost is small (object only — bytes
    # never get filled with real data; just used for shape lookups).
    input_ids_seqlen = 1
    if dynamic_batching_args is not None:
        input_ids_seqlen = int(dynamic_batching_args["max_tokens_per_microbatch"])
    skeleton = BatchedDataDict(
        {
            INPUT_IDS: torch.zeros(n, input_ids_seqlen, dtype=torch.int64),
            INPUT_LENGTHS: torch.tensor(seq_lens, dtype=torch.int64),
            SAMPLE_MASK: torch.ones(n, dtype=torch.float32),
            META_IDX: torch.arange(n, dtype=torch.int64),
        }
    )

    if dynamic_batching_args is not None:
        sharded, _ = skeleton.shard_by_batch_size(
            dp_world,
            batch_size=batch_size,
            # pyrefly: ignore  # bad-argument-type
            dynamic_batching_args=dynamic_batching_args,
        )
    elif sequence_packing_args is not None:
        sharded, _ = skeleton.shard_by_batch_size(
            dp_world,
            batch_size=batch_size,
            # pyrefly: ignore  # bad-argument-type
            sequence_packing_args=sequence_packing_args,
        )
    else:
        sharded = skeleton.shard_by_batch_size(dp_world, batch_size=batch_size)

    base_extra: dict[str, Any] = dict(meta.extra_info or {})
    out: list[KVBatchMeta] = []
    flat_idx: list[int] = []
    for shard in sharded:
        # pyrefly: ignore  # no-matching-overload
        idx_list: list[int] = shard[META_IDX].tolist()
        flat_idx.extend(idx_list)
        rank_keys = [meta.keys[i] for i in idx_list]
        rank_seqlens = [seq_lens[i] for i in idx_list]
        rank_extra = dict(base_extra)
        # Per-shard packing metadata — set by ``shard_by_batch_size`` when
        # sequence_packing or dynamic_batching is enabled. Workers'
        # *_presharded paths look these up off ``meta.extra_info`` to avoid
        # re-packing locally. Propagation is critical: local re-packing on
        # different real per-rank data produces varying microbatch counts,
        # which desynchronizes NCCL collectives across DP ranks and trips
        # the Watchdog timeout.
        for attr in (
            MICRO_BATCH_INDICES,
            MICRO_BATCH_LENGTHS,
            ELEM_COUNTS_PER_GB,
        ):
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
