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

from dataclasses import replace
from typing import Any, Optional

import torch

from nemo_rl.data.packing.shared_prefix_metadata import (
    SHARED_PREFIX_EXECUTION_SLOT,
    SHARED_PREFIX_PROMPT_LENGTHS,
)
from nemo_rl.data_plane.interfaces import KVBatchMeta
from nemo_rl.data_plane.schema import (
    ELEM_COUNTS_PER_GB,
    GROUP_ID_TAG,
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
    shared_prefix_groups: bool = False,
    shared_prefix_work_weights: tuple[int, int] | None = None,
) -> tuple[list[KVBatchMeta], Optional[list[int]]]:
    """Pure key-list split: assign ``meta.sample_ids`` to ``dp_world`` ranks.

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
        shared_prefix_groups: Assign complete prompt groups (rows sharing a
            ``GROUP_ID_TAG`` tag) coherently to DP ranks. This supersedes
            ordinary row-level sequence packing; each worker performs the
            structured star planning locally.

    Returns:
        ``(per_rank_metas, unsorted_indices)``. ``unsorted_indices`` contains
        the original-row index for each row in DP-rank concatenation order,
        matching :meth:`BatchedDataDict.reorder_data`. ``None`` means no
        reorder occurred.
    """
    n = len(meta.sample_ids)
    if n == 0:
        raise ValueError("shard_meta_for_dp: empty meta — nothing to shard")
    sequence_lengths = meta.sequence_lengths
    if sequence_lengths is None or len(sequence_lengths) != n:
        raise ValueError(
            "shard_meta_for_dp requires meta.sequence_lengths populated and "
            f"of length {n} (got {sequence_lengths!r}). The rollout "
            "actor's fan-out should populate this from input_lengths."
        )
    if sequence_packing_args is not None and dynamic_batching_args is not None:
        raise ValueError(
            "Pass at most one of sequence_packing_args / dynamic_batching_args."
        )
    if shared_prefix_work_weights is not None and not shared_prefix_groups:
        raise ValueError("Shared work weights require complete-group sharding")
    if shared_prefix_groups:
        if dynamic_batching_args is not None:
            raise ValueError(
                "shared-prefix group sharding does not support dynamic batching"
            )
        # The explicit row tag, like the advantage baseline, rather than the
        # "{group_id}_g{i}" sample-id naming convention.
        if meta.tags is None or any(GROUP_ID_TAG not in tag for tag in meta.tags):
            raise ValueError(
                f"shared-prefix group sharding requires a {GROUP_ID_TAG!r} tag on "
                "every row"
            )
        group_ids = [tag[GROUP_ID_TAG] for tag in meta.tags]
        if sequence_packing_args is None:
            raise ValueError(
                "shared-prefix group sharding requires sequence-packing arguments"
            )
        for key in ("max_tokens_per_microbatch", "sequence_length_pad_multiple"):
            if key not in sequence_packing_args:
                raise ValueError(
                    f"shared-prefix group sharding requires sequence_packing_args.{key}"
                )
        # Lazy: megatron.rl ships only with shared-prefix Megatron-LM builds,
        # and dense sharding must stay importable without it.
        from megatron.rl.shared_prefix_cost import estimate_shared_prefix_row_work
        from megatron.rl.shared_prefix_metadata import (
            plan_fixed_execution_slots,
            plan_group_coherent_shards,
        )

        slot_plan = plan_fixed_execution_slots(
            group_ids=group_ids,
            sequence_lengths=sequence_lengths,
            bin_capacity=int(sequence_packing_args["max_tokens_per_microbatch"]),
            batch_size=batch_size,
            sequence_length_pad_multiple=int(
                sequence_packing_args["sequence_length_pad_multiple"]
            ),
        )
        assignment_lengths = sequence_lengths
        if shared_prefix_work_weights is not None:
            if meta.tags is None or any(
                SHARED_PREFIX_PROMPT_LENGTHS not in tag for tag in meta.tags
            ):
                raise ValueError(
                    "Shared work balancing requires row-aligned prompt length tags"
                )
            assignment_lengths = estimate_shared_prefix_row_work(
                group_ids=group_ids,
                sequence_lengths=sequence_lengths,
                prompt_lengths=[tag[SHARED_PREFIX_PROMPT_LENGTHS] for tag in meta.tags],
                physical_weight=shared_prefix_work_weights[0],
                expanded_weight=shared_prefix_work_weights[1],
                # Each execution slot stores its own prompt copy.
                row_slot_ids=slot_plan.row_slot_ids,
            )
        plan = plan_group_coherent_shards(
            group_ids=group_ids,
            sequence_lengths=assignment_lengths,
            num_shards=dp_world,
            batch_size=batch_size,
        )
        sharded_metas: list[KVBatchMeta] = []
        for indices in plan.shard_indices:
            shard = meta.subset(indices)
            shard.extra_info[SHARED_PREFIX_EXECUTION_SLOT] = [
                slot_plan.row_slot_ids[index] for index in indices
            ]
            sharded_metas.append(shard)
        return (
            sharded_metas,
            (
                list(plan.rank_order_permutation)
                if plan.rank_order_permutation is not None
                else None
            ),
        )

    seq_lens = list(sequence_lengths)
    # Skeleton BatchedDataDict — `shard_by_batch_size` only needs
    # input_ids (placeholder), input_lengths (real), sample_mask (ones).
    # ``meta_idx`` lets us recover which original meta index each shard row
    # corresponds to, so we can slice ``meta.sample_ids`` per rank.
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
        # ``subset`` owns per-sample projection: it slices ``sample_ids``,
        # ``sequence_lengths`` and ``tags`` together, so a sidecar added to
        # ``KVBatchMeta`` later cannot go missing on the presharded path.
        # Only ``extra_info`` is per-shard rather than per-sample.
        out.append(replace(meta.subset(idx_list), extra_info=rank_extra))

    # ``reorder_data`` expects the forward permutation: row `j` in the
    # DP-rank-concatenated aggregate came from original row ``flat_idx[j]``.
    unsorted: Optional[list[int]] = None
    if flat_idx != list(range(n)):
        unsorted = flat_idx
    return out, unsorted
