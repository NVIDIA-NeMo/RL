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

from dataclasses import dataclass, replace
from typing import Any, Optional

import torch

from nemo_rl.data.packed_rollouts import (
    PACKED_ATTENTION_SEGMENT_LENGTHS,
    PACKED_ATTENTION_SELECTED_SEGMENTS,
    TREE_ATTENTION_FRAGMENT_SELECTIONS,
    TREE_ATTENTION_LAYOUTS,
    TreeAttentionFragmentBatchLayout,
    pad_tree_attention_fragments,
    plan_tree_attention_fragments,
    validate_packed_attention_segment_lengths,
)
from nemo_rl.data_plane.interfaces import KVBatchMeta
from nemo_rl.data_plane.schema import (
    ELEM_COUNTS_PER_GB,
    GLOBAL_FORWARD_PAD_SEQLEN,
    INPUT_IDS,
    INPUT_LENGTHS,
    META_IDX,
    MICRO_BATCH_INDICES,
    MICRO_BATCH_LENGTHS,
    SAMPLE_MASK,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@dataclass(frozen=True)
class PackedAttentionCallMeta:
    """Physical-call control-plane view of logical TQ rollout rows."""

    sequence_lengths: list[int]
    parent_indices: list[int]
    segment_indices: list[int]


def expand_meta_for_tree_attention(
    meta: KVBatchMeta,
    *,
    max_physical_tokens: int,
    fragment_count_multiple: int = 1,
) -> TreeAttentionFragmentBatchLayout | None:
    """Plan bounded virtual tree rows while leaving TQ payloads logical."""
    layouts = (meta.extra_info or {}).get(TREE_ATTENTION_LAYOUTS)
    if layouts is None:
        return None
    if len(layouts) != len(meta.sample_ids):
        raise ValueError(
            f"{TREE_ATTENTION_LAYOUTS} must align with sample_ids: "
            f"{len(layouts)} != {len(meta.sample_ids)}"
        )
    if meta.sequence_lengths is None:
        raise ValueError("tree attention metadata requires sequence lengths")
    if all(layout.unique_token_count <= max_physical_tokens for layout in layouts):
        for layout in layouts:
            layout.validate()
            if layout.max_path_length > max_physical_tokens:
                raise ValueError(
                    "tree logical path exceeds the physical fragment budget: "
                    f"{layout.max_path_length} > {max_physical_tokens}"
                )
        return None

    fragments = []
    parent_indices: list[int] = []
    edge_lengths: list[int] = []
    for parent, layout in enumerate(layouts):
        row_fragments = plan_tree_attention_fragments(
            layout, max_physical_tokens=max_physical_tokens
        )
        fragments.extend(row_fragments)
        parent_indices.extend([parent] * len(row_fragments))
        edge_lengths.append(len(layout.edge_source_indices))
    fragments, parent_indices = pad_tree_attention_fragments(
        fragments,
        parent_indices,
        count_multiple=fragment_count_multiple,
    )
    return TreeAttentionFragmentBatchLayout(
        fragments=tuple(fragments),
        parent_indices=tuple(parent_indices),
        original_batch_size=len(meta.sample_ids),
        original_edge_lengths=tuple(edge_lengths),
    )


def expand_meta_for_packed_attention(
    meta: KVBatchMeta,
) -> PackedAttentionCallMeta | None:
    """Expand packed rollout metadata into legacy-equivalent call descriptors.

    The payload stays in TQ under logical rollout IDs. This pure metadata view
    is the expansion boundary used by the generic DP sharder; workers later
    materialize only the physical calls assigned to their rank.
    """
    packed_segment_lengths = meta.extra_info.get(PACKED_ATTENTION_SEGMENT_LENGTHS)
    if packed_segment_lengths is None:
        return None
    if meta.sequence_lengths is None:
        raise ValueError("packed attention metadata requires meta.sequence_lengths")
    validate_packed_attention_segment_lengths(
        packed_segment_lengths, meta.sequence_lengths
    )

    sequence_lengths: list[int] = []
    parent_indices: list[int] = []
    segment_indices: list[int] = []
    for parent_index, row_segments in enumerate(packed_segment_lengths):
        for segment_index, segment_length in enumerate(row_segments):
            sequence_lengths.append(segment_length)
            parent_indices.append(parent_index)
            segment_indices.append(segment_index)
    return PackedAttentionCallMeta(
        sequence_lengths=sequence_lengths,
        parent_indices=parent_indices,
        segment_indices=segment_indices,
    )


def split_packed_attention_microbatch_metas(
    meta: KVBatchMeta,
) -> list[KVBatchMeta]:
    """Split one DP-rank meta into its already-planned packed microbatches.

    ``shard_meta_for_dp`` stores physical calls in packer execution order and
    records contiguous ranges for each packed bin.  Workers use this helper to
    fetch the logical parent rows once, then expand/broadcast/consume one bin at
    a time instead of materializing the full ``[calls, global_max_call]``
    rectangle.

    The returned metas retain only the parent rows referenced by their bin and
    remap ``PACKED_ATTENTION_SELECTED_SEGMENTS`` to those local rows.  Their
    packing metadata describes exactly one microbatch, so the existing model
    preparation path consumes them without re-packing or changing call order.
    """
    extra = meta.extra_info or {}
    segment_lengths = extra.get(PACKED_ATTENTION_SEGMENT_LENGTHS)
    selected_segments = extra.get(PACKED_ATTENTION_SELECTED_SEGMENTS)
    micro_batch_indices = extra.get(MICRO_BATCH_INDICES)
    micro_batch_lengths = extra.get(MICRO_BATCH_LENGTHS)
    elem_counts = extra.get(ELEM_COUNTS_PER_GB)

    if segment_lengths is None or selected_segments is None:
        return [meta]
    if micro_batch_indices is None or micro_batch_lengths is None:
        return [meta]
    if len(micro_batch_indices) != len(micro_batch_lengths):
        raise ValueError(
            "packed attention microbatch index/length chunk counts differ: "
            f"{len(micro_batch_indices)} != {len(micro_batch_lengths)}"
        )
    if elem_counts is not None and len(elem_counts) != len(micro_batch_indices):
        raise ValueError(
            "packed attention elem_counts_per_gb must align with packing chunks"
        )

    out: list[KVBatchMeta] = []
    chunk_offset = 0
    for chunk_index, (chunk_ranges, chunk_lengths) in enumerate(
        zip(micro_batch_indices, micro_batch_lengths, strict=True)
    ):
        if len(chunk_ranges) != len(chunk_lengths):
            raise ValueError(
                "packed attention microbatch ranges/lengths differ in chunk "
                f"{chunk_index}: {len(chunk_ranges)} != {len(chunk_lengths)}"
            )
        chunk_count = (
            int(elem_counts[chunk_index])
            if elem_counts is not None
            else (int(chunk_ranges[-1][1]) if chunk_ranges else 0)
        )
        for (start, stop), packed_length in zip(
            chunk_ranges, chunk_lengths, strict=True
        ):
            start = int(start)
            stop = int(stop)
            if not 0 <= start < stop <= chunk_count:
                raise ValueError(
                    "invalid packed attention microbatch range "
                    f"[{start}, {stop}) for chunk size {chunk_count}"
                )
            calls = selected_segments[chunk_offset + start : chunk_offset + stop]
            if len(calls) != stop - start:
                raise ValueError(
                    "packed attention selected-call metadata ended before its "
                    "microbatch ranges"
                )

            parent_indices = list(dict.fromkeys(int(parent) for parent, _ in calls))
            parent_to_local = {
                parent: local for local, parent in enumerate(parent_indices)
            }
            micro_meta = meta.subset(parent_indices)
            micro_meta.extra_info[PACKED_ATTENTION_SELECTED_SEGMENTS] = [
                (parent_to_local[int(parent)], int(segment))
                for parent, segment in calls
            ]
            micro_meta.extra_info[MICRO_BATCH_INDICES] = [[[0, len(calls)]]]
            micro_meta.extra_info[MICRO_BATCH_LENGTHS] = [[int(packed_length)]]
            micro_meta.extra_info[ELEM_COUNTS_PER_GB] = [len(calls)]
            micro_meta.extra_info[GLOBAL_FORWARD_PAD_SEQLEN] = max(
                int(segment_lengths[parent][segment]) for parent, segment in calls
            )
            out.append(micro_meta)
        chunk_offset += chunk_count

    if chunk_offset != len(selected_segments):
        raise ValueError(
            "packed attention packing metadata does not cover every selected call: "
            f"covered={chunk_offset}, selected={len(selected_segments)}"
        )
    if not out:
        raise ValueError("packed attention packing metadata contains no microbatches")
    return out


def split_tree_attention_microbatch_metas(meta: KVBatchMeta) -> list[KVBatchMeta]:
    """Split a tree DP shard into its already-planned packed microbatches.

    Tree rollouts stay one row per episode in TQ.  The driver has already
    ordered and packed those rows, so this transform only slices the control
    plane and narrows the per-fetch pad width.  It never rebuilds the packing
    plan on a worker.
    """
    extra = meta.extra_info or {}
    layouts = extra.get(TREE_ATTENTION_LAYOUTS)
    fragment_selections = extra.get(TREE_ATTENTION_FRAGMENT_SELECTIONS)
    micro_batch_indices = extra.get(MICRO_BATCH_INDICES)
    micro_batch_lengths = extra.get(MICRO_BATCH_LENGTHS)
    elem_counts = extra.get(ELEM_COUNTS_PER_GB)

    if layouts is None:
        return [meta]
    if micro_batch_indices is None or micro_batch_lengths is None:
        return [meta]
    if len(layouts) != len(meta.sample_ids):
        raise ValueError(
            f"{TREE_ATTENTION_LAYOUTS} must align with sample_ids: "
            f"{len(layouts)} != {len(meta.sample_ids)}"
        )
    if meta.sequence_lengths is None:
        raise ValueError("tree attention metadata requires sequence lengths")
    sequence_lengths = meta.sequence_lengths
    if len(micro_batch_indices) != len(micro_batch_lengths):
        raise ValueError(
            "tree attention microbatch index/length chunk counts differ: "
            f"{len(micro_batch_indices)} != {len(micro_batch_lengths)}"
        )
    if elem_counts is not None and len(elem_counts) != len(micro_batch_indices):
        raise ValueError(
            "tree attention elem_counts_per_gb must align with packing chunks"
        )

    out: list[KVBatchMeta] = []
    chunk_offset = 0
    for chunk_index, (chunk_ranges, chunk_lengths) in enumerate(
        zip(micro_batch_indices, micro_batch_lengths, strict=True)
    ):
        if len(chunk_ranges) != len(chunk_lengths):
            raise ValueError(
                "tree attention microbatch ranges/lengths differ in chunk "
                f"{chunk_index}: {len(chunk_ranges)} != {len(chunk_lengths)}"
            )
        chunk_count = (
            int(elem_counts[chunk_index])
            if elem_counts is not None
            else (int(chunk_ranges[-1][1]) if chunk_ranges else 0)
        )
        for (start, stop), packed_length in zip(
            chunk_ranges, chunk_lengths, strict=True
        ):
            start = int(start)
            stop = int(stop)
            if not 0 <= start < stop <= chunk_count:
                raise ValueError(
                    "invalid tree attention microbatch range "
                    f"[{start}, {stop}) for chunk size {chunk_count}"
                )
            if fragment_selections is None:
                row_indices = list(range(chunk_offset + start, chunk_offset + stop))
                micro_meta = meta.subset(row_indices)
                physical_count = len(row_indices)
                forward_pad_seqlen = max(
                    int(sequence_lengths[index]) for index in row_indices
                )
            else:
                selected = fragment_selections[
                    chunk_offset + start : chunk_offset + stop
                ]
                if len(selected) != stop - start:
                    raise ValueError(
                        "tree fragment selection metadata ended before its "
                        "microbatch ranges"
                    )
                parent_indices = list(
                    dict.fromkeys(int(parent) for parent, _ in selected)
                )
                parent_to_local = {
                    parent: local for local, parent in enumerate(parent_indices)
                }
                micro_meta = meta.subset(parent_indices)
                micro_meta.extra_info[TREE_ATTENTION_FRAGMENT_SELECTIONS] = [
                    (parent_to_local[int(parent)], fragment)
                    for parent, fragment in selected
                ]
                physical_count = len(selected)
                forward_pad_seqlen = max(
                    fragment.layout.unique_token_count for _, fragment in selected
                )
            micro_meta.extra_info[MICRO_BATCH_INDICES] = [[[0, physical_count]]]
            micro_meta.extra_info[MICRO_BATCH_LENGTHS] = [[int(packed_length)]]
            micro_meta.extra_info[ELEM_COUNTS_PER_GB] = [physical_count]
            micro_meta.extra_info[GLOBAL_FORWARD_PAD_SEQLEN] = forward_pad_seqlen
            out.append(micro_meta)
        chunk_offset += chunk_count

    expected_count = (
        len(fragment_selections)
        if fragment_selections is not None
        else len(meta.sample_ids)
    )
    if chunk_offset != expected_count:
        raise ValueError(
            "tree attention packing metadata does not cover every row: "
            f"covered={chunk_offset}, rows={expected_count}"
        )
    if not out:
        raise ValueError("tree attention packing metadata contains no microbatches")
    return out


def shard_meta_for_dp(
    meta: KVBatchMeta,
    *,
    dp_world: int,
    batch_size: Optional[int] = None,
    sequence_packing_args: Optional[dict[str, Any]] = None,
    dynamic_batching_args: Optional[dict[str, Any]] = None,
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

    Returns:
        ``(per_rank_metas, unsorted_indices)``. ``unsorted_indices`` is
        the original row index for every output in DP-rank order (feed to
        ``BatchedDataDict.reorder_data`` post-aggregation); ``None`` if
        no reorder occurred.
    """
    n = len(meta.sample_ids)
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

    logical_seq_lens = list(meta.sequence_lengths)
    packed_segment_lengths = meta.extra_info.get(PACKED_ATTENTION_SEGMENT_LENGTHS)
    tree_layouts = meta.extra_info.get(TREE_ATTENTION_LAYOUTS)
    if tree_layouts is not None and len(tree_layouts) != n:
        raise ValueError(
            f"{TREE_ATTENTION_LAYOUTS} must align with sample_ids: "
            f"{len(tree_layouts)} != {n}"
        )
    packed_call_meta = expand_meta_for_packed_attention(meta)
    tree_fragment_layout = None
    if tree_layouts is not None:
        if dynamic_batching_args is not None:
            raise NotImplementedError(
                "tree attention metadata is not supported with dynamic batching; "
                "use sequence packing"
            )
        if sequence_packing_args is None:
            raise ValueError("tree rollouts require sequence packing")
        tree_fragment_layout = expand_meta_for_tree_attention(
            meta,
            max_physical_tokens=int(sequence_packing_args["max_tokens_per_microbatch"]),
            fragment_count_multiple=dp_world,
        )
    if packed_call_meta is not None:
        if dynamic_batching_args is not None:
            raise NotImplementedError(
                "packed attention metadata is not supported with dynamic batching; "
                "use sequence packing"
            )
        seq_lens = packed_call_meta.sequence_lengths
    elif tree_fragment_layout is not None:
        seq_lens = [
            fragment.layout.unique_token_count
            for fragment in tree_fragment_layout.fragments
        ]
    else:
        seq_lens = logical_seq_lens
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
            INPUT_IDS: torch.zeros(len(seq_lens), input_ids_seqlen, dtype=torch.int64),
            INPUT_LENGTHS: torch.tensor(seq_lens, dtype=torch.int64),
            SAMPLE_MASK: torch.ones(len(seq_lens), dtype=torch.float32),
            META_IDX: torch.arange(len(seq_lens), dtype=torch.int64),
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
            batch_size=(
                None
                if packed_segment_lengths is not None
                or tree_fragment_layout is not None
                else batch_size
            ),
            allow_uneven_shards=(
                packed_segment_lengths is not None or tree_fragment_layout is not None
            ),
            # pyrefly: ignore  # bad-argument-type
            sequence_packing_args=sequence_packing_args,
        )
    elif packed_segment_lengths is not None or tree_fragment_layout is not None:
        sharded = skeleton.shard_by_batch_size(
            dp_world,
            batch_size=None,
            allow_uneven_shards=True,
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
        if packed_call_meta is not None:
            assert packed_segment_lengths is not None
            selected_global = [
                (
                    packed_call_meta.parent_indices[i],
                    packed_call_meta.segment_indices[i],
                )
                for i in idx_list
            ]
            parent_indices = list(
                dict.fromkeys(parent for parent, _ in selected_global)
            )
            parent_to_local = {
                parent: local for local, parent in enumerate(parent_indices)
            }
            rank_seqlens = [logical_seq_lens[i] for i in parent_indices]
            rank_extra[PACKED_ATTENTION_SEGMENT_LENGTHS] = [
                packed_segment_lengths[i] for i in parent_indices
            ]
            rank_extra[PACKED_ATTENTION_SELECTED_SEGMENTS] = [
                (parent_to_local[parent], segment)
                for parent, segment in selected_global
            ]
        elif tree_fragment_layout is not None:
            assert tree_layouts is not None
            selected_global = [
                (
                    tree_fragment_layout.parent_indices[index],
                    tree_fragment_layout.fragments[index],
                )
                for index in idx_list
            ]
            parent_indices = list(
                dict.fromkeys(parent for parent, _ in selected_global)
            )
            parent_to_local = {
                parent: local for local, parent in enumerate(parent_indices)
            }
            rank_seqlens = [logical_seq_lens[index] for index in parent_indices]
            rank_extra[TREE_ATTENTION_LAYOUTS] = [
                tree_layouts[index] for index in parent_indices
            ]
            rank_extra[TREE_ATTENTION_FRAGMENT_SELECTIONS] = [
                (parent_to_local[parent], fragment)
                for parent, fragment in selected_global
            ]
        else:
            parent_indices = idx_list
            rank_seqlens = [seq_lens[i] for i in idx_list]
            if tree_layouts is not None:
                rank_extra[TREE_ATTENTION_LAYOUTS] = [tree_layouts[i] for i in idx_list]
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
        out.append(
            replace(
                meta.subset(parent_indices),
                sequence_lengths=rank_seqlens,
                extra_info=rank_extra,
            )
        )

    # When worker results are concatenated in DP-rank order, aggregate row
    # ``j`` corresponds to original row ``flat_idx[j]``. ``reorder_data``
    # sorts positions by these original indices, matching the contract of
    # ``BatchedDataDict.shard_by_batch_size``.
    unsorted: Optional[list[int]] = None
    result_count = len(seq_lens)
    if flat_idx != list(range(result_count)):
        unsorted = flat_idx
    return out, unsorted
