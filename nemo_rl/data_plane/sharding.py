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
"""Driver-side seqlen-balanced DP sharding from metadata only.

Sort-by-seqlen + stride is the same algorithm NeMo-RL's
``BatchedDataDict.shard_by_batch_size(dynamic_batching_args=...)`` branch
applies (`batched_data_dict.py:404-414`) and rl-arena's ``shard_for_dp``
(`rl-arena/arena/dataplane_client.py:275-314`). Operates on
``list[str] + list[int]`` only — does not touch tensors. Per plan §Stage 4,
this is the entire data-plane sharding surface in Phase 1.
"""

from __future__ import annotations

from nemo_rl.data_plane.interfaces import KVBatchMeta


def shard_keys_by_seqlen(
    meta: KVBatchMeta, dp_world_size: int
) -> list[KVBatchMeta]:
    """Split a meta into per-DP-rank shards using sort-by-seqlen + stride.

    Each rank gets a mix of long+short samples and roughly equal total
    tokens. List index IS the dp_rank; shards inherit ``task_name`` and
    ``fields`` for traceability.

    Control-plane only — does NOT fetch tensor data.
    """
    if dp_world_size <= 0:
        raise ValueError(f"dp_world_size must be positive, got {dp_world_size}")
    if meta.sequence_lengths is None:
        raise ValueError(
            "shard_keys_by_seqlen requires meta.sequence_lengths "
            "(set the input_lengths tag at kv_batch_put time, or populate "
            "meta.sequence_lengths from train_data['input_lengths'] before "
            "calling)"
        )
    if len(meta.sequence_lengths) != len(meta.keys):
        raise ValueError(
            f"meta.keys ({len(meta.keys)}) and meta.sequence_lengths "
            f"({len(meta.sequence_lengths)}) length mismatch"
        )

    seqlens = meta.sequence_lengths
    order = sorted(range(meta.size), key=seqlens.__getitem__)
    shards: list[KVBatchMeta] = []
    for r in range(dp_world_size):
        idx = order[r::dp_world_size]
        # Record original indices in extra_info so ``dp_dispatch`` can invert
        # the seqlen-strided permutation when aggregating per-rank results
        # back into a single output. Without this, ``policy.get_logprobs(meta)``
        # returns rows in [rank0 samples..., rank1 samples...] order rather
        # than the caller's ``meta.keys`` order — silent correctness bug.
        shards.append(
            KVBatchMeta(
                partition_id=meta.partition_id,
                task_name=meta.task_name,
                keys=[meta.keys[i] for i in idx],
                fields=list(meta.fields) if meta.fields is not None else None,
                sequence_lengths=[seqlens[i] for i in idx],
                extra_info={
                    **dict(meta.extra_info),
                    "_dp_original_indices": list(idx),
                },
            )
        )
    return shards
