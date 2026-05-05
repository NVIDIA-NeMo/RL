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

import asyncio
from typing import Any, Optional, Sequence

import torch
from tensordict import TensorDict

from nemo_rl.data_plane.interfaces import DataPlaneClient, KVBatchMeta
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

# Tensor fields the data plane carries between driver and DP workers. The
# canonical schema for the ``train`` partition. Producers (sync trainer fan-out,
# async trainer fan-out) write only the subset they have computed; consumers
# (``train_presharded`` workers) fetch what they need via ``select_fields``.
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


def driver_balanced_preshards(
    train_data: BatchedDataDict,
    *,
    dp_world: int,
    policy_cfg: dict[str, Any],
) -> list[BatchedDataDict]:
    """Shard ``train_data`` into ``dp_world`` balanced shards.

    Mirrors legacy ``lm_policy.train``: ``shard_by_batch_size(shards=dp_world,
    sequence_packing_args=...)`` uses ``bin_count_multiple=dp_world`` which is
    what guarantees every DP rank ends up with the same number of microbatches.
    Without it, sequence packing / dynamic batching produce variable per-rank
    bin counts and Megatron diverges on its first cross-DP collective.

    Pure transform — no I/O, no TQ. Caller computes ``dp_world`` (typically
    ``policy.sharding_annotations.get_axis_size("data_parallel")``).
    """
    gbs = policy_cfg["train_global_batch_size"]
    seqpack_cfg = policy_cfg.get("sequence_packing", {}) or {}
    dynbatch_cfg = policy_cfg.get("dynamic_batching", {}) or {}

    spa: Optional[dict[str, Any]] = None
    dba: Optional[dict[str, Any]] = None
    if dynbatch_cfg.get("enabled", False):
        dba = {
            "input_key": "input_ids",
            "input_lengths_key": "input_lengths",
            "sequence_length_round": dynbatch_cfg["sequence_length_round"],
            "max_tokens_per_microbatch": dynbatch_cfg["train_mb_tokens"],
        }
    elif seqpack_cfg.get("enabled", False):
        spa = {
            "algorithm": seqpack_cfg["algorithm"],
            "input_key": "input_ids",
            "input_lengths_key": "input_lengths",
            "sequence_length_pad_multiple": policy_cfg[
                "make_sequence_length_divisible_by"
            ],
            "max_tokens_per_microbatch": seqpack_cfg["train_mb_tokens"],
        }

    if dba is not None:
        pre_shards, _ = train_data.shard_by_batch_size(
            dp_world,
            batch_size=gbs,
            dynamic_batching_args=dba,
        )
    elif spa is not None:
        pre_shards, _ = train_data.shard_by_batch_size(
            dp_world,
            batch_size=gbs,
            sequence_packing_args=spa,
        )
    else:
        pre_shards = train_data.shard_by_batch_size(
            dp_world,
            batch_size=gbs,
        )
    return pre_shards


def fan_out_per_rank_metas(
    pre_shards: Sequence[BatchedDataDict],
    *,
    dp_client: DataPlaneClient,
    partition_id: str,
    task_name: str,
    key_prefix: str,
    seed_fields: Sequence[str],
) -> list[KVBatchMeta]:
    """For each pre-shard: ``kv_batch_put`` seed fields, return per-rank meta.

    Each shard's key list is ``f"{key_prefix}_dp{r}_s{i}"`` for ``i in
    range(n_shard)``. Pre-computed packing metadata
    (``micro_batch_indices`` / ``micro_batch_lengths`` /
    ``elem_counts_per_gb``) rides on ``KVBatchMeta.extra_info`` so
    ``train_presharded`` can reattach it post-fetch and skip a local repack.

    The caller chooses ``key_prefix`` to namespace keys: ``f"step{N}"`` for
    sync GRPO, ``f"v{wv}_step{N}"`` for the planned async path.
    """
    dp_metas: list[KVBatchMeta] = []
    for dp_rank, shard in enumerate(pre_shards):
        n_shard = int(shard["sample_mask"].shape[0])
        shard_keys = [
            f"{key_prefix}_dp{dp_rank}_s{i}" for i in range(n_shard)
        ]
        shard_field_names = [
            f
            for f in seed_fields
            if f in shard and isinstance(shard[f], torch.Tensor)
        ]
        shard_fields = TensorDict(
            {f: shard[f].detach().contiguous() for f in shard_field_names},
            batch_size=[n_shard],
        )
        asyncio.run(
            dp_client.kv_batch_put(
                keys=shard_keys,
                partition_id=partition_id,
                fields=shard_fields,
            )
        )
        extra: dict[str, Any] = {}
        if (
            getattr(shard, "micro_batch_indices", None) is not None
            and getattr(shard, "micro_batch_lengths", None) is not None
        ):
            extra["micro_batch_indices"] = shard.micro_batch_indices
            extra["micro_batch_lengths"] = shard.micro_batch_lengths
            ecpg = getattr(shard, "elem_counts_per_gb", None)
            if ecpg is not None:
                extra["elem_counts_per_gb"] = ecpg
        dp_metas.append(
            KVBatchMeta(
                partition_id=partition_id,
                task_name=task_name,
                keys=shard_keys,
                fields=shard_field_names,
                sequence_lengths=[
                    int(s) for s in shard["input_lengths"].tolist()
                ],
                extra_info=extra,
            )
        )
    return dp_metas
