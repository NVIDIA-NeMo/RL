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
"""Tests for ``fan_out_per_rank_metas`` schema-extension behavior.

Lock in the multimodal-extras fix: tensor fields beyond ``seed_fields``
(e.g. VLM ``pixel_values``) ride along instead of being silently dropped.
"""

from __future__ import annotations

import torch

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.preshard import (
    DP_SEED_FIELDS,
    LP_SEED_FIELDS,
    fan_out_per_rank_metas,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _shard(n_samples: int = 4, *, with_extras: bool = False) -> BatchedDataDict:
    d: BatchedDataDict = BatchedDataDict()
    d["input_ids"] = torch.zeros((n_samples, 8), dtype=torch.long)
    d["input_lengths"] = torch.tensor([8] * n_samples, dtype=torch.long)
    d["token_mask"] = torch.ones((n_samples, 8), dtype=torch.long)
    d["sample_mask"] = torch.ones((n_samples,), dtype=torch.long)
    if with_extras:
        # Stand-in for a multimodal field — shape doesn't matter, only
        # that it's a tensor not in DP_SEED_FIELDS.
        d["pixel_values"] = torch.zeros((n_samples, 3, 4, 4), dtype=torch.float32)
    return d


def _setup_partition(client: NoOpDataPlaneClient, *, num_samples: int):
    client.register_partition(
        partition_id="train",
        fields=list(DP_SEED_FIELDS),
        num_samples=num_samples,
        consumer_tasks=["train"],
    )


def test_fan_out_includes_seed_fields():
    """Fields in the canonical seed set are written and listed in the meta."""
    client = NoOpDataPlaneClient()
    pre_shards = [_shard()]
    _setup_partition(client, num_samples=4)
    metas = fan_out_per_rank_metas(
        pre_shards,
        dp_client=client,
        partition_id="train",
        task_name="train",
        key_prefix="step1",
        seed_fields=DP_SEED_FIELDS,
    )
    assert len(metas) == 1
    fields = set(metas[0].fields)
    # input_ids/input_lengths/token_mask/sample_mask present in the shard.
    assert {"input_ids", "input_lengths", "token_mask", "sample_mask"} <= fields


def test_fan_out_includes_tensor_extras():
    """Tensor fields not in seed_fields (multimodal) are auto-included."""
    client = NoOpDataPlaneClient()
    pre_shards = [_shard(with_extras=True)]
    _setup_partition(client, num_samples=4)
    metas = fan_out_per_rank_metas(
        pre_shards,
        dp_client=client,
        partition_id="train",
        task_name="train",
        key_prefix="step1",
        seed_fields=DP_SEED_FIELDS,
    )
    fields = set(metas[0].fields)
    assert "pixel_values" in fields, (
        "Multimodal tensor extras must ride along; otherwise VLM training "
        "is silently broken on the TQ path."
    )


def test_fan_out_skips_non_tensor_extras():
    """Non-tensor entries (lists, primitives) are not written to TQ."""
    client = NoOpDataPlaneClient()
    shard = _shard()
    shard["some_string"] = "not-a-tensor"
    shard["some_list"] = [1, 2, 3, 4]
    pre_shards = [shard]
    _setup_partition(client, num_samples=4)
    metas = fan_out_per_rank_metas(
        pre_shards,
        dp_client=client,
        partition_id="train",
        task_name="train",
        key_prefix="step1",
        seed_fields=DP_SEED_FIELDS,
    )
    fields = set(metas[0].fields)
    assert "some_string" not in fields
    assert "some_list" not in fields


def test_lp_seed_fields_subset_of_dp_seed_fields():
    """LP_SEED_FIELDS must be a subset of DP_SEED_FIELDS — same partition,
    consumers fetch what they need via select_fields.
    """
    assert set(LP_SEED_FIELDS) <= set(DP_SEED_FIELDS)


def test_metas_per_rank_have_namespaced_keys():
    """Each DP rank's meta gets keys prefixed with ``_dp{rank}_``."""
    client = NoOpDataPlaneClient()
    pre_shards = [_shard(), _shard()]
    _setup_partition(client, num_samples=4)
    metas = fan_out_per_rank_metas(
        pre_shards,
        dp_client=client,
        partition_id="train",
        task_name="train",
        key_prefix="step1",
        seed_fields=DP_SEED_FIELDS,
    )
    assert len(metas) == 2
    for r, meta in enumerate(metas):
        assert all(k.startswith(f"step1_dp{r}_") for k in meta.keys), (
            f"rank {r} meta keys must be prefixed with step1_dp{r}_"
        )
