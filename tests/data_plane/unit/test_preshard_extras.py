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
"""Tests for the rollout first-write helper and the meta-only sharder.

After the sync 1-hop refactor, ``fan_out_per_rank_metas`` was retired in
favor of:

  * ``kv_first_write`` — single flat ``kv_batch_put`` of every tensor
    field in the rollout output (multimodal extras ride along).
  * ``shard_meta_for_dp`` — pure key-list split per DP rank, no I/O.

These tests lock in the schema-extensibility behavior (multimodal
fields propagate) and the meta-sharding contract (no key minting,
identity preserved across shards).
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.codec import materialize, maybe_pack_jagged
from nemo_rl.data_plane.preshard import shard_meta_for_dp
from nemo_rl.data_plane.schema import (
    DP_TRAIN_FIELDS,
    LP_SEED_FIELDS,
    ROLLOUT_SEED_FIELDS,
    ROUTED_EXPERTS_FIELD,
    fields_with_optional_routed_experts,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.sync_rollout_actor import (
    _build_rollout_bulk_batch,
    kv_first_write,
)


def _final_batch(n_samples: int = 4, *, with_extras: bool = False) -> BatchedDataDict:
    d: BatchedDataDict = BatchedDataDict()
    d["input_ids"] = torch.zeros((n_samples, 8), dtype=torch.long)
    d["input_lengths"] = torch.tensor([8] * n_samples, dtype=torch.long)
    d["token_mask"] = torch.ones((n_samples, 8), dtype=torch.long)
    d["sample_mask"] = torch.ones((n_samples,), dtype=torch.long)
    d["generation_logprobs"] = torch.zeros((n_samples, 8), dtype=torch.float32)
    if with_extras:
        d["pixel_values"] = torch.zeros((n_samples, 3, 4, 4), dtype=torch.float32)
    return d


def _setup_partition(client: NoOpDataPlaneClient, *, num_samples: int):
    client.register_partition(
        partition_id="train",
        fields=list(DP_TRAIN_FIELDS),
        num_samples=num_samples,
        consumer_tasks=["train"],
    )


# ── kv_first_write schema extensibility ────────────────────────────────


def test_kv_first_write_writes_seed_fields():
    client = NoOpDataPlaneClient()
    _setup_partition(client, num_samples=4)
    fb = _final_batch(4)
    uids = [f"u{i}" for i in range(4)]
    meta = kv_first_write(fb, uids=uids, dp_client=client, partition_id="train")
    # Every tensor field in the input lands in TQ under f"{uid}_g0".
    assert meta.keys == [f"u{i}_g0" for i in range(4)]
    fetched = client.kv_batch_get(
        keys=meta.keys,
        partition_id="train",
        select_fields=["input_ids", "input_lengths", "token_mask", "sample_mask"],
    )
    assert fetched["input_ids"].shape == (4, 8)


def test_kv_first_write_carries_multimodal_extras():
    """VLM extras (pixel_values) ride along with no schema declaration."""
    client = NoOpDataPlaneClient()
    _setup_partition(client, num_samples=4)
    fb = _final_batch(4, with_extras=True)
    uids = [f"u{i}" for i in range(4)]
    meta = kv_first_write(fb, uids=uids, dp_client=client, partition_id="train")
    assert "pixel_values" in (meta.fields or [])
    fetched = client.kv_batch_get(
        keys=meta.keys,
        partition_id="train",
        select_fields=["pixel_values"],
    )
    assert fetched["pixel_values"].shape == (4, 3, 4, 4)


def test_kv_first_write_packs_routed_experts_like_input_ids():
    client = NoOpDataPlaneClient()
    _setup_partition(client, num_samples=4)
    fb = _final_batch(4)
    lengths = torch.tensor([8, 6, 8, 5], dtype=torch.long)
    fb["input_lengths"] = lengths
    fb["routed_experts"] = torch.arange(4 * 8 * 2 * 2, dtype=torch.int32).reshape(
        4, 8, 2, 2
    )
    uids = [f"u{i}" for i in range(4)]

    meta = kv_first_write(fb, uids=uids, dp_client=client, partition_id="train")

    rows = client._partitions["train"].rows
    for i, key in enumerate(meta.keys):
        seq_len = int(lengths[i].item())
        assert rows[key]["input_ids"].shape == (seq_len,)
        assert rows[key]["routed_experts"].shape == (seq_len, 2, 2)
        assert torch.equal(rows[key]["input_ids"], fb["input_ids"][i, :seq_len])
        assert torch.equal(
            rows[key]["routed_experts"],
            fb["routed_experts"][i, :seq_len],
        )


def test_routed_experts_jagged_materialize_preserves_trailing_dims():
    lengths = torch.tensor([3, 5], dtype=torch.long)
    input_ids = torch.arange(2 * 5, dtype=torch.long).reshape(2, 5)
    routed_experts = torch.arange(2 * 5 * 2 * 2, dtype=torch.int32).reshape(2, 5, 2, 2)
    td = TensorDict(
        {
            "input_ids": maybe_pack_jagged(input_ids, lengths),
            "routed_experts": maybe_pack_jagged(routed_experts, lengths),
        },
        batch_size=[2],
    )

    materialized = materialize(
        td,
        pad_value_dict={"input_ids": -1},
        pad_to_multiple=4,
    )

    assert materialized["input_ids"].shape == (2, 8)
    assert materialized["routed_experts"].shape == (2, 8, 2, 2)
    assert torch.equal(materialized["input_ids"][0, :3], input_ids[0, :3])
    assert torch.equal(materialized["input_ids"][0, 3:], torch.full((5,), -1))
    assert torch.equal(materialized["routed_experts"][0, :3], routed_experts[0, :3])
    assert torch.equal(materialized["routed_experts"][1, :5], routed_experts[1, :5])


def test_kv_first_write_keys_match_uids_x_ngen():
    """Keys are f"{uid}_g{i}"; n_gen inferred from sample_mask shape vs uids."""
    client = NoOpDataPlaneClient()
    _setup_partition(client, num_samples=6)
    fb = _final_batch(6)  # 3 prompts × 2 generations
    uids = ["a", "b", "c"]
    meta = kv_first_write(fb, uids=uids, dp_client=client, partition_id="train")
    assert meta.keys == ["a_g0", "a_g1", "b_g0", "b_g1", "c_g0", "c_g1"]


def _flat_rollout_batch(*, with_routed_experts: bool) -> BatchedDataDict:
    flat: BatchedDataDict = BatchedDataDict()
    flat["token_ids"] = torch.zeros((2, 4), dtype=torch.long)
    flat["generation_logprobs"] = torch.zeros((2, 4), dtype=torch.float32)
    flat["token_loss_mask"] = torch.ones((2, 4), dtype=torch.bool)
    if with_routed_experts:
        flat["routed_experts"] = torch.zeros((2, 4, 3, 2), dtype=torch.int32)
    return flat


def test_build_rollout_bulk_batch_requires_routed_experts_when_enabled():
    flat = _flat_rollout_batch(with_routed_experts=False)

    try:
        _build_rollout_bulk_batch(
            flat,
            input_lengths=torch.tensor([4, 3], dtype=torch.long),
            sample_mask=torch.ones((2,), dtype=torch.bool),
            router_replay_enabled=True,
        )
    except RuntimeError as exc:
        assert "requires routed_experts" in str(exc)
    else:
        raise AssertionError("Expected missing routed_experts to raise")


def test_build_rollout_bulk_batch_drops_routed_experts_when_disabled():
    flat = _flat_rollout_batch(with_routed_experts=True)

    bulk = _build_rollout_bulk_batch(
        flat,
        input_lengths=torch.tensor([4, 3], dtype=torch.long),
        sample_mask=torch.ones((2,), dtype=torch.bool),
        router_replay_enabled=False,
    )

    assert list(bulk.keys()) == list(ROLLOUT_SEED_FIELDS)
    assert ROUTED_EXPERTS_FIELD not in bulk


def test_build_rollout_bulk_batch_includes_routed_experts_when_enabled():
    flat = _flat_rollout_batch(with_routed_experts=True)

    bulk = _build_rollout_bulk_batch(
        flat,
        input_lengths=torch.tensor([4, 3], dtype=torch.long),
        sample_mask=torch.ones((2,), dtype=torch.bool),
        router_replay_enabled=True,
    )

    expected_fields = fields_with_optional_routed_experts(
        ROLLOUT_SEED_FIELDS,
        enabled=True,
    )
    assert list(bulk.keys()) == expected_fields
    assert torch.equal(bulk[ROUTED_EXPERTS_FIELD], flat[ROUTED_EXPERTS_FIELD])


# ── optional router replay field selection ────────────────────────────


def test_rollout_seed_fields_are_dp_seed_subset():
    assert set(ROLLOUT_SEED_FIELDS).issubset(DP_TRAIN_FIELDS)


def test_fields_with_optional_routed_experts_disabled():
    fields = fields_with_optional_routed_experts(LP_SEED_FIELDS, enabled=False)
    assert fields == list(LP_SEED_FIELDS)
    assert ROUTED_EXPERTS_FIELD not in fields


def test_fields_with_optional_routed_experts_enabled_once():
    fields = fields_with_optional_routed_experts(DP_TRAIN_FIELDS, enabled=True)
    assert fields[-1] == ROUTED_EXPERTS_FIELD
    assert fields.count(ROUTED_EXPERTS_FIELD) == 1

    fields = fields_with_optional_routed_experts(fields, enabled=True)
    assert fields.count(ROUTED_EXPERTS_FIELD) == 1


# ── shard_meta_for_dp invariants ──────────────────────────────────────


def _meta(n: int) -> KVBatchMeta:
    return KVBatchMeta(
        partition_id="train",
        task_name="train",
        keys=[f"k{i}" for i in range(n)],
        fields=list(DP_TRAIN_FIELDS),
        sequence_lengths=[10 + i for i in range(n)],
        extra_info={},
    )


def test_shard_meta_for_dp_partitions_keys_disjointly():
    n, dp = 8, 4
    metas, _ = shard_meta_for_dp(_meta(n), dp_world=dp, batch_size=n)
    assert len(metas) == dp
    flat = [k for m in metas for k in m.keys]
    assert sorted(flat) == sorted(_meta(n).keys)  # same set, no dups, no minting


def test_shard_meta_for_dp_preserves_partition_id():
    metas, _ = shard_meta_for_dp(_meta(4), dp_world=2, batch_size=4)
    assert all(m.partition_id == "train" for m in metas)


def test_shard_meta_for_dp_unsorted_round_trip():
    """unsorted_indices must reconstruct the input order from DP-rank concat."""
    n, dp = 8, 4
    metas, unsorted = shard_meta_for_dp(_meta(n), dp_world=dp, batch_size=n)
    if unsorted is None:
        # No reorder happened — DP-rank concat IS the original order.
        return
    # Build a tensor whose row i is i; permute via dispatch order; reorder back.
    flat = [k for m in metas for k in m.keys]
    aggregated = torch.tensor([_meta(n).keys.index(k) for k in flat])
    restored = aggregated[torch.tensor(unsorted)]
    assert restored.tolist() == list(range(n))


# ── meta utility helpers ──────────────────────────────────────────────


def test_kvbatchmeta_subset_filters_keys_and_seqlens():
    m = _meta(6)
    sub = m.subset([1, 3, 5])
    assert sub.keys == ["k1", "k3", "k5"]
    assert sub.sequence_lengths == [11, 13, 15]
    assert sub.partition_id == m.partition_id


def test_kvbatchmeta_concat_joins_keys_and_seqlens():
    m1 = _meta(3)
    m2 = _meta(6).subset([3, 4, 5])
    j = m1.concat(m2)
    assert j.keys == ["k0", "k1", "k2", "k3", "k4", "k5"]
    assert j.sequence_lengths == [10, 11, 12, 13, 14, 15]


def test_kvbatchmeta_slice_takes_range():
    m = _meta(5)
    s = m.slice(1, 4)
    assert s.keys == ["k1", "k2", "k3"]
    assert s.sequence_lengths == [11, 12, 13]


def test_kvbatchmeta_concat_rejects_partition_mismatch():
    import pytest

    m1 = _meta(2)
    m2 = KVBatchMeta(
        partition_id="other",
        task_name="train",
        keys=["x", "y"],
        fields=None,
        sequence_lengths=[1, 2],
    )
    with pytest.raises(ValueError, match=r"partition_ids must match"):
        m1.concat(m2)
