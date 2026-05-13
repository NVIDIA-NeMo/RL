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
"""Correctness invariants for the sync 1-hop data-plane.

Each test guards a real bug we either hit (Mapping check, tensordict
import, kv_clear ordering) or could silently introduce. Tests target
the ABC contract through ``NoOpDataPlaneClient``, so they run without
TQ installed.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.column_io import read_columns, write_columns
from nemo_rl.data_plane.interfaces import KVBatchMeta
from nemo_rl.data_plane.preshard import shard_meta_for_dp
from nemo_rl.data_plane.schema import DP_TRAIN_FIELDS
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.sync_rollout_actor import kv_first_write

# ── helpers ────────────────────────────────────────────────────────────


def _final_batch(n: int = 4, *, with_image: bool = False) -> BatchedDataDict:
    d: BatchedDataDict = BatchedDataDict()
    d["input_ids"] = torch.arange(n * 8, dtype=torch.long).reshape(n, 8)
    d["input_lengths"] = torch.tensor([8] * n, dtype=torch.long)
    d["token_mask"] = torch.ones((n, 8), dtype=torch.long)
    d["sample_mask"] = torch.ones((n,), dtype=torch.long)
    d["generation_logprobs"] = torch.zeros((n, 8), dtype=torch.float32)
    if with_image:
        # Multimodal extras — exercises the "any tensor field" branch
        # in kv_first_write.
        d["image_features"] = torch.randn((n, 16, 32), dtype=torch.bfloat16)
    return d


def _setup(client: NoOpDataPlaneClient, n: int, *, fields=None) -> None:
    client.register_partition(
        partition_id="train",
        fields=list(fields if fields is not None else DP_TRAIN_FIELDS),
        num_samples=n,
        consumer_tasks=["train"],
    )


# ── fail-loud invariants ───────────────────────────────────────────────


def test_kv_batch_get_after_clear_raises() -> None:
    """Real bug guard: v3 driver tried to read input_ids for log_data
    AFTER kv_clear, hit ``ValueError: keys not found``. We now stash
    before clear — this test pins the contract that get-after-clear
    must fail loud, not silently return empty."""
    client = NoOpDataPlaneClient()
    _setup(client, n=2)
    fb = _final_batch(2)
    meta = kv_first_write(fb, uids=["a", "b"], dp_client=client, partition_id="train")

    client.kv_clear(keys=meta.keys, partition_id="train")

    with pytest.raises(KeyError):
        # NoOp raises KeyError when the partition entry is gone.
        client.kv_batch_get(
            keys=meta.keys,
            partition_id="train",
            select_fields=["input_ids"],
        )


def test_kv_batch_get_unproduced_field_raises() -> None:
    """Mid-pipeline guard: requesting a field that no producer has
    written must fail loud, not return zeros / silently skip."""
    client = NoOpDataPlaneClient()
    _setup(client, n=2)
    fb = _final_batch(2)
    meta = kv_first_write(fb, uids=["a", "b"], dp_client=client, partition_id="train")

    # ``advantages`` has not been written yet (driver delta-write).
    with pytest.raises(KeyError):
        client.kv_batch_get(
            keys=meta.keys,
            partition_id="train",
            select_fields=["advantages"],
        )


def test_get_data_without_select_fields_raises() -> None:
    """P2 invariant — never silently fetch all fields."""
    client = NoOpDataPlaneClient()
    _setup(client, n=2)
    fb = _final_batch(2)
    kv_first_write(fb, uids=["a", "b"], dp_client=client, partition_id="train")

    bare_meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        keys=["a_g0", "b_g0"],
        fields=None,  # no fields on meta
    )
    with pytest.raises(ValueError, match=r"select_fields|fields"):
        client.get_data(bare_meta, select_fields=None)


def test_kv_batch_put_rejects_non_tensor_leaves() -> None:
    """P3 — no pickle on the bus. Adapters MUST reject non-tensor
    leaves so callers can't accidentally ship Python objects."""
    client = NoOpDataPlaneClient()
    _setup(client, n=2, fields=["input_ids", "metadata"])

    # Build a TensorDict that smuggles a non-tensor — bypass via
    # tensordict's NonTensorData where possible.
    from tensordict import NonTensorData

    bad_td = TensorDict(
        {
            "input_ids": torch.zeros((2, 4), dtype=torch.long),
            "metadata": NonTensorData(["a", "b"], batch_size=[2]),
        },
        batch_size=[2],
    )
    with pytest.raises(TypeError, match=r"non-tensor"):
        client.kv_batch_put(
            keys=["x_g0", "y_g0"],
            partition_id="train",
            fields=bad_td,
        )


def test_claim_meta_unregistered_task_raises() -> None:
    """Catches typo'd consumer task names early."""
    client = NoOpDataPlaneClient()
    client.register_partition(
        partition_id="train",
        fields=["input_ids"],
        num_samples=2,
        consumer_tasks=["lp"],
    )
    with pytest.raises(KeyError, match=r"task"):
        client.claim_meta(
            partition_id="train",
            task_name="trian",  # typo
            required_fields=["input_ids"],
            batch_size=2,
        )


# ── lifecycle invariants ───────────────────────────────────────────────


def test_kv_clear_with_none_drops_partition() -> None:
    """Step-end teardown must remove the partition entirely so the
    next step's register_partition starts clean."""
    client = NoOpDataPlaneClient()
    _setup(client, n=2)
    fb = _final_batch(2)
    meta = kv_first_write(fb, uids=["a", "b"], dp_client=client, partition_id="train")

    client.kv_clear(keys=None, partition_id="train")

    # Partition is gone — re-registering must succeed.
    _setup(client, n=2)


def test_double_register_partition_is_idempotent_overwrite() -> None:
    """Re-registering the same partition_id within a step (e.g. retry)
    must overwrite cleanly, not append fields."""
    client = NoOpDataPlaneClient()
    client.register_partition(
        partition_id="train",
        fields=["a"],
        num_samples=2,
        consumer_tasks=["t"],
    )
    client.register_partition(
        partition_id="train",
        fields=["b"],
        num_samples=4,
        consumer_tasks=["t"],
    )
    rec = client._partitions["train"]
    assert rec.fields == ["b"]
    assert rec.num_samples == 4


def test_check_consumption_status_only_true_when_all_consumed() -> None:
    """Authoritative cross-worker stage-done signal — must NOT lie
    when consumers haven't fetched yet."""
    client = NoOpDataPlaneClient()
    _setup(client, n=2)
    fb = _final_batch(2)
    meta = kv_first_write(fb, uids=["a", "b"], dp_client=client, partition_id="train")
    # No consumer has fetched yet.
    assert not client.check_consumption_status("train", ["train"])

    # Simulate the worker fetch.
    client.claim_meta(
        partition_id="train",
        task_name="train",
        required_fields=["input_ids"],
        batch_size=meta.size,
    )
    assert client.check_consumption_status("train", ["train"])


# ── per-DP shard invariants ────────────────────────────────────────────


def test_shard_meta_for_dp_partitions_keys_disjointly() -> None:
    """Sum of shard sizes == total, and pairwise disjoint.

    ``shard_meta_for_dp`` returns ``(list[KVBatchMeta], unsorted_indices)``;
    here we only care about the metas.
    """
    client = NoOpDataPlaneClient()
    _setup(client, n=8)
    fb = _final_batch(8)
    meta = kv_first_write(
        fb,
        uids=[f"u{i}" for i in range(8)],
        dp_client=client,
        partition_id="train",
    )

    shards, _ = shard_meta_for_dp(meta, dp_world=4, batch_size=8)
    assert len(shards) == 4
    assert sum(len(s.keys) for s in shards) == len(meta.keys)
    seen: set[str] = set()
    for s in shards:
        for k in s.keys:
            assert k not in seen, f"duplicate key {k!r} across DP shards"
            seen.add(k)
    assert seen == set(meta.keys)


def test_shard_meta_for_dp_keeps_partition_id() -> None:
    client = NoOpDataPlaneClient()
    _setup(client, n=4)
    fb = _final_batch(4)
    meta = kv_first_write(
        fb,
        uids=[f"u{i}" for i in range(4)],
        dp_client=client,
        partition_id="train",
    )
    shards, _ = shard_meta_for_dp(meta, dp_world=2, batch_size=4)
    for s in shards:
        assert s.partition_id == meta.partition_id
        assert s.task_name == meta.task_name


# ── multimodal / VLM extras ────────────────────────────────────────────


def test_kv_first_write_carries_multimodal_extras_through_tq() -> None:
    """End-to-end flow for VLM: image features must round-trip via TQ
    with original shape + dtype, not be silently dropped or coerced."""
    client = NoOpDataPlaneClient()
    fields = list(DP_TRAIN_FIELDS) + ["image_features"]
    client.register_partition(
        partition_id="train",
        fields=fields,
        num_samples=4,
        consumer_tasks=["train"],
    )
    fb = _final_batch(4, with_image=True)
    expected = fb["image_features"].clone()

    meta = kv_first_write(
        fb,
        uids=[f"u{i}" for i in range(4)],
        dp_client=client,
        partition_id="train",
    )
    assert "image_features" in meta.fields

    fetched = read_columns(client, meta, select_fields=["image_features"])
    got = fetched["image_features"]
    assert got.shape == expected.shape
    assert got.dtype == expected.dtype, (
        f"dtype drift: expected {expected.dtype}, got {got.dtype}"
    )
    assert torch.equal(got, expected)


# ── dtype preservation ─────────────────────────────────────────────────


def test_kv_batch_put_preserves_bf16_dtype() -> None:
    """Catches silent fp32 promotion in the put path."""
    client = NoOpDataPlaneClient()
    client.register_partition(
        partition_id="train",
        fields=["x"],
        num_samples=2,
        consumer_tasks=["train"],
    )
    x = torch.randn((2, 4), dtype=torch.bfloat16)
    td = TensorDict({"x": x}, batch_size=[2])
    client.kv_batch_put(keys=["a", "b"], partition_id="train", fields=td)

    out = client.kv_batch_get(
        keys=["a", "b"], partition_id="train", select_fields=["x"]
    )
    assert out["x"].dtype == torch.bfloat16


def test_kv_batch_put_preserves_int64_dtype() -> None:
    """input_ids is int64; never coerce to int32 silently."""
    client = NoOpDataPlaneClient()
    client.register_partition(
        partition_id="train",
        fields=["input_ids"],
        num_samples=2,
        consumer_tasks=["train"],
    )
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    td = TensorDict({"input_ids": x}, batch_size=[2])
    client.kv_batch_put(keys=["a", "b"], partition_id="train", fields=td)

    out = client.kv_batch_get(
        keys=["a", "b"],
        partition_id="train",
        select_fields=["input_ids"],
    )
    assert out["input_ids"].dtype == torch.long
    assert torch.equal(out["input_ids"], x)


# ── BatchedDataDict / Mapping check ────────────────────────────────────


def test_write_columns_accepts_batched_data_dict_input() -> None:
    """Real bug guard (job 11614968 v2 crash): worker write-back
    silently skipped because BatchedDataDict inherits from UserDict,
    not dict. The fix uses ``isinstance(result, Mapping)``; this test
    pins that contract.
    """
    client = NoOpDataPlaneClient()
    _setup(client, n=2)
    fb = _final_batch(2)
    meta = kv_first_write(fb, uids=["a", "b"], dp_client=client, partition_id="train")

    bdd = BatchedDataDict()
    bdd["advantages"] = torch.full((2,), 3.0)

    # write_columns accepts plain dict; the Mapping-check on the worker
    # side ensures BatchedDataDict (UserDict) also goes through.
    write_columns(client, meta, dict(bdd))

    out = read_columns(client, meta, select_fields=["advantages"])
    assert torch.equal(out["advantages"], torch.full((2,), 3.0))


# ── kv_first_write key-mint contract ────────────────────────────────────


def test_kv_first_write_rejects_indivisible_batch() -> None:
    """If the flattened batch isn't divisible by len(uids), keys would
    silently mis-align. Must fail loud."""
    client = NoOpDataPlaneClient()
    _setup(client, n=5)
    # 5 samples, 2 uids → not divisible by num_generations.
    fb = _final_batch(5)
    with pytest.raises(ValueError, match=r"divisible"):
        kv_first_write(
            fb,
            uids=["a", "b"],
            dp_client=client,
            partition_id="train",
        )


def test_kv_first_write_meta_sequence_lengths_match_input_lengths() -> None:
    """meta.sequence_lengths is consumed by Megatron's balanced packing
    on the driver — it MUST mirror final_batch.input_lengths."""
    client = NoOpDataPlaneClient()
    _setup(client, n=4)
    fb = _final_batch(4)
    fb["input_lengths"] = torch.tensor([3, 5, 7, 8], dtype=torch.long)

    meta = kv_first_write(
        fb,
        uids=[f"u{i}" for i in range(4)],
        dp_client=client,
        partition_id="train",
    )
    assert meta.sequence_lengths == [3, 5, 7, 8]
