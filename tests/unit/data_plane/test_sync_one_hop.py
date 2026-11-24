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
"""Sync 1-hop unit tests.

Coverage:
  * write_columns / read_columns roundtrip — catches async-without-await
    bugs (put_samples returning a coroutine instead of running). The
    test that didn't exist when the bug was introduced.
  * Per-sample key lifecycle — ``kv_first_write`` mints keys, every
    subsequent ``shard_meta_for_dp`` slice references the SAME key set
    (verl pattern, no re-minting).
"""

from __future__ import annotations

import pytest
import torch

from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.column_io import kv_first_write, read_columns, write_columns
from nemo_rl.data_plane.preshard import shard_meta_for_dp
from nemo_rl.data_plane.schema import DP_TRAIN_FIELDS
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

from ._rollout_shapes import (
    keys_from_uids,
    make_realistic_tags,
    make_rollout_batch,
    register_train_partition,
)


def _final_batch(n: int = 4) -> BatchedDataDict:
    d: BatchedDataDict = BatchedDataDict()
    d["input_ids"] = torch.arange(n * 8, dtype=torch.long).reshape(n, 8)
    d["input_lengths"] = torch.tensor([8] * n, dtype=torch.long)
    d["token_mask"] = torch.ones((n, 8), dtype=torch.long)
    d["sample_mask"] = torch.ones((n,), dtype=torch.long)
    d["generation_logprobs"] = torch.zeros((n, 8), dtype=torch.float32)
    return d


# ── write_columns / read_columns roundtrip ─────────────────────────────
#
# These tests would have caught the asyncio-without-await bug:
# put_samples used to be an async def; calling it without await
# silently dropped the coroutine. The roundtrip below would have
# returned an empty / stale tensor in that case.


def test_write_columns_lands_in_tq():
    client = NoOpDataPlaneClient()
    register_train_partition(client, num_samples=4)
    fb = _final_batch(4)
    uids = [f"u{i}" for i in range(4)]
    meta = kv_first_write(
        fb, sample_ids=keys_from_uids(uids), dp_client=client, partition_id="train"
    )

    # Driver delta-write: simulates advantage compute on the trainer.
    delta = {"advantages": torch.full((4,), 7.5)}
    write_columns(client, meta, delta)

    fetched = client.get_samples(
        sample_ids=meta.sample_ids,
        partition_id="train",
        select_fields=["advantages"],
    )
    assert torch.equal(fetched["advantages"], torch.full((4,), 7.5))


def test_read_columns_returns_only_requested_fields():
    client = NoOpDataPlaneClient()
    register_train_partition(client, num_samples=4)
    fb = _final_batch(4)
    uids = [f"u{i}" for i in range(4)]
    meta = kv_first_write(
        fb, sample_ids=keys_from_uids(uids), dp_client=client, partition_id="train"
    )

    bdd = read_columns(client, meta, ["input_ids", "input_lengths"])
    assert "input_ids" in bdd
    assert "input_lengths" in bdd
    # token_mask was written but not requested — must not be returned.
    assert "token_mask" not in bdd


def test_write_then_read_roundtrip_after_train_window():
    """Full lifecycle: rollout puts → driver delta-writes → read deltas back."""
    client = NoOpDataPlaneClient()
    register_train_partition(client, num_samples=4)
    fb = _final_batch(4)
    uids = [f"u{i}" for i in range(4)]
    meta = kv_first_write(
        fb, sample_ids=keys_from_uids(uids), dp_client=client, partition_id="train"
    )

    # Simulate the full sync 1-hop trainer-step writes:
    write_columns(
        client,
        meta,
        {
            "prev_logprobs": torch.full((4, 8), 0.1),
            "reference_policy_logprobs": torch.full((4, 8), 0.2),
            "advantages": torch.full((4,), 0.3),
        },
    )

    # train_presharded would fetch the union — verify all columns present.
    fetched = read_columns(
        client,
        meta,
        [
            "input_ids",
            "input_lengths",
            "prev_logprobs",
            "reference_policy_logprobs",
            "advantages",
        ],
    )
    assert torch.allclose(fetched["prev_logprobs"], torch.full((4, 8), 0.1))
    assert torch.allclose(fetched["reference_policy_logprobs"], torch.full((4, 8), 0.2))
    assert torch.allclose(fetched["advantages"], torch.full((4,), 0.3))


# ── Per-sample key lifecycle invariant ────────────────────────────────


def test_meta_keys_identity_across_dp_shards():
    """``shard_meta_for_dp`` must NOT mint new keys — every per-rank
    slice references a subset of the original ``meta.sample_ids``."""
    client = NoOpDataPlaneClient()
    register_train_partition(client, num_samples=8)
    fb = _final_batch(8)
    uids = [f"u{i}" for i in range(8)]
    meta = kv_first_write(
        fb, sample_ids=keys_from_uids(uids), dp_client=client, partition_id="train"
    )

    rank_metas, _ = shard_meta_for_dp(meta, dp_world=4, batch_size=8)
    flat = {k for m in rank_metas for k in m.sample_ids}
    assert flat == set(meta.sample_ids), (
        "shard_meta_for_dp introduced or dropped keys — should be a "
        "pure permutation of the original meta.sample_ids."
    )
    # Every rank slice points at the same partition.
    assert all(m.partition_id == meta.partition_id for m in rank_metas)


def test_kv_clear_uses_meta_keys_minted_at_rollout():
    """The keys cleared at step end are the SAME keys the rollout
    actor minted — no minting at any stage in between."""
    client = NoOpDataPlaneClient()
    register_train_partition(client, num_samples=4)
    fb = _final_batch(4)
    uids = [f"u{i}" for i in range(4)]
    meta = kv_first_write(
        fb, sample_ids=keys_from_uids(uids), dp_client=client, partition_id="train"
    )
    rollout_keys = list(meta.sample_ids)

    # Workers / driver write deltas — keys still meta.sample_ids.
    write_columns(client, meta, {"advantages": torch.zeros(4)})
    rank_metas, _ = shard_meta_for_dp(meta, dp_world=2, batch_size=4)
    for rm in rank_metas:
        for k in rm.sample_ids:
            assert k in set(rollout_keys), (
                "Rank meta references a key not in the original rollout set"
            )

    client.clear_samples(sample_ids=meta.sample_ids, partition_id="train")
    # Cleared keys should no longer fetch.
    import pytest

    with pytest.raises(KeyError):
        client.get_samples(
            sample_ids=meta.sample_ids,
            partition_id="train",
            select_fields=["input_ids"],
        )


# ── Multi-stage TQ lifecycle on a realistic batch ──
# Walks the per-step sequence:
#   1. register_partition → 2. kv_first_write (seed) → 3. stamp filter tags
#   → 4. worker logprob delta-writes → 5. driver advantage delta-write
#   → 6. full read of train fields → 7. clear_samples at step-end.
# Each stage uses data shaped like the real rollout writer's output
# (bf16 logprobs, int64 ids, int32 masks, realistic value distributions).


def test_full_sync_step_lifecycle_on_realistic_batch() -> None:
    """End-to-end TQ lifecycle test on a realistic batch."""

    _PARTITION = "train"
    client = NoOpDataPlaneClient()
    n = 8
    max_seqlen = 128

    # ── Stage 1: register partition with the schema rollout will write ──
    client.register_partition(
        partition_id=_PARTITION,
        fields=list(DP_TRAIN_FIELDS),
        num_samples=n,
        consumer_tasks=["prev_lp", "ref_lp", "train"],
    )

    # ── Stage 2: rollout writes seed fields via kv_first_write ──
    batch = make_rollout_batch(n=n, max_seqlen=max_seqlen, seed=101)
    uids = [f"u{i}" for i in range(n)]
    seed_fields = {
        "input_ids": batch["input_ids"],
        "input_lengths": batch["input_lengths"],
        "token_mask": batch["token_mask"],
        "sample_mask": batch["sample_mask"],
        "generation_logprobs": batch["generation_logprobs"],
    }
    final = BatchedDataDict(seed_fields)
    meta = kv_first_write(
        final,
        sample_ids=keys_from_uids(uids),
        dp_client=client,
        partition_id=_PARTITION,
    )
    # Sanity: meta carries the per-row lengths the driver needs for packing.
    assert meta.sequence_lengths is not None
    assert len(meta.sample_ids) == n
    # Bf16 logprob survives the put.
    seeded = client.get_samples(
        sample_ids=meta.sample_ids,
        partition_id=_PARTITION,
        select_fields=["generation_logprobs"],
    )
    assert seeded["generation_logprobs"].dtype == torch.bfloat16

    # ── Stage 3: driver stamps per-row tags (filter input for dyn sampling) ──
    tags = make_realistic_tags(n, zero_std_fraction=0.25, seed=101)
    meta.tags = tags
    assert sum(1 for t in tags if t["std"] == 0.0) == n // 4

    # ── Stage 4: workers compute logprob deltas, write back ──
    write_columns(
        client,
        meta,
        fields={
            "prev_logprobs": batch["prev_logprobs"],
            "reference_policy_logprobs": batch["reference_policy_logprobs"],
        },
    )

    # ── Stage 5: driver computes advantages, writes back ──
    write_columns(
        client,
        meta,
        fields={"advantages": batch["advantages"]},
    )

    # ── Stage 6: full read of train fields (what train_presharded does) ──
    full = read_columns(
        client,
        meta,
        select_fields=[
            "input_ids",
            "input_lengths",
            "token_mask",
            "sample_mask",
            "generation_logprobs",
            "prev_logprobs",
            "reference_policy_logprobs",
            "advantages",
        ],
    )
    # All fields present, dtypes preserved end-to-end.
    assert full["input_ids"].dtype == torch.long
    assert full["token_mask"].dtype == torch.int32
    assert full["generation_logprobs"].dtype == torch.bfloat16
    assert full["prev_logprobs"].dtype == torch.bfloat16
    assert full["reference_policy_logprobs"].dtype == torch.bfloat16
    assert full["advantages"].dtype == torch.bfloat16
    # Row count survives the full pipeline.
    assert full["input_ids"].shape[0] == n

    # ── Stage 7: step-end clear (mirror of finish_step) ──
    client.clear_samples(sample_ids=meta.sample_ids, partition_id=_PARTITION)
    # Subsequent get must fail loud — the keys are gone.
    with pytest.raises(KeyError):
        client.get_samples(
            sample_ids=[meta.sample_ids[0]],
            partition_id=_PARTITION,
            select_fields=["input_ids"],
        )
