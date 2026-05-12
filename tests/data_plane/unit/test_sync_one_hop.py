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
    bugs (kv_batch_put returning a coroutine instead of running). The
    test that didn't exist when the bug was introduced.
  * Per-sample key lifecycle — ``kv_first_write`` mints keys, every
    subsequent ``shard_meta_for_dp`` slice references the SAME key set
    (verl pattern, no re-minting).
  * Slice-only dynamic sampling — filter / cache-merge / overflow-slice
    on per-sample tensors plus ``meta.keys``.
"""

from __future__ import annotations

import torch

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.column_io import read_columns, write_columns
from nemo_rl.data_plane.preshard import DP_SEED_FIELDS, shard_meta_for_dp
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.sync_rollout_actor import kv_first_write


def _final_batch(n: int = 4) -> BatchedDataDict:
    d: BatchedDataDict = BatchedDataDict()
    d["input_ids"] = torch.arange(n * 8, dtype=torch.long).reshape(n, 8)
    d["input_lengths"] = torch.tensor([8] * n, dtype=torch.long)
    d["token_mask"] = torch.ones((n, 8), dtype=torch.long)
    d["sample_mask"] = torch.ones((n,), dtype=torch.long)
    d["generation_logprobs"] = torch.zeros((n, 8), dtype=torch.float32)
    return d


def _setup(client: NoOpDataPlaneClient, n: int) -> None:
    client.register_partition(
        partition_id="train",
        fields=list(DP_SEED_FIELDS),
        num_samples=n,
        consumer_tasks=["train"],
    )


# ── write_columns / read_columns roundtrip ─────────────────────────────
#
# These tests would have caught the asyncio-without-await bug:
# kv_batch_put used to be an async def; calling it without await
# silently dropped the coroutine. The roundtrip below would have
# returned an empty / stale tensor in that case.


def test_write_columns_lands_in_tq():
    client = NoOpDataPlaneClient()
    _setup(client, n=4)
    fb = _final_batch(4)
    uids = [f"u{i}" for i in range(4)]
    meta = kv_first_write(fb, uids=uids, dp_client=client, partition_id="train")

    # Driver delta-write: simulates advantage compute on the trainer.
    delta = {"advantages": torch.full((4,), 7.5)}
    write_columns(client, meta, delta)

    fetched = client.kv_batch_get(
        keys=meta.keys,
        partition_id="train",
        select_fields=["advantages"],
    )
    assert torch.equal(fetched["advantages"], torch.full((4,), 7.5))


def test_read_columns_returns_only_requested_fields():
    client = NoOpDataPlaneClient()
    _setup(client, n=4)
    fb = _final_batch(4)
    uids = [f"u{i}" for i in range(4)]
    meta = kv_first_write(fb, uids=uids, dp_client=client, partition_id="train")

    bdd = read_columns(client, meta, ["input_ids", "input_lengths"])
    assert "input_ids" in bdd
    assert "input_lengths" in bdd
    # token_mask was written but not requested — must not be returned.
    assert "token_mask" not in bdd


def test_write_then_read_roundtrip_after_train_window():
    """Full lifecycle: rollout puts → driver delta-writes → read deltas back."""
    client = NoOpDataPlaneClient()
    _setup(client, n=4)
    fb = _final_batch(4)
    uids = [f"u{i}" for i in range(4)]
    meta = kv_first_write(fb, uids=uids, dp_client=client, partition_id="train")

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
    slice references a subset of the original ``meta.keys``."""
    client = NoOpDataPlaneClient()
    _setup(client, n=8)
    fb = _final_batch(8)
    uids = [f"u{i}" for i in range(8)]
    meta = kv_first_write(fb, uids=uids, dp_client=client, partition_id="train")

    rank_metas, _ = shard_meta_for_dp(meta, dp_world=4, batch_size=8)
    flat = {k for m in rank_metas for k in m.keys}
    assert flat == set(meta.keys), (
        "shard_meta_for_dp introduced or dropped keys — should be a "
        "pure permutation of the original meta.keys."
    )
    # Every rank slice points at the same partition.
    assert all(m.partition_id == meta.partition_id for m in rank_metas)


def test_kv_clear_uses_meta_keys_minted_at_rollout():
    """The keys cleared at step end are the SAME keys the rollout
    actor minted — no minting at any stage in between."""
    client = NoOpDataPlaneClient()
    _setup(client, n=4)
    fb = _final_batch(4)
    uids = [f"u{i}" for i in range(4)]
    meta = kv_first_write(fb, uids=uids, dp_client=client, partition_id="train")
    rollout_keys = list(meta.keys)

    # Workers / driver write deltas — keys still meta.keys.
    write_columns(client, meta, {"advantages": torch.zeros(4)})
    rank_metas, _ = shard_meta_for_dp(meta, dp_world=2, batch_size=4)
    for rm in rank_metas:
        for k in rm.keys:
            assert k in set(rollout_keys), (
                "Rank meta references a key not in the original rollout set"
            )

    client.kv_clear(keys=meta.keys, partition_id="train")
    # Cleared keys should no longer fetch.
    import pytest

    with pytest.raises(KeyError):
        client.kv_batch_get(
            keys=meta.keys,
            partition_id="train",
            select_fields=["input_ids"],
        )


# ── Slice-only dynamic sampling logic ─────────────────────────────────
#
# These exercise the private ``_apply_dynamic_sampling`` helper in
# grpo_sync.py without requiring a full trainer to spin up.


def _slice_data(rewards: list[float], stds: list[float]) -> BatchedDataDict:
    n = len(rewards)
    return BatchedDataDict(
        {
            "total_reward": torch.tensor(rewards, dtype=torch.float32),
            "std": torch.tensor(stds, dtype=torch.float32),
            "baseline": torch.zeros(n),
            "input_lengths": torch.tensor([8] * n, dtype=torch.long),
            "loss_multiplier": torch.ones(n),
            "truncated": torch.zeros(n, dtype=torch.bool),
            "length": torch.tensor([8] * n, dtype=torch.long),
            "prompt_ids_for_adv": torch.zeros(n, 4, dtype=torch.long),
        }
    )


def _seed_meta(client: NoOpDataPlaneClient, prefix: str, n: int) -> KVBatchMeta:
    """Stage n keys in TQ so kv_clear has something to remove."""
    _setup(client, n=n)
    fb = _final_batch(n)
    uids = [f"{prefix}{i}" for i in range(n)]
    return kv_first_write(fb, uids=uids, dp_client=client, partition_id="train")


def test_apply_dynamic_sampling_filters_zero_std():
    """Drops uids whose std == 0 and clears their TQ payload."""
    from nemo_rl.algorithms.grpo_sync import _apply_dynamic_sampling

    client = NoOpDataPlaneClient()
    meta = _seed_meta(client, "u", n=4)
    sd = _slice_data([1.0, 2.0, 3.0, 4.0], [0.5, 0.0, 0.5, 0.0])

    pm, ps, pur, complete, ds_metrics, _ = _apply_dynamic_sampling(
        meta=meta,
        slice_data=sd,
        pending_meta=None,
        pending_slice=None,
        pending_unfiltered_rewards=[],
        train_prompts_size=4,
        num_gen_batches=1,
        max_gen_batches=10,
        dp_client=client,
    )
    # Only 2 survivors → not complete (need 4).
    assert complete is False
    assert pm is not None and len(pm.keys) == 2
    # Surviving uids' total_reward is 1.0 and 3.0 (kept indices [0, 2]).
    assert torch.equal(ps["total_reward"], torch.tensor([1.0, 3.0]))
    assert ps["filtered_reward"] is ps["total_reward"] or torch.equal(
        ps["filtered_reward"], ps["total_reward"]
    )

    # Dropped uids' TQ payload was cleared.
    import pytest

    with pytest.raises(KeyError):
        client.kv_batch_get(
            keys=[meta.keys[1]],
            partition_id="train",
            select_fields=["input_ids"],
        )
    # Surviving uids' payload is still alive.
    survivors = client.kv_batch_get(
        keys=[meta.keys[0], meta.keys[2]],
        partition_id="train",
        select_fields=["input_ids"],
    )
    assert survivors["input_ids"].shape == (2, 8)


def test_apply_dynamic_sampling_completes_when_train_size_reached():
    """When pending cache reaches train_prompts_size, returns complete."""
    from nemo_rl.algorithms.grpo_sync import _apply_dynamic_sampling

    client = NoOpDataPlaneClient()
    meta = _seed_meta(client, "u", n=4)
    sd = _slice_data([1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5])

    pm, ps, _, complete, ds_metrics, unfiltered = _apply_dynamic_sampling(
        meta=meta,
        slice_data=sd,
        pending_meta=None,
        pending_slice=None,
        pending_unfiltered_rewards=[],
        train_prompts_size=4,
        num_gen_batches=1,
        max_gen_batches=10,
        dp_client=client,
    )
    assert complete is True
    assert pm is not None and len(pm.keys) == 4
    assert ds_metrics["dynamic_sampling_num_gen_batches"] == 1
    # Unfiltered rewards mirror the input (no filtering happened).
    assert torch.equal(unfiltered, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_apply_dynamic_sampling_overflow_slices_and_clears():
    """When the cache exceeds train_prompts_size, slice + kv_clear discards."""
    from nemo_rl.algorithms.grpo_sync import _apply_dynamic_sampling

    client = NoOpDataPlaneClient()
    meta = _seed_meta(client, "u", n=6)
    sd = _slice_data([1.0] * 6, [0.5] * 6)

    pm, ps, _, complete, ds_metrics, _ = _apply_dynamic_sampling(
        meta=meta,
        slice_data=sd,
        pending_meta=None,
        pending_slice=None,
        pending_unfiltered_rewards=[],
        train_prompts_size=4,  # only need 4; 2 should be discarded
        num_gen_batches=1,
        max_gen_batches=10,
        dp_client=client,
    )
    assert complete is True
    assert len(pm.keys) == 4
    assert ds_metrics.get("dynamic_sampling_num_discarded_valid_samples") == 2
    # Discarded uids (last 2) cleared from TQ.
    import pytest

    with pytest.raises(KeyError):
        client.kv_batch_get(
            keys=[meta.keys[4]],
            partition_id="train",
            select_fields=["input_ids"],
        )


def test_apply_dynamic_sampling_raises_on_max_gen_batches():
    """Exceeding dynamic_sampling_max_gen_batches must raise loudly."""
    from nemo_rl.algorithms.grpo_sync import _apply_dynamic_sampling

    client = NoOpDataPlaneClient()
    meta = _seed_meta(client, "u", n=2)
    sd = _slice_data([1.0, 2.0], [0.0, 0.0])  # all dropped

    import pytest

    with pytest.raises(ValueError, match=r"max_gen_batches"):
        _apply_dynamic_sampling(
            meta=meta,
            slice_data=sd,
            pending_meta=None,
            pending_slice=None,
            pending_unfiltered_rewards=[],
            train_prompts_size=4,
            num_gen_batches=11,
            max_gen_batches=10,  # exceeded
            dp_client=client,
        )
