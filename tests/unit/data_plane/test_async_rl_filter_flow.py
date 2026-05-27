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
"""Async-RL rollout-buffer filter pipeline — claim_meta + driver-side filter class.

Pins the recipe::

    meta = client.claim_meta(..., batch_size=GBS)        # 1. claim a batch
    keep, drop = rollout_filter.filter(meta, **state)    # 2. driver-side filter
    shards, _ = shard_meta_for_dp(keep, dp_world=N)      # 3. sync DP fan-out
    # ...workers fetch each shard via get_data...
    client.clear_samples(drop.sample_ids, ...)           # 4. drop the rejects

Two design choices that matter:

1. **Filter runs on the driver, not in TQ.** TQ ships ``BaseSampler``
   for server-side filtering (great when the predicate can be
   expressed from partition state + ``sampling_config`` alone). For
   filters that need in-process driver state (current weight version,
   dynamic configs, model-side scalars) the predicate lives on the
   driver. This mirrors verl's ``ReplayBuffer.sample()`` location
   choice.

2. **Filter is a class, not an inline lambda.** Mirrors TQ's
   :class:`BaseSampler` shape — easy to swap implementations
   (staleness / DAPO std==0 / format-check / length threshold) and
   trivially unit-testable in isolation. The example sits in
   ``_rollout_filters.py`` next to this test; not yet production code.

Sibling to ``test_tq_lifecycle.py``; same Ray + simple-backend fixture,
same mock-dataset shape.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch
from tensordict import TensorDict

transfer_queue = pytest.importorskip("transfer_queue")  # noqa: F841

from nemo_rl.data_plane import build_data_plane_client  # noqa: E402
from nemo_rl.data_plane.preshard import shard_meta_for_dp  # noqa: E402

from ._rollout_filters import BaseRolloutFilter, StalenessFilter  # noqa: E402


@pytest.fixture
def tq_client():
    """Single-node simple-backend TQ via the production nemo-rl entry point.

    Mirrors ``test_tq_lifecycle.py::tq_client``.
    """
    import ray

    if not ray.is_initialized():
        ray.init(local_mode=False, include_dashboard=False)

    client = build_data_plane_client(
        {
            "enabled": True,
            "impl": "transfer_queue",
            "backend": "simple",
            "storage_capacity": 1024,
            "num_storage_units": 1,
            "claim_meta_poll_interval_s": 0.5,
            "global_segment_size": 8589934592,
            "local_buffer_size": 1073741824,
        }
    )
    yield client
    client.close()


def test_async_rl_filter_train_clear_flow(tq_client) -> None:
    """claim → filter on tags → DP fan-out keep set → clear drop set."""
    pid = "async-rl-flow"
    n = 8
    # Varying sequence lengths so the sequence_lengths assertion below
    # actually pins ordering, not just identity-of-values.
    seq_lens = [1, 2, 3, 4, 5, 6, 7, 8]
    max_seq = max(seq_lens)
    sample_ids = [f"s{i}" for i in range(n)]
    # weight_versions: 0,0,1,1,2,2,3,3 — keyed by sample_id so we don't
    # depend on TQ's sampler returning samples in put-order.
    expected_versions = {sid: i // 2 for i, sid in enumerate(sample_ids)}
    expected_seqlen = {sid: seq_lens[i] for i, sid in enumerate(sample_ids)}

    # 1. Producer side: rollout actor writes samples and stamps a
    #    weight_version tag on every one. Half stale, half fresh.
    tq_client.register_partition(
        partition_id=pid,
        fields=["input_ids", "input_lengths"],
        num_samples=n,
        consumer_tasks=["train"],
    )
    # Right-pad input_ids to a rectangular (n, max_seq) tensor so the put
    # is uniform; input_lengths carries the real per-sample length.
    input_ids = torch.zeros(n, max_seq, dtype=torch.int64)
    for i, sl in enumerate(seq_lens):
        input_ids[i, :sl] = torch.arange(1, sl + 1, dtype=torch.int64)
    tq_client.put_samples(
        sample_ids=sample_ids,
        partition_id=pid,
        fields=TensorDict(
            {
                "input_ids": input_ids,
                "input_lengths": torch.tensor(seq_lens, dtype=torch.long),
            },
            batch_size=[n],
        ),
        tags=[
            {"weight_version": expected_versions[sid], "input_lengths": seq_lens[i]}
            for i, sid in enumerate(sample_ids)
        ],
    )

    # 2. Async trainer claims a batch. Mirrors verl's "training blocks
    #    until enough rollout is available" pattern — claim_meta blocks
    #    by default; timeout_s=30 matches the sibling test_tq_lifecycle
    #    convention and bounds the wait if production stalls.
    sample_meta = tq_client.claim_meta(
        partition_id=pid,
        task_name="train",
        required_fields=["input_ids", "input_lengths"],
        batch_size=n,
        timeout_s=30.0,
    )
    assert sample_meta.size == n
    assert sample_meta.tags is not None
    assert len(sample_meta.tags) == n
    assert sample_meta.sequence_lengths is not None
    assert len(sample_meta.sequence_lengths) == n

    # Verify tag/seqlen alignment by sample_id — does NOT depend on
    # TQ's sampler returning samples in put-order. If TQ ever ships a
    # non-Sequential default sampler, this still passes; the previous
    # positional check would have silently broken.
    for sid, tag, sl in zip(
        sample_meta.sample_ids, sample_meta.tags, sample_meta.sequence_lengths
    ):
        assert tag["weight_version"] == expected_versions[sid], (
            f"tag/sample_id misaligned for {sid}"
        )
        assert sl == expected_seqlen[sid], (
            f"sequence_lengths misaligned for {sid}: got {sl}, "
            f"expected {expected_seqlen[sid]}"
        )

    # 3. Driver-side filter — class-based predicate, swappable. Staleness
    #    here, but a DAPOStdFilter / FormatCheckFilter / length threshold
    #    would slot in identically — that's the point of BaseRolloutFilter.
    current_weight_version = 3  # keep where 3 - v <= max_age(1), i.e. v >= 2
    rollout_filter: BaseRolloutFilter = StalenessFilter(max_age=1)
    keep_meta, drop_meta = rollout_filter.filter(
        sample_meta, current_weight_version=current_weight_version
    )

    assert keep_meta.size == 4
    assert drop_meta.size == 4
    assert keep_meta.size + drop_meta.size == sample_meta.size
    # Tags must propagate so per-tag predicate checks aren't vacuous.
    assert keep_meta.tags is not None and len(keep_meta.tags) == keep_meta.size
    assert drop_meta.tags is not None and len(drop_meta.tags) == drop_meta.size
    for t in keep_meta.tags:
        assert current_weight_version - t["weight_version"] <= 1
    for t in drop_meta.tags:
        assert current_weight_version - t["weight_version"] > 1

    # 4. Sync trainer path: shard_meta_for_dp partitions keep_meta across
    #    DP ranks — same code the sync trainer uses post-claim_meta.
    dp_world = 2
    shards, _ = shard_meta_for_dp(keep_meta, dp_world=dp_world)
    assert len(shards) == dp_world
    flat = [s for shard in shards for s in shard.sample_ids]
    # Counter equality catches duplicates AND missing ids; set+len would
    # let a duplication-and-drop swap pass silently.
    assert Counter(flat) == Counter(keep_meta.sample_ids), (
        "DP shards must cover keep set exactly (no minted, dropped, or "
        "duplicated keys)"
    )

    # Each rank's shard materializes via the same get_data path as the
    # sync trainer — proves the meta is usable by workers.
    for shard in shards:
        if shard.size == 0:
            continue
        td = tq_client.get_data(shard, select_fields=["input_ids"])
        assert td["input_ids"].shape[0] == shard.size

    # 5. Drop the stale set — kv_clear removes bulk for filtered samples.
    tq_client.clear_samples(drop_meta.sample_ids, partition_id=pid)

    # 6. Cursor really did advance — check via the consumption-status
    #    signal directly rather than relying on claim_meta's polling-mode
    #    short-circuit (which would return empty regardless of cursor
    #    state when ready < batch_size).
    assert tq_client.check_consumption_status(
        partition_id=pid, task_names=["train"]
    ), (
        "all n samples were claimed in step 2; check_consumption_status "
        "must report the partition fully consumed by 'train'"
    )

    # 7. Keep set bulk is still fetchable (clear targeted only the drop set).
    survivors_td = tq_client.get_samples(
        keep_meta.sample_ids, partition_id=pid, select_fields=["input_ids"]
    )
    assert survivors_td["input_ids"].shape[0] == keep_meta.size

    # 8. Teardown: drop the keep set too so subsequent tests reusing this
    #    partition_id (or a flaky session-scope Ray cluster) don't inherit
    #    residual state.
    tq_client.clear_samples(keep_meta.sample_ids, partition_id=pid)
