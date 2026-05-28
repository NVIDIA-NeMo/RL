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
"""Async-RL filter flow with a separate Ray producer actor.

Closer to the production shape than a single-process test: a Ray actor
mirrors :class:`AsyncTrajectoryCollector` (writes rollout batches to TQ
on demand) while the driver claims → filters → DP-fans-out → clears.

Setup uses shared helpers from ``_rollout_shapes`` so the bulk-batch
schema (``DP_TRAIN_FIELDS``) matches what the sync trainer actually
writes — same shapes, same tag conventions.

Pipeline and verification are in distinct phases — the test body has
**no asserts inside the pipeline**; everything is checked in a
trailing ``Verify`` block. Easier to read as "what happens" vs
"what we require."
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

transfer_queue = pytest.importorskip("transfer_queue")  # noqa: F841

import numpy as np  # noqa: E402
import ray  # noqa: E402

from nemo_rl.data_plane import build_data_plane_client  # noqa: E402
from nemo_rl.data_plane.column_io import kv_first_write, read_columns  # noqa: E402
from nemo_rl.data_plane.preshard import shard_meta_for_dp  # noqa: E402
from nemo_rl.data_plane.schema import DP_TRAIN_FIELDS  # noqa: E402
from nemo_rl.distributed.batched_data_dict import BatchedDataDict  # noqa: E402

from ._rollout_filters import StalenessFilter  # noqa: E402
from ._rollout_shapes import (  # noqa: E402
    keys_from_uids,
    make_multi_turn_message_log,
    make_realistic_tags,
    make_rollout_batch,
    register_train_partition,
)

# Shared TQ client config — driver fixture builds it; actor uses the same
# values so build_data_plane_client inside the Ray worker attaches to the
# existing controller (no new bootstrap).
_TQ_CLIENT_CONFIG = {
    "enabled": True,
    "impl": "transfer_queue",
    "backend": "simple",
    "storage_capacity": 1024,
    "num_storage_units": 1,
    "claim_meta_poll_interval_s": 0.5,
    "global_segment_size": 8589934592,
    "local_buffer_size": 1073741824,
}


# ── producer actor ────────────────────────────────────────────────────────────


@ray.remote  # pragma: no cover
class _ProducerActor:
    """Stand-in for ``AsyncTrajectoryCollector`` — writes rollout batches to TQ.

    Uses the same write path as ``SyncRolloutActor``: a ``DataPlaneClient``
    built via ``build_data_plane_client`` (which attaches to the existing
    TQ controller in the Ray worker process), then a single flat
    ``kv_first_write`` per produce call. Each call stamps one
    ``weight_version`` so the driver's filter has something to discriminate
    on.
    """

    def __init__(self) -> None:
        # Attach to the existing TQ controller (driver bootstrapped it).
        # Same dp_client surface SyncRolloutActor's workers hold.
        self._dp_client = build_data_plane_client(_TQ_CLIENT_CONFIG)

    def produce(
        self,
        partition_id: str,
        sample_ids: list[str],
        weight_version: int,
        max_seqlen: int = 16,
    ) -> list[str]:
        n = len(sample_ids)
        bulk = make_rollout_batch(n=n, max_seqlen=max_seqlen)
        # Add ``content`` as np.ndarray(object) of per-row message_logs —
        # matches SyncRolloutActor's production write shape (see
        # nemo_rl/experience/sync_rollout_actor.py:271-273). Object fields
        # ride through TQ via the codec's NonTensorStack path.
        content = np.empty(n, dtype=object)
        for i, ml in enumerate(make_multi_turn_message_log(n=n)):
            content[i] = ml
        bulk["content"] = content
        tags = make_realistic_tags(n=n)
        for i, t in enumerate(tags):
            t["weight_version"] = weight_version
            t["input_lengths"] = int(bulk["input_lengths"][i])
        # Same write helper SyncRolloutActor uses (column_io.py:123) —
        # single flat put_samples of every tensor field, jagged-packed.
        kv_first_write(
            BatchedDataDict(bulk),
            sample_ids=list(sample_ids),
            dp_client=self._dp_client,
            partition_id=partition_id,
            task_name="train",
            tags=tags,
        )
        return sample_ids


# ── driver fixture ────────────────────────────────────────────────────────────
# Named ``dp_client`` (not ``tq_client``) since the test exercises the
# backend-agnostic DataPlaneClient surface, not TQ specifics.


@pytest.fixture
def dp_client():
    if not ray.is_initialized():
        ray.init(local_mode=False, include_dashboard=False)

    client = build_data_plane_client(_TQ_CLIENT_CONFIG)
    yield client
    client.close()


# ── test ──────────────────────────────────────────────────────────────────────


def test_async_rl_filter_with_producer_actor(dp_client) -> None:
    """Producer actor writes mixed-version rollouts → driver claims, filters, clears."""
    pid = "async-rl-flow"
    current_weight_version = 3  # max_age=1 → keep where v >= 2
    keys_stale = keys_from_uids(
        [f"v1_p{i}" for i in range(4)]
    )  # weight_version=1 → drop
    keys_fresh = keys_from_uids(
        [f"v3_p{i}" for i in range(4)]
    )  # weight_version=3 → keep
    n = len(keys_stale) + len(keys_fresh)

    # Driver pre-registers the partition with pid (the helper defaults
    # to 'train' which would leave 'async-rl-flow' unregistered).
    register_train_partition(dp_client, num_samples=n, partition_id=pid)

    # ── Pipeline (no asserts) ────────────────────────────────────────────────
    producer = _ProducerActor.remote()
    ray.get(
        [
            producer.produce.remote(pid, keys_stale, weight_version=1),
            producer.produce.remote(
                pid, keys_fresh, weight_version=current_weight_version
            ),
        ]
    )
    sample_meta = dp_client.claim_meta(
        partition_id=pid,
        task_name="train",
        required_fields=list(DP_TRAIN_FIELDS),
        batch_size=n,
        timeout_s=30.0,
    )
    keep_meta, drop_meta = StalenessFilter(max_age=1).filter(
        sample_meta, current_weight_version=current_weight_version
    )
    shards, _ = shard_meta_for_dp(keep_meta, dp_world=2)
    # Same per-rank fetch helper grpo_sync.py uses (column_io.py:51) —
    # wraps get_samples + materialize (jagged unpack, object decode).
    shard_fetches: list[Any] = [
        read_columns(dp_client, s, select_fields=["input_ids"])
        for s in shards
        if s.size > 0
    ]
    dp_client.clear_samples(drop_meta.sample_ids, partition_id=pid)
    fully_consumed = dp_client.check_consumption_status(
        partition_id=pid, task_names=["train"]
    )
    survivors_td = dp_client.get_samples(
        keep_meta.sample_ids, partition_id=pid, select_fields=["input_ids"]
    )

    # ── Verify ───────────────────────────────────────────────────────────────
    assert sample_meta.size == n
    assert sample_meta.tags is not None and len(sample_meta.tags) == n

    # Tag/seqlen alignment by sample_id — survives any TQ default-sampler
    # change (positional alignment would not).
    tags_by_id = dict(zip(sample_meta.sample_ids, sample_meta.tags))
    for k in keys_stale:
        assert tags_by_id[k]["weight_version"] == 1
    for k in keys_fresh:
        assert tags_by_id[k]["weight_version"] == current_weight_version

    assert keep_meta.size == len(keys_fresh)
    assert drop_meta.size == len(keys_stale)
    assert set(keep_meta.sample_ids) == set(keys_fresh)
    assert set(drop_meta.sample_ids) == set(keys_stale)

    # DP shards cover keep set exactly — Counter catches dup-and-drop swaps.
    flat = [s for shard in shards for s in shard.sample_ids]
    assert Counter(flat) == Counter(keep_meta.sample_ids)

    # Every non-empty shard materializes via the same get_data path the
    # sync trainer's workers use.
    assert sum(td["input_ids"].shape[0] for td in shard_fetches) == keep_meta.size

    # Cursor advanced on all n in a single claim_meta — check via the
    # consumption signal directly (claim_meta's polling-mode empty-return
    # would pass for the wrong reason).
    assert fully_consumed

    # Clear targeted only drop set → keep set still fetchable.
    assert survivors_td["input_ids"].shape[0] == keep_meta.size

    # Partition / actor cleanup is the conftest's job: session-scope
    # init_ray_cluster (tests/unit/conftest.py:380) calls ray.shutdown()
    # at the end of the session, and the dp_client fixture's client.close()
    # kills the TQ controller actor.
