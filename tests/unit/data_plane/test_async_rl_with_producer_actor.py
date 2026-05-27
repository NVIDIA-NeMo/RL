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
from tensordict import TensorDict

transfer_queue = pytest.importorskip("transfer_queue")  # noqa: F841

from nemo_rl.data_plane import build_data_plane_client  # noqa: E402
from nemo_rl.data_plane.preshard import shard_meta_for_dp  # noqa: E402
from nemo_rl.data_plane.schema import DP_TRAIN_FIELDS  # noqa: E402

from ._rollout_filters import StalenessFilter  # noqa: E402
from ._rollout_shapes import (  # noqa: E402
    keys_from_uids,
    make_realistic_tags,
    make_rollout_batch,
    register_train_partition,
)

import ray  # noqa: E402


# ── producer actor ────────────────────────────────────────────────────────────


@ray.remote  # pragma: no cover
class _ProducerActor:
    """Stand-in for ``AsyncTrajectoryCollector`` — writes rollout batches to TQ.

    Each ``produce`` call writes ``len(sample_ids)`` samples stamped with
    a single ``weight_version`` so the driver-side filter has something
    to discriminate on. The actor attaches to the existing TQ controller
    (initialized by the driver fixture) via ``tq.init()`` — same path
    production workers use.
    """

    def __init__(self) -> None:
        import transfer_queue as tq

        tq.init()  # no config → attach to existing controller

    def produce(
        self,
        partition_id: str,
        sample_ids: list[str],
        weight_version: int,
        max_seqlen: int = 16,
    ) -> list[str]:
        import transfer_queue as tq

        n = len(sample_ids)
        bulk = make_rollout_batch(n=n, max_seqlen=max_seqlen)
        tags = make_realistic_tags(n=n)
        for i, t in enumerate(tags):
            t["weight_version"] = weight_version
            t["input_lengths"] = int(bulk["input_lengths"][i])
        tq.kv_batch_put(
            keys=list(sample_ids),
            partition_id=partition_id,
            fields=TensorDict(bulk, batch_size=[n]),
            tags=tags,
        )
        return sample_ids


# ── driver fixture (mirrors test_tq_lifecycle.py::tq_client) ──────────────────


@pytest.fixture
def tq_client():
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


# ── test ──────────────────────────────────────────────────────────────────────


def test_async_rl_filter_with_producer_actor(tq_client) -> None:
    """Producer actor writes mixed-version rollouts → driver claims, filters, clears."""
    pid = "async-rl-flow"
    current_weight_version = 3  # max_age=1 → keep where v >= 2
    keys_stale = keys_from_uids([f"v1_p{i}" for i in range(4)])  # weight_version=1 → drop
    keys_fresh = keys_from_uids([f"v3_p{i}" for i in range(4)])  # weight_version=3 → keep
    n = len(keys_stale) + len(keys_fresh)

    # Driver registers the partition; producer attaches and writes.
    register_train_partition(tq_client, num_samples=n)

    # ── Pipeline (no asserts) ────────────────────────────────────────────────
    producer = _ProducerActor.remote()
    ray.get(
        [
            producer.produce.remote(pid, keys_stale, weight_version=1),
            producer.produce.remote(pid, keys_fresh, weight_version=current_weight_version),
        ]
    )
    sample_meta = tq_client.claim_meta(
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
    shard_fetches: list[Any] = [
        tq_client.get_data(s, select_fields=["input_ids"])
        for s in shards
        if s.size > 0
    ]
    tq_client.clear_samples(drop_meta.sample_ids, partition_id=pid)
    fully_consumed = tq_client.check_consumption_status(
        partition_id=pid, task_names=["train"]
    )
    survivors_td = tq_client.get_samples(
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

    # Teardown so the partition doesn't leak across tests.
    tq_client.clear_samples(keep_meta.sample_ids, partition_id=pid)
