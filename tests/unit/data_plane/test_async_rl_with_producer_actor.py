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
    simple_backend_dp_config,
)


# ── producer actor ────────────────────────────────────────────────────────────


@ray.remote  # pragma: no cover
class _TQBackedAsyncCollector:
    """Stand-in for a not-yet-built TQ-backed async rollout actor.

    The production codebase has these pieces SEPARATELY but not fused:

    * ``AsyncTrajectoryCollector`` (``async_utils/trajectory_collector.py``)
      — the async rollout loop with weight-version tracking + pause/refit
      semantics. Today writes to an in-memory ``ReplayBuffer``, not TQ.
    * ``SyncRolloutActor`` (``experience/sync_rollout_actor.py``) — the TQ
      write path (``kv_first_write`` into a ``DataPlaneClient``). Today
      only used by sync GRPO.

    A real async-RL deployment that uses TQ as the rollout buffer would
    fuse these two: the async loop's weight-version state + sync's TQ
    write path. This class pretends that fusion exists so the integration
    test can pin its shape today.

    API surface (subset):

    * ``set_weight_version(v)`` — mirrors ``AsyncTrajectoryCollector``.
    * ``rollout_to_tq(partition_id, sample_ids)`` — mirrors
      ``SyncRolloutActor.rollout_to_tq``: one Ray RPC, single flat
      ``kv_first_write`` of every tensor field, returns the sample_ids
      that landed. Rollout content itself is mocked via
      ``make_rollout_batch`` / ``make_multi_turn_message_log`` — the
      test exercises the wire shape, not real generation.
    """

    def __init__(self) -> None:
        # Attach to the existing TQ controller (driver bootstrapped it).
        # ``bootstrap=False`` matches every production rollout-side caller
        # — ``sync_rollout_actor.py:94`` and ``worker_mixin.py:142``.
        self._dp_client = build_data_plane_client(
            simple_backend_dp_config(), bootstrap=False
        )
        self.current_weight_version: int = 0

    def set_weight_version(self, version: int) -> None:
        """Trainer-driven initial seed of the producer's version state.

        Mirrors the one-shot setup call at ``grpo.py:2652`` ("Ensure
        collector knows initial weight version"). For *subsequent*
        version changes (real refit events), prefer ``policy_refit`` —
        the producer should own its version transitions, not have them
        written from outside.
        """
        self.current_weight_version = int(version)

    def policy_refit(self, new_version: int) -> None:
        """Producer-side refit: swap in new weights, bump version.

        Models the real refit event (``grpo.py:3022-3023`` does
        ``set_weight_version`` + ``resume_after_refit``). Body delegates
        to ``set_weight_version`` so both paths share a single source
        of truth for the version write; when a fuller implementation
        adds weight reload / resume semantics, diverge here.
        """
        self.set_weight_version(new_version)

    def rollout_to_tq(
        self,
        partition_id: str,
        sample_ids: list[str],
        *,
        max_seqlen: int = 16,
    ) -> list[str]:
        """Mocked rollout + flat TQ write, mirroring ``SyncRolloutActor.rollout_to_tq``.

        Stamps each sample with ``self.current_weight_version`` — owned by
        the producer, updated only via ``set_weight_version``. The
        version never crosses the driver/producer boundary as a call arg.
        """
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
            t["weight_version"] = self.current_weight_version
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

    client = build_data_plane_client(simple_backend_dp_config())
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
    # Mirror the async-RL trainer pattern in grpo.py:2649 — fire-and-forget
    # rollout calls into the collector; the driver does NOT wait on them.
    # Version state is producer-owned: set_weight_version is the one-shot
    # initial seed (grpo.py:2652); subsequent transitions go through the
    # producer's policy_refit (grpo.py:3022-3023). Driver never writes
    # the version directly mid-stream. Ray serializes actor methods on
    # the main thread → keys_stale lands with v=1 and keys_fresh with v=3
    # after the refit. Trainer blocks on claim_meta(timeout_s=...) — the
    # way the production trainer blocks on replay_buffer.size().
    producer = _TQBackedAsyncCollector.remote()
    producer.set_weight_version.remote(1)
    producer.rollout_to_tq.remote(pid, keys_stale)
    producer.policy_refit.remote(current_weight_version)
    producer.rollout_to_tq.remote(pid, keys_fresh)

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

    # Tags must propagate onto both halves — a regression in meta.subset()
    # that dropped tags would leave the existing tag-by-id checks above
    # silently green (those read sample_meta, not keep/drop).
    assert keep_meta.tags is not None and len(keep_meta.tags) == keep_meta.size
    assert drop_meta.tags is not None and len(drop_meta.tags) == drop_meta.size

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
