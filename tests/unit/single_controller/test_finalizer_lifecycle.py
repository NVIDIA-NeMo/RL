# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Lifecycle tests for actor-pool finalization and deferred-route ownership."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

from nemo_rl.algorithms.async_utils.replay_buffer import (
    DataPlaneCheckpointBarrier,
)
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils.config import SegmentRowsConfig
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import ROLLOUT_METRICS, ROUTE_PLAN_TAG
from nemo_rl.experience.rollout_reassembler import FinalizedGroup
from nemo_rl.experience.rollout_reassembler_actor import ReassemblyRequest
from nemo_rl.experience.route_plan import (
    ROUTE_PLAN_SCHEMA_VERSION,
    RouteAssemblyPlan,
    RouteSpan,
    encode_route_plan,
)


class _RemoteFinalize:
    def __init__(
        self,
        *,
        result: FinalizedGroup | None = None,
        error: BaseException | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.calls: list[ReassemblyRequest] = []

    def remote(self, request: ReassemblyRequest) -> Any:
        self.calls.append(request)

        async def _result():
            if self._error is not None:
                raise self._error
            assert self._result is not None
            return self._result

        return _result()


class _DataPlaneClient:
    def __init__(self) -> None:
        self.clear_calls: list[dict[str, Any]] = []

    async def clear_samples(self, *, sample_ids: list[str], partition_id: str) -> None:
        self.clear_calls.append(
            {"sample_ids": list(sample_ids), "partition_id": partition_id}
        )


def _request() -> ReassemblyRequest:
    return ReassemblyRequest(
        group_id="group",
        prompt_idx=17,
        rollout_ids=("group_g0",),
        canonical_sample_ids=("group_g0",),
        receipts=(
            {
                "manifest": [
                    {"staging_key": "group_g0/call"},
                    {"staging_key": "group_g0/call"},
                ]
            },
        ),
        rewards=(1.0,),
        mask_sample=(False,),
        fallback_weight_version=3,
    )


def _controller(actor: object) -> Any:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._available_finalizers = asyncio.Queue()
    ctrl._available_finalizers.put_nowait(actor)
    ctrl._active_finalizers = 0
    ctrl._finalizer_waiters = 0
    ctrl._finalizer_unknown_outcomes = 0
    ctrl._finalizer_metrics_by_group = {}
    ctrl._rollout_recovery_ledger = MagicMock()
    ctrl._rollout_recovery_ledger.__contains__.return_value = False
    ctrl._rollout_manager = MagicMock()
    ctrl._rollout_manager.pop_capture_rollout_metrics.return_value = {}
    ctrl._data_plane_checkpoint_barrier = DataPlaneCheckpointBarrier()
    ctrl._buffer = MagicMock()
    ctrl._buffer.commit_finalized = AsyncMock()
    ctrl._dp_client = _DataPlaneClient()
    ctrl._data_plane_checkpoint_barrier = DataPlaneCheckpointBarrier()
    ctrl._partition_id = "canonical"
    ctrl._master_config = SimpleNamespace(
        token_capture=SimpleNamespace(
            staging_partition="staging", segment_rows=SegmentRowsConfig()
        ),
        grpo=SimpleNamespace(num_prompts_per_step=1),
    )
    ctrl._trainer_version = 3
    ctrl._train_steps = 3
    return ctrl


def test_successful_actor_finalization_returns_actor_and_transfers_ownership() -> None:
    meta = KVBatchMeta(
        partition_id="canonical",
        task_name="train",
        sample_ids=["group_g0"],
        fields=["input_ids"],
        sequence_lengths=[3],
        tags=[{"weight_version": 3}],
    )
    result = FinalizedGroup(
        meta=meta,
        group_min_wv=3,
        group_max_wv=3,
        staging_keys=["group_g0/call"],
        metrics={"finalize/group_ms": 1.0},
    )
    finalize = _RemoteFinalize(result=result)
    actor = SimpleNamespace(finalize=finalize)
    ctrl = _controller(actor)
    request = _request()

    committed = asyncio.run(ctrl._finalize_with_actor(request))

    assert committed is result
    assert finalize.calls == [request]
    assert ctrl._available_finalizers.get_nowait() is actor
    assert ctrl._active_finalizers == 0
    assert ctrl._finalizer_unknown_outcomes == 0
    ctrl._buffer.commit_finalized.assert_awaited_once_with(
        ANY,
        "group",
        meta,
        3,
        3,
        staging_keys=["group_g0/call"],
    )
    assert ctrl._finalizer_metrics_by_group["group"]["finalize/group_ms"] == 1.0


def _privileged(ctrl: Any, env_info: dict[str, Any]) -> Any:
    """Give the controller a real privilege store and a dataset prompt lookup."""
    import torch

    from nemo_rl.algorithms import swe_privileged_critic as spc

    class _CharTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

        def decode(self, ids):
            return "".join(chr(i) for i in ids)

        def apply_chat_template(self, messages, **kwargs):
            return "".join(m["content"] for m in messages)

        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            return {"input_ids": torch.tensor([self.encode(text)])}

    ctrl._privilege_store = spc.SwePrivilegePrefixStore(
        _CharTokenizer(), spc.SwePrivilegedCriticConfig(enabled=True)
    )
    ctrl._dataset_prompt = AsyncMock(return_value={"extra_env_info": env_info})
    return spc


def _finalized_meta() -> KVBatchMeta:
    return KVBatchMeta(
        partition_id="canonical",
        task_name="train",
        sample_ids=["group_g0", "group_g0_t1"],
        fields=["input_ids"],
        sequence_lengths=[3, 2],
        tags=[{"weight_version": 3}, {"weight_version": 3}],
    )


def test_privileged_critic_stamps_finalized_rows_before_commit() -> None:
    result = FinalizedGroup(
        meta=_finalized_meta(),
        group_min_wv=3,
        group_max_wv=3,
        staging_keys=["group_g0/call"],
        metrics={},
    )
    ctrl = _controller(SimpleNamespace(finalize=_RemoteFinalize(result=result)))
    spc = _privileged(
        ctrl, {"instance_id": "repo__9", "patch": "diff --git a/z b/z\n+fix"}
    )

    asyncio.run(ctrl._finalize_with_actor(_request()))

    ctrl._dataset_prompt.assert_awaited_once_with(17)
    committed_meta = ctrl._buffer.commit_finalized.await_args.args[2]
    # Every row, segment rows included, references the instance's prefix.
    assert [tag[spc.PRIVILEGE_KEY_TAG] for tag in committed_meta.tags] == [
        "repo__9",
        "repo__9",
    ]
    assert committed_meta.sample_ids == ["group_g0", "group_g0_t1"]
    assert "repo__9" in ctrl._privilege_store.prefixes_for(committed_meta)


def test_privileged_critic_without_golden_patch_cleans_up_and_never_commits() -> None:
    result = FinalizedGroup(
        meta=_finalized_meta(),
        group_min_wv=3,
        group_max_wv=3,
        staging_keys=["group_g0/call"],
        metrics={},
    )
    ctrl = _controller(SimpleNamespace(finalize=_RemoteFinalize(result=result)))
    _privileged(ctrl, {"instance_id": "repo__9"})

    with pytest.raises(ValueError, match="no golden patch"):
        asyncio.run(ctrl._finalize_with_actor(_request()))

    ctrl._buffer.commit_finalized.assert_not_awaited()
    assert ctrl._dp_client.clear_calls


def test_actor_rpc_failure_is_fatal_and_does_not_retry_or_requeue_actor() -> None:
    finalize = _RemoteFinalize(error=RuntimeError("actor died after submission"))
    actor = SimpleNamespace(finalize=finalize)
    ctrl = _controller(actor)
    request = _request()

    with pytest.raises(RuntimeError, match="actor died after submission"):
        asyncio.run(ctrl._finalize_with_actor(request))

    assert finalize.calls == [request]
    assert ctrl._available_finalizers.empty()
    assert ctrl._active_finalizers == 0
    assert ctrl._finalizer_unknown_outcomes == 1
    ctrl._buffer.commit_finalized.assert_not_awaited()
    ctrl._buffer.abort.assert_not_called()
    assert ctrl._dp_client.clear_calls == []


def test_missing_actor_metadata_cleans_known_canonical_and_staging_ownership() -> None:
    result = FinalizedGroup(
        meta=None,
        group_min_wv=3,
        group_max_wv=3,
        staging_keys=["group_g0/call"],
        metrics={},
    )
    actor = SimpleNamespace(finalize=_RemoteFinalize(result=result))
    ctrl = _controller(actor)

    with pytest.raises(RuntimeError, match="no metadata for non-dropped group"):
        asyncio.run(ctrl._finalize_with_actor(_request()))

    assert ctrl._dp_client.clear_calls == [
        {"sample_ids": ["group_g0"], "partition_id": "canonical"},
        {"sample_ids": ["group_g0/call"], "partition_id": "staging"},
    ]
    ctrl._buffer.abort.assert_called_once_with("group")
    assert ctrl._available_finalizers.get_nowait() is actor


def test_dropped_actor_group_cleans_ownership_and_returns_uncommitted() -> None:
    # A finalizer-side structural drop (e.g. router replay with no routed
    # data yet) -- not a low valid-row fraction, which the controller now
    # decides, not the finalizer (see min_valid_fraction_per_group's removal
    # from RolloutReassembler).
    result = FinalizedGroup(
        meta=None,
        group_min_wv=3,
        group_max_wv=3,
        staging_keys=["group_g0/call"],
        metrics={},
        dropped=True,
        drop_reason=(
            "router replay on, no rollout carried routed_experts, "
            "and (L, K) is unknown yet"
        ),
    )
    actor = SimpleNamespace(finalize=_RemoteFinalize(result=result))
    ctrl = _controller(actor)

    committed = asyncio.run(ctrl._finalize_with_actor(_request()))

    assert committed is None
    assert ctrl._dp_client.clear_calls == [
        {"sample_ids": ["group_g0"], "partition_id": "canonical"},
        {"sample_ids": ["group_g0/call"], "partition_id": "staging"},
    ]
    ctrl._buffer.abort.assert_called_once_with("group")
    ctrl._buffer.commit_finalized.assert_not_awaited()
    assert "group" not in ctrl._finalizer_metrics_by_group
    assert ctrl._available_finalizers.get_nowait() is actor
    # The group's stashed rollout metrics are consumed even when it drops.
    ctrl._rollout_manager.pop_capture_rollout_metrics.assert_called_once_with("group")


def test_capture_rollout_metrics_ride_on_the_committed_meta() -> None:
    result = FinalizedGroup(
        meta=_finalized_meta(),
        group_min_wv=3,
        group_max_wv=3,
        staging_keys=["group_g0/call"],
        metrics={},
    )
    ctrl = _controller(SimpleNamespace(finalize=_RemoteFinalize(result=result)))
    metrics = {"agent/rollout_time_taken/mean": 12.5}
    ctrl._rollout_manager.pop_capture_rollout_metrics.return_value = metrics

    asyncio.run(ctrl._finalize_with_actor(_request()))

    ctrl._rollout_manager.pop_capture_rollout_metrics.assert_called_once_with("group")
    committed_meta = ctrl._buffer.commit_finalized.await_args.args[2]
    # Same sidecar the echo path's TQReplayBuffer.commit writes.
    assert committed_meta.extra_info[ROLLOUT_METRICS] == [metrics]
    assert committed_meta.sample_ids == result.meta.sample_ids
    assert committed_meta.tags == result.meta.tags


def test_capture_generation_sizes_replace_the_receipt_time_proxy() -> None:
    result = FinalizedGroup(
        meta=_finalized_meta(),
        group_min_wv=3,
        group_max_wv=3,
        staging_keys=["group_g0/call"],
        metrics={},
        rollout_gen_tokens=(1200,),
        rollout_max_gen_tokens_per_turn=(700,),
    )
    ctrl = _controller(SimpleNamespace(finalize=_RemoteFinalize(result=result)))
    # Receipt-time proxy: manifest deltas, tool output included.
    ctrl._rollout_manager.pop_capture_rollout_metrics.return_value = {
        "mean_gen_tokens_per_sample": 5000.0,
        "gen_tokens_per_sample/mean": 5000.0,
        "agent/rollout_time_taken/mean": 12.5,
    }

    asyncio.run(ctrl._finalize_with_actor(_request()))

    (metrics,) = ctrl._buffer.commit_finalized.await_args.args[2].extra_info[
        ROLLOUT_METRICS
    ]
    assert metrics["mean_gen_tokens_per_sample"] == 1200
    assert metrics["gen_tokens_per_sample/mean"] == 1200
    assert metrics["max_gen_tokens_per_turn/max"] == 700
    assert metrics["agent/rollout_time_taken/mean"] == 12.5


def test_post_train_cleanup_clears_canonical_rows_and_route_plan_staging_keys() -> None:
    plan = RouteAssemblyPlan(
        schema_version=ROUTE_PLAN_SCHEMA_VERSION,
        staging_partition="staging",
        spans=(
            RouteSpan(
                staging_key="group_g0/call",
                carry_len=2,
                generation_len=1,
                staged_route_len=3,
                extras_digest_version=1,
                extras_digest="0" * 64,
            ),
        ),
        cleanup_staging_keys=("group_g0/call", "group_g0/fork"),
        expected_token_length=3,
    )
    meta = KVBatchMeta(
        partition_id="canonical",
        task_name="train",
        sample_ids=["group_g0", "group_g1"],
        fields=["input_ids"],
        tags=[
            {ROUTE_PLAN_TAG: encode_route_plan(plan)},
            {ROUTE_PLAN_TAG: encode_route_plan(plan)},
        ],
    )
    ctrl = _controller(SimpleNamespace())

    async def _cleanup() -> None:
        async with ctrl._data_plane_checkpoint_barrier.mutation() as cut:
            await ctrl._cleanup_consumed_metas_unlocked(cut, [meta])

    asyncio.run(_cleanup())

    assert ctrl._dp_client.clear_calls == [
        {
            "sample_ids": ["group_g0", "group_g1"],
            "partition_id": "canonical",
        },
        {
            "sample_ids": ["group_g0/call", "group_g0/fork"],
            "partition_id": "staging",
        },
    ]


class _SyncDataPlaneClient:
    """Synchronous client like the production TQ adapter; records caller threads."""

    def __init__(self) -> None:
        self.clear_calls: list[dict[str, Any]] = []
        self.clear_thread_ids: list[int] = []

    def clear_samples(self, *, sample_ids: list[str], partition_id: str) -> None:
        self.clear_thread_ids.append(threading.get_ident())
        self.clear_calls.append(
            {"sample_ids": list(sample_ids), "partition_id": partition_id}
        )


def test_known_outcome_cleanup_runs_sync_clears_off_the_event_loop() -> None:
    ctrl = _controller(SimpleNamespace())
    ctrl._dp_client = _SyncDataPlaneClient()

    async def _main() -> int:
        await ctrl._cleanup_known_finalization_request(_request())
        return threading.get_ident()

    loop_thread_id = asyncio.run(_main())

    assert ctrl._dp_client.clear_calls == [
        {"sample_ids": ["group_g0"], "partition_id": "canonical"},
        {"sample_ids": ["group_g0/call"], "partition_id": "staging"},
    ]
    assert ctrl._dp_client.clear_thread_ids
    assert all(tid != loop_thread_id for tid in ctrl._dp_client.clear_thread_ids)
    ctrl._buffer.abort.assert_called_once_with("group")
