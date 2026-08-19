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
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_rl.algorithms.async_utils.replay_buffer import DataPlaneCheckpointBarrier
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import ROUTE_PLAN_TAG
from nemo_rl.experience.blackbox_finalizer import FinalizedGroup
from nemo_rl.experience.finalizer_actor import FinalizationRequest
from nemo_rl.experience.rollout_recovery import (
    PromptGroupStatus,
    RolloutRecoveryLedger,
)
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
        self.calls: list[FinalizationRequest] = []

    def remote(self, request: FinalizationRequest) -> Any:
        self.calls.append(request)

        async def _result():
            if self._error is not None:
                raise self._error
            assert self._result is not None
            return self._result

        return _result()


class _BlockingRemoteFinalize(_RemoteFinalize):
    def __init__(self, *, result: FinalizedGroup) -> None:
        super().__init__(result=result)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def remote(self, request: FinalizationRequest) -> Any:
        self.calls.append(request)

        async def _result() -> FinalizedGroup:
            self.started.set()
            await self.release.wait()
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


def _request() -> FinalizationRequest:
    return FinalizationRequest(
        group_id="group",
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
    ctrl._data_plane_checkpoint_barrier = DataPlaneCheckpointBarrier()
    ctrl._buffer = MagicMock()
    ctrl._buffer.commit_finalized = AsyncMock()
    ctrl._dp_client = _DataPlaneClient()
    ctrl._partition_id = "canonical"
    ctrl._master_config = SimpleNamespace(
        token_capture=SimpleNamespace(staging_partition="staging"),
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

    asyncio.run(ctrl._finalize_with_actor(request))

    assert finalize.calls == [request]
    assert ctrl._available_finalizers.get_nowait() is actor
    assert ctrl._active_finalizers == 0
    assert ctrl._finalizer_unknown_outcomes == 0
    ctrl._buffer.commit_finalized.assert_awaited_once_with(
        "group",
        meta,
        3,
        3,
        staging_keys=["group_g0/call"],
    )
    assert ctrl._finalizer_metrics_by_group["group"]["finalize/group_ms"] == 1.0


def test_checkpoint_waits_for_actor_publish_and_replay_index_commit() -> None:
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
        metrics={},
    )

    async def _scenario() -> list[str]:
        order: list[str] = []
        finalize = _BlockingRemoteFinalize(result=result)
        ctrl = _controller(SimpleNamespace(finalize=finalize))

        async def _commit(*args: Any, **kwargs: Any) -> None:
            order.append("commit")

        ctrl._buffer.commit_finalized = AsyncMock(side_effect=_commit)
        finalize_task = asyncio.create_task(ctrl._finalize_with_actor(_request()))
        await finalize.started.wait()

        async def _checkpoint() -> None:
            async with ctrl._data_plane_checkpoint_barrier.checkpoint():
                order.append("checkpoint")

        checkpoint_task = asyncio.create_task(_checkpoint())
        await asyncio.sleep(0)
        assert order == []

        finalize.release.set()
        await finalize_task
        await checkpoint_task
        return order

    assert asyncio.run(_scenario()) == ["commit", "checkpoint"]


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


def test_dropped_actor_group_cleans_known_canonical_and_staging_ownership() -> None:
    result = FinalizedGroup(
        meta=None,
        group_min_wv=3,
        group_max_wv=3,
        staging_keys=["group_g0/call"],
        metrics={},
        dropped=True,
    )
    actor = SimpleNamespace(finalize=_RemoteFinalize(result=result))
    ctrl = _controller(actor)

    with pytest.raises(RuntimeError, match="group group dropped"):
        asyncio.run(ctrl._finalize_with_actor(_request()))

    assert ctrl._dp_client.clear_calls == [
        {"sample_ids": ["group_g0"], "partition_id": "canonical"},
        {"sample_ids": ["group_g0/call"], "partition_id": "staging"},
    ]
    ctrl._buffer.abort.assert_called_once_with("group")
    assert ctrl._available_finalizers.get_nowait() is actor


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

    asyncio.run(ctrl._cleanup_consumed_metas([meta]))

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


def test_train_selection_transfers_finalized_group_to_open_step() -> None:
    meta = KVBatchMeta(
        partition_id="canonical",
        task_name="train",
        sample_ids=["group_g0"],
        fields=["input_ids"],
        sequence_lengths=[3],
        tags=[{"weight_version": 3}],
    )
    ctrl = _controller(SimpleNamespace())
    ledger = RolloutRecoveryLedger()
    ledger.adopt_finalized_group(
        group_id="group",
        meta=meta,
        start_weight_version=3,
        end_weight_version=3,
    )
    ctrl._rollout_recovery_ledger = ledger

    selected = ctrl._claim_train_meta(meta, num_groups=1)

    assert selected == ["group"]
    assert ledger.get_group("group").status == PromptGroupStatus.CLAIMED_FOR_TRAINING
    assert ledger.open_train_step is not None
    assert ledger.open_train_step.group_ids == ["group"]
