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
"""Cancellation must not separate storage deletion from SC ownership release."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_rl.algorithms.async_utils.replay_buffer import DataPlaneCheckpointBarrier
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.async_utils import complete_before_cancellation


class _GatedClient:
    async_io_enabled = True

    def __init__(self):
        self.entered = {name: asyncio.Event() for name in ("canonical", "staging")}
        self.release = {name: asyncio.Event() for name in self.entered}
        self.rows = {name: {"sample"} for name in self.entered}
        self.failure_partition = None

    def clear_samples(self, **kwargs):
        raise AssertionError("enabled async cleanup must not use the sync API")

    async def async_clear_samples(self, *, sample_ids, partition_id):
        async def clear():
            self.entered[partition_id].set()
            await self.release[partition_id].wait()
            if partition_id == self.failure_partition:
                raise RuntimeError("storage clear failed")
            self.rows[partition_id].difference_update(sample_ids)

        await complete_before_cancellation(clear())


class _Ledger(dict):
    def discard_group(self, cut, group_id):
        cut.require_live()
        self.pop(group_id)


def _controller():
    cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(cls)
    ctrl._dp_client = _GatedClient()
    ctrl._data_plane_checkpoint_barrier = DataPlaneCheckpointBarrier()
    ctrl._buffer = MagicMock()
    ctrl._rollout_recovery_ledger = _Ledger(group=True)
    ctrl._partition_id = "canonical"
    ctrl._master_config = SimpleNamespace(
        token_capture=SimpleNamespace(staging_partition="staging")
    )
    ctrl._finalizer_metrics_by_group = {"group": {"metric": 1}}
    ctrl._finalizer_rewards_by_group = {"group": [1]}
    ctrl._buffer_capacity = asyncio.Semaphore(0)
    return ctrl


def _request():
    return SimpleNamespace(
        group_id="group",
        canonical_sample_ids=("sample",),
        receipts=({"manifest": [{"staging_key": "sample"}]},),
    )


async def _post_train_cleanup(ctrl):
    metas = [
        KVBatchMeta(partition_id=name, task_name="train", sample_ids=["sample"])
        for name in ("canonical", "staging")
    ]
    async with ctrl._data_plane_checkpoint_barrier.mutation("sample_clears") as cut:
        await ctrl._release_consumed_metas_unlocked(cut, metas, ["claim"], {"group"}, 2)


@pytest.mark.parametrize("cleanup_kind", ["known_request", "post_train"])
@pytest.mark.parametrize("cancel_partition", ["canonical", "staging"])
def test_cancelled_cleanup_releases_ownership_before_checkpoint(
    cleanup_kind, cancel_partition
):
    async def scenario():
        ctrl = _controller()
        client = ctrl._dp_client
        if cancel_partition == "staging":
            client.release["canonical"].set()
        cleanup = (
            ctrl._cleanup_known_finalization_request(_request())
            if cleanup_kind == "known_request"
            else _post_train_cleanup(ctrl)
        )
        task = asyncio.create_task(cleanup)
        await client.entered[cancel_partition].wait()
        checkpoint_entered = asyncio.Event()

        async def checkpoint():
            async with ctrl._data_plane_checkpoint_barrier.checkpoint():
                checkpoint_entered.set()
                assert client.rows == {"canonical": set(), "staging": set()}
                if cleanup_kind == "known_request":
                    ctrl._buffer.abort.assert_called_once_with("group")
                    assert not ctrl._rollout_recovery_ledger
                else:
                    ctrl._buffer.release_training_claims.assert_called_once_with(
                        ["claim"]
                    )
                    assert ctrl._buffer_capacity._value == 2
                assert not ctrl._finalizer_metrics_by_group
                assert not ctrl._finalizer_rewards_by_group

        snapshot = asyncio.create_task(checkpoint())
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert not checkpoint_entered.is_set()
            assert ctrl._data_plane_checkpoint_barrier._active_mutations == 1
        for event in client.release.values():
            event.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await snapshot
        assert checkpoint_entered.is_set()
        assert ctrl._data_plane_checkpoint_barrier._active_mutations == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("cleanup_kind", ["known_request", "post_train"])
def test_failed_cleanup_preserves_ownership_and_propagates_error(cleanup_kind):
    async def scenario():
        ctrl = _controller()
        ctrl._dp_client.failure_partition = "staging"
        for event in ctrl._dp_client.release.values():
            event.set()
        with pytest.raises(ExceptionGroup, match="cleanup failed"):
            if cleanup_kind == "known_request":
                await ctrl._cleanup_known_finalization_request(_request())
            else:
                await _post_train_cleanup(ctrl)
        ctrl._buffer.abort.assert_not_called()
        ctrl._buffer.release_training_claims.assert_not_called()
        assert ctrl._rollout_recovery_ledger == {"group": True}
        assert "group" in ctrl._finalizer_metrics_by_group
        assert "group" in ctrl._finalizer_rewards_by_group
        assert ctrl._buffer_capacity._value == 0
        assert ctrl._dp_client.rows == {"canonical": set(), "staging": {"sample"}}
        assert ctrl._data_plane_checkpoint_barrier._active_mutations == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("capability", [False, MagicMock()])
def test_baseline_or_remote_like_capability_does_not_enable_transaction_drain(
    monkeypatch, capability
):
    async def scenario():
        ctrl = _controller()
        ctrl._dp_client.async_io_enabled = capability
        started = asyncio.Event()

        async def cleanup(cut, request):
            cut.require_live()
            started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(
            ctrl, "_cleanup_known_finalization_request_transaction", cleanup
        )
        task = asyncio.create_task(ctrl._cleanup_known_finalization_request(_request()))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert ctrl._data_plane_checkpoint_barrier._active_mutations == 0

    asyncio.run(scenario())
