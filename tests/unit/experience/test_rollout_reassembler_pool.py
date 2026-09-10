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

import asyncio
from unittest.mock import Mock

import pytest

from nemo_rl.experience.rollout_reassembler_pool import RolloutReassemblerPool


def test_contention_cancellation_and_unknown_outcome_quarantine():
    async def scenario():
        actor = object()
        pool = RolloutReassemblerPool([actor])
        lease = await pool.acquire()
        waiter = asyncio.create_task(pool.acquire())
        await asyncio.sleep(0)
        assert pool.active == 1
        assert pool.waiters == 1
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert pool.waiters == 0
        pool.release(lease)
        reused = await pool.acquire()
        assert reused.actor is actor
        reused.rpc_submitted = True
        pool.release(reused)
        assert pool.active == 0
        assert pool.available_count == 0
        assert pool.unknown_outcomes == 1

    asyncio.run(scenario())


def test_known_outcome_releases_waiter_and_records_queue_depth():
    async def scenario():
        pool = RolloutReassemblerPool([object()])
        lease = await pool.acquire()
        waiter = asyncio.create_task(pool.acquire())
        await asyncio.sleep(0)
        lease.rpc_submitted = True
        lease.outcome_known = True
        pool.release(lease)
        next_lease = await waiter
        assert next_lease.actor is lease.actor
        assert next_lease.queue_depth == 1
        assert next_lease.queue_wait_ms >= 0
        assert next_lease.active_actor_count == 1
        pool.release(next_lease)
        assert pool.unknown_outcomes == 0
        assert pool.available_count == 1

    asyncio.run(scenario())


def test_shutdown_wakes_waiters_and_terminates_quarantined_actors(monkeypatch):
    kill = Mock()
    monkeypatch.setattr("nemo_rl.experience.rollout_reassembler_pool.ray.kill", kill)

    async def scenario():
        actor = object()
        pool = RolloutReassemblerPool([actor])
        lease = await pool.acquire()
        lease.rpc_submitted = True
        pool.release(lease)
        waiter = asyncio.create_task(pool.acquire())
        await asyncio.sleep(0)
        pool.shutdown()
        pool.shutdown()
        with pytest.raises(RuntimeError, match="closed"):
            await waiter
        with pytest.raises(RuntimeError, match="closed"):
            await pool.acquire()
        assert pool.waiters == 0
        assert pool.available_count == 0
        kill.assert_called_once_with(actor, no_restart=True)

    asyncio.run(scenario())
