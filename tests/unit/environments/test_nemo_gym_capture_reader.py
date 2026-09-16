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
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_rl.environments.nemo_gym_capture import (
    CaptureSnapshotRef,
    GymCaptureReader,
    acknowledge_gym_captures,
)

pytestmark = pytest.mark.nemo_gym


def _build(*, masked=False, redundant=False):
    return {
        "rebuilt_response": None if redundant else {"output": []},
        "mask_sample": masked,
        "_redundant_capture": redundant,
        "_capture_snapshot": {"snapshot_id": "frozen", "version": 7},
    }


@pytest.mark.parametrize("redundant", [False, True])
def test_only_acknowledged_owned_snapshot_is_retired(redundant):
    source = SimpleNamespace(drop=AsyncMock(return_value=True))
    reader = GymCaptureReader(source, owns_source=False)
    ref = reader.register("rollout", _build(redundant=redundant))
    assert ref == CaptureSnapshotRef("rollout", "frozen", 7)
    source.drop.assert_not_called()
    foreign = CaptureSnapshotRef("other-rollout", "frozen", 7)
    stale = CaptureSnapshotRef("rollout", "frozen", 6)
    assert asyncio.run(reader.acknowledge([foreign, stale])) == 0
    source.drop.assert_not_called()
    assert asyncio.run(reader.acknowledge([ref, ref])) == 1
    source.drop.assert_awaited_once_with("rollout", snapshot_id="frozen", version=7)
    assert asyncio.run(reader.acknowledge([ref])) == 0
    assert source.drop.await_count == 1


@pytest.mark.parametrize("failure", [False, OSError("storage unavailable")])
def test_retirement_failure_retains_retryable_snapshot(failure):
    source = SimpleNamespace(drop=AsyncMock())
    if isinstance(failure, Exception):
        source.drop.side_effect = failure
    else:
        source.drop.return_value = False
    reader = GymCaptureReader(source, owns_source=False)
    ref = reader.register("rollout", _build())
    assert asyncio.run(reader.acknowledge([ref])) == 0
    source.drop.side_effect = None
    source.drop.return_value = True
    assert asyncio.run(reader.acknowledge([ref])) == 1


@pytest.mark.parametrize(
    "built", [None, _build(masked=True), {"rebuilt_response": None}]
)
def test_unsuccessful_build_cannot_be_acknowledged(built):
    source = SimpleNamespace(drop=AsyncMock())
    reader = GymCaptureReader(source, owns_source=False)
    assert reader.register("rollout", built) is None
    assert (
        asyncio.run(reader.acknowledge([CaptureSnapshotRef("rollout", "frozen", 7)]))
        == 0
    )
    source.drop.assert_not_called()


def test_retain_consumed_does_not_register_for_cleanup():
    reader = GymCaptureReader(
        SimpleNamespace(), owns_source=False, retain_consumed=True
    )
    assert reader.register("rollout", _build()) is None


@pytest.mark.parametrize("owned", [False, True])
def test_close_only_owned_source_and_never_unaccepted_capture(owned):
    source = SimpleNamespace(close=AsyncMock(), drop=AsyncMock())
    reader = GymCaptureReader(source, owns_source=owned)
    reader.register("rollout", _build())
    asyncio.run(reader.close())
    asyncio.run(reader.close())
    assert source.close.await_count == int(owned)
    source.drop.assert_not_called()


def test_disabled_capture_needs_no_source():
    assert GymCaptureReader.from_config({}) is None


@pytest.mark.parametrize(
    "rebuild_response,message",
    [(True, "requires a directory or sink"), (False, "readable local capture")],
)
def test_enabled_capture_without_readable_source_fails_at_setup(
    rebuild_response, message
):
    with pytest.raises(ValueError, match=message):
        GymCaptureReader.from_config(
            {
                "token_id_capture": {
                    "enabled": True,
                    "rebuild_response": rebuild_response,
                },
            }
        )


def test_actor_ack_failure_does_not_fail_accepted_rollout(caplog):
    env = SimpleNamespace(
        acknowledge_token_captures=SimpleNamespace(
            remote=AsyncMock(side_effect=OSError("actor lost"))
        )
    )
    asyncio.run(
        acknowledge_gym_captures(env, [CaptureSnapshotRef("rollout", "frozen", 7)])
    )
    assert "after payload acceptance" in caplog.text


def test_ack_cancellation_is_not_swallowed():
    env = SimpleNamespace(
        acknowledge_token_captures=SimpleNamespace(
            remote=AsyncMock(side_effect=asyncio.CancelledError())
        )
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            acknowledge_gym_captures(env, [CaptureSnapshotRef("rollout", "frozen", 7)])
        )
