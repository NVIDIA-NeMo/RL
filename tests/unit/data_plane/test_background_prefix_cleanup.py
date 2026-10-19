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
"""Background deletes run under the snapshot gate and never block its fence."""

import logging
import threading
import time

import pytest

from nemo_rl.data_plane.background_prefix_cleanup import (
    BackgroundPrefixCleanup,
    GenerationPrefixCleanupConfig,
)
from nemo_rl.models.generation.generation_cut_capture import _TokenCaptureSnapshotGate


def _wait_for(predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "condition was not reached"
        time.sleep(0.005)


def _fence(gate):
    """Close the gate the way a checkpoint fence does."""
    gate.close_and_wait(gate.begin_epoch())


def test_closed_gate_defers_clears_until_reopen():
    gate = _TokenCaptureSnapshotGate()
    cleared, done = [], threading.Event()

    def clear(keys):
        cleared.append(keys)
        done.set()

    cleanup = BackgroundPrefixCleanup(
        clear, gate, config=GenerationPrefixCleanupConfig(batch_size=1)
    )
    _fence(gate)
    cleanup.submit(["a"])
    assert not done.wait(0.2)
    gate.reopen()
    assert done.wait(5)
    assert cleared == [["a"]]


def test_fence_waits_for_a_running_clear_and_blocks_the_next():
    gate = _TokenCaptureSnapshotGate()
    calls, entered, release = [], threading.Event(), threading.Event()

    def clear(keys):
        calls.append(keys)
        entered.set()
        assert release.wait(5)

    cleanup = BackgroundPrefixCleanup(
        clear, gate, config=GenerationPrefixCleanupConfig(batch_size=1)
    )
    cleanup.submit(["running"])
    assert entered.wait(5)
    fenced = threading.Event()
    fence = threading.Thread(target=lambda: (_fence(gate), fenced.set()))
    fence.start()
    # The fence has closed the gate and is waiting for the running CLEAR.
    assert not fenced.wait(0.2)
    entered.clear()
    cleanup.submit(["queued"])
    release.set()
    assert fenced.wait(5)
    # No CLEAR starts while the fence holds the gate closed.
    assert not entered.wait(0.2)
    gate.reopen()
    assert entered.wait(5)
    fence.join(5)
    assert calls == [["running"], ["queued"]]


def test_full_queue_blocks_producers_without_holding_a_gate_permit():
    gate = _TokenCaptureSnapshotGate()
    cleared = []
    cleanup = BackgroundPrefixCleanup(
        cleared.append,
        gate,
        config=GenerationPrefixCleanupConfig(
            batch_size=1, max_pending=1, wait_seconds=0
        ),
    )
    _fence(gate)
    # The cleanup thread takes "first" and blocks at the closed gate; "second"
    # fills the queue, so "third" blocks its producer.
    cleanup.submit(["first"])
    cleanup.submit(["second"])
    producer = threading.Thread(target=cleanup.submit, args=(["third"],))
    producer.start()
    producer.join(0.2)
    assert producer.is_alive()
    # Neither the blocked producer nor the waiting thread holds a permit, so a
    # later fence still completes at once.
    second_fence = threading.Thread(target=_fence, args=(gate,))
    second_fence.start()
    second_fence.join(5)
    assert not second_fence.is_alive()
    gate.reopen()
    producer.join(5)
    assert not producer.is_alive()
    _wait_for(lambda: len(cleared) == 3)
    assert cleared == [["first"], ["second"], ["third"]]


def test_failed_clear_is_dropped_without_retry_and_cleanup_keeps_serving(caplog):
    gate = _TokenCaptureSnapshotGate()
    calls, done = [], threading.Event()

    def clear(keys):
        calls.append(keys)
        if keys == ["failed"]:
            raise RuntimeError("transport failed")
        done.set()

    cleanup = BackgroundPrefixCleanup(
        clear, gate, config=GenerationPrefixCleanupConfig(batch_size=1, max_pending=1)
    )
    with caplog.at_level(logging.ERROR):
        cleanup.submit(["failed"])
        cleanup.submit(["next"])
        assert done.wait(5)
    assert calls == [["failed"], ["next"]]
    assert "background prefix cleanup failed for 1 keys" in caplog.text
    # The failed CLEAR released its permit: a fence does not wait on it.
    _fence(gate)
    gate.reopen()


def test_submitted_keys_are_copied_and_deduplicated():
    gate = _TokenCaptureSnapshotGate()
    cleared, done = [], threading.Event()

    def clear(keys):
        cleared.append(keys)
        done.set()

    cleanup = BackgroundPrefixCleanup(
        clear, gate, config=GenerationPrefixCleanupConfig(batch_size=1)
    )
    _fence(gate)
    keys = ["a", "b", "a"]
    cleanup.submit(keys)
    keys.append("added-after-submit")
    cleanup.submit([])
    gate.reopen()
    assert done.wait(5)
    time.sleep(0.1)
    assert cleared == [["a", "b"]]


def test_many_completed_calls_coalesce_into_bounded_clears():
    gate = _TokenCaptureSnapshotGate()
    calls = []
    cleanup = BackgroundPrefixCleanup(
        calls.append,
        gate,
        config=GenerationPrefixCleanupConfig(
            batch_size=32, max_pending=1024, wait_seconds=5.0
        ),
    )
    _fence(gate)
    for request in range(1024):
        cleanup.submit([f"prefix-{request}-0", f"prefix-{request}-1"])
    gate.reopen()
    _wait_for(lambda: sum(len(batch) for batch in calls) == 2048)
    assert len(calls) == 32
    assert all(len(batch) == 64 for batch in calls)
    assert len({key for batch in calls for key in batch}) == 2048


@pytest.mark.parametrize(
    "changes",
    [
        {"batch_size": 0},
        {"max_pending": 0},
        {"batch_size": 3, "max_pending": 2},
        {"wait_seconds": -1},
        {"wait_seconds": float("inf")},
        {"wait_seconds": float("nan")},
        {"unknown_setting": 1},
    ],
)
def test_invalid_limits_fail_before_creating_a_worker(changes):
    with pytest.raises(ValueError):
        GenerationPrefixCleanupConfig.model_validate(changes)
