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
"""Background deletes obey snapshot epochs and bounded terminal admission."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from nemo_rl.data_plane.background_prefix_cleanup import (
    BackgroundPrefixCleanup,
    GenerationPrefixCleanupConfig,
)
from nemo_rl.models.generation.generation_cut_capture import _TokenCaptureSnapshotGate


def test_fence_waits_only_for_active_batch_and_resume_drains_queue():
    entered, release = threading.Event(), threading.Event()
    calls = []

    def clear(keys):
        calls.append(keys)
        entered.set()
        assert release.wait(5)

    cleanup = BackgroundPrefixCleanup(
        clear,
        config=GenerationPrefixCleanupConfig(
            batch_size=2, max_pending=4, wait_seconds=0
        ),
    )
    try:
        cleanup.reserve().submit(["first"])
        assert entered.wait(5)
        cleanup.pause(1)
        cleanup.reserve().submit(["second"])
        with ThreadPoolExecutor(max_workers=1) as pool:
            fence = pool.submit(cleanup.wait_paused, 1)
            assert not fence.done()
            release.set()
            fence.result(5)
        assert calls == [["first"]]
        assert cleanup.stats().queued_requests == 1
        cleanup.resume()
    finally:
        release.set()
        cleanup.resume()
        cleanup.close()
    assert calls == [["first"], ["second"]]


def test_full_queue_backpressure_cannot_hold_terminal_gate():
    gate = _TokenCaptureSnapshotGate()
    cleanup = BackgroundPrefixCleanup(
        lambda keys: None,
        config=GenerationPrefixCleanupConfig(batch_size=1, max_pending=1),
    )
    cleanup.pause(1)
    cleanup.reserve().submit(["queued"])
    attempting = threading.Event()

    def terminal():
        attempting.set()
        reservation = cleanup.reserve()
        try:
            gate.enter()
            try:
                reservation.submit(["later"])
            finally:
                gate.exit()
        finally:
            reservation.close()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            writer = pool.submit(terminal)
            assert attempting.wait(5)
            try:
                fence = pool.submit(gate.close_and_wait, gate.begin_epoch())
                fence.result(5)
                cleanup.wait_paused(1)
                assert not writer.done()
            finally:
                gate.reopen()
                cleanup.resume()
            writer.result(5)
    finally:
        cleanup.resume()
        cleanup.close()
    assert cleanup.stats().peak_pending_requests == 1


def test_reserved_slot_can_enqueue_while_cleanup_paused_and_deduplicates():
    calls = []
    cleanup = BackgroundPrefixCleanup(
        calls.append, config=GenerationPrefixCleanupConfig(batch_size=2, max_pending=2)
    )
    first, second = cleanup.reserve(), cleanup.reserve()
    cleanup.pause(1)
    first.submit(["a", "shared"])
    second.submit(["b", "shared"])
    first.close()
    second.close()
    cleanup.wait_paused(1)
    assert calls == []
    cleanup.resume()
    cleanup.close()
    assert calls == [["a", "shared", "b"]]


def test_unused_reservation_releases_capacity_and_old_epoch_cannot_reclose():
    calls = []
    cleanup = BackgroundPrefixCleanup(
        calls.append, config=GenerationPrefixCleanupConfig(batch_size=1, max_pending=1)
    )
    reservation = cleanup.reserve()
    reservation.close()
    reservation.close()
    assert cleanup.stats().pending_requests == 0
    cleanup.pause(1)
    cleanup.resume()
    cleanup.pause(1)
    cleanup.reserve().submit(["after-abort"])
    cleanup.close()
    assert calls == [["after-abort"]]


def test_failed_delete_is_latched_and_never_retried():
    calls = []

    def clear(keys):
        calls.append(keys)
        raise RuntimeError("transport failed")

    cleanup = BackgroundPrefixCleanup(
        clear, config=GenerationPrefixCleanupConfig(batch_size=1, max_pending=1)
    )
    cleanup.reserve().submit(["failed"])
    with pytest.raises(RuntimeError, match="background prefix cleanup failed"):
        cleanup.close()
    with pytest.raises(RuntimeError, match="background prefix cleanup failed"):
        cleanup.reserve()
    with pytest.raises(RuntimeError, match="background prefix cleanup failed"):
        cleanup.wait_paused(1)
    assert calls == [["failed"]]
    assert cleanup.stats().failed_keys == 1


def test_close_refuses_to_delete_under_a_snapshot():
    cleanup = BackgroundPrefixCleanup(
        lambda keys: None, config=GenerationPrefixCleanupConfig()
    )
    cleanup.pause(1)
    try:
        with pytest.raises(RuntimeError, match="resume"):
            cleanup.close()
    finally:
        cleanup.resume()
        cleanup.close()


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


def test_released_fence_wait_does_not_block_on_new_epoch_work():
    entered, release = threading.Event(), threading.Event()

    def clear(keys):
        entered.set()
        assert release.wait(5)

    cleanup = BackgroundPrefixCleanup(
        clear, config=GenerationPrefixCleanupConfig(batch_size=1)
    )
    cleanup.pause(1)
    cleanup.resume()
    try:
        cleanup.reserve().submit(["new-work"])
        assert entered.wait(5)
        # A late wait from an aborted checkpoint must not reclose the queue.
        cleanup.pause(1)
        cleanup.wait_paused(1)
        assert cleanup.stats().active_batch
        assert not cleanup.stats().paused
    finally:
        release.set()
        cleanup.close()


def test_batch_bound_deduplicates_without_losing_keys():
    calls = []
    cleanup = BackgroundPrefixCleanup(
        calls.append, config=GenerationPrefixCleanupConfig(batch_size=2, max_pending=5)
    )
    cleanup.pause(1)
    for key in ["a", "b", "c", "d", "e"]:
        cleanup.reserve().submit([key, "shared"])
    cleanup.resume()
    cleanup.close()
    assert len(calls) == 3
    assert set(key for batch in calls for key in batch) == {
        "a",
        "b",
        "c",
        "d",
        "e",
        "shared",
    }
    assert all(len(batch) == len(set(batch)) for batch in calls)
    assert cleanup.stats().requests == 5
    assert cleanup.stats().pending_requests == 0


def test_close_wakes_a_producer_waiting_for_capacity():
    cleanup = BackgroundPrefixCleanup(
        lambda keys: None,
        config=GenerationPrefixCleanupConfig(batch_size=1, max_pending=1),
    )
    first = cleanup.reserve()
    attempting = threading.Event()

    def reserve():
        attempting.set()
        return cleanup.reserve()

    with ThreadPoolExecutor(max_workers=1) as pool:
        waiter = pool.submit(reserve)
        assert attempting.wait(5)
        cleanup.close()
        with pytest.raises(RuntimeError, match="closed"):
            waiter.result(5)
    first.close()
    assert cleanup.stats().pending_requests == 0


def test_submitted_keys_are_owned_and_a_slot_cannot_be_reused():
    calls = []
    cleanup = BackgroundPrefixCleanup(
        calls.append, config=GenerationPrefixCleanupConfig()
    )
    cleanup.pause(1)
    reservation = cleanup.reserve()
    keys = ["original"]
    reservation.submit(keys)
    keys[0] = "mutated"
    with pytest.raises(RuntimeError, match="already consumed"):
        reservation.submit(["second"])
    cleanup.resume()
    cleanup.close()
    assert calls == [["original"]]


def test_nonblocking_close_does_not_wait_for_stalled_clear():
    entered, release = threading.Event(), threading.Event()

    def clear(keys):
        entered.set()
        assert release.wait(5)

    cleanup = BackgroundPrefixCleanup(
        clear, config=GenerationPrefixCleanupConfig(batch_size=1)
    )
    try:
        cleanup.reserve().submit(["obsolete"])
        assert entered.wait(5)
        cleanup.close(wait=False)
        assert cleanup.stats().active_batch
        with pytest.raises(RuntimeError, match="closed"):
            cleanup.reserve()
    finally:
        release.set()
        cleanup.close()


def test_many_completed_calls_coalesce_into_bounded_clears():
    calls = []
    cleanup = BackgroundPrefixCleanup(
        calls.append,
        config=GenerationPrefixCleanupConfig(batch_size=32, max_pending=1024),
    )
    cleanup.pause(1)
    for request in range(1024):
        cleanup.reserve().submit([f"prefix-{request}-0", f"prefix-{request}-1"])
    cleanup.resume()
    cleanup.close()
    assert len(calls) == 32
    assert all(len(batch) == 64 for batch in calls)
    assert len({key for batch in calls for key in batch}) == 2048
    assert cleanup.stats().pending_requests == 0
