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
"""Cancellation must not orphan a retained view or race temporary-file cleanup."""

import asyncio
import threading

import pytest

from nemo_rl.data_plane.checkpoint import (
    CheckpointCapture,
    CheckpointReleaseError,
    capture_checkpoint,
    checkpoint_io,
)
from nemo_rl.data_plane.observability import MetricsDataPlaneClient


class _Capture(CheckpointCapture):
    def __init__(self):
        self.token = "token-1"
        self.released = False

    def persist(self):
        pass

    def release(self):
        self.released = True


def test_context_manager_returns_capture_and_releases_after_success():
    captured = _Capture()
    with captured as entered:
        assert entered is captured
        assert not captured.released
        entered.persist()
    assert captured.released


@pytest.mark.parametrize("error", [ValueError("body failed"), KeyboardInterrupt()])
def test_context_manager_releases_without_suppressing_body_error(error):
    captured = _Capture()
    with pytest.raises(type(error)) as raised:
        with captured:
            raise error
    assert raised.value is error
    assert captured.released


def test_context_manager_release_failure_keeps_body_error_as_context():
    class FailingCapture(_Capture):
        def release(self):
            raise TimeoutError("release timed out")

    error = ValueError("body failed")
    with pytest.raises(TimeoutError, match="release timed out") as raised:
        with FailingCapture():
            raise error
    assert raised.value.__context__ is error


@pytest.mark.asyncio
async def test_cancelled_capture_releases_late_result():
    entered, finish = threading.Event(), threading.Event()
    captured = _Capture()

    class Client:
        def capture_checkpoint(self):
            entered.set()
            assert finish.wait(5)
            return captured

    task = asyncio.create_task(capture_checkpoint(Client()))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert captured.released


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_during_release", [False, True])
async def test_cancelled_capture_release_failure_reaches_caller(cancel_during_release):
    entered, finish = threading.Event(), threading.Event()
    releasing, fail_release = threading.Event(), threading.Event()

    class FailingCapture(_Capture):
        def release(self):
            releasing.set()
            assert fail_release.wait(5)
            raise TimeoutError("release timed out")

    class Client:
        def capture_checkpoint(self):
            entered.set()
            assert finish.wait(5)
            return FailingCapture()

    task = asyncio.create_task(capture_checkpoint(Client()))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    finish.set()
    assert await asyncio.to_thread(releasing.wait, 5)
    if cancel_during_release:
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    fail_release.set()
    with pytest.raises(CheckpointReleaseError, match="token-1") as raised:
        await task
    assert isinstance(raised.value.__cause__, TimeoutError)


@pytest.mark.asyncio
async def test_adapter_release_error_passes_through_cancellation():
    entered, finish = threading.Event(), threading.Event()

    class Client:
        def capture_checkpoint(self):
            entered.set()
            assert finish.wait(5)
            raise CheckpointReleaseError("capture token-2 still retained")

    task = asyncio.create_task(capture_checkpoint(Client()))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    finish.set()
    with pytest.raises(CheckpointReleaseError, match="token-2"):
        await task


@pytest.mark.asyncio
async def test_cancelled_capture_failure_still_reports_cancellation():
    entered, finish = threading.Event(), threading.Event()

    class Client:
        def capture_checkpoint(self):
            entered.set()
            assert finish.wait(5)
            raise ValueError("capture failed")

    task = asyncio.create_task(capture_checkpoint(Client()))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_cancellation_waits_for_writer_even_when_cancelled_twice(tmp_path):
    entered, finish = threading.Event(), threading.Event()
    path = tmp_path / "payload"

    def write():
        entered.set()
        assert finish.wait(5)
        path.write_text("complete")

    task = asyncio.create_task(checkpoint_io(write))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert path.read_text() == "complete"


@pytest.mark.asyncio
async def test_io_error_propagates():
    def fail():
        raise OSError("write failed")

    with pytest.raises(OSError, match="write failed"):
        await checkpoint_io(fail)


@pytest.mark.asyncio
async def test_unsupported_adapter_requests_synchronous_fallback():
    class Client:
        def capture_checkpoint(self):
            return None

    assert await capture_checkpoint(Client()) is None


def test_metrics_wrapper_forwards_capture_without_replacing_handle(tmp_path):
    captured = _Capture()

    class Client:
        def capture_checkpoint(self, checkpoint_dir, *, metadata):
            assert checkpoint_dir == tmp_path
            assert metadata == {"step": 7}
            return captured

    wrapped = MetricsDataPlaneClient(Client())
    assert wrapped.capture_checkpoint(tmp_path, metadata={"step": 7}) is captured
