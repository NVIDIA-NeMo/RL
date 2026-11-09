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
    capture_checkpoint,
    checkpoint_io,
)
from nemo_rl.data_plane.observability import MetricsDataPlaneClient


class _Capture(CheckpointCapture):
    def __init__(self):
        self.released = False

    def persist(self):
        pass

    def release(self):
        self.released = True


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
