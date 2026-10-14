# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Native capture sink ownership tests with controllable storage completion."""

import asyncio
import threading
from typing import Any

import pytest
import torch

pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_gym.token_id_capture.staging import (  # noqa: E402
    CaptureAdmission,
    RolloutTokenCapture,
    StagedCallRecord,
    StageResult,
)

from nemo_rl.data_plane.tq_token_sink import TQTokenSink  # noqa: E402
from nemo_rl.data_plane.async_utils import complete_before_cancellation  # noqa: E402

pytestmark = pytest.mark.nemo_gym


class _RecordSink:
    def stage(self, record: StagedCallRecord) -> StageResult:
        self.record = record
        return StageResult(ok=True, staging_key=record.staging_key)


def _record() -> StagedCallRecord:
    sink = _RecordSink()
    capture = RolloutTokenCapture(sink=sink, weight_version_fn=lambda: 3)
    capture.complete_call(
        capture.begin_call(
            CaptureAdmission(rollout_id="rollout-é", model_call_id="call", mode="text")
        ),
        prompt_token_ids=[1, 2],
        generated_token_ids=[3],
        generated_logprobs=[-0.25],
        extras={"routed_experts": [[[1, 2]], [[3, 4]], [[5, 6]]]},
    )
    return sink.record


class _Storage:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.events: list[str] = []
        self.rows: dict[str, Any] = {}
        self.tags: dict[str, Any] = {}
        self.fail_put = False
        self.fail_clear = False
        self.clear_entered = asyncio.Event()
        self.clear_release: asyncio.Event | None = None

    def put_samples(self, *, sample_ids, partition_id, fields, tags) -> None:
        self.events.append("put")
        for key in sample_ids:
            self.rows[key] = fields.clone()
            self.tags[key] = tags
        if self.fail_put:
            raise OSError("partial write")

    async def async_put_samples(self, **kwargs) -> None:
        # Match the data-plane mutation API's completion-before-cancellation
        # contract; the sink must reclaim a settled write after cancellation.
        async def put():
            self.entered.set()
            await self.release.wait()
            self.put_samples(**kwargs)

        await complete_before_cancellation(put())

    async def async_clear_samples(self, *, sample_ids, partition_id) -> None:
        self.clear_entered.set()
        if self.clear_release is not None:
            await self.clear_release.wait()
        self.events.append("clear")
        if self.fail_clear:
            raise OSError("cleanup failed")
        for key in sample_ids:
            self.rows.pop(key, None)
            self.tags.pop(key, None)


@pytest.mark.parametrize("media_mode", ["text", "media_empty", "media"])
def test_async_stage_waits_for_ack_and_preserves_exact_sync_payload(
    media_mode: str,
) -> None:
    async def scenario() -> None:
        storage = _Storage()
        sink_options = dict(
            staging_partition="staging",
            capture_media=media_mode != "text",
            media_pixel_dtype=torch.bfloat16 if media_mode != "text" else None,
        )
        sink = TQTokenSink(storage, **sink_options)
        attachments = (
            {
                "imgs": torch.arange(12, dtype=torch.bfloat16).reshape(1, 1, 12),
                "imgs_sizes": torch.tensor([[2, 2]], dtype=torch.int32),
            }
            if media_mode == "media"
            else None
        )
        record = _record()
        task = asyncio.create_task(sink.async_stage(record, attachments=attachments))
        await storage.entered.wait()
        assert not task.done()
        assert not storage.rows
        storage.release.set()
        assert (await task).ok
        baseline = _Storage()
        assert (
            TQTokenSink(baseline, **sink_options)
            .stage(record, attachments=attachments)
            .ok
        )
        actual = storage.rows[record.staging_key]
        expected = baseline.rows[record.staging_key]
        assert set(actual.keys()) == set(expected.keys())
        for key in expected.keys():
            assert actual[key].dtype == expected[key].dtype
            assert torch.equal(actual[key], expected[key])
        assert storage.tags == baseline.tags

    asyncio.run(scenario())


def test_cancellation_during_cleanup_still_waits_for_clear() -> None:
    async def scenario() -> None:
        storage = _Storage()
        storage.clear_release = asyncio.Event()
        task = asyncio.create_task(
            TQTokenSink(storage, staging_partition="staging").async_stage(_record())
        )
        await storage.entered.wait()
        task.cancel()
        storage.release.set()
        await storage.clear_entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert storage.events == ["put"]
        storage.clear_release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert storage.events == ["put", "clear"]
        assert not storage.rows

    asyncio.run(scenario())


@pytest.mark.parametrize("repeat_cancel", [False, True])
def test_cancelled_write_settles_before_cleanup_and_cancellation(
    repeat_cancel: bool,
) -> None:
    async def scenario() -> None:
        storage = _Storage()
        sink = TQTokenSink(storage, staging_partition="staging")
        task = asyncio.create_task(sink.async_stage(_record()))
        await storage.entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        if repeat_cancel:
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done()
        assert storage.events == []
        storage.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert storage.events == ["put", "clear"]
        assert not storage.rows

    asyncio.run(scenario())


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_failed_write_attempts_cleanup_and_never_reports_success(
    cleanup_fails: bool,
) -> None:
    async def scenario() -> None:
        storage = _Storage()
        storage.fail_put = True
        storage.fail_clear = cleanup_fails
        storage.release.set()
        result = await TQTokenSink(storage, staging_partition="staging").async_stage(
            _record()
        )
        assert not result.ok
        assert "partial write" in result.error
        assert storage.events == ["put", "clear"]
        assert bool(storage.rows) == cleanup_fails

    asyncio.run(scenario())


def test_bad_attachments_fail_before_io() -> None:
    async def scenario() -> None:
        storage = _Storage()
        result = await TQTokenSink(storage, staging_partition="staging").async_stage(
            _record(), attachments={"unrecognized": object()}
        )
        assert not result.ok
        assert "media-enabled" in result.error
        assert not storage.entered.is_set()
        assert storage.events == []

    asyncio.run(scenario())


def test_tensor_preparation_stays_off_the_event_loop() -> None:
    async def scenario() -> None:
        storage = _Storage()
        storage.release.set()
        sink = TQTokenSink(storage, staging_partition="staging")
        original = sink._stage_payload
        loop_thread = threading.get_ident()
        prepared = []

        def prepare(record, attachments):
            prepared.append(threading.get_ident())
            return original(record, attachments)

        sink._stage_payload = prepare
        assert (await sink.async_stage(_record())).ok
        assert len(prepared) == 1
        assert prepared[0] != loop_thread

    asyncio.run(scenario())


def test_cancelling_tensor_preparation_never_submits_a_write() -> None:
    async def scenario() -> None:
        storage = _Storage()
        sink = TQTokenSink(storage, staging_partition="staging")
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        original = sink._stage_payload

        def prepare(record, attachments):
            started.set()
            try:
                assert release.wait(5)
                return original(record, attachments)
            finally:
                finished.set()

        sink._stage_payload = prepare
        task = asyncio.create_task(sink.async_stage(_record()))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            assert await asyncio.to_thread(finished.wait, 5)
        assert not storage.entered.is_set()
        assert storage.events == []

    asyncio.run(scenario())
