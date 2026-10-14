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

"""Exercise native capture on a real worker instance without starting its engine."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter  # noqa: E402
from nemo_gym.token_id_capture.staging import (  # noqa: E402
    CaptureAdmission,
    RolloutTokenCapture,
    StageResult,
)

from nemo_rl.models.generation.vllm.vllm_worker_async import (  # noqa: E402
    VllmAsyncGenerationWorkerImpl,
)

pytestmark = pytest.mark.nemo_gym


@pytest.fixture
def worker_class() -> type:
    return VllmAsyncGenerationWorkerImpl


class _Sink:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.records = []
        self.attachments = []

    def stage(self, record, *, attachments=None) -> StageResult:
        self.records.append(record)
        self.attachments.append(attachments)
        return StageResult(ok=True, staging_key=record.staging_key)

    async def async_stage(self, record, *, attachments=None) -> StageResult:
        self.entered.set()
        await self.release.wait()
        return self.stage(record, attachments=attachments)


def _worker(worker_class: type, sink: _Sink) -> Any:
    worker = object.__new__(worker_class)
    worker.is_model_owner = True
    worker._capture_calls = {}
    worker._rollout_weight_version = 4
    worker._capture_async_semaphore = asyncio.Semaphore(1)
    worker._chain_prefix = SimpleNamespace(install=lambda source: None)
    worker.install_token_capture = lambda capture: setattr(
        worker, "token_capture", capture
    )
    worker.token_capture = RolloutTokenCapture(
        sink=sink,
        async_sink=sink,
        weight_version_fn=lambda: 4,
        adapter=VLLMCaptureAdapter(),
    )
    return worker


def _admit(worker: Any, call_id: str, *, media: bool = False) -> tuple[object, Any]:
    request = object()
    call = worker.token_capture.begin_call(
        CaptureAdmission(rollout_id="rollout", model_call_id=call_id, mode="text")
    )
    worker._capture_calls[id(request)] = SimpleNamespace(
        call=call,
        prompt_token_ids=[1, 2],
        media=SimpleNamespace(items=[], tensors={"opaque": object()})
        if media
        else None,
    )
    return request, call


def _content() -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "generation_token_ids": [3],
                    "generation_log_probs": [-0.25],
                },
                "logprobs": {"content": [{"token": "token_id:3", "logprob": -0.25}]},
            }
        ]
    }


@pytest.mark.parametrize("media", [False, True])
def test_host_awaits_capture_before_stripping_or_publishing(
    worker_class: type, media: bool
) -> None:
    async def scenario() -> None:
        sink = _Sink()
        worker = _worker(worker_class, sink)
        request, _ = _admit(worker, "first", media=media)
        attachments = worker._capture_calls[id(request)].media
        content = _content()
        baseline_content = deepcopy(content)
        task = asyncio.create_task(
            worker._finish_request_capture_async(request, content)
        )
        await sink.entered.wait()
        assert not task.done()
        assert "ng_commit_coords" not in content
        assert "generation_token_ids" in content["choices"][0]["message"]
        sink.release.set()
        result = await task
        assert result["ng_commit_coords"]["disposition"] == "staged"
        assert "logprobs" not in result["choices"][0]
        assert "generation_token_ids" not in result["choices"][0]["message"]
        assert sink.records[0].token_ids_delta == [1, 2, 3]
        assert sink.attachments == [attachments.tensors if media else None]
        baseline_sink = _Sink()
        baseline_worker = _worker(worker_class, baseline_sink)
        baseline_request, _ = _admit(baseline_worker, "first", media=media)
        baseline = baseline_worker._finish_request_capture(
            baseline_request, baseline_content
        )
        assert result == baseline
        assert sink.records == baseline_sink.records

    asyncio.run(scenario())


def test_host_bounds_preparation_and_cleans_cancelled_admission(
    worker_class: type,
) -> None:
    async def scenario() -> None:
        sink = _Sink()
        worker = _worker(worker_class, sink)
        first_request, _ = _admit(worker, "first")
        first = asyncio.create_task(
            worker._finish_request_capture_async(first_request, _content())
        )
        await sink.entered.wait()
        second_request, second_call = _admit(worker, "second")
        second = asyncio.create_task(
            worker._finish_request_capture_async(second_request, _content())
        )
        await asyncio.sleep(0)
        assert not second_call.completed
        assert id(second_request) in worker._capture_calls
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        assert second_call.completed
        assert id(second_request) not in worker._capture_calls
        sink.release.set()
        await first
        assert [record.model_call_id for record in sink.records] == ["first"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "backend,enabled,native",
    [("simple", True, True), ("simple", False, False), ("mooncake_cpu", True, False)],
)
def test_setup_selects_native_only_for_enabled_simple(
    worker_class: type, monkeypatch, backend: str, enabled: bool, native: bool
) -> None:
    sink = _Sink()
    worker = _worker(worker_class, sink)
    monkeypatch.setattr(
        "nemo_rl.data_plane.build_data_plane_client", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        "nemo_rl.data_plane.tq_token_sink.TQTokenSink", lambda *args, **kwargs: sink
    )
    monkeypatch.setattr(
        "nemo_rl.data_plane.tq_token_sink.TQTokenSource",
        lambda *args, **kwargs: object(),
    )
    asyncio.run(
        worker.setup_token_capture(
            {"backend": backend, "async_io": {"enabled": enabled, "max_inflight": 2}},
            "staging",
        )
    )
    assert (worker._capture_async_semaphore is not None) == native
    assert (worker.token_capture._async_sink is sink) == native
