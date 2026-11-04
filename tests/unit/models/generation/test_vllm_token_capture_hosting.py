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

"""S2 worker hosting: install_capture wiring, fan-outs, version stamping.

Marked nemo_gym (run with ``--nemo-gym-only``): the hosting seam imports
Gym's capture core. No engine or GPU is needed — the worker methods are
driven unbound against light fakes, and the VllmGeneration fan-outs against
a mock worker group.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

nemo_gym = pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_gym._checkpoint.generation_cut import (  # noqa: E402
    GENERATION_CUT_ROUTE,
    GenerationCutInventory,
    GenerationCutPrefix,
)
from nemo_gym.token_id_capture.staging.capture import (  # noqa: E402
    CaptureError,
    RolloutTokenCapture,
)
from nemo_gym.token_id_capture.staging.records import (  # noqa: E402
    CaptureAdmission,
    StagedCallRecord,
    StageResult,
)

from nemo_rl.data.captured_media import MediaColumnSpec  # noqa: E402
from nemo_rl.data_plane.tq_token_sink import (  # noqa: E402
    ChainPrefixCache,
    StagedPrefix,
    TQTokenSink,
)
from nemo_rl.models.generation.generation_cut_capture import (  # noqa: E402
    _remaining_generation_limits_after_prefix,
    _RequestCaptureState,
    _TokenCaptureSnapshotGate,
    initial_generation_cut_counts,
)
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration  # noqa: E402
from nemo_rl.models.generation.vllm.vllm_worker_async import (  # noqa: E402
    _GENERATION_CUT_ROUTE,
    _CAPTURED_MODEL_CALL,
    _EngineProgressRead,
    _AsyncLLMHTTPClient,
    VllmAsyncGenerationWorkerImpl,
    _classify_restored_prefix_terminal,
    _count_restored_prefix_as_completion,
    _generation_cut_blockers,
    _RestoredPrefixTerminal,
)

pytestmark = pytest.mark.nemo_gym


class _MemorySink:
    def __init__(self) -> None:
        self.records: list[StagedCallRecord] = []
        self.attachments: list[dict | None] = []
        self.generation_prefix_records: list[tuple[str, StagedCallRecord]] = []
        self.generation_prefix_keys: list[str] = []
        self.cleared_generation_prefix_keys: list[str] = []

    def stage(
        self, record: StagedCallRecord, *, attachments: dict | None = None
    ) -> StageResult:
        self.records.append(record)
        self.attachments.append(attachments)
        return StageResult(ok=True, staging_key=record.staging_key)

    def stage_generation_prefix(
        self,
        record: StagedCallRecord,
        *,
        checkpoint_id: str,
        chunk_sequence: int,
        attachments: dict | None = None,
    ) -> StageResult:
        assert attachments is None
        self.generation_prefix_records.append((checkpoint_id, record))
        key = (
            f"__generation_cut__/{checkpoint_id}/{record.capture_key}/"
            f"{record.model_call_id}/{chunk_sequence}"
        )
        self.generation_prefix_keys.append(key)
        return StageResult(ok=True, staging_key=key)

    def clear(self, staging_keys: list[str]) -> None:
        self.cleared_generation_prefix_keys.extend(staging_keys)


class _ByteTokenizer:
    """Byte-level stand-in: each token ID is one UTF-8 byte of the output."""

    def decode(self, token_ids: list[int], **kwargs) -> str:
        return bytes(token_ids).decode("utf-8", errors="replace")


class _NoCallsInVllm:
    """An HTTP engine client none of whose captured calls has reached vLLM."""

    def read_generation_progress(self, calls, *, timeout_s):
        return _EngineProgressRead([None] * len(calls), 0.0, 0.0)


def _fake_worker(*, is_model_owner: bool = True) -> SimpleNamespace:
    """The attribute surface setup_token_capture touches, minus the engine."""
    worker = SimpleNamespace(
        is_model_owner=is_model_owner,
        token_capture=None,
        _rollout_weight_version=0,
        _chain_prefix=ChainPrefixCache(),
        _capture_calls={},
        _capture_calls_by_model_call_id={},
        _completed_capture_calls={},
        _generation_cut_receipts={},
        _capture_registry_lock=threading.Lock(),
        _capture_sink=None,
        _generation_prefix_cuts_enabled=False,
        _generation_cut_control_token=None,
        _generation_cut_control_timeout_s=None,
        _generation_cut_tokenizer=_ByteTokenizer(),
        _restored_generation_prefix_calls=0,
        _restored_generation_prefix_tokens=0,
        _http_engine_client=_NoCallsInVllm(),
        _generation_cut_counts=initial_generation_cut_counts(),
        _token_capture_snapshot_gate=_TokenCaptureSnapshotGate(),
        _generation_cut_cleanup_executor=ThreadPoolExecutor(max_workers=1),
        _staging_source=None,
        _capture_top_logprobs=0,
        _capture_media=False,
    )
    worker.install_token_capture = lambda capture: setattr(
        worker, "token_capture", capture
    )
    return worker


def test_generation_cut_route_matches_gym():
    """The worker serves cuts where Gym sends them; a drifted route 404s every cut."""
    assert _GENERATION_CUT_ROUTE == GENERATION_CUT_ROUTE


def test_setup_token_capture_installs_capture_with_vllm_adapter(monkeypatch):
    sink = _MemorySink()
    segment_sizes = []
    monkeypatch.setattr(
        "nemo_rl.data_plane.build_data_plane_client",
        lambda dp_cfg, bootstrap, segment_size=None: (
            segment_sizes.append(segment_size) or MagicMock(name="dp_client")
        ),
    )
    monkeypatch.setattr(
        "nemo_rl.data_plane.tq_token_sink.TQTokenSink",
        lambda dp_client, *, staging_partition, capture_media, media_pixel_dtype=None: (
            sink
        ),
    )
    worker = _fake_worker()

    installed = asyncio.run(
        VllmAsyncGenerationWorkerImpl.setup_token_capture(
            worker,
            dp_cfg={"backend": "simple"},
            staging_partition="rollout_staging",
            top_logprobs=4,
        )
    )

    assert installed is None  # text-only capture pins no media column
    # The capture client owns no Mooncake memory, so no save waits on vLLM.
    assert segment_sizes == [0]
    assert isinstance(worker.token_capture, RolloutTokenCapture)
    assert worker.token_capture.adapter is not None
    # The top-k count stays on the worker for the request path.
    assert worker._capture_top_logprobs == 4
    # The adapter is the vLLM one (prefix ids enter via the worker's field).
    payload = worker.token_capture.adapter.enter_prefix({}, [1, 2])
    assert payload["required_prefix_token_ids"] == [1, 2]


def test_setup_token_capture_skips_non_model_owners(monkeypatch):
    worker = _fake_worker(is_model_owner=False)
    installed = asyncio.run(
        VllmAsyncGenerationWorkerImpl.setup_token_capture(
            worker, dp_cfg={}, staging_partition="rollout_staging", top_logprobs=0
        )
    )
    assert installed is None
    assert worker.token_capture is None


def test_weight_version_is_stamped_from_worker_state(monkeypatch):
    """The install closure reads _rollout_weight_version live: a
    set_rollout_weight_version between calls changes the stamp."""
    sink = _MemorySink()
    monkeypatch.setattr(
        "nemo_rl.data_plane.build_data_plane_client",
        lambda dp_cfg, bootstrap, segment_size=None: MagicMock(),
    )
    monkeypatch.setattr(
        "nemo_rl.data_plane.tq_token_sink.TQTokenSink",
        lambda dp_client, *, staging_partition, capture_media, media_pixel_dtype=None: (
            sink
        ),
    )
    worker = _fake_worker()
    asyncio.run(
        VllmAsyncGenerationWorkerImpl.setup_token_capture(
            worker, dp_cfg={}, staging_partition="rollout_staging", top_logprobs=0
        )
    )

    asyncio.run(VllmAsyncGenerationWorkerImpl.set_rollout_weight_version(worker, 4))
    first = worker.token_capture.begin_call(
        CaptureAdmission(capture_key="r", model_call_id="c1", mode="text")
    )
    asyncio.run(VllmAsyncGenerationWorkerImpl.set_rollout_weight_version(worker, 5))
    second = worker.token_capture.begin_call(
        CaptureAdmission(capture_key="r", model_call_id="c2", mode="text")
    )

    assert (first.weight_version, second.weight_version) == (4, 5)

    coords = worker.token_capture.complete_call(
        first, prompt_token_ids=[1], generated_token_ids=[2], generated_logprobs=[-0.1]
    )
    assert coords.weight_version == 4
    assert sink.records[0].weight_version == 4


def _generation_with_mock_group(*, async_engine: bool = True) -> VllmGeneration:
    gen = object.__new__(VllmGeneration)
    gen.cfg = {"vllm_cfg": {"async_engine": async_engine}}
    gen.worker_group = MagicMock()
    gen.worker_group.run_all_workers_single_data.return_value = []
    return gen


def test_generation_setup_token_capture_fans_out(monkeypatch):
    gen = _generation_with_mock_group()
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures: futures,
    )
    gen.setup_token_capture(
        {"backend": "simple"},
        "rollout_staging",
        top_logprobs=4,
        generation_cut_control_timeout_s=60.0,
    )
    gen.worker_group.run_all_workers_single_data.assert_called_once_with(
        "setup_token_capture",
        dp_cfg={"backend": "simple"},
        staging_partition="rollout_staging",
        capture_media=False,
        top_logprobs=4,
        generation_prefix_cuts_enabled=False,
        generation_cut_control_token=None,
        generation_cut_control_timeout_s=60.0,
        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
    )


def test_generation_setup_token_capture_requires_async_engine():
    gen = _generation_with_mock_group(async_engine=False)
    with pytest.raises(AssertionError, match="async vLLM engine"):
        gen.setup_token_capture({}, "rollout_staging", top_logprobs=0)


@pytest.mark.parametrize(
    ("operation", "worker_method"),
    [
        (
            "begin_token_capture_snapshot_fence",
            "begin_token_capture_snapshot_fence_async",
        ),
        (
            "end_token_capture_snapshot_fence",
            "end_token_capture_snapshot_fence_async",
        ),
    ],
)
def test_token_capture_snapshot_fence_control_fans_out(
    monkeypatch, operation: str, worker_method: str
):
    gen = _generation_with_mock_group()
    gen.worker_group.workers = [object()]
    gen.worker_group.run_all_workers_single_data.return_value = [True]
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, timeout=None: futures,
    )

    assert getattr(gen, operation)(timeout_s=17.0)
    gen.worker_group.run_all_workers_single_data.assert_called_once_with(
        worker_method,
        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
    )


def test_generation_prefix_restore_counts_sum_over_workers(monkeypatch):
    gen = _generation_with_mock_group()
    gen.worker_group.workers = [object(), object()]
    gen.worker_group.run_all_workers_single_data.return_value = [
        {"restored_calls": 2, "restored_prefix_tokens": 50, "failed_cuts": 1},
        {"restored_calls": 1, "restored_prefix_tokens": 7, "failed_cuts": 0},
    ]
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, timeout=None: futures,
    )

    assert gen.get_generation_prefix_counts() == {
        "restored_calls": 3,
        "restored_prefix_tokens": 57,
        "failed_cuts": 1,
    }
    gen.worker_group.run_all_workers_single_data.assert_called_once_with(
        "get_generation_prefix_counts",
        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
    )


def test_prefix_capture_setup_requires_a_control_token(monkeypatch):
    # Prefix recovery with media or sampler top-k is refused at config/setup.
    monkeypatch.setattr(
        "nemo_rl.data_plane.build_data_plane_client",
        lambda dp_cfg, bootstrap, segment_size=None: MagicMock(),
    )
    worker = _fake_worker()
    with pytest.raises(ValueError, match="control bearer token"):
        asyncio.run(
            VllmAsyncGenerationWorkerImpl.setup_token_capture(
                worker,
                {},
                "rollout_staging",
                generation_prefix_cuts_enabled=True,
            )
        )


@pytest.mark.parametrize(
    ("max_tokens", "min_tokens", "prefix_token_count", "expected"),
    [
        (10, 6, 3, (7, 3)),
        (10, None, 3, (7, None)),
        (10, 2, 3, (7, 0)),
    ],
)
def test_restored_prefix_reduces_remaining_output_limits(
    max_tokens, min_tokens, prefix_token_count, expected
):
    assert (
        _remaining_generation_limits_after_prefix(
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            prefix_token_count=prefix_token_count,
        )
        == expected
    )


def _usage(prompt_tokens: int, completion_tokens: int, cached_tokens: int | None):
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )


def test_resumed_call_reports_its_restored_prefix_as_completion():
    # Prompt [1, 2, 3], restored generated prefix [4, 5], regenerated tail of 3.
    usage = _usage(prompt_tokens=5, completion_tokens=3, cached_tokens=5)

    prompt_token_ids = _count_restored_prefix_as_completion(
        usage, [1, 2, 3, 4, 5], restored_token_count=2, terminal=None
    )

    assert (usage.prompt_tokens, usage.completion_tokens) == (3, 5)
    assert usage.total_tokens == 8
    assert usage.prompt_tokens_details.cached_tokens == 3
    assert prompt_token_ids == [1, 2, 3]


def test_terminal_restored_prefix_is_the_whole_completion():
    terminal = _RestoredPrefixTerminal(
        prompt_token_ids=(1, 2, 3, 4, 5),
        prefix_token_count=2,
        reason="eos",
        finish_reason="stop",
    )
    # vLLM served a one-token request on the prefix-extended prompt.
    usage = _usage(prompt_tokens=4, completion_tokens=1, cached_tokens=None)

    prompt_token_ids = _count_restored_prefix_as_completion(
        usage, terminal.prompt_token_ids, restored_token_count=2, terminal=terminal
    )

    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (
        3,
        2,
        5,
    )
    assert prompt_token_ids == [1, 2, 3]


def test_restored_prefix_at_output_limit_is_terminal():
    terminal = _classify_restored_prefix_terminal(
        prompt_token_ids=[1, 2, 3, 4],
        prefix_token_count=2,
        requested_output_tokens=2,
        model_max_tokens=8,
    )
    assert terminal is not None
    assert terminal.reason == "output_limit"
    assert terminal.original_prompt_token_count == 2


def _stopped_terminal(
    finish_reason: str = "stop", stop_reason: str | int | None = None
) -> _RestoredPrefixTerminal:
    return _RestoredPrefixTerminal(
        prompt_token_ids=(1, 2),
        prefix_token_count=1,
        reason=f"observed_{finish_reason}",
        finish_reason=finish_reason,
        stop_reason=stop_reason,
    )


def test_restored_call_replays_its_prefix_with_the_original_settings(monkeypatch):
    """The prefix is replayed against the original prompt with the call's settings.

    The replay itself is vLLM's detokenizer; test_vllm_restored_prefix_replay.py
    checks it reproduces the uninterrupted response.
    """
    replays = []

    def replay(tokenizer, **kwargs):
        replays.append((tokenizer, kwargs))
        return "answer: 4"

    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_worker_async._detokenize_restored_prefix",
        replay,
    )
    tokenizer = object()
    terminal = _stopped_terminal(stop_reason="</answer>")
    state = SimpleNamespace(
        resumed_generation_token_ids=[12, 13],
        # The engine prompt ends with the spliced prefix.
        prompt_token_ids=[10, 11, 12, 13],
        lock=threading.Lock(),
    )
    worker = SimpleNamespace(_get_request_capture=lambda request: state)
    request = SimpleNamespace(
        _restored_prefix_terminal=terminal,
        # Admission reduces min_tokens by the prefix length; the replay needs
        # the call's original value, which gated its stop strings.
        _original_min_tokens=4,
        min_tokens=2,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )
    # The engine is skipped for a call that already stopped: its tail is empty.
    output = SimpleNamespace(token_ids=[], text="")

    VllmAsyncGenerationWorkerImpl._restore_response_prefix(
        worker,
        request,
        SimpleNamespace(outputs=[output]),
        tokenizer=tokenizer,
    )

    assert output.text == "answer: 4"
    assert replays == [
        (
            tokenizer,
            {
                "original_prompt_token_ids": [10, 11],
                "prefix_token_ids": [12, 13],
                "terminal": terminal,
                "stop": ["</answer>"],
                "min_tokens": 4,
                "include_stop_str_in_output": True,
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
            },
        )
    ]


def _captured_text_call(
    worker, model_call_id: str = "c1", prompt_token_ids: tuple[int, ...] = (10, 11)
) -> _FakeRequest:
    """Begin capturing a parentless text call of rollout ``r0``."""
    request = _FakeRequest(
        ng_capture={"rollout_id": "r0", "model_call_id": model_call_id, "mode": "text"},
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(
        worker, request, list(prompt_token_ids)
    )
    return request


def _live_call(
    worker,
    generated_token_ids: list[int],
    logprobs: list[float],
    *,
    model_call_id: str = "c1",
    prompt_token_ids: tuple[int, ...] = (10, 11),
):
    """Begin one captured call that has decoded ``generated_token_ids``."""
    request = _captured_text_call(worker, model_call_id, prompt_token_ids)
    state = worker._capture_calls[id(request)]
    with state.lock:
        state.effective_output_limit = 8
        state.observe(generated_token_ids, logprobs)
    return request, state


def test_finished_output_records_only_tokens_past_the_last_cut():
    worker = _worker_with_capture(_MemorySink())
    worker._generation_prefix_cuts_enabled = True
    request = _captured_text_call(worker)
    state = worker._capture_calls[id(request)]
    with state.lock:
        state.observe([12], [-0.1], start=0)  # read earlier by a cut
    final = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                token_ids=[12, 13, 14],
                logprobs=[
                    {12: SimpleNamespace(logprob=-9.0)},
                    {13: SimpleNamespace(logprob=-0.2)},
                    {14: SimpleNamespace(logprob=-0.3)},
                ],
                finish_reason="stop",
            )
        ]
    )

    VllmAsyncGenerationWorkerImpl._observe_request_capture(worker, request, final)

    assert state.observation_error is None
    assert state.generated_token_ids == [12, 13, 14]
    assert state.generated_logprobs == [-0.1, -0.2, -0.3]
    assert state.terminal_finish_reason == "stop"


def test_tokens_read_by_both_a_cut_and_the_finished_output_are_recorded_once():
    state = _RequestCaptureState(call=None, prompt_token_ids=[1])
    state.observe([12, 13], [-0.1, -0.2], start=0)
    state.observe([12, 13, 14], [-0.1, -0.2, -0.3], start=0)
    state.observe([14], [-0.3], start=2)

    assert state.generated_token_ids == [12, 13, 14]
    assert state.generated_logprobs == [-0.1, -0.2, -0.3]
    assert state.observation_error is None


def _engine_request(
    token_ids: list[int],
    logprobs: list[float],
    *,
    request_id: str = "chatcmpl-1",
    finish_reason: str | None = None,
) -> SimpleNamespace:
    """The RequestState fields a progress read touches."""
    final = SimpleNamespace(
        outputs=[SimpleNamespace(finish_reason=finish_reason, stop_reason=None)]
    )
    return SimpleNamespace(
        request_id=f"{request_id}-internal",
        external_req_id=request_id,
        queue=SimpleNamespace(output=final),
        detokenizer=SimpleNamespace(output_token_ids=list(token_ids)),
        logprobs_processor=SimpleNamespace(
            logprobs=[
                {token_id: SimpleNamespace(logprob=logprob)}
                for token_id, logprob in zip(token_ids, logprobs, strict=True)
            ]
        ),
    )


class _OutputProcessor:
    """vLLM 0.29's OutputProcessor surface: live requests and the two hooks."""

    def __init__(self, *requests: SimpleNamespace) -> None:
        self.request_states = {r.request_id: r for r in requests}
        self.external_req_ids = {r.external_req_id: [r.request_id] for r in requests}
        self.processed_under_lock: list[bool] = []

    def process_outputs(self, client) -> None:
        self.processed_under_lock.append(client._progress_lock._is_owned())

    def _finish_request(self, req_state) -> None:
        self.request_states.pop(req_state.request_id)
        del self.external_req_ids[req_state.external_req_id]


def _progress_client(
    monkeypatch,
    processor: _OutputProcessor,
    *,
    vllm_version: str = "0.29.0",
    enable: bool = True,
) -> _AsyncLLMHTTPClient:
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_worker_async.importlib.metadata.version",
        lambda name: vllm_version,
    )
    engine = SimpleNamespace(
        output_processor=processor,
        model_config=None,
        renderer=None,
        input_processor=None,
        vllm_config=None,
    )
    client = _AsyncLLMHTTPClient(engine, MagicMock(), flat_logprobs=False)
    if enable:
        client.enable_generation_progress_reads()
    return client


def test_vllm_processes_outputs_under_the_progress_lock(monkeypatch):
    processor = _OutputProcessor()
    client = _progress_client(monkeypatch, processor)

    processor.process_outputs(client)

    assert processor.processed_under_lock == [True]


def test_finished_request_stays_readable_until_its_consumer_is_done(monkeypatch):
    request = _engine_request([20, 21], [-0.1, -0.2], finish_reason="stop")
    unserved = _engine_request([30], [-0.3], request_id="offline-1")
    processor = _OutputProcessor(request, unserved)
    client = _progress_client(monkeypatch, processor)
    client._captured_requests["chatcmpl-1"] = "c1"

    processor._finish_request(request)
    processor._finish_request(unserved)
    read = client.read_generation_progress([("c1", 1), ("c2", 0)], timeout_s=None)

    [finished, gone] = read.progress
    assert (finished.token_ids, finished.finish_reason) == ([21], "stop")
    # Only captured requests are kept after they finish.
    assert gone is None
    assert "offline-1" not in client._finished_requests


def test_finished_request_is_released_once_its_consumer_has_the_output(monkeypatch):
    request = _engine_request([20], [-0.1], finish_reason="length")
    processor = _OutputProcessor(request)
    client = _progress_client(monkeypatch, processor)

    async def consume() -> list:
        client._engine_loop = asyncio.get_running_loop()
        seen = []

        async def engine_generate(*args, **kwargs):
            processor._finish_request(request)
            yield "final output"

        client._engine_client.generate = engine_generate
        _CAPTURED_MODEL_CALL.set("c1")
        async for output in client._generate("prompt", None, "chatcmpl-1", {}):
            # The consumer records the output while vLLM's state is still kept.
            seen.append(client._finished_requests["chatcmpl-1"].finish_reason)
        return seen

    assert asyncio.run(consume()) == ["length"]
    assert client._finished_requests == {}
    assert client._captured_requests == {}


def test_progress_reads_refuse_an_unchecked_vllm(monkeypatch):
    client = _progress_client(
        monkeypatch, _OutputProcessor(), vllm_version="0.30.1", enable=False
    )

    with pytest.raises(RuntimeError, match="checked on vLLM 0.29.x"):
        client.enable_generation_progress_reads()


def test_cut_reads_each_call_progress_from_vllm_request_state(monkeypatch):
    worker = _worker_with_capture(_MemorySink())
    decoding, resumed, finished, unsubmitted = (
        worker._capture_calls[id(_captured_text_call(worker, call_id))]
        for call_id in ("c1", "c2", "c3", "c4")
    )
    resumed.observe([30], [-0.3])  # recorded by an earlier cut
    done = _engine_request(
        [40, 41], [-0.5, -0.6], request_id="chatcmpl-3", finish_reason="stop"
    )
    processor = _OutputProcessor(
        _engine_request([20, 21], [-0.1, -0.2]),
        # The earlier cut's position is not read again.
        _engine_request([30, 31], [-9.0, -0.4], request_id="chatcmpl-2"),
        done,
    )
    worker._http_engine_client = _progress_client(monkeypatch, processor)
    # c4 has not reached vLLM yet, so the HTTP client has no request for it.
    worker._http_engine_client._captured_requests.update(
        {"chatcmpl-1": "c1", "chatcmpl-2": "c2", "chatcmpl-3": "c3"}
    )
    processor._finish_request(done)

    VllmAsyncGenerationWorkerImpl._record_generation_progress(
        worker, [decoding, resumed, finished, unsubmitted], deadline=None
    )

    assert (decoding.generated_token_ids, decoding.generated_logprobs) == (
        [20, 21],
        [-0.1, -0.2],
    )
    assert (resumed.generated_token_ids, resumed.generated_logprobs) == (
        [30, 31],
        [-0.3, -0.4],
    )
    # Finished but not yet taken by its consumer: read whole, with its reason.
    assert finished.generated_token_ids == [40, 41]
    assert finished.terminal_finish_reason == "stop"
    # Never submitted: it keeps what it recorded.
    assert unsubmitted.generated_token_ids == []
    assert all(
        state.observation_error is None
        for state in (decoding, resumed, finished, unsubmitted)
    )
    counts = _cut_counts(worker, progress_reads=True)
    assert (counts["progress_reads"], counts["progress_read_calls"]) == (1, 4)


def test_cut_past_its_deadline_skips_reading_vllm_progress(monkeypatch, caplog):
    worker = _worker_with_capture(_MemorySink())
    state = worker._capture_calls[id(_captured_text_call(worker))]
    client = _progress_client(
        monkeypatch, _OutputProcessor(_engine_request([20], [-0.1]))
    )
    client._captured_requests["chatcmpl-1"] = "c1"
    worker._http_engine_client = client
    vllm_busy, release = threading.Event(), threading.Event()

    def process_a_long_chunk():
        with client._progress_lock:
            vllm_busy.set()
            release.wait(timeout=5)

    vllm = threading.Thread(target=process_a_long_chunk)
    vllm.start()
    vllm_busy.wait(timeout=5)
    try:
        with caplog.at_level(logging.WARNING):
            VllmAsyncGenerationWorkerImpl._record_generation_progress(
                worker, [state], deadline=time.monotonic()
            )
    finally:
        release.set()
        vllm.join(timeout=5)

    assert state.generated_token_ids == []
    assert "could not read vLLM's progress before its deadline" in caplog.text


def test_terminal_token_without_a_logprob_makes_its_call_uncuttable():
    worker = _worker_with_capture(_MemorySink())
    worker._generation_prefix_cuts_enabled = True
    request = _captured_text_call(worker)
    # Nothing appends after the terminal step, so this logprob never arrives.
    final = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                token_ids=[12, 13],
                logprobs=[{12: SimpleNamespace(logprob=-0.1)}],
                finish_reason="stop",
            )
        ]
    )

    VllmAsyncGenerationWorkerImpl._observe_request_capture(worker, request, final)

    state = worker._capture_calls[id(request)]
    assert "token_count=2, logprob_count=1" in state.observation_error
    assert state.terminal_finish_reason is None


def test_generation_cut_stages_only_tokens_past_the_last_sealed_cut():
    state = _RequestCaptureState(call=None, prompt_token_ids=[1])
    state.observe([10, 11], [-0.1, -0.2])

    first = state.freeze_for_checkpoint("c1")
    assert first == ("active/c1/0", 0, [10, 11], [-0.1, -0.2], [10, 11], [-0.1, -0.2])
    state.observe([12], [-0.3])  # decoding continues during the TQ write
    state.seal_frozen_cut("c1")
    state.observe([13], [-0.4])

    key, sequence, chunk, chunk_logprobs, prefix, _ = state.freeze_for_checkpoint("c2")
    assert (key, sequence, chunk, chunk_logprobs, prefix) == (
        "active/c2/1",
        1,
        [12, 13],
        [-0.3, -0.4],
        [10, 11, 12, 13],
    )
    state.rollback_frozen_cut("c2")

    # A failed cut's tokens stay unstaged, and its chunk name is never reused.
    key, sequence, chunk, _, _, _ = state.freeze_for_checkpoint("c3")
    assert (key, sequence, chunk) == ("active/c3/2", 2, [12, 13])


def test_token_capture_snapshot_fence_does_not_pause_decoding():
    class FakeLLM:
        pause_calls = 0

        async def pause_generation(self, **kwargs):
            self.pause_calls += 1

    async def scenario():
        worker = object.__new__(VllmAsyncGenerationWorkerImpl)
        worker.cfg = {"vllm_cfg": {"async_engine": True}}
        worker.llm = FakeLLM()
        worker._token_capture_snapshot_gate = _TokenCaptureSnapshotGate()
        worker._token_capture_fence_executor = ThreadPoolExecutor(max_workers=1)
        try:
            assert await worker.begin_token_capture_snapshot_fence_async()
            assert worker.llm.pause_calls == 0
            entered = threading.Event()
            released = threading.Event()

            def terminal_write():
                entered.set()
                worker._token_capture_snapshot_gate.enter()
                released.set()
                worker._token_capture_snapshot_gate.exit()

            thread = threading.Thread(target=terminal_write)
            thread.start()
            assert entered.wait(timeout=5)
            assert not released.wait(timeout=0.05)
            assert await worker.end_token_capture_snapshot_fence_async()
            assert released.wait(timeout=5)
            thread.join(timeout=5)
        finally:
            worker._token_capture_snapshot_gate.reopen()
            worker._token_capture_fence_executor.shutdown()

    asyncio.run(scenario())


def _gate_admits_write(gate: _TokenCaptureSnapshotGate, *, timeout: float) -> bool:
    """Whether a terminal write passes the gate within ``timeout`` seconds."""
    admitted = threading.Event()

    def terminal_write():
        gate.enter()
        admitted.set()
        gate.exit()

    threading.Thread(target=terminal_write, daemon=True).start()
    return admitted.wait(timeout=timeout)


def test_snapshot_gate_close_after_its_release_is_a_no_op():
    gate = _TokenCaptureSnapshotGate()
    epoch = gate.begin_epoch()
    # The driver timed out and released before the worker ran the close.
    gate.reopen()
    gate.close_and_wait(epoch)
    assert _gate_admits_write(gate, timeout=5)

    # A later fence still closes the gate normally.
    gate.close_and_wait(gate.begin_epoch())
    assert not _gate_admits_write(gate, timeout=0.05)
    gate.reopen()


def test_snapshot_gate_release_unblocks_a_close_still_draining():
    gate = _TokenCaptureSnapshotGate()
    gate.enter()
    closer = threading.Thread(
        target=gate.close_and_wait, args=(gate.begin_epoch(),), daemon=True
    )
    closer.start()
    closer.join(timeout=0.05)
    assert closer.is_alive()
    gate.reopen()
    closer.join(timeout=5)
    assert not closer.is_alive()
    gate.exit()
    assert _gate_admits_write(gate, timeout=5)


def _fence_worker() -> VllmAsyncGenerationWorkerImpl:
    worker = object.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {"vllm_cfg": {"async_engine": True}}
    worker._token_capture_snapshot_gate = _TokenCaptureSnapshotGate()
    worker._generation_cut_control_executor = ThreadPoolExecutor(max_workers=1)
    worker._token_capture_fence_executor = ThreadPoolExecutor(max_workers=1)
    return worker


def _shutdown_fence_worker(worker: VllmAsyncGenerationWorkerImpl) -> None:
    worker._token_capture_snapshot_gate.reopen()
    worker._generation_cut_control_executor.shutdown()
    worker._token_capture_fence_executor.shutdown()


def test_snapshot_fence_is_not_queued_behind_a_running_cut():
    async def scenario():
        worker = _fence_worker()
        stale_cut_running = threading.Event()
        release_stale_cut = threading.Event()
        try:
            stale_cut = asyncio.ensure_future(
                worker._run_generation_cut_control(
                    lambda: (stale_cut_running.set(), release_stale_cut.wait(10))
                )
            )
            assert await asyncio.to_thread(stale_cut_running.wait, 5)
            await asyncio.wait_for(
                worker.begin_token_capture_snapshot_fence_async(), timeout=5
            )
            assert not stale_cut.done()
            release_stale_cut.set()
            await stale_cut
        finally:
            release_stale_cut.set()
            _shutdown_fence_worker(worker)

    asyncio.run(scenario())


def test_snapshot_fence_closed_after_its_release_leaves_the_gate_open():
    async def scenario():
        worker = _fence_worker()
        release_fence_thread = threading.Event()
        try:
            # Hold the fence thread so the close is still queued when the
            # driver gives up and releases the fence.
            worker._token_capture_fence_executor.submit(release_fence_thread.wait, 10)
            begin = asyncio.ensure_future(
                worker.begin_token_capture_snapshot_fence_async()
            )
            await asyncio.sleep(0)
            assert not begin.done()
            assert await worker.end_token_capture_snapshot_fence_async()
            release_fence_thread.set()
            assert await asyncio.wait_for(begin, timeout=5)
            assert _gate_admits_write(worker._token_capture_snapshot_gate, timeout=5)
        finally:
            release_fence_thread.set()
            _shutdown_fence_worker(worker)

    asyncio.run(scenario())


def test_generation_set_rollout_weight_version_fans_out(monkeypatch):
    gen = _generation_with_mock_group()
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures: futures,
    )
    gen.set_rollout_weight_version(7)
    gen.worker_group.run_all_workers_single_data.assert_called_once_with(
        "set_rollout_weight_version",
        version=7,
        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
    )


# ---------------------------------------------------------------------------
# S4: the request-path hookup (begin -> finish/abort around a served call)
# ---------------------------------------------------------------------------


class _FakeRequest(SimpleNamespace):
    pass


def _cut_counts(worker, *, progress_reads: bool = False) -> dict[str, int]:
    """The worker's nonzero generation-cut counters.

    Every cut reads vLLM's progress first; those counters are left out unless
    ``progress_reads`` asks for them.
    """
    counts = VllmAsyncGenerationWorkerImpl.get_generation_prefix_counts(worker)
    return {
        name: count
        for name, count in counts.items()
        if count
        and name not in ("restored_calls", "restored_prefix_tokens")
        and (progress_reads or not name.startswith("progress_read"))
    }


def _single_call_inventory(
    model_call_id: str = "c1",
    *,
    checkpoint_id: str = "checkpoint-1",
    attempt: int = 0,
) -> GenerationCutInventory:
    """Gym's inventory naming one active call of rollout ``r0``."""
    return GenerationCutInventory.build(
        checkpoint_id=checkpoint_id,
        server_name="policy_model",
        active_prefixes=[
            GenerationCutPrefix(
                ticket_id="ticket-1",
                rollout_id="r0",
                attempt=attempt,
                model_call_id=model_call_id,
                admitted_at=1.0,
            )
        ],
    )


def _worker_with_capture(sink: _MemorySink):
    """A real worker, minus the engine, with in-memory token capture."""
    from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter

    worker = object.__new__(VllmAsyncGenerationWorkerImpl)
    fake = vars(_fake_worker())
    fake.pop("install_token_capture")
    vars(worker).update(fake)
    worker._capture_sink = sink
    worker.token_capture = RolloutTokenCapture(
        sink=sink,
        weight_version_fn=lambda: worker._rollout_weight_version,
        adapter=VLLMCaptureAdapter(),
    )
    return worker


@pytest.mark.parametrize(
    ("logprobs", "top_logprobs"),
    [(False, 4), (True, None), (True, 0), (True, 2), (True, 8)],
)
def test_capture_rejects_mismatched_http_topk(logprobs, top_logprobs):
    worker = _worker_with_capture(_MemorySink())
    worker._capture_top_logprobs = 4
    request = SimpleNamespace(logprobs=logprobs, top_logprobs=top_logprobs)
    admission = CaptureAdmission(rollout_id="r0", model_call_id="c1", mode="text")

    with pytest.raises(ValueError, match="top_logprobs=4 over HTTP"):
        VllmAsyncGenerationWorkerImpl._begin_request_capture(
            worker, request, [10, 11], admission=admission
        )
    assert worker._capture_calls == {}


def test_capture_accepts_matching_http_topk():
    worker = _worker_with_capture(_MemorySink())
    worker._capture_top_logprobs = 4
    request = SimpleNamespace(logprobs=True, top_logprobs=4)
    admission = CaptureAdmission(rollout_id="r0", model_call_id="c1", mode="text")

    VllmAsyncGenerationWorkerImpl._begin_request_capture(
        worker, request, [10, 11], admission=admission
    )
    assert id(request) in worker._capture_calls
    assert request.top_logprobs == 4


class _MemoryPrefixSource:
    def __init__(
        self,
        deltas: dict[str, list[int]],
        *,
        records: dict[str, StagedCallRecord] | None = None,
    ) -> None:
        self.deltas = deltas
        self.records = records or {}
        self.calls: list[list[str]] = []

    def fetch_prefix_chains(self, staging_keys: list[str]) -> StagedPrefix:
        """Text-only chains: the vLLM worker never stages media through this source."""
        self.calls.append(list(staging_keys))
        return StagedPrefix(
            expanded=[token for key in staging_keys for token in self.deltas[key]]
        )

    def fetch(self, staging_keys: list[str]) -> list[StagedCallRecord]:
        self.calls.append(list(staging_keys))
        return [self.records[key] for key in staging_keys]


def _served_content(gen_ids, logprobs):
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "x"},
                "logprobs": {
                    "content": [
                        {"token": f"token_id:{t}", "logprob": lp}
                        for t, lp in zip(gen_ids, logprobs)
                    ]
                },
            }
        ]
    }


@pytest.mark.parametrize("with_message_tokens", [False, True])
@pytest.mark.parametrize("with_routed_experts", [False, True])
def test_request_capture_round_trip_stages_and_rides_coords(
    with_message_tokens: bool, with_routed_experts: bool
) -> None:
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _captured_text_call(worker, "c1", (10, 11, 12))
    content = _served_content([13, 14], [-0.1, -0.2])
    message = content["choices"][0]["message"]
    if with_message_tokens:
        # The HTTP serializer preserves these dynamic fields on the message.
        message.update(
            prompt_token_ids=[10, 11, 12],
            generation_token_ids=[13, 14],
            generation_log_probs=[-0.1, -0.2],
        )
    if with_routed_experts:
        message["routed_experts"] = [[[0]]] * 5
    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, content
    )
    # Bytes were staged before the coords existed (fail-closed ordering).
    assert len(sink.records) == 1
    assert sink.records[0].token_ids_delta == [10, 11, 12, 13, 14]
    assert sink.records[0].token_mask_delta == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert sink.records[0].generation_log_probs_delta == [
        0.0,
        0.0,
        0.0,
        -0.1,
        -0.2,
    ]
    if with_routed_experts:
        assert sink.records[0].extras["routed_experts"]
    else:
        assert sink.records[0].extras is None
    coords = content["ng_commit_coords"]
    assert coords["disposition"] == "staged"
    assert (coords["delta_len"], coords["cum_len"]) == (5, 5)
    # Coords are token-free: hashes ride the wire, deltas stay in the sink.
    assert "token_ids_delta" not in coords
    assert coords["chain_hash"] == sink.records[0].chain_hash
    assert coords["cumulative_hash"] == sink.records[0].cumulative_hash
    # Only the ordinary response fields and coords transit worker -> gate.
    assert content["choices"] == [
        {"index": 0, "message": {"role": "assistant", "content": "x"}}
    ]
    assert worker._capture_calls == {}


def test_generation_cut_stages_prefix_then_terminal_row_replaces_it():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request, state = _live_call(worker, [20, 21], [-0.1, -0.2])

    inventory = _single_call_inventory()
    receipt = worker._checkpoint_generation_cut(inventory)
    replayed = worker._checkpoint_generation_cut(inventory)

    assert replayed is receipt
    assert len(sink.generation_prefix_records) == 1
    assert receipt.prefixes[0].disposition == "durable_prefix"
    assert receipt.prefixes[0].prefix_token_count == 2
    assert sink.generation_prefix_records[0][1].token_ids_delta == [10, 11, 20, 21]
    assert state.generation_cut_staging_keys == sink.generation_prefix_keys

    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker,
        request,
        _served_content([20, 21, 22], [-0.1, -0.2, -0.3]),
    )
    assert content["ng_commit_coords"]["disposition"] == "staged"
    assert sink.records[-1].token_ids_delta == [10, 11, 20, 21, 22]
    assert sink.cleared_generation_prefix_keys == sink.generation_prefix_keys
    assert worker._capture_calls == {}
    assert worker._capture_calls_by_model_call_id == {}


def test_generation_cut_stages_the_tokens_it_reads_from_vllm(monkeypatch):
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _captured_text_call(worker)
    state = worker._capture_calls[id(request)]
    with state.lock:
        state.effective_output_limit = 8
    worker._http_engine_client = _progress_client(
        monkeypatch, _OutputProcessor(_engine_request([20, 21], [-0.1, -0.2]))
    )
    worker._http_engine_client._captured_requests["chatcmpl-1"] = "c1"
    inventory = _single_call_inventory()

    receipt = worker._checkpoint_generation_cut(inventory)

    assert receipt.prefixes[0].disposition == "durable_prefix"
    assert receipt.prefixes[0].prefix_token_count == 2
    assert sink.generation_prefix_records[0][1].token_ids_delta == [10, 11, 20, 21]
    assert sink.generation_prefix_records[0][1].generation_log_probs_delta[2:] == [
        -0.1,
        -0.2,
    ]


@pytest.mark.parametrize(
    ("completed_capture_key", "matches"), [("r0-a1", True), ("r0", False)]
)
def test_completed_retry_is_matched_by_its_attempt_qualified_capture_key(
    completed_capture_key: str, matches: bool
):
    """A retry's terminal row is keyed ``<rollout>-a<attempt>``, as Gym keys it."""
    worker = _worker_with_capture(_MemorySink())
    worker._remember_completed_capture(
        "c2",
        SimpleNamespace(
            capture_key=completed_capture_key,
            disposition="staged",
            staging_key="terminal-key",
            digest="a" * 64,
        ),
        3,
        effective_output_limit=128,
        terminal_finish_reason="stop",
    )
    prefix = GenerationCutPrefix(
        ticket_id="ticket-1",
        rollout_id="r0",
        attempt=1,
        model_call_id="c2",
        admitted_at=1.0,
    )

    if not matches:
        with pytest.raises(RuntimeError, match="does not match the completed call"):
            worker._completed_generation_cut_ack(prefix, "checkpoint-1")
        return
    ack = worker._completed_generation_cut_ack(prefix, "checkpoint-1")

    assert ack.disposition == "durable_prefix"
    assert ack.cut_kind == "terminal_completion"
    assert ack.staging_keys == ("terminal-key",)
    assert ack.prefix_token_count == 3
    assert _cut_counts(worker) == {"completed_cuts_kept": 1}


def test_generation_cut_past_its_deadline_fails_uncut_calls_without_staging(caplog):
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    _, state = _live_call(worker, [11, 12], [-0.1, -0.2], prompt_token_ids=(10,))
    inventory = _single_call_inventory()

    # Gym has already stopped waiting: the queued cut must not stage rows.
    with caplog.at_level(logging.WARNING):
        receipt = worker._checkpoint_generation_cut(
            inventory, deadline=time.monotonic() - 1.0
        )

    receipt.validate_for(inventory)
    assert [ack.disposition for ack in receipt.prefixes] == ["durable_failure"]
    assert sink.generation_prefix_records == []
    assert state.frozen_cut is None
    assert state.generated_token_ids == [11, 12]
    assert "uncut=1 of 1 model_call_ids=['c1']" in caplog.text
    assert _cut_counts(worker) == {"failed_cuts_control_deadline": 1, "failed_cuts": 1}


def _active_call_inventory(
    worker, *calls: tuple[str, list[int]]
) -> tuple[GenerationCutInventory, list]:
    """Admit one live call per (model_call_id, generated IDs) and inventory them."""
    states = []
    prefixes = []
    for index, (model_call_id, generated) in enumerate(calls):
        request = _FakeRequest(
            ng_capture={
                "rollout_id": f"r{index}",
                "model_call_id": model_call_id,
                "mode": "text",
            },
            stream=False,
        )
        VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10])
        state = worker._capture_calls[id(request)]
        with state.lock:
            state.effective_output_limit = 16
            state.observe(generated, [-0.1] * len(generated))
        states.append(state)
        prefixes.append(
            GenerationCutPrefix(
                ticket_id=f"ticket-{index}",
                rollout_id=f"r{index}",
                attempt=0,
                model_call_id=model_call_id,
                admitted_at=1.0,
            )
        )
    inventory = GenerationCutInventory.build(
        checkpoint_id="checkpoint-1",
        server_name="policy_model",
        active_prefixes=prefixes,
    )
    return inventory, states


def test_generation_cut_refuses_a_prefix_ending_inside_a_character(caplog):
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    dinosaur = list("🦖".encode())
    inventory, (split, whole) = _active_call_inventory(
        worker,
        ("split", list(b"Sure ") + dinosaur[:2]),
        ("whole", list(b"Sure ") + dinosaur),
    )

    with caplog.at_level(logging.WARNING):
        receipt = worker._checkpoint_generation_cut(inventory)

    # vLLM would prime the resumed tail on half a character and echo the
    # prompt into the response, so that call regenerates instead.
    assert [ack.disposition for ack in receipt.prefixes] == [
        "durable_failure",
        "durable_prefix",
    ]
    assert len(sink.generation_prefix_records) == 1
    assert split.frozen_cut is None
    assert split.generated_token_ids == list(b"Sure ") + dinosaur[:2]
    assert whole.generation_cut_staging_keys == sink.generation_prefix_keys
    assert (
        "cannot cut model call split: its 7 generated tokens end inside a "
        "multi-byte character"
    ) in caplog.text
    assert "model call whole" not in caplog.text
    assert _cut_counts(worker) == {
        "failed_cuts_incomplete_utf8_boundary": 1,
        "failed_cuts": 1,
    }


def test_generation_cut_of_a_call_with_no_tokens_logs_at_debug(caplog):
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    inventory, (empty,) = _active_call_inventory(worker, ("empty", []))

    with caplog.at_level(
        logging.DEBUG, logger="nemo_rl.models.generation.generation_cut_capture"
    ):
        receipt = worker._checkpoint_generation_cut(inventory)

    assert [ack.disposition for ack in receipt.prefixes] == ["durable_failure"]
    assert _cut_counts(worker) == {
        "failed_cuts_zero_generated_tokens": 1,
        "failed_cuts": 1,
    }
    assert sink.generation_prefix_records == []
    assert empty.frozen_cut is None
    # Still prefilling is expected, so it stays out of WARNING.
    [record] = [r for r in caplog.records if "model call empty" in r.getMessage()]
    assert record.levelno == logging.DEBUG
    assert "it has no generated tokens yet" in record.getMessage()


@pytest.mark.parametrize(
    ("completed", "reason", "category"),
    [
        (None, "this worker has no record of it", "no_capture_or_completed_state"),
        (
            {"disposition": "capture_failed"},
            "its terminal write ended as 'capture_failed'",
            "completed_not_staged",
        ),
        (
            {"effective_output_limit": None},
            "vLLM did not resolve its effective output limit",
            "completed_missing_output_limit",
        ),
        (
            {"terminal_finish_reason": None},
            "it finished without a stop or length finish reason",
            "completed_missing_finish_reason",
        ),
    ],
)
def test_completed_call_that_cannot_be_cut_logs_why(
    completed, reason, category, caplog
):
    worker = _worker_with_capture(_MemorySink())
    if completed is not None:
        worker._remember_completed_capture(
            "c2",
            SimpleNamespace(
                capture_key="r0",
                disposition=completed.get("disposition", "staged"),
                staging_key="terminal-key",
                digest="a" * 64,
            ),
            3,
            effective_output_limit=completed.get("effective_output_limit", 128),
            terminal_finish_reason=completed.get("terminal_finish_reason", "stop"),
        )
    prefix = GenerationCutPrefix(
        ticket_id="ticket-1",
        rollout_id="r0",
        attempt=0,
        model_call_id="c2",
        admitted_at=1.0,
    )

    with caplog.at_level(logging.WARNING):
        ack = worker._completed_generation_cut_ack(prefix, "checkpoint-1")

    assert ack.disposition == "durable_failure"
    assert f"cannot cut completed model call c2: {reason}" in caplog.text
    assert _cut_counts(worker) == {f"failed_cuts_{category}": 1, "failed_cuts": 1}


def test_call_with_a_recording_error_is_not_cut():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    inventory, (poisoned, healthy) = _active_call_inventory(
        worker, ("poisoned", [11, 12]), ("healthy", [13, 14])
    )
    poisoned.observation_error = "RuntimeError: vLLM returned no logprobs"

    receipt = worker._checkpoint_generation_cut(inventory)

    assert [ack.disposition for ack in receipt.prefixes] == [
        "durable_failure",
        "durable_prefix",
    ]
    assert poisoned.frozen_cut is None
    assert healthy.generation_cut_staging_keys == sink.generation_prefix_keys
    assert _cut_counts(worker) == {"failed_cuts_observation_error": 1, "failed_cuts": 1}


class _WordTokenizer:
    """One token per whitespace-separated word, the shape vLLM's expansion reads."""

    max_token_id = 1000

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        return [len(word) for word in text.split()]


class _SamplingParams(SimpleNamespace):
    """The SamplingParams surface the cut checks read."""

    _bad_words_token_ids = None

    def clone(self) -> "_SamplingParams":
        return _SamplingParams(**vars(self))

    def update_from_tokenizer(self, tokenizer) -> None:
        self._bad_words_token_ids = [
            tokenizer.encode(word, add_special_tokens=False) for word in self.bad_words
        ]


def _sampling_params(**overrides) -> _SamplingParams:
    defaults = dict(stop=[], bad_words=[], presence_penalty=0.0, frequency_penalty=0.0)
    return _SamplingParams(**{**defaults, **overrides})


def _record_cut_constraints(state, sampling_params) -> None:
    VllmAsyncGenerationWorkerImpl._record_request_cut_constraints(
        SimpleNamespace(
            _get_request_capture=lambda request: state,
            _generation_cut_tokenizer=_WordTokenizer(),
        ),
        object(),
        sampling_params,
    )


@pytest.mark.parametrize(
    "setting",
    [
        {"stop": ["</answer>"]},
        {"bad_words": ["forbidden word"]},
        {"presence_penalty": 0.5},
        {"frequency_penalty": -0.2},
        {"seed": 7, "temperature": 1.0},
    ],
)
def test_sampling_settings_a_resumed_tail_cannot_honor_keep_the_call_uncut(
    setting, monkeypatch
):
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_worker_async._LOGGED_UNCUTTABLE_REASONS",
        set(),
    )
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    inventory, (constrained, plain) = _active_call_inventory(
        worker, ("constrained", [11, 12]), ("plain", [13, 14])
    )
    _record_cut_constraints(constrained, _sampling_params(**setting))
    _record_cut_constraints(plain, _sampling_params())
    assert constrained.uncuttable_reason is not None
    assert plain.uncuttable_reason is None

    receipt = worker._checkpoint_generation_cut(inventory)

    # The constrained call regenerates after a restore instead of resuming a
    # tail that would miss a straddling stop string or uncount its prefix.
    assert [ack.disposition for ack in receipt.prefixes] == [
        "durable_failure",
        "durable_prefix",
    ]
    assert constrained.frozen_cut is None
    assert constrained.generated_token_ids == [11, 12]
    assert plain.generation_cut_staging_keys == sink.generation_prefix_keys
    assert _cut_counts(worker) == {
        "failed_cuts_uncuttable_request": 1,
        "failed_cuts": 1,
    }


@pytest.mark.parametrize(
    ("seed", "temperature", "blocked"),
    [
        # A restored call would restart the seeded generator and reuse its draws.
        (7, 1.0, True),
        # Greedy decoding draws nothing, so the seed is irrelevant.
        (7, 0.0, False),
        # Unseeded calls draw from the engine's shared generator.
        (None, 1.0, False),
    ],
)
def test_only_a_seeded_sampled_call_is_kept_out_of_cuts(seed, temperature, blocked):
    blockers = _generation_cut_blockers(
        _sampling_params(seed=seed, temperature=temperature), _WordTokenizer()
    )

    assert ("seed" in blockers) is blocked


@pytest.mark.parametrize(
    ("bad_words", "blocked"),
    [
        # Banned at every step whatever precedes it, before or after a cut.
        (["forbidden", "banned"], False),
        # Its first token can sit in the restored prefix, where vLLM no longer looks.
        (["forbidden", "two words"], True),
    ],
)
def test_only_multi_token_bad_words_keep_a_call_uncut(bad_words, blocked):
    sampling_params = _sampling_params(bad_words=bad_words)

    blockers = _generation_cut_blockers(sampling_params, _WordTokenizer())

    assert ("multi-token bad_words" in blockers) is blocked
    # The check runs on a copy; vLLM tokenizes the request's own params later.
    assert sampling_params._bad_words_token_ids is None


@pytest.mark.vllm
def test_bad_word_check_uses_vllms_own_expansion():
    # Both dependency markers select the combined vLLM + Gym lane.
    pytest.importorskip("vllm")
    from vllm import SamplingParams

    class SpaceSplitsTokenizer(_WordTokenizer):
        # " split" encodes as two tokens, "split" as one.
        def encode(self, text: str, add_special_tokens: bool) -> list[int]:
            ids = super().encode(text, add_special_tokens)
            return [999, *ids] if text == " split" else ids

    tokenizer = SpaceSplitsTokenizer()

    assert not _generation_cut_blockers(
        SamplingParams(bad_words=["forbidden"]), tokenizer
    )
    assert "multi-token bad_words" in _generation_cut_blockers(
        SamplingParams(bad_words=["two words"]), tokenizer
    )
    # vLLM drops a space-prefixed form whose length differs, so only "split"
    # (one token) is enforced and the call stays cuttable.
    assert not _generation_cut_blockers(SamplingParams(bad_words=["split"]), tokenizer)


def test_uncuttable_sampling_settings_warn_once_per_reason(monkeypatch, caplog):
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_worker_async._LOGGED_UNCUTTABLE_REASONS",
        set(),
    )
    states = [SimpleNamespace(uncuttable_reason=None, lock=threading.Lock())]
    states.append(SimpleNamespace(uncuttable_reason=None, lock=threading.Lock()))

    with caplog.at_level(logging.WARNING):
        for state in states:
            _record_cut_constraints(state, _sampling_params(stop=["x"]))

    assert all(state.uncuttable_reason for state in states)
    assert caplog.text.count("are not cut; they regenerate after a restore") == 1


class _FailOncePrefixSink(_MemorySink):
    def __init__(self) -> None:
        super().__init__()
        self._fail_next_prefix = True

    def stage_generation_prefix(self, *args, **kwargs) -> StageResult:
        if self._fail_next_prefix:
            self._fail_next_prefix = False
            raise RuntimeError("injected generation-prefix staging failure")
        return super().stage_generation_prefix(*args, **kwargs)


def test_generation_cut_rolls_back_tokens_after_staging_failure():
    sink = _FailOncePrefixSink()
    worker = _worker_with_capture(sink)
    _, state = _live_call(worker, [11, 12], [-0.1, -0.2], prompt_token_ids=(10,))

    def inventory(checkpoint_id: str) -> GenerationCutInventory:
        return _single_call_inventory(checkpoint_id=checkpoint_id)

    # The failed call is acknowledged, not raised, and its buffer rolls back.
    failed = worker._checkpoint_generation_cut(inventory("checkpoint-1"))
    assert [ack.disposition for ack in failed.prefixes] == ["durable_failure"]
    assert state.frozen_cut is None
    assert state.generated_token_ids == [11, 12]
    assert state.staged_token_count == 0
    assert _cut_counts(worker) == {"failed_cuts_cut_error": 1, "failed_cuts": 1}

    receipt = worker._checkpoint_generation_cut(inventory("checkpoint-2"))
    assert receipt.prefixes[0].prefix_token_count == 2
    assert sink.generation_prefix_records[-1][1].token_ids_delta == [10, 11, 12]


def test_one_failed_call_does_not_fail_the_rest_of_the_cut():
    sink = _FailOncePrefixSink()
    worker = _worker_with_capture(sink)
    inventory, (failing, healthy) = _active_call_inventory(
        worker, ("failing", [11, 12]), ("healthy", [13, 14])
    )

    receipt = worker._checkpoint_generation_cut(inventory)

    assert [ack.disposition for ack in receipt.prefixes] == [
        "durable_failure",
        "durable_prefix",
    ]
    assert failing.frozen_cut is None
    assert healthy.generation_cut_staging_keys == sink.generation_prefix_keys


def test_call_without_an_output_limit_is_not_frozen_for_the_cut():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    inventory, (state,) = _active_call_inventory(worker, ("c1", [11, 12]))
    with state.lock:
        state.effective_output_limit = None

    receipt = worker._checkpoint_generation_cut(inventory)

    assert [ack.disposition for ack in receipt.prefixes] == ["durable_failure"]
    assert state.frozen_cut is None
    assert sink.generation_prefix_records == []
    assert _cut_counts(worker) == {
        "failed_cuts_missing_output_limit": 1,
        "failed_cuts": 1,
    }


def test_call_that_finishes_while_being_cut_counts_one_race():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    inventory, (state,) = _active_call_inventory(worker, ("c1", [11, 12]))
    state.terminal_started = True  # its terminal write won the lifecycle lock

    receipt = worker._checkpoint_generation_cut(inventory)

    # It falls back to the completed-call path, which has no record of it yet.
    assert [ack.disposition for ack in receipt.prefixes] == ["durable_failure"]
    assert _cut_counts(worker) == {
        "cut_lifecycle_races": 1,
        "failed_cuts_no_capture_or_completed_state": 1,
        "failed_cuts": 1,
    }


def test_unexpected_error_after_the_freeze_rolls_the_call_back():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    inventory, (state,) = _active_call_inventory(worker, ("c1", [11, 12]))

    class _BrokenTokenizer:
        def decode(self, token_ids, **kwargs):
            raise ValueError("injected decode failure")

    worker._generation_cut_tokenizer = _BrokenTokenizer()
    receipt = worker._checkpoint_generation_cut(inventory)

    assert [ack.disposition for ack in receipt.prefixes] == ["durable_failure"]
    # Without the rollback, every later cut of this call would fail to freeze.
    assert state.frozen_cut is None
    worker._generation_cut_tokenizer = _ByteTokenizer()
    retried = worker._checkpoint_generation_cut(
        GenerationCutInventory.build(
            checkpoint_id="checkpoint-2",
            server_name="policy_model",
            active_prefixes=list(inventory.active_prefixes),
        )
    )
    assert [ack.disposition for ack in retried.prefixes] == ["durable_prefix"]


def _restorable_cut(sink: _MemorySink):
    """Cut one call at weight version 7 and return what a restore would send."""
    worker = _worker_with_capture(sink)
    worker._rollout_weight_version = 7
    inventory, (state,) = _active_call_inventory(worker, ("c1", [12, 13]))
    receipt = worker._checkpoint_generation_cut(inventory)
    (ack,) = receipt.prefixes
    (cut_key,) = ack.staging_keys
    cut_record = sink.generation_prefix_records[-1][1]
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0-a1",
            "model_call_id": "c2",
            "mode": "text",
            "generation_cut": {
                "source_capture_key": "r0",
                "source_model_call_id": "c1",
                "staging_keys": [cut_key],
                "prefix_token_count": ack.prefix_token_count,
                "prefix_digest": cut_record.digest,
                "effective_output_limit": ack.effective_output_limit,
            },
        },
        stream=False,
    )
    return cut_key, cut_record, request


def _resumed_worker(sink, *, records, weight_version: int = 9):
    worker = _worker_with_capture(sink)
    worker._rollout_weight_version = weight_version
    worker._staging_source = _MemoryPrefixSource({}, records=records)
    return worker


def test_usable_restored_prefix_is_resumed():
    sink = _MemorySink()
    cut_key, cut_record, request = _restorable_cut(sink)
    worker = _resumed_worker(sink, records={cut_key: cut_record})
    admission = worker._capture_admission(request)

    served, snapshot = worker._resolve_usable_generation_cut(admission, [])

    assert served is admission
    assert snapshot.digest == cut_record.digest
    _drain_cleanup(worker)
    assert sink.cleared_generation_prefix_keys == []


@pytest.mark.parametrize(
    "failure",
    ["missing chunk", "chunk newer than the worker", "digest mismatch"],
)
def test_unusable_restored_prefix_regenerates_instead_of_failing(failure, caplog):
    sink = _MemorySink()
    cut_key, cut_record, request = _restorable_cut(sink)
    worker = _resumed_worker(
        sink,
        records={} if failure == "missing chunk" else {cut_key: cut_record},
        weight_version=0 if failure == "chunk newer than the worker" else 9,
    )
    admission = worker._capture_admission(request)
    if failure == "digest mismatch":
        admission = admission.model_copy(
            update={
                "generation_cut": admission.generation_cut.model_copy(
                    update={"prefix_digest": "0" * 64}
                )
            }
        )

    with caplog.at_level(logging.WARNING):
        served, snapshot = worker._resolve_usable_generation_cut(admission, [])

    assert snapshot is None
    assert served.generation_cut is None
    assert "regenerating it from its turn boundary" in caplog.text
    # The regenerated call is admitted as an ordinary call from its boundary.
    VllmAsyncGenerationWorkerImpl._begin_request_capture(
        worker, request, [10], admission=served, prefix_token_ids=[]
    )
    assert worker._capture_calls[id(request)].generation_cut_staging_keys == []
    # The restored rows it will never extend are retired through the gate.
    _drain_cleanup(worker)
    assert sink.cleared_generation_prefix_keys == [cut_key]


def test_restored_generation_cut_is_extended_and_retired_on_completion(
    caplog, monkeypatch
):
    caplog.set_level(logging.INFO)
    sink = _MemorySink()
    cut_key, cut_record, request = _restorable_cut(sink)
    resumed_worker = _resumed_worker(sink, records={cut_key: cut_record})
    admission = resumed_worker._capture_admission(request)
    with caplog.at_level(logging.INFO):
        cut = resumed_worker._resolve_generation_cut(admission, [])
    expected_sha = hashlib.sha256(b"12,13").hexdigest()
    assert f"prefix_ids_sha256={expected_sha}" in caplog.text
    # Gym's begin_call rejects a cut whose digest disagrees with its admission.
    bad_digest_admission = admission.model_copy(
        update={
            "generation_cut": admission.generation_cut.model_copy(
                update={"prefix_digest": "0" * 64}
            )
        }
    )
    with pytest.raises(CaptureError, match="identity does not match admission"):
        VllmAsyncGenerationWorkerImpl._begin_request_capture(
            resumed_worker,
            request,
            [10, 12, 13],
            admission=bad_digest_admission,
            prefix_token_ids=[],
            generation_cut=resumed_worker._resolve_generation_cut(
                bad_digest_admission, []
            ),
            resumed_generation_token_ids=[12, 13],
        )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(
        resumed_worker,
        request,
        [10, 12, 13],
        admission=admission,
        prefix_token_ids=[],
        generation_cut=cut,
        resumed_generation_token_ids=[12, 13],
    )
    # Only the restore Gym accepted counts.
    counts = VllmAsyncGenerationWorkerImpl.get_generation_prefix_counts(resumed_worker)
    assert (counts["restored_calls"], counts["restored_prefix_tokens"]) == (1, 2)
    replays = []

    def replay(tokenizer, **kwargs):
        replays.append(kwargs)
        return "partial"

    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_worker_async._detokenize_restored_prefix",
        replay,
    )
    # vLLM's processed tail: EOS and any matched stop string already stripped.
    output = SimpleNamespace(token_ids=[14, 151645], text=" tail")
    resumed_worker._restore_response_prefix(
        request,
        SimpleNamespace(outputs=[output]),
        tokenizer=_ByteTokenizer(),
    )
    (replay_kwargs,) = replays
    assert replay_kwargs["original_prompt_token_ids"] == [10]
    assert replay_kwargs["prefix_token_ids"] == [12, 13]
    assert replay_kwargs["terminal"] is None
    assert output.text == "partial tail"

    # A second checkpoint taken before the resumed call finishes must retain
    # the original cut key and append the newly generated tail. Gym v2 stores
    # this complete key list in the replacement attempt's model record.
    resumed = resumed_worker._capture_calls[id(request)]
    with resumed.lock:
        resumed.observe([14], [-0.3])
    second_receipt = resumed_worker._checkpoint_generation_cut(
        _single_call_inventory("c2", checkpoint_id="checkpoint-2", attempt=1)
    )
    assert second_receipt.prefixes[0].prefix_token_count == 3
    assert second_receipt.prefixes[0].staging_keys[0] == cut_key
    assert len(second_receipt.prefixes[0].staging_keys) == 2

    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        resumed_worker,
        request,
        _served_content([14], [-0.3]),
    )

    final_record = sink.records[-1]
    assert content["ng_commit_coords"]["capture_key"] == "r0-a1"
    assert final_record.token_ids_delta == [10, 12, 13, 14]
    assert final_record.token_mask_delta == [0.0, 1.0, 1.0, 1.0]
    assert final_record.generation_log_probs_delta == [0.0, -0.1, -0.1, -0.3]
    assert final_record.weight_version == 7
    assert content["ng_commit_coords"]["weight_version"] == 7
    assert resumed_worker._completed_capture_calls["c2"].generation_token_count == 3
    assert sink.cleared_generation_prefix_keys == list(
        second_receipt.prefixes[0].staging_keys
    )
    assert "generation prefix restored:" in caplog.text


def test_request_capture_token_in_prev_len_chains():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _FakeRequest(
        ng_capture={
            "capture_key": "r0",
            "model_call_id": "c2",
            "parent_call_id": "c1",
            "prev_len": 3,
            "mode": "token_in",
            "required_prefix_token_ids": [10, 11, 12],
            "parent_chain_hash": "1" * 64,
        },
        stream=False,
    )
    spliced_prompt = [10, 11, 12, 20, 21]  # exact prefix + fresh suffix
    VllmAsyncGenerationWorkerImpl._begin_request_capture(
        worker, request, spliced_prompt
    )
    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, _served_content([22], [-0.5])
    )
    coords = content["ng_commit_coords"]
    assert coords["parent_call_id"] == "c1"
    assert (coords["delta_len"], coords["cum_len"]) == (3, 6)
    assert sink.records[0].token_ids_delta == [20, 21, 22]


def _staging_chain_request(prev_len: int = 3) -> _FakeRequest:
    return _FakeRequest(
        ng_capture={
            "capture_key": "r0",
            "model_call_id": "c3",
            "parent_call_id": "c2",
            "prev_len": prev_len,
            "mode": "token_in",
            "staging_chain": ["r0/c1", "r0/c2"],
            "parent_chain_hash": "2" * 64,
        },
        stream=False,
    )


def test_staging_chain_prefix_flows_through_adapter_and_begin_call():
    """The resolved prefix reaches both the engine request and capture admission."""
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    source = _MemoryPrefixSource({"r0/c1": [10, 11], "r0/c2": [12]})
    worker._chain_prefix.install(source)
    request = _staging_chain_request()
    context_before = dict(request.ng_capture)

    admission = worker._capture_admission(request)
    prefix = worker._resolve_admission_prefix(admission)
    worker._enter_request_prefix(request, prefix)
    VllmAsyncGenerationWorkerImpl._begin_request_capture(
        worker,
        request,
        prefix + [20],
        admission=admission,
        prefix_token_ids=prefix,
    )

    assert prefix == [10, 11, 12]
    assert source.calls == [["r0/c1", "r0/c2"]]
    # Never patched back into the wire context.
    assert request.ng_capture == context_before
    assert admission.required_prefix_token_ids == []
    # enter_prefix is the production writer of the request field.
    assert request.required_prefix_token_ids == prefix
    state = worker._capture_calls[id(request)]
    assert state.call.prefix_token_ids == prefix
    assert state.prompt_token_ids == [10, 11, 12, 20]


def test_inline_prefix_admission_resolves_without_a_fetch():
    worker = _worker_with_capture(_MemorySink())
    source = _MemoryPrefixSource({})
    worker._chain_prefix.install(source)
    request = _FakeRequest(
        ng_capture={
            "capture_key": "r0",
            "model_call_id": "c2",
            "parent_call_id": "c1",
            "prev_len": 2,
            "mode": "token_in",
            "required_prefix_token_ids": [10, 11],
            "parent_chain_hash": "1" * 64,
        },
        stream=False,
    )
    admission = worker._capture_admission(request)
    assert worker._resolve_admission_prefix(admission) == [10, 11]
    assert source.calls == []
    text_root = worker._capture_admission(
        _FakeRequest(
            ng_capture={"capture_key": "r0", "model_call_id": "c1", "mode": "text"}
        )
    )
    assert worker._resolve_admission_prefix(text_root) == []


def test_staging_chain_cache_fetches_only_uncached_suffix():
    worker = _worker_with_capture(_MemorySink())
    source = _MemoryPrefixSource({"r0/c1": [10, 11], "r0/c2": [12]})
    worker._chain_prefix.install(source)

    first = VllmAsyncGenerationWorkerImpl._fetch_chain_prefix(worker, ["r0/c1"])
    second = VllmAsyncGenerationWorkerImpl._fetch_chain_prefix(
        worker, ["r0/c1", "r0/c2"]
    )

    assert first == [10, 11]
    assert second == [10, 11, 12]
    assert source.calls == [["r0/c1"], ["r0/c2"]]


def test_staging_chain_prefix_length_mismatch_is_rejected_by_begin_call():
    """Gym's begin_call rejects a fetched prefix whose length is not prev_len."""
    worker = _worker_with_capture(_MemorySink())
    worker._chain_prefix.install(_MemoryPrefixSource({"r0/c1": [10, 11], "r0/c2": []}))
    request = _staging_chain_request(prev_len=3)
    context_before = dict(request.ng_capture)

    admission = worker._capture_admission(request)
    prefix = worker._resolve_admission_prefix(admission)
    assert prefix == [10, 11]
    with pytest.raises(CaptureError, match="does not equal prev_len 3"):
        VllmAsyncGenerationWorkerImpl._begin_request_capture(
            worker, request, prefix + [20], admission=admission, prefix_token_ids=prefix
        )

    assert request.ng_capture == context_before
    assert worker._capture_calls == {}


def test_staging_chain_admission_requires_the_resolved_prefix_keyword():
    worker = _worker_with_capture(_MemorySink())
    request = _staging_chain_request()

    with pytest.raises(
        CaptureError,
        match="requires the caller to pass the resolved prefix_token_ids",
    ):
        VllmAsyncGenerationWorkerImpl._begin_request_capture(
            worker, request, [10, 11, 12, 20]
        )

    assert worker._capture_calls == {}


@pytest.mark.parametrize("capture_enabled", [False, True])
def test_request_capture_is_a_noop_without_context_or_capture(
    capture_enabled: bool,
) -> None:
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    plain = _FakeRequest(stream=False)  # no ng_capture attribute
    if not capture_enabled:
        worker.token_capture = None
        plain.ng_capture = {"capture_key": "r0", "model_call_id": "c1", "mode": "text"}
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, plain, [1, 2])
    content = _served_content([3], [-0.1])
    content["choices"][0]["message"].update(
        prompt_token_ids=[1, 2],
        generation_token_ids=[3],
        generation_log_probs=[-0.1],
        routed_experts=[[[0]]] * 3,
    )
    original = deepcopy(content)
    out = VllmAsyncGenerationWorkerImpl._finish_request_capture(worker, plain, content)
    assert out == original
    assert worker._capture_calls == {}
    assert sink.records == []


def _text_root_request() -> _FakeRequest:
    return _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c1",
            "parent_call_id": None,
            "prev_len": 0,
            "mode": "text",
        },
        stream=False,
    )


def _content_with_topk(gen_ids, logprobs):
    content = _served_content(gen_ids, logprobs)
    content["choices"][0]["message"].update(
        sampler_topk_token_ids=torch.tensor(
            [[token_id, 1] for token_id in gen_ids], dtype=torch.int32
        ),
        sampler_topk_logprobs=torch.tensor([[lp, -5.0] for lp in logprobs]),
    )
    return content


def test_request_capture_stages_topk_with_the_call():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _text_root_request()
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10, 11, 12])
    content = _content_with_topk([13, 14], [-0.1, -0.2])
    ids = content["choices"][0]["message"]["sampler_topk_token_ids"]
    logprobs = content["choices"][0]["message"]["sampler_topk_logprobs"]

    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, content
    )

    assert content["ng_commit_coords"]["disposition"] == "staged"
    assert len(sink.records) == 1
    assert sink.attachments == [
        {"sampler_topk_token_ids": ids, "sampler_topk_logprobs": logprobs}
    ]
    # It never crosses the worker -> gate hop.
    assert content["choices"] == [
        {"index": 0, "message": {"role": "assistant", "content": "x"}}
    ]


def test_request_capture_skips_topk_when_the_call_did_not_stage():
    class _RejectingSink(_MemorySink):
        def stage(self, record: StagedCallRecord, *, attachments=None) -> StageResult:
            assert attachments["sampler_topk_token_ids"].shape == (1, 2)
            return StageResult(ok=False, staging_key=record.staging_key, error="down")

    sink = _RejectingSink()
    worker = _worker_with_capture(sink)
    request = _text_root_request()
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10, 11, 12])

    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, _content_with_topk([13], [-0.1])
    )

    assert content["ng_commit_coords"]["disposition"] == "capture_failed"
    assert sink.records == []
    assert "sampler_topk_token_ids" not in content["choices"][0]["message"]


def test_request_capture_drops_topk_from_uncaptured_responses():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    plain = _FakeRequest(stream=False)  # no ng_capture attribute

    out = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, plain, _content_with_topk([3], [-0.1])
    )

    assert "sampler_topk_token_ids" not in out["choices"][0]["message"]
    assert "sampler_topk_logprobs" not in out["choices"][0]["message"]
    assert sink.records == []


def test_request_capture_missing_required_topk_fails_before_staging():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _text_root_request()
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10, 11, 12])
    worker._capture_top_logprobs = 2
    out = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, _served_content([13], [-0.1])
    )
    assert out["ng_commit_coords"]["disposition"] == "capture_failed"
    assert sink.records == []
    assert worker._capture_calls == {}


def test_request_capture_topk_partial_write_returns_capture_failed():
    sink = object.__new__(TQTokenSink)
    sink._capture_media = False
    sink._store = MagicMock()

    def failed_put(key, fields, **kwargs):
        assert "token_ids_delta" in fields
        assert "sampler_topk_token_ids" in fields
        raise RuntimeError("top-k storage unavailable")

    sink._store.put.side_effect = failed_put
    worker = _worker_with_capture(sink)
    request = _text_root_request()
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10, 11, 12])
    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, _content_with_topk([13], [-0.1])
    )
    assert content["ng_commit_coords"]["disposition"] == "capture_failed"
    sink._store.put.assert_called_once()
    sink._store.clear.assert_called_once_with(["r0/c1"])
    assert "sampler_topk_token_ids" not in content["choices"][0]["message"]
    assert "sampler_topk_logprobs" not in content["choices"][0]["message"]
    assert worker._capture_calls == {}


def _cut_active_call(worker, sink, model_call_id: str = "c1"):
    """Begin one captured call and stage its prefix at a generation cut."""
    request, state = _live_call(
        worker, [20, 21], [-0.1, -0.2], model_call_id=model_call_id
    )
    inventory = _single_call_inventory(model_call_id)
    worker._checkpoint_generation_cut(inventory)
    assert state.generation_cut_staging_keys == sink.generation_prefix_keys
    return request


def _drain_cleanup(worker) -> None:
    worker._generation_cut_cleanup_executor.submit(lambda: None).result(timeout=5)


def test_aborted_call_clears_its_rows_only_after_the_snapshot_fence():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _cut_active_call(worker, sink)
    gate = worker._token_capture_snapshot_gate
    gate.close_and_wait(gate.begin_epoch())

    # The abort itself never waits for the fence.
    assert VllmAsyncGenerationWorkerImpl._abort_request_capture(
        worker, request, reason="client_disconnected", blocking=False
    )
    time.sleep(0.2)
    assert sink.cleared_generation_prefix_keys == []

    gate.reopen()
    _drain_cleanup(worker)
    assert sink.generation_prefix_keys
    assert sink.cleared_generation_prefix_keys == sink.generation_prefix_keys


class _FailingTerminalSink(_MemorySink):
    def stage(
        self, record: StagedCallRecord, *, attachments: dict | None = None
    ) -> StageResult:
        return StageResult(ok=False, error="tq unavailable")


def test_call_whose_terminal_staging_fails_clears_its_cut_rows():
    sink = _FailingTerminalSink()
    worker = _worker_with_capture(sink)
    request = _cut_active_call(worker, sink)

    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker,
        request,
        _served_content([20, 21, 22], [-0.1, -0.2, -0.3]),
    )

    assert content["ng_commit_coords"]["disposition"] == "capture_failed"
    assert sink.cleared_generation_prefix_keys == sink.generation_prefix_keys


def test_request_capture_abort_fails_the_call_and_drains_state():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _captured_text_call(worker, "c1", (1, 2))
    VllmAsyncGenerationWorkerImpl._abort_request_capture(
        worker, request, reason="engine_error"
    )
    assert worker._capture_calls == {}
    assert sink.records == []
    # A call that was never cut has no chunk rows to clear.
    _drain_cleanup(worker)
    assert sink.cleared_generation_prefix_keys == []
    # A late finish after abort is a no-op (state already drained).
    out = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, _served_content([3], [-0.1])
    )
    assert "ng_commit_coords" not in out


@pytest.mark.vllm
@pytest.mark.parametrize("pruning_rate", [0.0, 0.5])
def test_omni_capture_setup_rejects_video_pruning(monkeypatch, pruning_rate):
    # Both dependency markers select the combined vLLM + Gym lane.
    pytest.importorskip("vllm")
    from vllm.model_executor.models.nano_nemotron_vl import NanoNemotronVLProcessingInfo

    info = object.__new__(NanoNemotronVLProcessingInfo)
    monkeypatch.setattr(NanoNemotronVLProcessingInfo, "is_dynamic_tiler", True)
    monkeypatch.setattr(
        NanoNemotronVLProcessingInfo,
        "get_video_pruning_rate",
        lambda self: pruning_rate,
    )
    monkeypatch.setattr(
        NanoNemotronVLProcessingInfo,
        "get_hf_processor",
        lambda self: SimpleNamespace(_img_context_token_ids=[18]),
    )
    monkeypatch.setattr(
        NanoNemotronVLProcessingInfo,
        "get_hf_config",
        lambda self: SimpleNamespace(patch_size=2),
    )
    worker = _fake_worker()
    worker.llm = SimpleNamespace(
        renderer=SimpleNamespace(get_mm_processor=lambda: SimpleNamespace(info=info)),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    monkeypatch.setattr(
        "nemo_rl.data_plane.build_data_plane_client",
        lambda dp_cfg, bootstrap, segment_size=None: MagicMock(),
    )
    if pruning_rate:
        with pytest.raises(ValueError, match="video token pruning"):
            asyncio.run(
                VllmAsyncGenerationWorkerImpl.setup_token_capture(
                    worker, {}, staging_partition="staging", capture_media=True
                )
            )
    else:
        spec = asyncio.run(
            VllmAsyncGenerationWorkerImpl.setup_token_capture(
                worker, {}, staging_partition="staging", capture_media=True
            )
        )
        # The engine's model dtype and the HF patch size, as pinned on the sink.
        assert spec == MediaColumnSpec(pixel_dtype=torch.bfloat16, patch_size=2)
        assert worker._capture_image_token_id == 18


def test_request_capture_text_call_in_media_run_stages_without_attachments():
    from nemo_rl.data.captured_media import CapturedMedia

    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _text_root_request()
    # A text-only call in a media-capture run adds no media tensors.
    VllmAsyncGenerationWorkerImpl._begin_request_capture(
        worker, request, [10, 11, 12], media=CapturedMedia((), None)
    )

    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, _served_content([13], [-0.1])
    )

    assert content["ng_commit_coords"]["disposition"] == "staged"
    assert sink.attachments == [None]


@pytest.mark.parametrize("prev_len", [0, 3])
def test_route_delta_preserves_actual_predecessor_tail(prev_len):
    import torch
    from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter

    from nemo_rl.utils.routed_experts_codec import decode_routed_experts

    routes = [[[i]] for i in range(7)]
    payload = {
        "choices": [
            {
                "message": {
                    "routed_experts": routes,
                }
            }
        ]
    }
    VllmAsyncGenerationWorkerImpl._delta_align_routed_experts(
        payload,
        prev_len=prev_len,
        prompt_len=5,
        generated_len=2,
    )
    extras = VLLMCaptureAdapter().extract_extras(payload)
    assert (
        decode_routed_experts(extras["routed_experts"], torch.int16).tolist()
        == routes[max(0, prev_len - 1) :]
    )
    assert "predecessor_tail_route" not in extras
