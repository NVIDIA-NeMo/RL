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
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

nemo_gym = pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_gym._checkpoint.model_control_contracts import (  # noqa: E402
    GenerationCutInventory,
    GenerationCutPrefix,
    GenerationCutReceipt,
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

from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration  # noqa: E402
from nemo_rl.models.generation.vllm.vllm_worker_async import (  # noqa: E402
    VllmAsyncGenerationWorkerImpl,
    _CheckpointCaptureGate,
    _RequestOutputDeltaAccumulator,
    _remaining_generation_limits_after_prefix,
)

pytestmark = pytest.mark.nemo_gym


class _MemorySink:
    def __init__(self) -> None:
        self.records: list[StagedCallRecord] = []
        self.generation_prefix_records: list[tuple[str, StagedCallRecord]] = []
        self.generation_prefix_keys: list[str] = []
        self.cleared_generation_prefix_keys: list[str] = []

    def stage(self, record: StagedCallRecord) -> StageResult:
        self.records.append(record)
        return StageResult(ok=True, staging_key=record.staging_key)

    def stage_generation_prefix(
        self,
        record: StagedCallRecord,
        *,
        checkpoint_id: str,
        chunk_sequence: int,
    ) -> StageResult:
        self.generation_prefix_records.append((checkpoint_id, record))
        staging_key = (
            f"__generation_cut__/{checkpoint_id}/"
            f"{record.rollout_id}/{record.model_call_id}/{chunk_sequence}"
        )
        self.generation_prefix_keys.append(staging_key)
        return StageResult(ok=True, staging_key=staging_key)

    def clear(self, staging_keys: list[str]) -> None:
        self.cleared_generation_prefix_keys.extend(staging_keys)


class _BlockingPrefixSink(_MemorySink):
    """Hold one prefix write so a test can observe the post-swap buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.write_started = threading.Event()
        self.release_write = threading.Event()

    def stage_generation_prefix(
        self,
        record: StagedCallRecord,
        *,
        checkpoint_id: str,
        chunk_sequence: int,
    ) -> StageResult:
        self.write_started.set()
        if not self.release_write.wait(timeout=5.0):
            raise TimeoutError("test did not release the prefix write")
        return super().stage_generation_prefix(
            record,
            checkpoint_id=checkpoint_id,
            chunk_sequence=chunk_sequence,
        )


class _FailClearSink(_MemorySink):
    """Fail prefix cleanup after accepting a canonical terminal row."""

    def clear(self, staging_keys: list[str]) -> None:
        raise RuntimeError("injected cleanup failure")


class _FailOncePrefixSink(_MemorySink):
    """Reject the first prefix write and accept its retry."""

    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def stage_generation_prefix(
        self,
        record: StagedCallRecord,
        *,
        checkpoint_id: str,
        chunk_sequence: int,
    ) -> StageResult:
        if not self.failed:
            self.failed = True
            return StageResult(ok=False, error="injected failure")
        return super().stage_generation_prefix(
            record,
            checkpoint_id=checkpoint_id,
            chunk_sequence=chunk_sequence,
        )


class _FailModelCallOncePrefixSink(_MemorySink):
    """Fail one named call after earlier calls in the same cut have staged."""

    def __init__(self, model_call_id: str) -> None:
        super().__init__()
        self.model_call_id = model_call_id
        self.failed = False

    def stage_generation_prefix(
        self,
        record: StagedCallRecord,
        *,
        checkpoint_id: str,
        chunk_sequence: int,
    ) -> StageResult:
        if record.model_call_id == self.model_call_id and not self.failed:
            self.failed = True
            return StageResult(ok=False, error="injected later-call failure")
        return super().stage_generation_prefix(
            record,
            checkpoint_id=checkpoint_id,
            chunk_sequence=chunk_sequence,
        )


def _fake_worker(*, is_model_owner: bool = True) -> SimpleNamespace:
    """The attribute surface setup_token_capture touches, minus the engine."""
    worker = SimpleNamespace(
        is_model_owner=is_model_owner,
        token_capture=None,
        _rollout_weight_version=0,
        _capture_calls={},
        _capture_calls_by_model_call_id={},
        _completed_capture_calls={},
        _generation_cut_receipts={},
        _capture_registry_lock=threading.Lock(),
        _capture_sink=None,
        _generation_prefix_cuts_enabled=False,
        _generation_cut_control_token=None,
        _generation_chunk_flush_tokens=0,
        _generation_checkpoint_gate=_CheckpointCaptureGate(),
        _staging_source=None,
        _prefix_cache={},
        _prefix_cache_lock=threading.Lock(),
    )
    worker.install_token_capture = lambda capture: setattr(
        worker, "token_capture", capture
    )
    return worker


def test_setup_token_capture_installs_capture_with_vllm_adapter(monkeypatch):
    sink = _MemorySink()
    monkeypatch.setattr(
        "nemo_rl.data_plane.build_data_plane_client",
        lambda dp_cfg, bootstrap: MagicMock(name="dp_client"),
    )
    monkeypatch.setattr(
        "nemo_rl.data_plane.tq_token_sink.TQTokenSink",
        lambda dp_client, *, staging_partition: sink,
    )
    worker = _fake_worker()

    installed = asyncio.run(
        VllmAsyncGenerationWorkerImpl.setup_token_capture(
            worker, dp_cfg={"backend": "simple"}, staging_partition="rollout_staging"
        )
    )

    assert installed is True
    assert isinstance(worker.token_capture, RolloutTokenCapture)
    assert worker.token_capture.adapter is not None
    # The adapter is the vLLM one (prefix ids enter via the worker's field).
    payload = worker.token_capture.adapter.enter_prefix({}, [1, 2])
    assert payload["required_prefix_token_ids"] == [1, 2]


def test_setup_token_capture_skips_non_model_owners(monkeypatch):
    worker = _fake_worker(is_model_owner=False)
    installed = asyncio.run(
        VllmAsyncGenerationWorkerImpl.setup_token_capture(
            worker, dp_cfg={}, staging_partition="rollout_staging"
        )
    )
    assert installed is False
    assert worker.token_capture is None


def test_weight_version_is_stamped_from_worker_state(monkeypatch):
    """The install closure reads _rollout_weight_version live: a
    set_rollout_weight_version between calls changes the stamp."""
    sink = _MemorySink()
    monkeypatch.setattr(
        "nemo_rl.data_plane.build_data_plane_client",
        lambda dp_cfg, bootstrap: MagicMock(),
    )
    monkeypatch.setattr(
        "nemo_rl.data_plane.tq_token_sink.TQTokenSink",
        lambda dp_client, *, staging_partition: sink,
    )
    worker = _fake_worker()
    asyncio.run(
        VllmAsyncGenerationWorkerImpl.setup_token_capture(
            worker, dp_cfg={}, staging_partition="rollout_staging"
        )
    )

    asyncio.run(VllmAsyncGenerationWorkerImpl.set_rollout_weight_version(worker, 4))
    first = worker.token_capture.begin_call(
        CaptureAdmission(rollout_id="r", model_call_id="c1", mode="text")
    )
    asyncio.run(VllmAsyncGenerationWorkerImpl.set_rollout_weight_version(worker, 5))
    second = worker.token_capture.begin_call(
        CaptureAdmission(rollout_id="r", model_call_id="c2", mode="text")
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
        lambda futures, timeout=None: futures,
    )
    gen.setup_token_capture({"backend": "simple"}, "rollout_staging")
    gen.worker_group.run_all_workers_single_data.assert_called_once_with(
        "setup_token_capture",
        dp_cfg={"backend": "simple"},
        staging_partition="rollout_staging",
        generation_prefix_cuts_enabled=False,
        generation_cut_control_token=None,
        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
    )


def test_generation_setup_token_capture_requires_async_engine():
    gen = _generation_with_mock_group(async_engine=False)
    with pytest.raises(AssertionError, match="async vLLM engine"):
        gen.setup_token_capture({}, "rollout_staging")


@pytest.mark.parametrize(
    ("operation", "worker_method"),
    [
        ("begin_generation_checkpoint", "begin_generation_checkpoint_async"),
        ("finish_generation_checkpoint", "finish_generation_checkpoint_async"),
    ],
)
def test_generation_checkpoint_control_fans_out(
    monkeypatch, operation: str, worker_method: str
):
    gen = _generation_with_mock_group()
    gen.worker_group.workers = [object()]
    gen.worker_group.run_all_workers_single_data.return_value = [True]
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, timeout=None: futures,
    )

    assert getattr(gen, operation)()

    gen.worker_group.run_all_workers_single_data.assert_called_once_with(
        worker_method,
        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
    )


def test_generation_checkpoint_control_isolated_from_blocked_default_executor():
    async def scenario() -> None:
        worker = object.__new__(VllmAsyncGenerationWorkerImpl)
        worker._generation_checkpoint_executor = ThreadPoolExecutor(max_workers=1)
        gate = _CheckpointCaptureGate()
        gate.close_and_wait()
        default_thread_started = threading.Event()

        def wait_at_closed_gate() -> None:
            default_thread_started.set()
            gate.enter()
            gate.exit()

        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        blocked_completion = asyncio.create_task(asyncio.to_thread(wait_at_closed_gate))
        while not default_thread_started.is_set():
            await asyncio.sleep(0)

        try:
            result = await asyncio.wait_for(
                worker._run_generation_checkpoint_control(lambda: "cut-ready"),
                timeout=1.0,
            )
            assert result == "cut-ready"
        finally:
            gate.reopen()
            await blocked_completion
            worker._generation_checkpoint_executor.shutdown()

    asyncio.run(scenario())


def test_begin_generation_checkpoint_fences_terminal_stage_without_pausing_decode():
    class _FakeLLM:
        def __init__(self) -> None:
            self.pause_calls = 0

        async def pause_generation(self, **_kwargs) -> None:
            self.pause_calls += 1

    async def scenario() -> None:
        worker = object.__new__(VllmAsyncGenerationWorkerImpl)
        worker.cfg = {"vllm_cfg": {"async_engine": True}}
        worker.llm = _FakeLLM()
        worker._generation_checkpoint_gate = _CheckpointCaptureGate()
        worker._generation_checkpoint_executor = ThreadPoolExecutor(max_workers=1)

        try:
            await worker.begin_generation_checkpoint_async()
            assert worker.llm.pause_calls == 0

            entered = threading.Event()
            released = threading.Event()

            def enter_terminal_stage() -> None:
                entered.set()
                worker._generation_checkpoint_gate.enter()
                released.set()
                worker._generation_checkpoint_gate.exit()

            thread = threading.Thread(target=enter_terminal_stage)
            thread.start()
            assert entered.wait(timeout=5.0)
            assert not released.wait(timeout=0.05)

            await worker.finish_generation_checkpoint_async()
            assert released.wait(timeout=5.0)
            thread.join(timeout=5.0)
            assert not thread.is_alive()
        finally:
            worker._generation_checkpoint_gate.reopen()
            worker._generation_checkpoint_executor.shutdown()

    asyncio.run(scenario())


def test_request_output_deltas_are_assembled_without_mutating_engine_outputs():
    class _FakeRequestOutput:
        def __init__(self, text: str, token_ids: list[int]) -> None:
            self.outputs = [
                SimpleNamespace(
                    index=0,
                    text=text,
                    token_ids=list(token_ids),
                    logprobs=[f"lp-{token_id}" for token_id in token_ids],
                )
            ]

    first = _FakeRequestOutput("a", [1])
    second = _FakeRequestOutput("bc", [2, 3])

    accumulator = _RequestOutputDeltaAccumulator()
    accumulator.append(first)
    accumulator.append(second)
    accumulated = accumulator.build()

    assert accumulated.outputs[0].text == "abc"
    assert accumulated.outputs[0].token_ids == [1, 2, 3]
    assert accumulated.outputs[0].logprobs == ["lp-1", "lp-2", "lp-3"]
    assert first.outputs[0].token_ids == [1]
    assert second.outputs[0].token_ids == [2, 3]


@pytest.mark.parametrize(
    ("max_tokens", "min_tokens", "generation_token_count", "expected"),
    [
        (10, 6, 3, (7, 3)),
        (None, 6, 3, (None, 3)),
        (10, None, 3, (7, None)),
        (None, None, 3, (None, None)),
        (10, 2, 3, (7, 0)),
    ],
)
def test_restored_generation_prefix_reduces_independent_output_limits(
    max_tokens: int | None,
    min_tokens: int | None,
    generation_token_count: int,
    expected: tuple[int | None, int | None],
) -> None:
    assert (
        _remaining_generation_limits_after_prefix(
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            generation_token_count=generation_token_count,
        )
        == expected
    )


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


class _PartiallyPublishedLogprobs:
    """FlatLogprobs-like container whose newest position is not complete."""

    def __init__(self, entries: list[dict[int, SimpleNamespace]]) -> None:
        self._entries = entries
        self.end_indices = [1] * (len(entries) - 1)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> dict[int, SimpleNamespace]:
        if index >= len(self.end_indices):
            raise AssertionError("observer read an incompletely published position")
        return self._entries[index]


def _worker_with_capture(sink: _MemorySink):
    from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter

    worker = _fake_worker()
    worker._capture_calls = {}
    worker._capture_calls_by_model_call_id = {}
    worker._completed_capture_calls = {}
    worker._generation_cut_receipts = {}
    worker._capture_registry_lock = threading.Lock()
    worker._capture_sink = sink
    worker._generation_checkpoint_gate = _CheckpointCaptureGate()
    worker._prefix_cache = {}
    worker._prefix_cache_lock = threading.Lock()
    worker._staging_source = None
    worker._delta_align_routed_experts = (
        VllmAsyncGenerationWorkerImpl._delta_align_routed_experts
    )
    for name in (
        "_fetch_chain_prefix",
        "_capture_admission",
        "_resolve_admission_prefix",
        "_resolve_generation_cut",
        "_enter_request_prefix",
        "_get_request_capture",
        "_pop_request_capture",
        "_remember_completed_capture",
        "_completed_generation_cut_ack",
        "_checkpoint_active_generation_cut",
        "_finish_request_capture_after_checkpoint_gate",
        "_finish_request_capture_with_lifecycle_owned",
    ):
        setattr(
            worker, name, getattr(VllmAsyncGenerationWorkerImpl, name).__get__(worker)
        )
    worker.token_capture = RolloutTokenCapture(
        sink=sink,
        weight_version_fn=lambda: worker._rollout_weight_version,
        adapter=VLLMCaptureAdapter(),
    )
    return worker


class _MemoryPrefixSource:
    def __init__(
        self,
        deltas: dict[str, list[int]],
        records: dict[str, StagedCallRecord] | None = None,
    ) -> None:
        self.deltas = deltas
        self.records = records or {}
        self.calls: list[list[str]] = []
        self.fetch_calls: list[list[str]] = []

    def fetch_prefix_token_ids(self, staging_keys: list[str]) -> list[int]:
        self.calls.append(list(staging_keys))
        return [token for key in staging_keys for token in self.deltas[key]]

    def fetch(self, staging_keys: list[str]):
        self.fetch_calls.append(list(staging_keys))
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


def test_request_capture_round_trip_stages_and_rides_coords():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c1",
            "parent_call_id": None,
            "prev_len": 0,
            "mode": "text",
        },
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10, 11, 12])
    content = _served_content([13, 14], [-0.1, -0.2])
    # Full-length routes on the served response must not survive the strip.
    content["choices"][0]["message"]["routed_experts"] = [[[0]]] * 5
    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, content
    )
    # Bytes were staged before the coords existed (fail-closed ordering).
    assert len(sink.records) == 1
    assert sink.records[0].token_ids_delta == [10, 11, 12, 13, 14]
    coords = content["ng_commit_coords"]
    assert coords["disposition"] == "staged"
    assert (coords["delta_len"], coords["cum_len"]) == (5, 5)
    # Coords are token-free: hashes ride the wire, deltas stay in the sink.
    assert "token_ids_delta" not in coords
    assert coords["chain_hash"] == sink.records[0].chain_hash
    assert coords["cumulative_hash"] == sink.records[0].cumulative_hash
    # Logprobs and routes never transit worker -> gate; state map is drained.
    assert (
        "logprobs" not in content["choices"][0]
        or content["choices"][0]["logprobs"] is None
    )
    assert "routed_experts" not in content["choices"][0]["message"]
    assert worker._capture_calls == {}


def _capture_worker_with_periodic_flush(sink, threshold):
    worker = _worker_with_capture(sink)
    worker._generation_chunk_flush_tokens = threshold
    worker._generation_chunk_flush_sequence = 0
    for name in (
        "flush_due_generation_chunks",
        "_flush_generation_chunk",
        "_flush_generation_chunk_with_lifecycle_owned",
        "_observe_request_capture",
    ):
        setattr(
            worker, name, getattr(VllmAsyncGenerationWorkerImpl, name).__get__(worker)
        )
    return worker


def _begin_captured_request(worker, prompt_token_ids):
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c1",
            "parent_call_id": None,
            "prev_len": 0,
            "mode": "text",
        },
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(
        worker, request, list(prompt_token_ids)
    )
    return request


def _observe(worker, request, token_ids, logprobs):
    worker._observe_request_capture(
        request,
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    token_ids=list(token_ids),
                    logprobs=[
                        {token: SimpleNamespace(logprob=logprob)}
                        for token, logprob in zip(token_ids, logprobs, strict=True)
                    ],
                )
            ]
        ),
    )


def test_periodic_flush_stages_once_the_unstaged_segment_passes_the_bound():
    sink = _MemorySink()
    worker = _capture_worker_with_periodic_flush(sink, threshold=3)
    request = _begin_captured_request(worker, [10, 11])
    state = worker._capture_calls[id(request)]

    _observe(worker, request, [20, 21], [-0.1, -0.2])
    assert state.periodic_flush_due is False
    assert worker.flush_due_generation_chunks() == 0
    assert sink.generation_prefix_keys == []

    _observe(worker, request, [22], [-0.3])
    assert state.periodic_flush_due is True
    assert worker.flush_due_generation_chunks() == 1

    # The first staged row for a call is cumulative: prompt plus everything
    # generated so far, exactly as a checkpoint cut would have written it.
    assert len(sink.generation_prefix_records) == 1
    _checkpoint_id, record = sink.generation_prefix_records[0]
    assert record.token_ids_delta == [10, 11, 20, 21, 22]
    # The segment is adopted, so nothing is left unstaged and the mark clears.
    assert state.unstaged_token_count() == 0
    assert state.periodic_flush_due is False
    assert state.generation_cut_staging_keys == sink.generation_prefix_keys


def test_periodic_flush_stages_only_the_new_delta_after_the_first_chunk():
    sink = _MemorySink()
    worker = _capture_worker_with_periodic_flush(sink, threshold=2)
    request = _begin_captured_request(worker, [10])
    state = worker._capture_calls[id(request)]

    _observe(worker, request, [20, 21], [-0.1, -0.2])
    assert worker.flush_due_generation_chunks() == 1
    _observe(worker, request, [22, 23], [-0.3, -0.4])
    assert worker.flush_due_generation_chunks() == 1

    assert len(sink.generation_prefix_records) == 2
    assert sink.generation_prefix_records[0][1].token_ids_delta == [10, 20, 21]
    # Only the tokens produced since the previous flush.
    assert sink.generation_prefix_records[1][1].token_ids_delta == [22, 23]
    assert len(state.generation_cut_staging_keys) == 2


def test_periodic_flush_rolls_back_and_clears_when_staging_fails():
    sink = _FailOncePrefixSink()
    worker = _capture_worker_with_periodic_flush(sink, threshold=2)
    request = _begin_captured_request(worker, [10])
    state = worker._capture_calls[id(request)]

    _observe(worker, request, [20, 21], [-0.1, -0.2])
    # The pass swallows the failure so one bad call cannot stall the loop.
    assert worker.flush_due_generation_chunks() == 0
    # Tokens are back in the active buffer, so the next attempt retries them.
    assert state.unstaged_token_count() == 2
    assert state.frozen_buffer is None
    assert state.generation_cut_staging_keys == []
    assert state.periodic_flush_due is True

    assert worker.flush_due_generation_chunks() == 1
    assert sink.generation_prefix_records[-1][1].token_ids_delta == [10, 20, 21]


def test_periodic_flush_rearms_when_fresh_buffer_fills_during_tq_write():
    sink = _BlockingPrefixSink()
    worker = _capture_worker_with_periodic_flush(sink, threshold=2)
    request = _begin_captured_request(worker, [10])
    state = worker._capture_calls[id(request)]
    _observe(worker, request, [20, 21], [-0.1, -0.2])

    first_result: list[int] = []
    thread = threading.Thread(
        target=lambda: first_result.append(worker.flush_due_generation_chunks())
    )
    thread.start()
    try:
        assert sink.write_started.wait(timeout=5.0)
        # Decoding continues into the post-swap buffer while TQ is blocked.
        _observe(worker, request, [22, 23], [-0.3, -0.4])
        # The frozen write owns the first flush; sealing it must rediscover
        # that the fresh buffer independently crossed the threshold.
        assert state.periodic_flush_due is False
    finally:
        sink.release_write.set()
        thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert first_result == [1]
    assert state.periodic_flush_due is True
    assert worker.flush_due_generation_chunks() == 1
    assert [record.token_ids_delta for _, record in sink.generation_prefix_records] == [
        [10, 20, 21],
        [22, 23],
    ]


def test_terminal_completion_waits_for_periodic_flush_and_clears_its_row():
    sink = _BlockingPrefixSink()
    worker = _capture_worker_with_periodic_flush(sink, threshold=2)
    request = _begin_captured_request(worker, [10])
    state = worker._capture_calls[id(request)]
    _observe(worker, request, [20, 21], [-0.1, -0.2])

    flush_result: list[int] = []
    terminal_result: list[dict] = []
    flush_thread = threading.Thread(
        target=lambda: flush_result.append(worker.flush_due_generation_chunks())
    )
    terminal_thread = threading.Thread(
        target=lambda: terminal_result.append(
            VllmAsyncGenerationWorkerImpl._finish_request_capture(
                worker,
                request,
                _served_content([20, 21, 22], [-0.1, -0.2, -0.3]),
            )
        )
    )

    flush_thread.start()
    try:
        assert sink.write_started.wait(timeout=5.0)
        terminal_thread.start()
        terminal_thread.join(timeout=0.05)
        assert terminal_thread.is_alive()
        assert sink.records == []
    finally:
        sink.release_write.set()
        flush_thread.join(timeout=5.0)
        terminal_thread.join(timeout=5.0)

    assert not flush_thread.is_alive()
    assert not terminal_thread.is_alive()
    assert flush_result == [1]
    assert terminal_result[0]["ng_commit_coords"]["disposition"] == "staged"
    assert len(sink.records) == 1
    assert sink.cleared_generation_prefix_keys == sink.generation_prefix_keys
    assert state.terminal_started is True
    assert worker._capture_calls == {}


def test_stale_periodic_snapshot_skips_after_terminal_completion():
    sink = _MemorySink()
    worker = _capture_worker_with_periodic_flush(sink, threshold=2)
    request = _begin_captured_request(worker, [10])
    state = worker._capture_calls[id(request)]
    _observe(worker, request, [20, 21], [-0.1, -0.2])

    VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker,
        request,
        _served_content([20, 21], [-0.1, -0.2]),
    )

    assert worker._flush_generation_chunk(state, "stale-periodic") is False
    assert sink.generation_prefix_keys == []


def test_terminal_prefix_cleanup_failure_preserves_completion(caplog):
    sink = _FailClearSink()
    worker = _capture_worker_with_periodic_flush(sink, threshold=2)
    request = _begin_captured_request(worker, [10])
    state = worker._capture_calls[id(request)]
    _observe(worker, request, [20, 21], [-0.1, -0.2])
    assert worker.flush_due_generation_chunks() == 1

    with caplog.at_level(logging.ERROR):
        content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
            worker,
            request,
            _served_content([20, 21, 22], [-0.1, -0.2, -0.3]),
        )

    assert content["ng_commit_coords"]["disposition"] == "staged"
    assert worker._capture_calls == {}
    assert "failed to clear obsolete generation chunks" in caplog.text


def test_periodic_flush_skips_a_call_a_checkpoint_cut_already_froze():
    sink = _MemorySink()
    worker = _capture_worker_with_periodic_flush(sink, threshold=2)
    request = _begin_captured_request(worker, [10])
    state = worker._capture_calls[id(request)]

    _observe(worker, request, [20, 21], [-0.1, -0.2])
    assert state.periodic_flush_due is True
    state.freeze_for_checkpoint("cut-1")

    # A cut owns the frozen buffer; the mark survives for a later pass.
    assert worker.flush_due_generation_chunks() == 0
    assert sink.generation_prefix_keys == []
    assert state.periodic_flush_due is True


def test_periodic_flush_is_inert_when_disabled():
    sink = _MemorySink()
    worker = _capture_worker_with_periodic_flush(sink, threshold=0)
    request = _begin_captured_request(worker, [10])
    state = worker._capture_calls[id(request)]

    _observe(worker, request, [20, 21, 22], [-0.1, -0.2, -0.3])
    assert state.periodic_flush_due is False
    assert worker.flush_due_generation_chunks() == 0
    assert sink.generation_prefix_keys == []


def test_generation_cut_stages_latest_prefix_without_completing_live_call():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c1",
            "parent_call_id": None,
            "prev_len": 0,
            "mode": "text",
        },
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10, 11])
    progress = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                token_ids=[12, 13],
                logprobs=[
                    {12: SimpleNamespace(logprob=-0.1)},
                    {13: SimpleNamespace(logprob=-0.2)},
                ],
            )
        ]
    )
    VllmAsyncGenerationWorkerImpl._observe_request_capture(worker, request, progress)
    inventory = GenerationCutInventory.build(
        checkpoint_id="checkpoint-1",
        server_name="policy",
        active_prefixes=[
            GenerationCutPrefix(
                ticket_id="ticket-1",
                rollout_id="r0",
                attempt_index=0,
                model_call_id="c1",
                admitted_at=1.0,
            )
        ],
    )

    receipt = VllmAsyncGenerationWorkerImpl._checkpoint_generation_cut(
        worker, inventory
    )
    replayed = VllmAsyncGenerationWorkerImpl._checkpoint_generation_cut(
        worker, inventory
    )

    assert replayed is receipt
    assert len(sink.generation_prefix_records) == 1
    checkpoint_id, record = sink.generation_prefix_records[0]
    assert checkpoint_id == "checkpoint-1"
    assert record.token_ids_delta == [10, 11, 12, 13]
    assert record.generation_log_probs_delta == [0.0, 0.0, -0.1, -0.2]
    assert receipt.prefixes[0].cut_kind == "active_prefix"
    assert receipt.prefixes[0].prefix_token_count == 2
    assert receipt.prefixes[0].staging_keys == (
        "__generation_cut__/checkpoint-1/r0/c1/0",
    )
    # A cut is a snapshot, not terminal completion. The ordinary response can
    # still finish later and stages the final call under its canonical key.
    assert id(request) in worker._capture_calls
    VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker,
        request,
        _served_content([12, 13, 14], [-0.1, -0.2, -0.3]),
    )
    assert len(sink.records) == 1
    assert sink.records[0].token_ids_delta == [10, 11, 12, 13, 14]


def test_generation_cut_marks_completed_race_as_terminal_completion():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c1",
            "parent_call_id": None,
            "prev_len": 0,
            "mode": "text",
        },
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10])
    VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker,
        request,
        _served_content([11], [-0.1]),
    )
    inventory = GenerationCutInventory.build(
        checkpoint_id="checkpoint-1",
        server_name="policy",
        active_prefixes=[
            GenerationCutPrefix(
                ticket_id="ticket-1",
                rollout_id="r0",
                attempt_index=0,
                model_call_id="c1",
                admitted_at=1.0,
            )
        ],
    )

    receipt = VllmAsyncGenerationWorkerImpl._checkpoint_generation_cut(
        worker, inventory
    )

    assert receipt.prefixes[0].disposition == "durable_prefix"
    assert receipt.prefixes[0].cut_kind == "terminal_completion"
    assert receipt.prefixes[0].staging_keys == ("r0/c1",)


def test_abort_waits_for_generation_cut_before_failing_the_call():
    sink = _BlockingPrefixSink()
    worker = _worker_with_capture(sink)
    request = _begin_captured_request(worker, [10])
    _observe(worker, request, [11], [-0.1])
    inventory = GenerationCutInventory.build(
        checkpoint_id="checkpoint-1",
        server_name="policy",
        active_prefixes=[
            GenerationCutPrefix(
                ticket_id="ticket-1",
                rollout_id="r0",
                attempt_index=0,
                model_call_id="c1",
                admitted_at=1.0,
            )
        ],
    )
    receipt: list[GenerationCutReceipt] = []
    cut_thread = threading.Thread(
        target=lambda: receipt.append(
            VllmAsyncGenerationWorkerImpl._checkpoint_generation_cut(worker, inventory)
        )
    )
    abort_thread = threading.Thread(
        target=lambda: VllmAsyncGenerationWorkerImpl._abort_request_capture(
            worker, request, reason="injected abort"
        )
    )

    cut_thread.start()
    try:
        assert sink.write_started.wait(timeout=5.0)
        abort_thread.start()
        abort_thread.join(timeout=0.05)
        assert abort_thread.is_alive()
    finally:
        sink.release_write.set()
        cut_thread.join(timeout=5.0)
        abort_thread.join(timeout=5.0)

    assert not cut_thread.is_alive()
    assert not abort_thread.is_alive()
    assert receipt[0].prefixes[0].disposition == "durable_prefix"
    assert len(sink.generation_prefix_records) == 1
    assert worker._completed_capture_calls["c1"].coords.disposition == "failed"


def test_completed_capture_evidence_expires_by_age_not_entry_count(monkeypatch):
    worker = _worker_with_capture(_MemorySink())
    now = [100.0]
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_worker_async.time.monotonic",
        lambda: now[0],
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_worker_async."
        "_COMPLETED_CAPTURE_RETENTION_S",
        10.0,
    )
    coords = SimpleNamespace(rollout_id="r0", disposition="failed")

    worker._remember_completed_capture("c1", coords, 0)
    now[0] += 11.0
    worker._remember_completed_capture("c2", coords, 0)

    assert list(worker._completed_capture_calls) == ["c2"]


def test_generation_cut_recovers_terminal_evidence_from_durable_tq_row():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _begin_captured_request(worker, [10])
    VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker,
        request,
        _served_content([11, 12], [-0.1, -0.2]),
    )
    worker._staging_source = _MemoryPrefixSource({}, records={"r0/c1": sink.records[0]})
    worker._completed_capture_calls.clear()
    inventory = GenerationCutInventory.build(
        checkpoint_id="checkpoint-1",
        server_name="policy",
        active_prefixes=[
            GenerationCutPrefix(
                ticket_id="ticket-1",
                rollout_id="r0",
                attempt_index=0,
                model_call_id="c1",
                admitted_at=1.0,
            )
        ],
    )

    receipt = VllmAsyncGenerationWorkerImpl._checkpoint_generation_cut(
        worker, inventory
    )

    assert receipt.prefixes[0].cut_kind == "terminal_completion"
    assert receipt.prefixes[0].staging_keys == ("r0/c1",)
    assert receipt.prefixes[0].prefix_token_count == 2


def test_capture_observer_rejects_unaligned_delta_output():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c1",
            "parent_call_id": None,
            "prev_len": 0,
            "mode": "text",
        },
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10])

    # DELTA output is consumed exactly once, so an incomplete token/logprob
    # pair must fail loudly instead of dropping a token that will not reappear.
    VllmAsyncGenerationWorkerImpl._observe_request_capture(
        worker,
        request,
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    token_ids=[11, 12, 13],
                    logprobs=_PartiallyPublishedLogprobs(
                        [
                            {11: SimpleNamespace(logprob=-0.1)},
                            {12: SimpleNamespace(logprob=-0.2)},
                            {13: SimpleNamespace(logprob=-0.3)},
                        ]
                    ),
                )
            ]
        ),
    )
    state = worker._capture_calls[id(request)]
    assert state.active_buffer.generated_token_ids == []
    assert state.active_buffer.generated_logprobs == []
    assert "mismatched generation token IDs" in state.observation_error


def test_generation_cut_swaps_buffer_before_blocking_prefix_write():
    sink = _BlockingPrefixSink()
    worker = _capture_worker_with_periodic_flush(sink, threshold=2)
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c1",
            "parent_call_id": None,
            "prev_len": 0,
            "mode": "text",
        },
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10, 11])
    VllmAsyncGenerationWorkerImpl._observe_request_capture(
        worker,
        request,
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    token_ids=[12, 13],
                    logprobs=[
                        {12: SimpleNamespace(logprob=-0.1)},
                        {13: SimpleNamespace(logprob=-0.2)},
                    ],
                )
            ]
        ),
    )
    first_inventory = GenerationCutInventory.build(
        checkpoint_id="checkpoint-1",
        server_name="policy",
        active_prefixes=[
            GenerationCutPrefix(
                ticket_id="ticket-1",
                rollout_id="r0",
                attempt_index=0,
                model_call_id="c1",
                admitted_at=1.0,
            )
        ],
    )
    result: list[GenerationCutReceipt] = []

    def cut() -> None:
        result.append(
            VllmAsyncGenerationWorkerImpl._checkpoint_generation_cut(
                worker, first_inventory
            )
        )

    thread = threading.Thread(target=cut)
    thread.start()
    try:
        assert sink.write_started.wait(timeout=5.0)
        # vLLM emits only the new delta into the fresh active buffer while the
        # detached buffer is blocked in TQ.
        VllmAsyncGenerationWorkerImpl._observe_request_capture(
            worker,
            request,
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        token_ids=[14, 15],
                        logprobs=[
                            {14: SimpleNamespace(logprob=-0.3)},
                            {15: SimpleNamespace(logprob=-0.4)},
                        ],
                    )
                ]
            ),
        )
        assert worker._capture_calls[id(request)].periodic_flush_due is False
    finally:
        sink.release_write.set()
        thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert len(result) == 1
    assert worker._capture_calls[id(request)].periodic_flush_due is True
    assert sink.generation_prefix_records[0][1].token_ids_delta == [
        10,
        11,
        12,
        13,
    ]
    assert result[0].prefixes[0].frozen_buffer_id.endswith("/0")

    second_inventory = GenerationCutInventory.build(
        checkpoint_id="checkpoint-2",
        server_name="policy",
        active_prefixes=[
            GenerationCutPrefix(
                ticket_id="ticket-2",
                rollout_id="r0",
                attempt_index=0,
                model_call_id="c1",
                admitted_at=1.0,
            )
        ],
    )
    second_receipt = VllmAsyncGenerationWorkerImpl._checkpoint_generation_cut(
        worker, second_inventory
    )
    assert sink.generation_prefix_records[1][1].token_ids_delta == [14, 15]
    assert second_receipt.prefixes[0].frozen_buffer_id.endswith("/1")
    assert second_receipt.prefixes[0].staging_keys == (
        "__generation_cut__/checkpoint-1/r0/c1/0",
        "__generation_cut__/checkpoint-2/r0/c1/1",
    )
    continuation = second_receipt.prefixes[0]
    source = _MemoryPrefixSource(
        {},
        records={
            key: record
            for key, (_, record) in zip(
                continuation.staging_keys,
                sink.generation_prefix_records,
                strict=True,
            )
        },
    )
    resumed_worker = _worker_with_capture(_MemorySink())
    resumed_worker._staging_source = source
    admission = CaptureAdmission(
        rollout_id="r0-a1",
        model_call_id="c2",
        mode="text",
        generation_cut={
            "source_capture_key": "r0",
            "source_model_call_id": "c1",
            "staging_keys": continuation.staging_keys,
            "generation_token_count": continuation.prefix_token_count,
            "digest": continuation.prefix_digest,
        },
    )
    restored = resumed_worker._resolve_generation_cut(admission, [])
    assert source.fetch_calls == [list(continuation.staging_keys)]
    assert restored.token_ids_delta == [10, 11, 12, 13, 14, 15]
    assert restored.token_mask_delta == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]


def test_generation_cut_restore_accepts_monotonic_mixed_policy_versions():
    first_sink = _MemorySink()
    first_capture = RolloutTokenCapture(
        sink=first_sink,
        weight_version_fn=lambda: 7,
    )
    first_call = first_capture.begin_call(
        CaptureAdmission(rollout_id="r0", model_call_id="c1", mode="text")
    )
    first_chunk = first_capture.build_prefix_record(
        first_call,
        prompt_token_ids=[10, 11],
        generated_token_ids=[12, 13],
        generated_logprobs=[-0.1, -0.2],
    )
    first_key = "__generation_cut__/checkpoint-1/r0/c1"
    second_admission = CaptureAdmission(
        rollout_id="r0-a1",
        model_call_id="c2",
        mode="text",
        generation_cut={
            "source_capture_key": "r0",
            "source_model_call_id": "c1",
            "staging_keys": [first_key],
            "generation_token_count": 2,
            "digest": first_chunk.digest,
        },
    )
    second_capture = RolloutTokenCapture(
        sink=_MemorySink(),
        weight_version_fn=lambda: 9,
    )
    second_call = second_capture.begin_call(
        second_admission,
        prefix_token_ids=[],
        generation_cut=first_chunk,
        generation_cut_staging_keys=second_admission.generation_cut.staging_keys,
    )
    second_chunk = second_capture.build_generation_chunk_record(
        second_call,
        generated_token_ids=[14, 15],
        generated_logprobs=[-0.3, -0.4],
    )
    cumulative = second_capture.build_prefix_record(
        second_call,
        prompt_token_ids=[10, 11, 12, 13],
        generated_token_ids=[14, 15],
        generated_logprobs=[-0.3, -0.4],
    )
    second_key = "__generation_cut__/checkpoint-2/r0-a1/c2"
    worker = _worker_with_capture(_MemorySink())
    worker._rollout_weight_version = 9
    worker._staging_source = _MemoryPrefixSource(
        {},
        records={first_key: first_chunk, second_key: second_chunk},
    )
    admission = CaptureAdmission(
        rollout_id="r0-a2",
        model_call_id="c3",
        mode="text",
        generation_cut={
            "source_capture_key": "r0-a1",
            "source_model_call_id": "c2",
            "staging_keys": [first_key, second_key],
            "generation_token_count": 4,
            "digest": cumulative.digest,
        },
    )

    worker._rollout_weight_version = 8
    with pytest.raises(RuntimeError, match="newer than the current rollout version"):
        worker._resolve_generation_cut(admission, [])
    worker._rollout_weight_version = 9
    restored = worker._resolve_generation_cut(admission, [])

    assert restored.weight_version == 7
    assert restored.token_ids_delta == [10, 11, 12, 13, 14, 15]
    assert restored.generation_log_probs_delta == [0.0, 0.0, -0.1, -0.2, -0.3, -0.4]


def test_generation_cut_rolls_frozen_buffer_back_after_staging_failure():
    sink = _FailOncePrefixSink()
    worker = _worker_with_capture(sink)
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c1",
            "parent_call_id": None,
            "prev_len": 0,
            "mode": "text",
        },
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10])
    VllmAsyncGenerationWorkerImpl._observe_request_capture(
        worker,
        request,
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    token_ids=[11, 12],
                    logprobs=[
                        {11: SimpleNamespace(logprob=-0.1)},
                        {12: SimpleNamespace(logprob=-0.2)},
                    ],
                )
            ]
        ),
    )
    inventory = GenerationCutInventory.build(
        checkpoint_id="checkpoint-1",
        server_name="policy",
        active_prefixes=[
            GenerationCutPrefix(
                ticket_id="ticket-1",
                rollout_id="r0",
                attempt_index=0,
                model_call_id="c1",
                admitted_at=1.0,
            )
        ],
    )

    with pytest.raises(RuntimeError, match="injected failure"):
        VllmAsyncGenerationWorkerImpl._checkpoint_generation_cut(worker, inventory)

    receipt = VllmAsyncGenerationWorkerImpl._checkpoint_generation_cut(
        worker, inventory
    )
    assert sink.generation_prefix_records[-1][1].token_ids_delta == [10, 11, 12]
    assert receipt.prefixes[0].prefix_token_count == 2


def test_generation_cut_validates_ack_before_sealing_state(monkeypatch):
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c1",
            "parent_call_id": None,
            "prev_len": 0,
            "mode": "text",
        },
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10])
    VllmAsyncGenerationWorkerImpl._observe_request_capture(
        worker,
        request,
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    token_ids=[11],
                    logprobs=[{11: SimpleNamespace(logprob=-0.1)}],
                )
            ]
        ),
    )
    inventory = GenerationCutInventory.build(
        checkpoint_id="checkpoint-1",
        server_name="policy",
        active_prefixes=[
            GenerationCutPrefix(
                ticket_id="ticket-1",
                rollout_id="r0",
                attempt_index=0,
                model_call_id="c1",
                admitted_at=1.0,
            )
        ],
    )

    class _RejectingAck:
        def __init__(self, **_: object) -> None:
            raise ValueError("injected acknowledgement failure")

    monkeypatch.setattr(
        "nemo_gym._checkpoint.model_control_contracts.GenerationCutPrefixAck",
        _RejectingAck,
    )
    with pytest.raises(ValueError, match="injected acknowledgement failure"):
        VllmAsyncGenerationWorkerImpl._checkpoint_generation_cut(worker, inventory)

    state = worker._capture_calls[id(request)]
    assert state.generation_cut_staging_keys == []
    assert state.sealed_generated_token_ids == []
    assert state.active_buffer.generated_token_ids == [11]
    assert state.frozen_buffer is None
    assert sink.cleared_generation_prefix_keys == [
        "__generation_cut__/checkpoint-1/r0/c1/0"
    ]


def test_generation_cut_retry_after_later_call_failure_uses_new_chunk_key():
    sink = _FailModelCallOncePrefixSink("c2")
    worker = _worker_with_capture(sink)
    first_request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c1",
            "parent_call_id": None,
            "prev_len": 0,
            "mode": "text",
        },
        stream=False,
    )
    second_request = _FakeRequest(
        ng_capture={
            "rollout_id": "r1",
            "model_call_id": "c2",
            "parent_call_id": None,
            "prev_len": 0,
            "mode": "text",
        },
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, first_request, [10])
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, second_request, [20])
    VllmAsyncGenerationWorkerImpl._observe_request_capture(
        worker,
        first_request,
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    token_ids=[11],
                    logprobs=[{11: SimpleNamespace(logprob=-0.1)}],
                )
            ]
        ),
    )
    VllmAsyncGenerationWorkerImpl._observe_request_capture(
        worker,
        second_request,
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    token_ids=[21],
                    logprobs=[{21: SimpleNamespace(logprob=-0.2)}],
                )
            ]
        ),
    )
    inventory = GenerationCutInventory.build(
        checkpoint_id="checkpoint-1",
        server_name="policy",
        active_prefixes=[
            GenerationCutPrefix(
                ticket_id="ticket-1",
                rollout_id="r0",
                attempt_index=0,
                model_call_id="c1",
                admitted_at=1.0,
            ),
            GenerationCutPrefix(
                ticket_id="ticket-2",
                rollout_id="r1",
                attempt_index=0,
                model_call_id="c2",
                admitted_at=1.0,
            ),
        ],
    )

    with pytest.raises(RuntimeError, match="injected later-call failure"):
        VllmAsyncGenerationWorkerImpl._checkpoint_generation_cut(worker, inventory)

    # The first call keeps decoding after its successful buffer swap. Retrying
    # the same checkpoint must append that new chunk under a distinct key.
    VllmAsyncGenerationWorkerImpl._observe_request_capture(
        worker,
        first_request,
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    token_ids=[12],
                    logprobs=[{12: SimpleNamespace(logprob=-0.3)}],
                )
            ]
        ),
    )
    receipt = VllmAsyncGenerationWorkerImpl._checkpoint_generation_cut(
        worker, inventory
    )

    first_ack = next(
        prefix for prefix in receipt.prefixes if prefix.model_call_id == "c1"
    )
    assert first_ack.staging_keys == (
        "__generation_cut__/checkpoint-1/r0/c1/0",
        "__generation_cut__/checkpoint-1/r0/c1/1",
    )
    assert len(first_ack.staging_keys) == len(set(first_ack.staging_keys))
    assert first_ack.prefix_token_count == 2
    assert [
        record.token_ids_delta
        for _, record in sink.generation_prefix_records
        if record.model_call_id == "c1"
    ] == [[10, 11], [12]]


def test_restored_generation_cut_is_extended_and_retired_on_completion(caplog):
    caplog.set_level(logging.INFO)
    sink = _MemorySink()
    original_worker = _worker_with_capture(sink)
    original_worker._rollout_weight_version = 7
    original_request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c1",
            "mode": "text",
        },
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(
        original_worker, original_request, [10, 11]
    )
    progress = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                token_ids=[12, 13],
                logprobs=[
                    {12: SimpleNamespace(logprob=-0.1)},
                    {13: SimpleNamespace(logprob=-0.2)},
                ],
            )
        ]
    )
    VllmAsyncGenerationWorkerImpl._observe_request_capture(
        original_worker, original_request, progress
    )
    inventory = GenerationCutInventory.build(
        checkpoint_id="checkpoint-1",
        server_name="policy",
        active_prefixes=[
            GenerationCutPrefix(
                ticket_id="ticket-1",
                rollout_id="r0",
                attempt_index=0,
                model_call_id="c1",
                admitted_at=1.0,
            )
        ],
    )
    receipt = VllmAsyncGenerationWorkerImpl._checkpoint_generation_cut(
        original_worker, inventory
    )
    (cut_key,) = receipt.prefixes[0].staging_keys
    cut_record = sink.generation_prefix_records[-1][1]

    resumed_worker = _worker_with_capture(sink)
    resumed_worker._rollout_weight_version = 9
    resumed_worker._staging_source = _MemoryPrefixSource(
        {}, records={cut_key: cut_record}
    )
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0-a1",
            "model_call_id": "c2",
            "mode": "text",
            "generation_cut": {
                "source_capture_key": "r0",
                "source_model_call_id": "c1",
                "staging_keys": [cut_key],
                "generation_token_count": 2,
                "digest": cut_record.digest,
            },
        },
        stream=False,
    )
    admission = resumed_worker._capture_admission(request)
    bad_admission = admission.model_copy(
        update={
            "generation_cut": admission.generation_cut.model_copy(
                update={"digest": "0" * 64}
            )
        }
    )
    with pytest.raises(
        RuntimeError,
        match="checkpoint coordinates do not match the staged prefix",
    ):
        resumed_worker._resolve_generation_cut(bad_admission, [])
    cut = resumed_worker._resolve_generation_cut(admission, [])
    VllmAsyncGenerationWorkerImpl._begin_request_capture(
        resumed_worker,
        request,
        [10, 11, 12, 13],
        admission=admission,
        prefix_token_ids=[],
        generation_cut=cut,
        resumed_generation_token_ids=[12, 13],
    )
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "partial tail"
    output = SimpleNamespace(token_ids=[14], text=" tail")
    VllmAsyncGenerationWorkerImpl._restore_response_prefix(
        resumed_worker,
        request,
        SimpleNamespace(outputs=[output]),
        tokenizer=tokenizer,
    )
    tokenizer.decode.assert_called_once_with([12, 13, 14])
    assert output.text == "partial tail"
    output.token_ids.append(15)
    VllmAsyncGenerationWorkerImpl._restore_response_prefix(
        resumed_worker,
        request,
        SimpleNamespace(outputs=[output]),
        tokenizer=tokenizer,
    )
    tokenizer.decode.assert_called_with([12, 13, 14, 15])

    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        resumed_worker,
        request,
        _served_content([14], [-0.3]),
    )

    final_record = sink.records[-1]
    assert content["ng_commit_coords"]["rollout_id"] == "r0-a1"
    assert final_record.token_ids_delta == [10, 11, 12, 13, 14]
    assert final_record.token_mask_delta == [0.0, 0.0, 1.0, 1.0, 1.0]
    assert final_record.generation_log_probs_delta == [0.0, 0.0, -0.1, -0.2, -0.3]
    assert final_record.weight_version == 7
    assert content["ng_commit_coords"]["weight_version"] == 7
    assert resumed_worker._completed_capture_calls["c2"].generation_token_count == 3
    assert "generation prefix restored:" in caplog.text
    assert f"prefix_digest={cut_record.digest}" in caplog.text
    assert "prefix_tokens=2 tail_tokens=1 total_generation_tokens=3" in caplog.text


def test_checkpoint_gate_holds_terminal_stage_until_reopened():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c1",
            "parent_call_id": None,
            "prev_len": 0,
            "mode": "text",
        },
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10])
    worker._generation_checkpoint_gate.close_and_wait()
    finished = threading.Event()

    def finish() -> None:
        VllmAsyncGenerationWorkerImpl._finish_request_capture(
            worker, request, _served_content([11], [-0.1])
        )
        finished.set()

    thread = threading.Thread(target=finish)
    thread.start()
    try:
        assert not finished.wait(timeout=0.05)
        assert sink.records == []
        worker._generation_checkpoint_gate.reopen()
        assert finished.wait(timeout=5.0)
    finally:
        worker._generation_checkpoint_gate.reopen()
        thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert len(sink.records) == 1


def test_request_capture_token_in_prev_len_chains():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
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
            "rollout_id": "r0",
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
    """The admission dict is read once and never mutated: the resolved prefix
    reaches the request via the adapter and begin_call via its keyword."""
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    source = _MemoryPrefixSource({"r0/c1": [10, 11], "r0/c2": [12]})
    worker._staging_source = source
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
    worker._staging_source = _MemoryPrefixSource({})
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
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
    assert worker._staging_source.calls == []
    text_root = worker._capture_admission(
        _FakeRequest(
            ng_capture={"rollout_id": "r0", "model_call_id": "c1", "mode": "text"}
        )
    )
    assert worker._resolve_admission_prefix(text_root) == []


def test_staging_chain_cache_fetches_only_uncached_suffix():
    worker = _worker_with_capture(_MemorySink())
    source = _MemoryPrefixSource({"r0/c1": [10, 11], "r0/c2": [12]})
    worker._staging_source = source

    first = VllmAsyncGenerationWorkerImpl._fetch_chain_prefix(worker, ["r0/c1"])
    second = VllmAsyncGenerationWorkerImpl._fetch_chain_prefix(
        worker, ["r0/c1", "r0/c2"]
    )

    assert first == [10, 11]
    assert second == [10, 11, 12]
    assert source.calls == [["r0/c1"], ["r0/c2"]]


def test_staging_chain_prefix_length_mismatch_is_rejected_by_begin_call():
    """prev_len enforcement lives in Gym's begin_call, not in the worker."""
    worker = _worker_with_capture(_MemorySink())
    worker._staging_source = _MemoryPrefixSource({"r0/c1": [10, 11], "r0/c2": []})
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

    with pytest.raises(CaptureError, match="pass the resolved prefix_token_ids"):
        VllmAsyncGenerationWorkerImpl._begin_request_capture(
            worker, request, [10, 11, 12, 20]
        )

    assert worker._capture_calls == {}


def test_request_capture_is_a_noop_without_context_or_capture():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    plain = _FakeRequest(stream=False)  # no ng_capture attribute
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, plain, [1, 2])
    content = {
        "choices": [{"message": {"role": "assistant"}, "logprobs": {"content": []}}]
    }
    out = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, plain, dict(content)
    )
    assert "ng_commit_coords" not in out
    assert out["choices"][0]["logprobs"] is not None  # untouched off the capture path
    assert sink.records == []


def test_request_capture_abort_fails_the_call_and_drains_state():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c1",
            "parent_call_id": None,
            "prev_len": 0,
            "mode": "text",
        },
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [1, 2])
    VllmAsyncGenerationWorkerImpl._abort_request_capture(
        worker, request, reason="engine_error"
    )
    assert worker._capture_calls == {}
    assert sink.records == []
    # A late finish after abort is a no-op (state already drained).
    out = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker, request, _served_content([3], [-0.1])
    )
    assert "ng_commit_coords" not in out
