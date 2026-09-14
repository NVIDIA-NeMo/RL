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
    _remaining_generation_limits_after_prefix,
)

pytestmark = pytest.mark.nemo_gym


class _MemorySink:
    def __init__(self) -> None:
        self.records: list[StagedCallRecord] = []
        self.generation_prefix_records: list[tuple[str, StagedCallRecord]] = []

    def stage(self, record: StagedCallRecord) -> StageResult:
        self.records.append(record)
        return StageResult(ok=True, staging_key=record.staging_key)

    def stage_generation_prefix(
        self, record: StagedCallRecord, *, checkpoint_id: str
    ) -> StageResult:
        self.generation_prefix_records.append((checkpoint_id, record))
        return StageResult(
            ok=True,
            staging_key=(
                f"__generation_cut__/{checkpoint_id}/"
                f"{record.rollout_id}/{record.model_call_id}"
            ),
        )


class _BlockingPrefixSink(_MemorySink):
    """Hold one prefix write so a test can observe the post-swap buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.write_started = threading.Event()
        self.release_write = threading.Event()

    def stage_generation_prefix(
        self, record: StagedCallRecord, *, checkpoint_id: str
    ) -> StageResult:
        self.write_started.set()
        if not self.release_write.wait(timeout=5.0):
            raise TimeoutError("test did not release the prefix write")
        return super().stage_generation_prefix(record, checkpoint_id=checkpoint_id)


class _FailOncePrefixSink(_MemorySink):
    """Reject the first prefix write and accept its retry."""

    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def stage_generation_prefix(
        self, record: StagedCallRecord, *, checkpoint_id: str
    ) -> StageResult:
        if not self.failed:
            self.failed = True
            return StageResult(ok=False, error="injected failure")
        return super().stage_generation_prefix(record, checkpoint_id=checkpoint_id)


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
        _generation_checkpoint_gate=_CheckpointCaptureGate(),
        _generation_checkpoint_decoding_paused=False,
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
        ("pause_generation_for_checkpoint", "pause_generation_for_checkpoint_async"),
        (
            "resume_generation_after_checkpoint",
            "resume_generation_after_checkpoint_async",
        ),
        ("resume_generation_after_cut", "resume_generation_after_cut_async"),
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


def test_resume_after_cut_keeps_terminal_stage_fenced_until_finish():
    class _FakeLLM:
        def __init__(self) -> None:
            self.resume_calls = 0

        async def resume_generation(self) -> None:
            self.resume_calls += 1

    async def scenario() -> None:
        worker = object.__new__(VllmAsyncGenerationWorkerImpl)
        worker.cfg = {"vllm_cfg": {"async_engine": True}}
        worker.llm = _FakeLLM()
        worker._generation_checkpoint_gate = _CheckpointCaptureGate()
        worker._generation_checkpoint_gate.close_and_wait()
        worker._generation_checkpoint_decoding_paused = True

        await worker.resume_generation_after_cut_async()
        assert worker.llm.resume_calls == 1

        entered = threading.Event()
        released = threading.Event()

        def enter_terminal_stage() -> None:
            entered.set()
            worker._generation_checkpoint_gate.enter()
            released.set()
            worker._generation_checkpoint_gate.exit()

        thread = threading.Thread(target=enter_terminal_stage)
        thread.start()
        try:
            assert entered.wait(timeout=5.0)
            assert not released.wait(timeout=0.05)
            await worker.finish_generation_checkpoint_async()
            assert released.wait(timeout=5.0)
        finally:
            worker._generation_checkpoint_gate.reopen()
            thread.join(timeout=5.0)
        assert not thread.is_alive()

        # A cleanup retry after the split resume is harmless and does not send
        # a second resume RPC to vLLM.
        await worker.resume_generation_after_checkpoint_async()
        assert worker.llm.resume_calls == 1

    asyncio.run(scenario())


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
        "_finish_request_capture_after_checkpoint_gate",
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
    assert receipt.prefixes[0].prefix_token_count == 2
    assert receipt.prefixes[0].staging_key == ("__generation_cut__/checkpoint-1/r0/c1")
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


def test_generation_cut_swaps_buffer_before_blocking_prefix_write():
    sink = _BlockingPrefixSink()
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
        # vLLM remains cumulative, but only the new suffix is appended to the
        # fresh active buffer while the detached buffer is blocked in TQ.
        VllmAsyncGenerationWorkerImpl._observe_request_capture(
            worker,
            request,
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        token_ids=[12, 13, 14, 15],
                        logprobs=[
                            {12: SimpleNamespace(logprob=-0.1)},
                            {13: SimpleNamespace(logprob=-0.2)},
                            {14: SimpleNamespace(logprob=-0.3)},
                            {15: SimpleNamespace(logprob=-0.4)},
                        ],
                    )
                ]
            ),
        )
    finally:
        sink.release_write.set()
        thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert len(result) == 1
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
    assert sink.generation_prefix_records[1][1].token_ids_delta == [
        10,
        11,
        12,
        13,
        14,
        15,
    ]
    assert second_receipt.prefixes[0].frozen_buffer_id.endswith("/1")


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


def test_restored_generation_cut_is_extended_and_retired_on_completion(caplog):
    caplog.set_level(logging.INFO)
    sink = _MemorySink()
    original_worker = _worker_with_capture(sink)
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
    cut_key = receipt.prefixes[0].staging_key
    assert cut_key is not None
    cut_record = sink.generation_prefix_records[-1][1]

    resumed_worker = _worker_with_capture(sink)
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
                "staging_key": cut_key,
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
        resumed_worker._resolve_generation_cut(bad_admission)
    cut = resumed_worker._resolve_generation_cut(admission)
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
