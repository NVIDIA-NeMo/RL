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

from nemo_rl.data_plane.tq_token_sink import ChainPrefixCache  # noqa: E402
from nemo_rl.models.generation.generation_cut_capture import (  # noqa: E402
    GenerationPrefixBatchLimits,
    _remaining_generation_limits_after_prefix,
    _TokenCaptureSnapshotGate,
)
from nemo_rl.models.generation.prefix_read_batcher import PrefixReadBatcher  # noqa: E402
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration  # noqa: E402
from nemo_rl.models.generation.vllm.vllm_worker_async import (  # noqa: E402
    VllmAsyncGenerationWorkerImpl,
    _classify_restored_prefix_terminal,
    _RequestOutputDeltaAccumulator,
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
            f"__generation_cut__/{checkpoint_id}/{record.rollout_id}/"
            f"{record.model_call_id}/{chunk_sequence}"
        )
        self.generation_prefix_keys.append(key)
        return StageResult(ok=True, staging_key=key)

    def stage_generation_prefix_batch(
        self,
        records: list[StagedCallRecord],
        *,
        checkpoint_id: str,
        chunk_sequences: list[int],
    ) -> list[StageResult]:
        return [
            self.stage_generation_prefix(
                record,
                checkpoint_id=checkpoint_id,
                chunk_sequence=sequence,
            )
            for record, sequence in zip(records, chunk_sequences, strict=True)
        ]

    def clear(self, staging_keys: list[str]) -> None:
        self.cleared_generation_prefix_keys.extend(staging_keys)


class _ByteTokenizer:
    """Byte-level stand-in: each token ID is one UTF-8 byte of the output."""

    def decode(self, token_ids: list[int], **kwargs) -> str:
        return bytes(token_ids).decode("utf-8", errors="replace")


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
        _generation_prefix_batch_limits=GenerationPrefixBatchLimits(
            max_rows=256, max_tokens=4_194_304
        ),
        _capture_registry_lock=threading.Lock(),
        _capture_sink=None,
        _generation_prefix_cuts_enabled=False,
        _generation_cut_control_token=None,
        _generation_cut_control_timeout_s=None,
        _generation_cut_tokenizer=_ByteTokenizer(),
        _token_capture_snapshot_gate=_TokenCaptureSnapshotGate(),
        _staging_source=None,
        _generation_prefix_reader=None,
        _capture_media=False,
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
        lambda dp_client, *, staging_partition, capture_media, media_pixel_dtype=None: (
            sink
        ),
    )
    worker = _fake_worker()

    installed = asyncio.run(
        VllmAsyncGenerationWorkerImpl.setup_token_capture(
            worker, dp_cfg={"backend": "simple"}, staging_partition="rollout_staging"
        )
    )

    assert installed is True
    assert isinstance(worker.token_capture, RolloutTokenCapture)
    assert isinstance(worker._generation_prefix_reader, PrefixReadBatcher)
    assert worker._generation_prefix_batch_limits == GenerationPrefixBatchLimits(
        max_rows=256, max_tokens=4_194_304
    )
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
        lambda dp_client, *, staging_partition, capture_media, media_pixel_dtype=None: (
            sink
        ),
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
        lambda futures: futures,
    )
    gen.setup_token_capture(
        {"backend": "simple"},
        "rollout_staging",
        generation_cut_control_timeout_s=60.0,
        generation_prefix_batch_size=128,
        generation_prefix_batch_max_tokens=2048,
    )
    gen.worker_group.run_all_workers_single_data.assert_called_once_with(
        "setup_token_capture",
        dp_cfg={"backend": "simple"},
        staging_partition="rollout_staging",
        capture_media=False,
        generation_prefix_cuts_enabled=False,
        generation_cut_control_token=None,
        generation_cut_control_timeout_s=60.0,
        generation_prefix_batch_size=128,
        generation_prefix_batch_max_tokens=2048,
        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
    )


def test_generation_setup_token_capture_requires_async_engine():
    gen = _generation_with_mock_group(async_engine=False)
    with pytest.raises(AssertionError, match="async vLLM engine"):
        gen.setup_token_capture({}, "rollout_staging")


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


def test_prefix_capture_setup_requires_control_token_and_rejects_media(monkeypatch):
    monkeypatch.setattr(
        "nemo_rl.data_plane.build_data_plane_client",
        lambda dp_cfg, bootstrap: MagicMock(),
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

    worker = _fake_worker()
    worker.llm = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="does not yet support multimodal"):
        asyncio.run(
            VllmAsyncGenerationWorkerImpl.setup_token_capture(
                worker,
                {},
                "rollout_staging",
                capture_media=True,
                generation_prefix_cuts_enabled=True,
                generation_cut_control_token="secret",
            )
        )


@pytest.mark.parametrize(
    ("max_tokens", "min_tokens", "generation_token_count", "expected"),
    [
        (10, 6, 3, (7, 3)),
        (None, 6, 3, (None, 3)),
        (10, None, 3, (7, None)),
        (10, 2, 3, (7, 0)),
    ],
)
def test_restored_prefix_reduces_remaining_output_limits(
    max_tokens, min_tokens, generation_token_count, expected
):
    assert (
        _remaining_generation_limits_after_prefix(
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            generation_token_count=generation_token_count,
        )
        == expected
    )


def test_restored_prefix_at_output_limit_is_terminal():
    terminal = _classify_restored_prefix_terminal(
        prompt_token_ids=[1, 2, 3, 4],
        generation_token_count=2,
        requested_output_tokens=2,
        model_max_tokens=8,
    )
    assert terminal is not None
    assert terminal.reason == "output_limit"
    assert terminal.original_prompt_token_count == 2


def test_request_output_deltas_are_assembled_without_mutating_inputs():
    def output(text, token_ids):
        return SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    index=0,
                    text=text,
                    token_ids=list(token_ids),
                    logprobs=[f"lp-{token_id}" for token_id in token_ids],
                )
            ]
        )

    first = output("a", [1])
    second = output("bc", [2, 3])
    accumulator = _RequestOutputDeltaAccumulator()
    accumulator.append(first)
    accumulator.append(second)
    accumulated = accumulator.build()
    assert accumulated.outputs[0].text == "abc"
    assert accumulated.outputs[0].token_ids == [1, 2, 3]
    assert first.outputs[0].token_ids == [1]
    assert second.outputs[0].token_ids == [2, 3]


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


def _worker_with_capture(sink: _MemorySink):
    from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter

    worker = _fake_worker()
    worker._chain_prefix = ChainPrefixCache()
    worker._capture_sink = sink
    worker._delta_align_routed_experts = (
        VllmAsyncGenerationWorkerImpl._delta_align_routed_experts
    )
    for name in (
        "_fetch_chain_prefix",
        "_capture_admission",
        "_resolve_admission_prefix",
        "_fetch_generation_cut_chunks",
        "_rebuild_generation_cut_snapshot",
        "_resolve_generation_cut",
        "_enter_request_prefix",
        "_pop_request_capture",
        "_get_request_capture",
        "_remember_completed_capture",
        "_abort_request_capture",
        "_finish_request_capture_after_snapshot_fence",
        "_finish_request_capture_with_lifecycle_owned",
        "_restore_response_prefix",
        "_checkpoint_active_generation_cut",
        "_require_generation_prefix_batch_limits",
        "_generation_cut_transaction",
        "_checkpoint_generation_cut_batched",
        "_completed_generation_cut_ack",
        "_checkpoint_generation_cut",
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


class _BatchPrefixSink(_MemorySink):
    def __init__(self) -> None:
        super().__init__()
        self.batches: list[list[int]] = []
        self.failure: str | None = None
        self.before_write = lambda: None

    def stage_generation_prefix_batch(self, records, *, checkpoint_id, chunk_sequences):
        self.batches.append([len(record.token_ids_delta) for record in records])
        self.before_write()
        results = super().stage_generation_prefix_batch(
            records,
            checkpoint_id=checkpoint_id,
            chunk_sequences=chunk_sequences,
        )
        if self.failure == "raise":
            raise RuntimeError("injected batch transport failure after writes")
        if self.failure == "short":
            return results[:-1]
        if self.failure == "partial":
            results[-1] = StageResult(
                ok=False,
                staging_key=results[-1].staging_key,
                error="injected partial failure",
            )
        if self.failure == "wrong-key":
            results[-1] = StageResult(ok=True, staging_key="wrong-key")
        return results


def _batch_worker_fixture(lengths, *, batch_size=256, max_tokens=4_194_304):
    sink = _BatchPrefixSink()
    worker = _worker_with_capture(sink)
    worker._generation_prefix_batch_limits = GenerationPrefixBatchLimits(
        max_rows=batch_size,
        max_tokens=max_tokens,
    )
    requests, prefixes = [], []
    for i, length in enumerate(lengths):
        request = _FakeRequest(
            ng_capture={
                "rollout_id": f"r{i}",
                "model_call_id": f"c{i}",
                "parent_call_id": None,
                "prev_len": 0,
                "mode": "text",
            },
            stream=False,
        )
        VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10])
        state = worker._capture_calls[id(request)]
        state.effective_output_limit = 128
        with state.lock:
            state.observe(list(range(20, 20 + length)), [-0.1] * length)
        requests.append(request)
        prefixes.append(
            GenerationCutPrefix(
                ticket_id=f"t{i}",
                rollout_id=f"r{i}",
                attempt_index=0,
                model_call_id=f"c{i}",
                admitted_at=1.0,
            )
        )
    inventory = GenerationCutInventory.build(
        checkpoint_id="batch-cut",
        server_name="policy",
        active_prefixes=prefixes,
    )
    return worker, sink, requests, inventory


def test_worker_batches_1000_prefixes_and_caches_receipt():
    worker, sink, requests, inventory = _batch_worker_fixture([2] * 1000)
    receipt = worker._checkpoint_generation_cut(inventory)

    assert [len(batch) for batch in sink.batches] == [256, 256, 256, 232]
    assert [ack.ticket_id for ack in receipt.prefixes] == [
        prefix.ticket_id for prefix in inventory.active_prefixes
    ]
    assert all(ack.prefix_token_count == 2 for ack in receipt.prefixes)
    assert all(worker._capture_calls[id(r)].frozen_buffer is None for r in requests)
    assert worker._checkpoint_generation_cut(inventory) is receipt
    assert len(sink.batches) == 4


def test_worker_batch_token_limit_handles_ragged_and_oversized_rows():
    worker, sink, _, inventory = _batch_worker_fixture([1, 2, 8, 1], max_tokens=5)
    worker._checkpoint_generation_cut(inventory)
    assert sink.batches == [[2, 3], [9], [2]]  # First chunks include the prompt.


@pytest.mark.parametrize("failure", ["raise", "short", "partial", "wrong-key"])
def test_worker_batch_failure_rolls_back_every_unsealed_row(failure):
    worker, sink, requests, inventory = _batch_worker_fixture([2, 3, 4])
    sink.failure = failure

    with pytest.raises(RuntimeError):
        worker._checkpoint_generation_cut(inventory)

    assert not worker._generation_cut_receipts
    assert set(sink.generation_prefix_keys) <= set(sink.cleared_generation_prefix_keys)
    for request, length in zip(requests, [2, 3, 4], strict=True):
        state = worker._capture_calls[id(request)]
        assert state.frozen_buffer is None
        assert not state.generation_cut_staging_keys
        assert not state.sealed_generated_token_ids
        assert len(state.active_buffer.generated_token_ids) == length
        assert state.lifecycle_lock.acquire(blocking=False)
        state.lifecycle_lock.release()

    sink.failure = None
    receipt = worker._checkpoint_generation_cut(inventory)
    assert [ack.prefix_token_count for ack in receipt.prefixes] == [2, 3, 4]


def test_worker_batch_validates_every_ack_before_sealing(monkeypatch):
    from nemo_gym._checkpoint import generation_cut

    worker, sink, requests, inventory = _batch_worker_fixture([2, 3])
    original = generation_cut.GenerationCutPrefixAck

    def reject_second(**kwargs):
        if kwargs["model_call_id"] == "c1":
            raise ValueError("injected acknowledgement failure")
        return original(**kwargs)

    monkeypatch.setattr(generation_cut, "GenerationCutPrefixAck", reject_second)
    with pytest.raises(ValueError, match="injected acknowledgement failure"):
        worker._checkpoint_generation_cut(inventory)

    for request in requests:
        state = worker._capture_calls[id(request)]
        assert state.frozen_buffer is None
        assert not state.sealed_generated_token_ids
        assert not state.generation_cut_staging_keys
    assert set(sink.generation_prefix_keys) <= set(sink.cleared_generation_prefix_keys)


def test_worker_batch_retains_earlier_success_on_later_batch_failure():
    worker, sink, requests, inventory = _batch_worker_fixture([2] * 3, batch_size=2)

    def fail_second():
        if len(sink.batches) == 2:
            sink.failure = "partial"

    sink.before_write = fail_second
    with pytest.raises(RuntimeError):
        worker._checkpoint_generation_cut(inventory)

    earlier_keys = set(sink.generation_prefix_keys[:2])
    assert earlier_keys.isdisjoint(sink.cleared_generation_prefix_keys)
    sink.failure = None
    sink.before_write = lambda: None
    for request in requests:
        state = worker._capture_calls[id(request)]
        with state.lock:
            state.observe([99], [-0.2])
    receipt = worker._checkpoint_generation_cut(inventory)
    assert [ack.prefix_token_count for ack in receipt.prefixes] == [3, 3, 3]
    assert [len(ack.staging_keys) for ack in receipt.prefixes] == [2, 2, 1]
    assert earlier_keys.isdisjoint(sink.cleared_generation_prefix_keys)


def test_worker_batch_preparation_failure_releases_pending_calls():
    worker, sink, requests, inventory = _batch_worker_fixture([2, 3])
    worker._capture_calls[id(requests[1])].effective_output_limit = None

    with pytest.raises(RuntimeError, match="effective output limit"):
        worker._checkpoint_generation_cut(inventory)

    assert not sink.batches
    for request in requests:
        state = worker._capture_calls[id(request)]
        assert state.frozen_buffer is None
        assert not state.sealed_generated_token_ids
        assert state.lifecycle_lock.acquire(blocking=False)
        state.lifecycle_lock.release()


def test_worker_batch_past_deadline_fails_calls_not_yet_written(monkeypatch):
    from nemo_rl.models.generation import generation_cut_capture

    worker, sink, requests, inventory = _batch_worker_fixture([2, 3, 4], batch_size=2)
    clock = {"now": 0.0}
    monkeypatch.setattr(
        generation_cut_capture, "time", SimpleNamespace(monotonic=lambda: clock["now"])
    )

    def expire_during_first_write():
        clock["now"] = 10.0

    sink.before_write = expire_during_first_write
    receipt = worker._checkpoint_generation_cut(inventory, deadline=5.0)

    # The batch already writing completes; the call not yet reached fails.
    receipt.validate_for(inventory)
    assert sink.batches == [[3, 4]]  # First chunks include the prompt.
    assert [ack.disposition for ack in receipt.prefixes] == [
        "durable_prefix",
        "durable_prefix",
        "durable_failure",
    ]
    state = worker._capture_calls[id(requests[2])]
    assert state.frozen_buffer is None
    assert not state.generation_cut_staging_keys
    assert len(state.active_buffer.generated_token_ids) == 4


def test_worker_batch_past_deadline_rolls_back_pending_calls(monkeypatch):
    from nemo_rl.models.generation import generation_cut_capture

    worker, sink, requests, inventory = _batch_worker_fixture([2, 3, 4], batch_size=3)
    # Every call is frozen before the deadline; it passes before their write.
    readings = iter([0.0, 0.0, 0.0])
    monkeypatch.setattr(
        generation_cut_capture,
        "time",
        SimpleNamespace(monotonic=lambda: next(readings, 10.0)),
    )

    receipt = worker._checkpoint_generation_cut(inventory, deadline=5.0)

    receipt.validate_for(inventory)
    assert not sink.batches
    assert [ack.disposition for ack in receipt.prefixes] == ["durable_failure"] * 3
    for request, length in zip(requests, [2, 3, 4], strict=True):
        state = worker._capture_calls[id(request)]
        assert state.frozen_buffer is None
        assert not state.sealed_generated_token_ids
        assert len(state.active_buffer.generated_token_ids) == length
        assert state.lifecycle_lock.acquire(blocking=False)
        state.lifecycle_lock.release()


def test_worker_batch_observation_continues_while_lifecycle_waits():
    worker, sink, requests, inventory = _batch_worker_fixture([2, 3])
    state = worker._capture_calls[id(requests[0])]
    write_started = threading.Event()
    release_write = threading.Event()
    lifecycle_done = threading.Event()

    def block_write():
        write_started.set()
        assert release_write.wait(5)

    sink.before_write = block_write
    with ThreadPoolExecutor(max_workers=2) as executor:
        cut = executor.submit(worker._checkpoint_generation_cut, inventory)
        assert write_started.wait(5)

        def terminal_operation():
            with state.lifecycle_lock:
                lifecycle_done.set()

        terminal = executor.submit(terminal_operation)
        try:
            assert not lifecycle_done.wait(0.05)
            with state.lock:
                state.observe([99], [-0.2])
        finally:
            release_write.set()
        receipt = cut.result(timeout=5)
        terminal.result(timeout=5)

    assert receipt.prefixes[0].prefix_token_count == 2
    assert state.active_buffer.generated_token_ids == [99]
    assert state.sealed_generated_token_ids == [20, 21]


@pytest.mark.parametrize("batch_size", [1, 256])
def test_worker_batch_consecutive_cuts_reuse_then_append(batch_size):
    worker, sink, requests, inventory = _batch_worker_fixture(
        [2, 3], batch_size=batch_size
    )
    first = worker._checkpoint_generation_cut(inventory)
    second_inventory = GenerationCutInventory.build(
        checkpoint_id="second-cut",
        server_name="policy",
        active_prefixes=inventory.active_prefixes,
    )
    second = worker._checkpoint_generation_cut(second_inventory)
    assert [ack.staging_keys for ack in second.prefixes] == [
        ack.staging_keys for ack in first.prefixes
    ]
    assert len(sink.generation_prefix_keys) == 2

    for request in requests:
        state = worker._capture_calls[id(request)]
        with state.lock:
            state.observe([99], [-0.2])
    third_inventory = GenerationCutInventory.build(
        checkpoint_id="third-cut",
        server_name="policy",
        active_prefixes=inventory.active_prefixes,
    )
    third = worker._checkpoint_generation_cut(third_inventory)
    assert [ack.prefix_token_count for ack in third.prefixes] == [3, 4]
    assert all(len(ack.staging_keys) == 2 for ack in third.prefixes)
    assert [
        list(record.token_ids_delta)
        for _, record in sink.generation_prefix_records[-2:]
    ] == [[99], [99]]


def test_worker_batch_mixes_zero_token_and_active_calls():
    worker, sink, _, inventory = _batch_worker_fixture([2, 0, 3])
    receipt = worker._checkpoint_generation_cut(inventory)
    assert [len(batch) for batch in sink.batches] == [2]
    assert [ack.disposition for ack in receipt.prefixes] == [
        "durable_prefix",
        "durable_failure",
        "durable_prefix",
    ]


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

    def fetch_prefix_token_ids(self, staging_keys: list[str]) -> list[int]:
        self.calls.append(list(staging_keys))
        return [token for key in staging_keys for token in self.deltas[key]]

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
    worker._generation_prefix_cuts_enabled = True
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
    state = worker._capture_calls[id(request)]
    with state.lock:
        state.effective_output_limit = 8
        state.observe([20, 21], [-0.1, -0.2])

    inventory = GenerationCutInventory.build(
        checkpoint_id="checkpoint-1",
        server_name="policy_model",
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


def test_generation_cut_past_its_deadline_fails_uncut_calls_without_staging():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    worker._generation_prefix_cuts_enabled = True
    request = _FakeRequest(
        ng_capture={"rollout_id": "r0", "model_call_id": "c1", "mode": "text"},
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10])
    state = worker._capture_calls[id(request)]
    with state.lock:
        state.effective_output_limit = 8
        state.observe([11, 12], [-0.1, -0.2])
    inventory = GenerationCutInventory.build(
        checkpoint_id="checkpoint-1",
        server_name="policy_model",
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

    # Gym has already stopped waiting: the queued cut must not stage rows.
    receipt = worker._checkpoint_generation_cut(
        inventory, deadline=time.monotonic() - 1.0
    )

    receipt.validate_for(inventory)
    assert [ack.disposition for ack in receipt.prefixes] == ["durable_failure"]
    assert sink.generation_prefix_records == []
    assert state.frozen_buffer is None
    assert state.active_buffer.generated_token_ids == [11, 12]


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
                attempt_index=0,
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


def test_generation_cut_refuses_a_prefix_ending_inside_a_character():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    worker._generation_prefix_cuts_enabled = True
    dinosaur = list("🦖".encode())
    inventory, (split, whole) = _active_call_inventory(
        worker,
        ("split", list(b"Sure ") + dinosaur[:2]),
        ("whole", list(b"Sure ") + dinosaur),
    )

    receipt = worker._checkpoint_generation_cut(inventory)

    # vLLM would prime the resumed tail on half a character and echo the
    # prompt into the response, so that call regenerates instead.
    assert [ack.disposition for ack in receipt.prefixes] == [
        "durable_failure",
        "durable_prefix",
    ]
    assert len(sink.generation_prefix_records) == 1
    assert split.frozen_buffer is None
    assert split.active_buffer.generated_token_ids == list(b"Sure ") + dinosaur[:2]
    assert whole.generation_cut_staging_keys == sink.generation_prefix_keys


def test_dropped_output_delta_keeps_its_call_uncuttable():
    sink = _MemorySink()
    worker = _worker_with_capture(sink)
    worker._generation_prefix_cuts_enabled = True
    inventory, (poisoned, healthy) = _active_call_inventory(
        worker, ("poisoned", [11, 12]), ("healthy", [13, 14])
    )
    with poisoned.lock:
        poisoned.observe([15], [])
        # A later valid delta must not hide the tokens the bad one dropped.
        poisoned.observe([16], [-0.1])
        assert poisoned.observation_error is not None

    receipt = worker._checkpoint_generation_cut(inventory)

    assert [ack.disposition for ack in receipt.prefixes] == [
        "durable_failure",
        "durable_prefix",
    ]
    assert poisoned.frozen_buffer is None
    assert healthy.generation_cut_staging_keys == sink.generation_prefix_keys


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
    request = _FakeRequest(
        ng_capture={
            "rollout_id": "r0",
            "model_call_id": "c1",
            "mode": "text",
        },
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10])
    state = worker._capture_calls[id(request)]
    with state.lock:
        state.effective_output_limit = 8
        state.observe([11, 12], [-0.1, -0.2])
    inventory = GenerationCutInventory.build(
        checkpoint_id="checkpoint-1",
        server_name="policy_model",
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

    with pytest.raises(RuntimeError, match="injected generation-prefix"):
        worker._checkpoint_generation_cut(inventory)
    assert state.frozen_buffer is None
    assert state.active_buffer.generated_token_ids == [11, 12]
    assert state.sealed_generated_token_ids == []

    receipt = worker._checkpoint_generation_cut(inventory)
    assert receipt.prefixes[0].prefix_token_count == 2
    assert sink.generation_prefix_records[-1][1].token_ids_delta == [10, 11, 12]


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
    original_worker._capture_calls[id(original_request)].effective_output_limit = 128
    VllmAsyncGenerationWorkerImpl._observe_request_capture(
        original_worker,
        original_request,
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
    inventory = GenerationCutInventory.build(
        checkpoint_id="checkpoint-1",
        server_name="policy_model",
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
    receipt = original_worker._checkpoint_generation_cut(inventory)
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
                "effective_output_limit": receipt.prefixes[0].effective_output_limit,
            },
        },
        stream=False,
    )
    admission = resumed_worker._capture_admission(request)
    bad_count_admission = admission.model_copy(
        update={
            "generation_cut": admission.generation_cut.model_copy(
                update={"generation_token_count": 3}
            )
        }
    )
    with pytest.raises(
        RuntimeError,
        match="generation-cut token count mismatch: expected=3 actual=2",
    ):
        resumed_worker._resolve_generation_cut(bad_count_admission, [])
    bad_digest_admission = admission.model_copy(
        update={
            "generation_cut": admission.generation_cut.model_copy(
                update={"digest": "0" * 64}
            )
        }
    )
    with pytest.raises(RuntimeError, match="generation-cut digest mismatch"):
        resumed_worker._resolve_generation_cut(bad_digest_admission, [])

    with caplog.at_level(logging.INFO):
        cut = resumed_worker._resolve_generation_cut(admission, [])
    expected_sha = hashlib.sha256(b"12,13").hexdigest()
    assert f"prefix_ids_sha256={expected_sha}" in caplog.text
    resumed_worker._staging_source.calls.clear()
    prefetched_cut = resumed_worker._resolve_generation_cut(admission, [], [cut_record])
    assert prefetched_cut == cut
    assert resumed_worker._staging_source.calls == []
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
    tokenizer.decode.return_value = "partial"
    # vLLM's processed tail: EOS and any matched stop string already stripped.
    output = SimpleNamespace(token_ids=[14, 151645], text=" tail")
    resumed_worker._restore_response_prefix(
        request,
        SimpleNamespace(outputs=[output]),
        tokenizer=tokenizer,
    )
    tokenizer.decode.assert_called_once_with(
        [12, 13], skip_special_tokens=True, spaces_between_special_tokens=True
    )
    assert output.text == "partial tail"

    # A second checkpoint taken before the resumed call finishes must retain
    # the original cut key and append the newly generated tail. Gym v2 stores
    # this complete key list in the replacement attempt's model record.
    VllmAsyncGenerationWorkerImpl._observe_request_capture(
        resumed_worker,
        request,
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    token_ids=[14],
                    logprobs=[{14: SimpleNamespace(logprob=-0.3)}],
                )
            ]
        ),
    )
    second_receipt = resumed_worker._checkpoint_generation_cut(
        GenerationCutInventory.build(
            checkpoint_id="checkpoint-2",
            server_name="policy_model",
            active_prefixes=[
                GenerationCutPrefix(
                    ticket_id="ticket-2",
                    rollout_id="r0",
                    attempt_index=1,
                    model_call_id="c2",
                    admitted_at=2.0,
                )
            ],
        )
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
    assert content["ng_commit_coords"]["rollout_id"] == "r0-a1"
    assert final_record.token_ids_delta == [10, 11, 12, 13, 14]
    assert final_record.token_mask_delta == [0.0, 0.0, 1.0, 1.0, 1.0]
    assert final_record.generation_log_probs_delta == [
        0.0,
        0.0,
        -0.1,
        -0.2,
        -0.3,
    ]
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
    assert source.calls == []
    text_root = worker._capture_admission(
        _FakeRequest(
            ng_capture={"rollout_id": "r0", "model_call_id": "c1", "mode": "text"}
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
        plain.ng_capture = {"rollout_id": "r0", "model_call_id": "c1", "mode": "text"}
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
        lambda dp_cfg, bootstrap: MagicMock(),
    )
    if pruning_rate:
        with pytest.raises(ValueError, match="video token pruning"):
            asyncio.run(
                VllmAsyncGenerationWorkerImpl.setup_token_capture(
                    worker, {}, staging_partition="staging", capture_media=True
                )
            )
    else:
        assert asyncio.run(
            VllmAsyncGenerationWorkerImpl.setup_token_capture(
                worker, {}, staging_partition="staging", capture_media=True
            )
        )
        assert worker._capture_image_token_id == 18
