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
)
from nemo_rl.models.generation.generation_cut_capture import (  # noqa: E402
    _remaining_generation_limits_after_prefix,
    _RequestCaptureState,
    _TokenCaptureSnapshotGate,
)
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration  # noqa: E402
from nemo_rl.models.generation.vllm.vllm_worker_async import (  # noqa: E402
    _GENERATION_CUT_ROUTE,
    VllmAsyncGenerationWorkerImpl,
    _classify_restored_prefix_terminal,
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
            f"__generation_cut__/{checkpoint_id}/{record.rollout_id}/"
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
        _token_capture_snapshot_gate=_TokenCaptureSnapshotGate(),
        _staging_source=None,
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
            worker, dp_cfg={"backend": "simple"}, staging_partition="rollout_staging"
        )
    )

    assert installed is None  # text-only capture pins no media column
    # The capture client owns no Mooncake memory, so no save waits on vLLM.
    assert segment_sizes == [0]
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
    )
    gen.worker_group.run_all_workers_single_data.assert_called_once_with(
        "setup_token_capture",
        dp_cfg={"backend": "simple"},
        staging_partition="rollout_staging",
        capture_media=False,
        generation_prefix_cuts_enabled=False,
        generation_cut_control_token=None,
        generation_cut_control_timeout_s=60.0,
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


def test_generation_prefix_restore_counts_sum_over_workers(monkeypatch):
    gen = _generation_with_mock_group()
    gen.worker_group.workers = [object(), object()]
    gen.worker_group.run_all_workers_single_data.return_value = [
        {"restored_calls": 2, "restored_prefix_tokens": 50},
        {"restored_calls": 1, "restored_prefix_tokens": 7},
    ]
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        lambda futures, timeout=None: futures,
    )

    assert gen.get_generation_prefix_restore_counts() == {
        "restored_calls": 3,
        "restored_prefix_tokens": 57,
    }
    gen.worker_group.run_all_workers_single_data.assert_called_once_with(
        "get_generation_prefix_restore_counts",
        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
    )


def test_prefix_capture_setup_requires_control_token_and_rejects_media(monkeypatch):
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
    ("max_tokens", "min_tokens", "prefix_token_count", "expected"),
    [
        (10, 6, 3, (7, 3)),
        (None, 6, 3, (None, 3)),
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


def test_cumulative_outputs_are_observed_once_per_token():
    """vLLM's cumulative outputs share its live lists; each token is taken once."""
    worker = _worker_with_capture(_MemorySink())
    request = _FakeRequest(
        ng_capture={"rollout_id": "r0", "model_call_id": "c1", "mode": "text"},
        stream=False,
    )
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, [10, 11])
    token_ids = [12]
    logprobs = [{12: SimpleNamespace(logprob=-0.1)}]
    first = SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=token_ids, logprobs=logprobs)]
    )

    VllmAsyncGenerationWorkerImpl._observe_request_capture(worker, request, first)
    # vLLM appends to the same lists; a late read of the old output sees them.
    token_ids += [13, 14]
    logprobs += [
        {13: SimpleNamespace(logprob=-0.2)},
        {14: SimpleNamespace(logprob=-0.3)},
    ]
    VllmAsyncGenerationWorkerImpl._observe_request_capture(worker, request, first)
    final = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                token_ids=token_ids, logprobs=logprobs, finish_reason="stop"
            )
        ]
    )
    VllmAsyncGenerationWorkerImpl._observe_request_capture(worker, request, final)

    state = worker._capture_calls[id(request)]
    assert state.observation_error is None
    assert state.generated_token_ids == [12, 13, 14]
    assert state.generated_logprobs == [-0.1, -0.2, -0.3]
    assert state.terminal_finish_reason == "stop"


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
                attempt=0,
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
            rollout_id=completed_capture_key,
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
                attempt=0,
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
    assert state.frozen_cut is None
    assert state.generated_token_ids == [11, 12]


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
    assert split.frozen_cut is None
    assert split.generated_token_ids == list(b"Sure ") + dinosaur[:2]
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
    assert poisoned.frozen_cut is None
    assert healthy.generation_cut_staging_keys == sink.generation_prefix_keys


def _sampling_params(**overrides) -> SimpleNamespace:
    defaults = dict(stop=[], bad_words=[], presence_penalty=0.0, frequency_penalty=0.0)
    return SimpleNamespace(**{**defaults, **overrides})


def _record_cut_constraints(state, sampling_params) -> None:
    VllmAsyncGenerationWorkerImpl._record_request_cut_constraints(
        SimpleNamespace(_get_request_capture=lambda request: state),
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
    worker._generation_prefix_cuts_enabled = True
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
                attempt=0,
                model_call_id="c1",
                admitted_at=1.0,
            )
        ],
    )

    with pytest.raises(RuntimeError, match="injected generation-prefix"):
        worker._checkpoint_generation_cut(inventory)
    assert state.frozen_cut is None
    assert state.generated_token_ids == [11, 12]
    assert state.staged_token_count == 0

    receipt = worker._checkpoint_generation_cut(inventory)
    assert receipt.prefixes[0].prefix_token_count == 2
    assert sink.generation_prefix_records[-1][1].token_ids_delta == [10, 11, 12]


def test_restored_generation_cut_is_extended_and_retired_on_completion(
    caplog, monkeypatch
):
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
                attempt=0,
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
                "prefix_token_count": 2,
                "prefix_digest": cut_record.digest,
                "effective_output_limit": receipt.prefixes[0].effective_output_limit,
            },
        },
        stream=False,
    )
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
            [10, 11, 12, 13],
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
        [10, 11, 12, 13],
        admission=admission,
        prefix_token_ids=[],
        generation_cut=cut,
        resumed_generation_token_ids=[12, 13],
    )
    # Only the restore Gym accepted counts.
    assert VllmAsyncGenerationWorkerImpl.get_generation_prefix_restore_counts(
        resumed_worker
    ) == {"restored_calls": 1, "restored_prefix_tokens": 2}
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
    assert replay_kwargs["original_prompt_token_ids"] == [10, 11]
    assert replay_kwargs["prefix_token_ids"] == [12, 13]
    assert replay_kwargs["terminal"] is None
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
                    attempt=1,
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
