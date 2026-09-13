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
import threading
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

nemo_gym = pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_gym.token_id_capture.staging.capture import (  # noqa: E402
    CaptureError,
    RolloutTokenCapture,
)
from nemo_gym.token_id_capture.staging.records import (  # noqa: E402
    CaptureAdmission,
    StagedCallRecord,
    StageResult,
)
from nemo_gym._checkpoint.model_control_contracts import (  # noqa: E402
    GenerationCutInventory,
    GenerationCutPrefix,
)

from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration  # noqa: E402
from nemo_rl.models.generation.vllm.vllm_worker_async import (  # noqa: E402
    VllmAsyncGenerationWorkerImpl,
    _CheckpointCaptureGate,
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
        "_enter_request_prefix",
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
    def __init__(self, deltas: dict[str, list[int]]) -> None:
        self.deltas = deltas
        self.calls: list[list[str]] = []

    def fetch_prefix_token_ids(self, staging_keys: list[str]) -> list[int]:
        self.calls.append(list(staging_keys))
        return [token for key in staging_keys for token in self.deltas[key]]


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
