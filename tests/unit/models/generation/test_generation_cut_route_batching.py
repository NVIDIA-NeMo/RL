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

"""CPU regressions for frozen cut groups and real cross-thread owner handoffs."""

from __future__ import annotations

import asyncio
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("nemo_gym.token_id_capture.staging.capture")

from nemo_gym._checkpoint.generation_cut import (  # noqa: E402
    GenerationCutInventory,
    GenerationCutPrefix,
)
from nemo_gym.token_id_capture.staging.capture import RolloutTokenCapture  # noqa: E402
from nemo_gym.token_id_capture.staging.records import CaptureAdmission  # noqa: E402

from nemo_rl.models.generation import generation_cut_capture  # noqa: E402
from nemo_rl.models.generation.generation_cut_capture import (  # noqa: E402
    GenerationCutCaptureMixin,
    _RequestCaptureState,
)
from nemo_rl.models.generation.vllm.vllm_worker_async import (  # noqa: E402
    VllmAsyncGenerationWorkerImpl,
    _AsyncLLMHTTPClient,
)
from nemo_rl.utils.routed_experts_codec import decode_routed_experts  # noqa: E402

pytestmark = pytest.mark.nemo_gym


class _Sink:
    def __init__(self):
        self.records = {}
        self.batches = []
        self.fail_batch = False

    def stage_generation_prefix_batch(self, records, *, checkpoint_id, chunk_sequences):
        self.batches.append(len(records))
        if self.fail_batch and len(records) > 1:
            raise RuntimeError("injected batch failure")
        keys = []
        for record, sequence in zip(records, chunk_sequences, strict=True):
            key = f"{checkpoint_id}/{record.model_call_id}/{sequence}"
            self.records[key] = record
            keys.append(key)
        return keys

    def clear(self, keys):
        for key in keys:
            self.records.pop(key, None)


class _Worker(GenerationCutCaptureMixin):
    # Exercise the production vLLM bridge without constructing a GPU engine.
    _snapshot_engine_routed_experts_many = (
        VllmAsyncGenerationWorkerImpl._snapshot_engine_routed_experts_many
    )
    _snapshot_engine_routed_experts = (
        VllmAsyncGenerationWorkerImpl._snapshot_engine_routed_experts
    )
    _materialize_engine_routed_experts = (
        VllmAsyncGenerationWorkerImpl._materialize_engine_routed_experts
    )


def _routes(*values):
    return torch.tensor(values, dtype=torch.int16).reshape(-1, 1, 1)


@pytest.fixture
def owner_loop():
    loop = asyncio.new_event_loop()
    started = threading.Event()
    loop.call_soon(started.set)
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    assert started.wait(5)
    try:
        yield loop
    finally:
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(timeout=5)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        assert not thread.is_alive()
        loop.close()


def _worker(loop, *, count=3, max_rows=256, max_tokens=4_194_304):
    worker = _Worker()
    worker._capture_sink = _Sink()
    worker.token_capture = RolloutTokenCapture(
        sink=worker._capture_sink, weight_version_fn=lambda: 7
    )
    worker._capture_registry_lock = threading.Lock()
    worker._capture_calls_by_model_call_id = {}
    worker._generation_prefix_batch_max_rows = max_rows
    worker._generation_prefix_batch_max_tokens = max_tokens
    worker._generation_cut_tokenizer = SimpleNamespace(
        decode=lambda ids, **kw: bytes(ids).decode()
    )
    external, requests, states, prefixes = {}, {}, [], []
    for i in range(count):
        call = worker.token_capture.begin_call(
            CaptureAdmission(rollout_id=f"r{i}", model_call_id=f"c{i}", mode="text")
        )
        state = _RequestCaptureState(
            call=call,
            prompt_token_ids=[65, 66],
            capture_routed_experts=True,
            engine_request_id=f"engine-{i}",
            effective_output_limit=128,
        )
        state.observe([67, 68], [-0.1, -0.2])
        states.append(state)
        worker._capture_calls_by_model_call_id[call.model_call_id] = state
        external[f"engine-{i}"] = {f"internal-{i}"}
        requests[f"internal-{i}"] = SimpleNamespace(
            routed_experts_chunks=[_routes(1, 2), _routes(3)]
        )
        prefixes.append(
            GenerationCutPrefix(
                ticket_id=f"t{i}",
                rollout_id=f"r{i}",
                attempt=0,
                model_call_id=f"c{i}",
                admitted_at=1.0,
            )
        )
    processor = SimpleNamespace(external_req_ids=external, request_states=requests)
    # These are the only engine members needed by the route-reading seam.
    client = object.__new__(_AsyncLLMHTTPClient)
    client._engine_client = SimpleNamespace(output_processor=processor)
    client._engine_loop = loop
    worker._http_engine_client = client
    inventory = GenerationCutInventory.build(
        checkpoint_id="group-cut", server_name="policy", active_prefixes=prefixes
    )
    return worker, states, processor, inventory


def _assert_released(states):
    for state in states:
        assert state.frozen_cut is None
        assert state.lifecycle_lock.acquire(blocking=False)
        state.lifecycle_lock.release()


def _count_handoffs(monkeypatch, counts):
    submit = asyncio.run_coroutine_threadsafe

    def record(coroutine, loop):
        async def run():
            result = await coroutine
            counts.append(len(result))
            return result

        return submit(run(), loop)

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", record)


def test_freeze_group_before_one_owner_handoff(owner_loop, monkeypatch):
    worker, states, processor, inventory = _worker(owner_loop)
    reads = []
    client = worker._http_engine_client
    original = client._routed_expert_chunks_on_owner

    def read(request_id):
        assert asyncio.get_running_loop() is owner_loop
        assert all(state.frozen_cut == ("group-cut", 2) for state in states)
        assert all(not state.lock.locked() for state in states)
        assert all(state.lifecycle_lock.locked() for state in states)
        reads.append(request_id)
        return original(request_id)

    client._routed_expert_chunks_on_owner = read
    handoffs = []
    with monkeypatch.context() as patch:
        _count_handoffs(patch, handoffs)
        acks = worker._checkpoint_generation_cut_batched(inventory)
    assert handoffs == [3]
    assert reads == [f"engine-{i}" for i in range(3)]
    assert [ack.ticket_id for ack in acks] == ["t0", "t1", "t2"]
    assert all(ack.disposition == "durable_prefix" for ack in acks)
    for ack in acks:
        record = worker._capture_sink.records[ack.staging_keys[0]]
        assert decode_routed_experts(
            record.extras["routed_experts"]
        ).flatten().tolist() == [1, 2, 3, -1]
    _assert_released(states)


@pytest.mark.parametrize("bound", ["calls", "rows", "tokens", "oversized"])
def test_route_groups_are_bounded_independently_of_tq_puts(
    owner_loop, monkeypatch, bound
):
    count = 65 if bound == "calls" else 5
    worker, states, _, inventory = _worker(
        owner_loop,
        count=count,
        max_rows=2 if bound == "rows" else 256,
        max_tokens={"tokens": 8, "oversized": 3}.get(bound, 4_194_304),
    )
    handoffs = []
    with monkeypatch.context() as patch:
        _count_handoffs(patch, handoffs)
        acks = worker._checkpoint_generation_cut_batched(inventory)
    expected = (
        [64, 1]
        if bound == "calls"
        else ([1] * 5 if bound == "oversized" else [2, 2, 1])
    )
    assert handoffs == expected
    assert worker._capture_sink.batches == ([65] if bound == "calls" else expected)
    assert all(ack.disposition == "durable_prefix" for ack in acks)
    _assert_released(states)


@pytest.mark.parametrize(
    "failure", ["missing", "multiple_ids", "missing_chunks", "shape", "short"]
)
def test_one_bad_route_lookup_fails_only_its_call(owner_loop, failure):
    worker, states, processor, inventory = _worker(owner_loop)
    if failure == "missing":
        processor.external_req_ids.pop("engine-1")
    elif failure == "multiple_ids":
        processor.external_req_ids["engine-1"].add("other")
    else:
        processor.request_states["internal-1"].routed_experts_chunks = {
            "missing_chunks": None,
            "shape": [torch.zeros((2, 1), dtype=torch.int16)],
            "short": [_routes(1)],
        }[failure]
    acks = worker._checkpoint_generation_cut_batched(inventory)
    assert [ack.disposition for ack in acks] == [
        "durable_prefix",
        "durable_failure",
        "durable_prefix",
    ]
    assert states[1].staged_token_count == 0
    _assert_released(states)


def test_terminal_routes_and_replay_off_skip_engine_reads(owner_loop, monkeypatch):
    worker, states, _, inventory = _worker(owner_loop)
    states[0].terminal_routed_experts = _routes(1, 2, 3)
    for state in states[1:]:
        state.capture_routed_experts = False

    def unexpected(*args, **kwargs):
        pytest.fail("no call needs live engine routes")

    monkeypatch.setattr(
        worker._http_engine_client, "snapshot_routed_experts_many", unexpected
    )
    acks = worker._checkpoint_generation_cut_batched(inventory)
    assert all(ack.disposition == "durable_prefix" for ack in acks)
    _assert_released(states)


def test_grouped_and_single_route_reads_preserve_records_and_digests(owner_loop):
    grouped, states, _, inventory = _worker(owner_loop)
    single, single_states, _, _ = _worker(owner_loop)
    grouped_acks = grouped._checkpoint_generation_cut_batched(inventory)
    for state, prefix, grouped_ack in zip(
        single_states, inventory.active_prefixes, grouped_acks, strict=True
    ):
        transaction = single._generation_cut_transaction(
            prefix, state, inventory.checkpoint_id
        )
        record, sequence = next(transaction)
        [key] = single._capture_sink.stage_generation_prefix_batch(
            [record], checkpoint_id=inventory.checkpoint_id, chunk_sequences=[sequence]
        )
        single_ack = transaction.send(key)
        with pytest.raises(StopIteration):
            transaction.send(None)
        assert single_ack == grouped_ack
        assert record == grouped._capture_sink.records[grouped_ack.staging_keys[0]]
    _assert_released(states + single_states)


def test_replay_off_retains_existing_1000_call_tq_batches(owner_loop, monkeypatch):
    worker, states, _, inventory = _worker(owner_loop, count=1000)
    for state in states:
        state.capture_routed_experts = False

    def unexpected(*args, **kwargs):
        pytest.fail("replay-off calls must not fetch routes")

    monkeypatch.setattr(worker, "_snapshot_engine_routed_experts_many", unexpected)
    acks = worker._checkpoint_generation_cut_batched(inventory)
    assert worker._capture_sink.batches == [256, 256, 256, 232]
    assert all(ack.disposition == "durable_prefix" for ack in acks)
    _assert_released(states)


@pytest.mark.parametrize("when", ["before_lookup", "after_lookup", "during_put"])
def test_shared_deadline_releases_route_and_transport_batches(
    owner_loop, monkeypatch, when
):
    worker, states, _, inventory = _worker(owner_loop, count=5)
    clock = {"now": time.monotonic()}
    deadline = clock["now"] + 5
    monkeypatch.setattr(
        generation_cut_capture, "time", SimpleNamespace(monotonic=lambda: clock["now"])
    )
    if when == "before_lookup":
        original = states[0].freeze_for_checkpoint

        def freeze(checkpoint_id):
            result = original(checkpoint_id)
            clock["now"] = deadline
            return result

        states[0].freeze_for_checkpoint = freeze
    elif when == "after_lookup":
        original = worker._snapshot_engine_routed_experts_many

        def snapshot(*args, **kwargs):
            result = original(*args, **kwargs)
            clock["now"] = deadline
            return result

        worker._snapshot_engine_routed_experts_many = snapshot
    else:
        worker._generation_prefix_batch_max_rows = 2
        original = worker._capture_sink.stage_generation_prefix_batch

        def put(*args, **kwargs):
            result = original(*args, **kwargs)
            clock["now"] = deadline
            return result

        worker._capture_sink.stage_generation_prefix_batch = put
    acks = worker._checkpoint_generation_cut_batched(inventory, deadline=deadline)
    expected = (
        ["durable_prefix"] * 2 + ["durable_failure"] * 3
        if when == "during_put"
        else ["durable_failure"] * 5
    )
    assert [ack.disposition for ack in acks] == expected
    assert len(worker._capture_sink.records) == (2 if when == "during_put" else 0)
    _assert_released(states)


def test_observation_and_terminal_publication_continue_during_lookup(owner_loop):
    worker, states, _, inventory = _worker(owner_loop)
    client = worker._http_engine_client
    original = client._routed_expert_chunks_on_owner

    def read(request_id):
        state = states[int(request_id.rsplit("-", 1)[1])]
        with state.lock:
            state.observe([69], [-0.3])
            state.terminal_routed_experts = _routes(1, 2, 3, 4)
        return None if request_id == "engine-1" else original(request_id)

    client._routed_expert_chunks_on_owner = read
    acks = worker._checkpoint_generation_cut_batched(inventory)
    assert all(
        ack.disposition == "durable_prefix" and ack.prefix_token_count == 2
        for ack in acks
    )
    assert all(
        state.staged_token_count == 2 and state.generated_token_ids == [67, 68, 69]
        for state in states
    )
    _assert_released(states)


def test_owner_timeout_rolls_back_whole_group_and_allows_retry(owner_loop):
    worker, states, _, inventory = _worker(owner_loop)
    entered, release = threading.Event(), threading.Event()

    def block():
        entered.set()
        assert release.wait(5)

    owner_loop.call_soon_threadsafe(block)
    assert entered.wait(5)
    try:
        acks = worker._checkpoint_generation_cut_batched(
            inventory, deadline=time.monotonic() + 0.05
        )
        assert all(ack.disposition == "durable_failure" for ack in acks)
        assert not worker._capture_sink.records
        assert all(state.staged_token_count == 0 for state in states)
        _assert_released(states)
    finally:
        release.set()
    asyncio.run_coroutine_threadsafe(asyncio.sleep(0), owner_loop).result(timeout=5)
    acks = worker._checkpoint_generation_cut_batched(inventory)
    assert all(ack.disposition == "durable_prefix" for ack in acks)
    _assert_released(states)


def test_terminal_waits_for_lifecycle_but_observation_does_not(owner_loop, monkeypatch):
    worker, states, _, inventory = _worker(owner_loop)
    entered, release = threading.Event(), threading.Event()
    submitted, terminal_done = threading.Event(), threading.Event()
    original = worker._snapshot_engine_routed_experts_many

    def snapshot(*args, **kwargs):
        submitted.set()
        return original(*args, **kwargs)

    def block():
        entered.set()
        assert release.wait(5)

    def terminal():
        with states[0].lifecycle_lock:
            terminal_done.set()

    monkeypatch.setattr(worker, "_snapshot_engine_routed_experts_many", snapshot)
    owner_loop.call_soon_threadsafe(block)
    assert entered.wait(5)
    with ThreadPoolExecutor(max_workers=2) as executor:
        cut_future = executor.submit(
            worker._checkpoint_generation_cut_batched, inventory
        )
        try:
            assert submitted.wait(5)
            terminal_future = executor.submit(terminal)
            assert not terminal_done.wait(0.05)
            for state in states:
                with state.lock:
                    state.observe([69], [-0.3])
        finally:
            release.set()
        acks = cut_future.result(timeout=5)
        terminal_future.result(timeout=5)
    assert all(ack.prefix_token_count == 2 for ack in acks)
    assert all(
        state.generated_token_ids[state.staged_token_count :] == [69]
        for state in states
    )
    _assert_released(states)


@pytest.mark.parametrize("when", ["lookup", "record", "token_bound"])
def test_interruption_releases_every_frozen_transaction(owner_loop, monkeypatch, when):
    worker, states, _, inventory = _worker(owner_loop)
    if when == "record":
        original = worker.token_capture.build_prefix_record

        def interrupt(call, **kwargs):
            if call.model_call_id == "c1":
                raise KeyboardInterrupt("injected record interruption")
            return original(call, **kwargs)

        monkeypatch.setattr(worker.token_capture, "build_prefix_record", interrupt)
    else:
        if when == "token_bound":
            worker._generation_prefix_batch_max_tokens = 5

        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt("injected lookup interruption")

        monkeypatch.setattr(worker, "_snapshot_engine_routed_experts_many", interrupt)
    with pytest.raises(KeyboardInterrupt, match="injected"):
        worker._checkpoint_generation_cut_batched(inventory)
    assert not worker._capture_sink.records
    assert all(state.staged_token_count == 0 for state in states)
    _assert_released(states)


def test_failed_tq_batch_still_retries_rows_individually(owner_loop):
    worker, states, _, inventory = _worker(owner_loop)
    worker._capture_sink.fail_batch = True
    acks = worker._checkpoint_generation_cut_batched(inventory)
    assert worker._capture_sink.batches == [3, 1, 1, 1]
    assert all(ack.disposition == "durable_prefix" for ack in acks)
    _assert_released(states)


def test_group_failure_logs_one_traceback_and_keeps_later_groups(
    owner_loop, monkeypatch, caplog
):
    worker, states, _, inventory = _worker(owner_loop, count=65)
    original = worker._snapshot_engine_routed_experts_many
    error = RuntimeError("injected owner handoff failure")

    def snapshot(request_ids, *, deadline):
        if len(request_ids) > 1:
            raise error
        return original(request_ids, deadline=deadline)

    monkeypatch.setattr(worker, "_snapshot_engine_routed_experts_many", snapshot)
    acks = worker._checkpoint_generation_cut_batched(inventory)
    assert [ack.disposition for ack in acks] == ["durable_failure"] * 64 + [
        "durable_prefix"
    ]
    assert len([record for record in caplog.records if record.exc_info]) == 1
    assert len(traceback.extract_tb(error.__traceback__)) == 2
    assert (
        len(
            [
                record
                for record in caplog.records
                if "group route snapshot failed" in record.message
            ]
        )
        == 64
    )
    assert all(
        state.staged_token_count
        == (2 if state.call.model_call_id == acks[-1].model_call_id else 0)
        for state in states
    )
    assert worker._capture_sink.batches == [1]
    _assert_released(states)


def test_owner_snapshot_is_shallow_and_empty_expired_reads_are_not_submitted(
    owner_loop, monkeypatch
):
    worker, _, processor, _ = _worker(owner_loop)
    client = worker._http_engine_client
    result = client.snapshot_routed_experts_many(["engine-0"], timeout_s=5)
    original = processor.request_states["internal-0"].routed_experts_chunks
    assert result["engine-0"].routes is not original
    assert result["engine-0"].routes[0] is original[0]

    def unexpected(*args, **kwargs):
        pytest.fail("empty/expired read must not submit a callback")

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", unexpected)
    assert client.snapshot_routed_experts_many([], timeout_s=5) == {}
    with pytest.raises(TimeoutError):
        client.snapshot_routed_experts_many(["engine-0"], timeout_s=0)
