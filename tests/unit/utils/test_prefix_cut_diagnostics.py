"""Standalone regression tests for bounded prefix-cut timing summaries."""

import ast
import asyncio
import contextvars
import hashlib
import io
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

import pytest

from nemo_rl.utils import prefix_cut_diagnostics as diagnostics


def probe(*, enabled=True):
    return diagnostics.prefix_cut_probe(
        enabled=enabled,
        checkpoint_id="checkpoint-1",
        inventory_digest="inventory-1",
        calls=3,
        deadline=time.monotonic() + 120,
    )


def summary(caplog):
    messages = [
        record.message
        for record in caplog.records
        if record.message.startswith("prefix_cut_probe ")
    ]
    assert len(messages) == 1
    return json.loads(messages[0].split(" ", 1)[1])


def test_disabled_probe_does_not_read_clocks_or_log(monkeypatch, caplog):
    manager = probe(enabled=False)

    def unexpected_clock():
        raise AssertionError("disabled instrumentation read a clock")

    monkeypatch.setattr(diagnostics.time, "perf_counter", unexpected_clock)
    monkeypatch.setattr(diagnostics.time, "thread_time", unexpected_clock)
    monkeypatch.setattr(diagnostics.time, "process_time", unexpected_clock)
    with manager, diagnostics.prefix_cut_timing("unused"):
        diagnostics.prefix_cut_count("bytes", 100)
        assert not diagnostics.prefix_cut_probe_active()
    assert not caplog.records


def test_summary_aggregates_sizes_and_inclusive_timings(caplog):
    caplog.set_level(logging.INFO)
    with probe():
        for amount in (10, 20):
            with diagnostics.prefix_cut_timing("record"):
                diagnostics.prefix_cut_count("route_bytes", amount)
                diagnostics.prefix_cut_count("max_route_bytes", amount, maximum=True)
        diagnostics.prefix_cut_wait("control_queue_wait", 0.25)
    result = summary(caplog)
    assert result["calls"] == 3
    assert result["outcome"] == "completed"
    assert result["counts"] == {"route_bytes": 30, "max_route_bytes": 20}
    assert result["phases"]["record"]["calls"] == 2
    assert result["phases"]["record"]["wall_s"] >= 0
    assert result["phases"]["control_queue_wait"]["wall_s"] == 0.25
    assert result["phases"]["control_queue_wait"]["thread_cpu_s"] == 0
    assert not diagnostics.prefix_cut_probe_active()


def test_phase_failure_is_measured_without_changing_exception(caplog):
    caplog.set_level(logging.INFO)
    error = ValueError("injected")
    with pytest.raises(ValueError) as caught:
        with probe(), diagnostics.prefix_cut_timing("route_decode"):
            raise error
    assert caught.value is error
    result = summary(caplog)
    assert result["outcome"] == "interrupted"
    assert result["phases"]["route_decode"]["errors"] == 1
    assert not diagnostics.prefix_cut_probe_active()


def test_cancelled_probe_is_labelled_interrupted(caplog):
    caplog.set_level(logging.INFO)
    with pytest.raises(asyncio.CancelledError):
        with probe(), diagnostics.prefix_cut_timing("batch"):
            raise asyncio.CancelledError
    result = summary(caplog)
    assert result["outcome"] == "interrupted"
    assert result["phases"]["batch"]["errors"] == 1


def test_context_propagates_into_existing_executor(caplog):
    caplog.set_level(logging.INFO)
    with ThreadPoolExecutor(max_workers=1) as executor, probe():
        context = contextvars.copy_context()

        def work():
            with diagnostics.prefix_cut_timing("worker"):
                diagnostics.prefix_cut_count("rows", 4)

        executor.submit(context.run, work).result(timeout=5)
    result = summary(caplog)
    assert result["counts"]["rows"] == 4
    assert result["phases"]["worker"]["calls"] == 1


def test_queued_work_is_distinct_from_execution(caplog):
    caplog.set_level(logging.INFO)
    release = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        blocker = executor.submit(release.wait, 5)
        with probe():
            submitted = time.perf_counter()
            context = contextvars.copy_context()

            def work():
                diagnostics.prefix_cut_wait("queue", time.perf_counter() - submitted)
                with diagnostics.prefix_cut_timing("execution"):
                    diagnostics.prefix_cut_count("rows", 1)

            result = executor.submit(context.run, work)
            try:
                time.sleep(0.04)
            finally:
                release.set()
            result.result(timeout=5)
        blocker.result(timeout=5)
    phases = summary(caplog)["phases"]
    assert phases["queue"]["wall_s"] >= 0.03
    assert phases["execution"]["wall_s"] < phases["queue"]["wall_s"]


def test_logging_failure_does_not_replace_capture_failure(monkeypatch):
    def broken_logger(*args, **kwargs):
        raise RuntimeError("logger unavailable")

    monkeypatch.setattr(diagnostics.logger, "info", broken_logger)
    with pytest.raises(ValueError, match="capture error"):
        with probe():
            raise ValueError("capture error")


def test_actual_cut_control_wrapper_preserves_queue_context_and_result(caplog):
    """Execute the production method without importing GPU-only worker modules."""
    source = (
        Path(__file__).resolve().parents[3]
        / "nemo_rl/models/generation/generation_cut_capture.py"
    )
    tree = ast.parse(source.read_text())
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GenerationCutCaptureMixin"
    )
    method = next(
        node
        for node in owner.body
        if getattr(node, "name", None) == "_run_generation_cut_control"
    )
    namespace = dict(
        Any=Any,
        Callable=Callable,
        cast=cast,
        asyncio=asyncio,
        contextvars=contextvars,
        time=time,
        checkpoint_phase=lambda phase: nullcontext(),
        prefix_cut_wait=diagnostics.prefix_cut_wait,
        prefix_cut_timing=diagnostics.prefix_cut_timing,
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            str(source),
            "exec",
        ),
        namespace,
    )
    run_control = namespace["_run_generation_cut_control"]
    caplog.set_level(logging.INFO)

    async def scenario():
        with ThreadPoolExecutor(max_workers=1) as executor:
            worker = SimpleNamespace(_generation_cut_control_executor=executor)
            release = threading.Event()
            blocker = executor.submit(release.wait, 5)

            def operation():
                diagnostics.prefix_cut_count("executed", 1)
                return 42

            with probe():
                task = asyncio.create_task(run_control(worker, operation))
                try:
                    await asyncio.sleep(0.04)
                finally:
                    release.set()
                assert await task == 42
            blocker.result(timeout=5)

    asyncio.run(scenario())
    result = summary(caplog)
    assert result["counts"]["executed"] == 1
    assert result["phases"]["control_queue_wait"]["wall_s"] >= 0.03
    assert result["phases"]["control_execution"]["calls"] == 1
    assert result["phases"]["control_result_delivery"]["calls"] == 1
    assert result["phases"]["control_executor_dispatch_wait"]["wall_s"] >= 0.03


def test_active_tq_phase_is_visible_cross_thread_and_clears_on_error(caplog):
    caplog.set_level(logging.INFO)
    entered, release = threading.Event(), threading.Event()
    with probe():
        observer = diagnostics.current_prefix_cut_probe()
        context = contextvars.copy_context()

        def write():
            with diagnostics.prefix_cut_timing("tq_put_ack"):
                entered.set()
                assert release.wait(5)
                raise ValueError("TQ unavailable")

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(context.run, write)
            try:
                assert entered.wait(2)
                assert observer.active_phase_age("tq_put_ack") is not None
            finally:
                release.set()
            with pytest.raises(ValueError, match="TQ unavailable"):
                future.result(timeout=5)
        assert observer.active_phase_age("tq_put_ack") is None
    timing = summary(caplog)["phases"]["tq_put_ack"]
    assert timing["errors"] == 1
    assert timing["process_cpu_s"] >= 0


def test_timing_records_process_and_thread_cpu_for_actual_cpu_work(caplog):
    caplog.set_level(logging.INFO)
    with probe(), diagnostics.prefix_cut_timing("cpu_work"):
        assert sum(i * i for i in range(100000)) > 0
    timing = summary(caplog)["phases"]["cpu_work"]
    assert timing["thread_cpu_s"] > 0
    assert timing["process_cpu_s"] >= timing["thread_cpu_s"] - 0.001


def test_lifecycle_lock_reports_terminal_owner_and_tq_phase(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(diagnostics, "_LIFECYCLE_SLOW_SECONDS", 0.005)
    monkeypatch.setattr(diagnostics, "_LIFECYCLE_LOG_INTERVAL_SECONDS", 0)
    lock = diagnostics.CaptureLifecycleLock(enabled=True)
    entered, release = threading.Event(), threading.Event()

    def terminal():
        with lock.hold("terminal", model_call_id="call"):
            with diagnostics.prefix_cut_timing("tq_kv_batch_put_api"):
                entered.set()
                assert release.wait(5)

    def cut():
        with lock.hold("cut", model_call_id="call"):
            return 42

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(terminal)
        assert entered.wait(2)
        with lock.hold("abort", model_call_id="call", blocking=False) as acquired:
            assert not acquired
        second = executor.submit(cut)
        try:
            time.sleep(0.03)
            assert not second.done()
        finally:
            release.set()
        first.result(2)
        assert second.result(2) == 42
    assert not lock.locked()
    samples = [
        json.loads(r.message.split(" ", 1)[1])
        for r in caplog.records
        if r.message.startswith("capture_lifecycle_lock ")
    ]
    wait = next(x for x in samples if x["event"] == "wait")
    assert wait["owner_at_wait_start"]["operation"] == "terminal"
    assert wait["owner_at_wait_start"]["model_call_id"] == "call"
    assert wait["owner_at_wait_start"]["slowest_phase"] == "tq_kv_batch_put_api"
    assert wait["wait_s"] >= 0.02


@pytest.mark.parametrize("enabled", [False, True])
def test_lifecycle_lock_preserves_cross_thread_release_and_errors(caplog, enabled):
    lock = diagnostics.CaptureLifecycleLock(enabled=enabled)
    assert lock.acquire(False)
    thread = threading.Thread(target=lock.release)
    thread.start()
    thread.join(2)
    assert not lock.locked()
    with pytest.raises(ValueError, match="body failed"):
        with lock.hold("terminal", model_call_id="call"):
            raise ValueError("body failed")
    assert not lock.locked()
    assert diagnostics._LIFECYCLE_OWNER.get() is None
    assert not caplog.records


def test_cut_lock_context_does_not_leak_across_suspended_generators():
    locks = [diagnostics.CaptureLifecycleLock(enabled=True) for _ in range(2)]

    def transaction(lock):
        with lock.hold("cut", model_call_id="call"):
            yield

    first, second = [transaction(lock) for lock in locks]
    next(first)
    next(second)
    assert diagnostics._LIFECYCLE_OWNER.get() is None
    first.close()
    second.close()
    assert not any(lock.locked() for lock in locks)


@pytest.mark.parametrize("slow_phase", ["preprocess", "api", "postprocess"])
def test_actual_put_adapter_separates_preparation_api_and_return(caplog, slow_phase):
    caplog.set_level(logging.INFO)
    path = (
        Path(__file__).resolve().parents[3]
        / "nemo_rl/data_plane/adapters/transfer_queue.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "TQDataPlaneClient"
    )
    method = next(n for n in cls.body if getattr(n, "name", None) == "put_samples")
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    events = []

    class Fields:
        def detach(self):
            events.append("detach")
            if slow_phase == "preprocess":
                time.sleep(0.03)
            return self

        def keys(self):
            return ["tokens"]

    fields = Fields()

    def put(**kwargs):
        events.append("put")
        assert kwargs["keys"] == ["key"] and kwargs["fields"] is fields
        if slow_phase == "api":
            time.sleep(0.03)

    def meta(**kwargs):
        events.append("return")
        if slow_phase == "postprocess":
            time.sleep(0.03)
        return SimpleNamespace(**kwargs)

    namespace = dict(
        cast=cast,
        TensorDict=Fields,
        KVBatchMeta=meta,
        tq=SimpleNamespace(kv_batch_put=put),
        prefix_cut_timing=diagnostics.prefix_cut_timing,
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, method], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    client = SimpleNamespace(
        _gdr_requested=False,
        _gdr_put_confirmed=False,
        _mark_data_operation_started=lambda: None,
    )
    tags = [{"task": "test"}]
    with probe():
        result = namespace["put_samples"](client, ["key"], "staging", fields, tags)
    assert result.sample_ids == ["key"] and result.fields == ["tokens"]
    assert result.tags == tags and result.tags[0] is not tags[0]
    assert events == ["detach", "put", "return"]
    phase = {
        "preprocess": "tq_put_preprocess",
        "api": "tq_kv_batch_put_api",
        "postprocess": "tq_put_postprocess",
    }[slow_phase]
    times = summary(caplog)["phases"]
    assert times[phase]["wall_s"] >= 0.02
    for name in ["tq_put_preprocess", "tq_kv_batch_put_api", "tq_put_postprocess"]:
        assert times[name]["calls"] == 1 and times[name]["errors"] == 0
        if name != phase:
            assert times[name]["wall_s"] < times[phase]["wall_s"]


def test_capture_phase_split_preserves_serialized_recovery_and_save_order():
    torch = pytest.importorskip("torch")
    path = (
        Path(__file__).resolve().parents[3] / "nemo_rl/algorithms/single_controller.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SingleControllerActor"
    )
    method = next(
        n
        for n in cls.body
        if getattr(n, "name", None) == "_capture_rollout_checkpoint_cut"
    )
    method.decorator_list = []
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    phases, calls = [], []

    class Phase:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            phases.append(self.name)

        def __exit__(self, *args):
            pass

    saved = {}

    async def validate_replay(metadata):
        calls.append("validate_replay")

    async def validate_staging(*args, **kwargs):
        calls.append("validate_staging")
        return 7

    async def save(*args, **kwargs):
        calls.append("save")
        saved.update(kwargs)

    ledger_state = {"groups": [{"id": "group", "tokens": [1, 2, 3]}]}
    namespace = dict(
        asyncio=asyncio,
        io=io,
        hashlib=hashlib,
        torch=torch,
        time=time,
        checkpoint_phase=Phase,
        build_rollout_recovery_state=lambda *a, **kw: ledger_state,
        _RolloutCheckpointCut=lambda **kw: SimpleNamespace(**kw),
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, method], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    meta = {
        "groups": [
            {"group_id": "group", "meta": SimpleNamespace(sample_ids=["a", "b"])}
        ]
    }
    controller = SimpleNamespace(
        _dataloader=SimpleNamespace(state_dict=lambda: {"cursor": 1}),
        _replacement_reserve=[],
        _buffer=SimpleNamespace(
            training_owned_replay_groups=lambda: [],
            metadata_state_dict=lambda **kw: meta,
        ),
        _async_cfg=SimpleNamespace(max_buffered_rollouts=10),
        _validate_replay_inventory=validate_replay,
        _rollout_manager=SimpleNamespace(recovery_ledger=object()),
        _batch_shortfall=0,
        _sampler_stamps_target_steps={},
        _finalizer_metrics_by_group={},
        _finalizer_rewards_by_group={},
        _capture_reward_settings=lambda: None,
        _master_config=SimpleNamespace(token_capture=SimpleNamespace(enabled=True)),
        _validate_rollout_recovery_inventory=validate_staging,
        _save_data_plane_checkpoint=save,
        _train_steps=4,
        _trainer_version=4,
        _current_epoch=0,
        _sampler=SimpleNamespace(dispatch_index=20),
        _data_plane_checkpoint_barrier=SimpleNamespace(mutation_version=10),
    )
    result = asyncio.run(
        namespace["_capture_rollout_checkpoint_cut"](
            controller, SimpleNamespace(require_live=lambda: None), "/snapshot"
        )
    )
    assert calls == ["validate_replay", "validate_staging", "save"]
    assert (
        torch.load(io.BytesIO(result.rollout_recovery_payload), weights_only=True)
        == ledger_state
    )
    assert (
        saved["rollout_recovery_payload_sha256"]
        == hashlib.sha256(result.rollout_recovery_payload).hexdigest()
    )
    assert result.replay_row_count == 2 and result.staging_row_count == 7
    assert phases == [
        "capture_replay_metadata",
        "capture_validate_replay_inventory",
        "capture_build_recovery_state",
        "capture_serialize_recovery_state",
        "capture_hash_recovery_payload",
    ]


@pytest.fixture
def owner_loop():
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=run)
    thread.start()
    assert ready.wait(2)
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        assert not thread.is_alive()
        loop.close()


def test_snapshot_splits_owner_dispatch_callback_and_return(owner_loop, caplog):
    caplog.set_level(logging.INFO)
    blocked = threading.Event()
    release = threading.Event()

    def block_owner():
        blocked.set()
        release.wait(5)

    owner_loop.call_soon_threadsafe(block_owner)
    assert blocked.wait(2)
    value = object()

    def callback():
        assert asyncio.get_running_loop() is owner_loop
        time.sleep(0.01)
        return value

    def unblock():
        time.sleep(0.04)
        release.set()

    releaser = threading.Thread(target=unblock)
    releaser.start()
    try:
        with probe():
            assert (
                diagnostics.run_engine_snapshot(callback, loop=owner_loop, timeout_s=2)
                is value
            )
    finally:
        release.set()
        releaser.join(2)
    phases = summary(caplog)["phases"]
    assert phases["engine_snapshot_dispatch_wait"]["wall_s"] >= 0.02
    assert phases["engine_snapshot_callback"]["wall_s"] >= 0.008
    assert phases["engine_snapshot_callback"]["errors"] == 0
    assert phases["engine_snapshot_return_wakeup"]["wall_s"] >= 0


def test_snapshot_preserves_callback_exception(owner_loop, caplog):
    caplog.set_level(logging.INFO)
    error = ValueError("lookup failed")

    def callback():
        raise error

    with pytest.raises(ValueError) as caught:
        with probe():
            diagnostics.run_engine_snapshot(callback, loop=owner_loop, timeout_s=2)
    assert caught.value is error
    assert summary(caplog)["phases"]["engine_snapshot_callback"]["errors"] == 1


def test_snapshot_timeout_records_an_unresolved_round_trip(owner_loop, caplog):
    caplog.set_level(logging.INFO)
    blocked = threading.Event()
    release = threading.Event()

    def block_owner():
        blocked.set()
        release.wait(5)

    owner_loop.call_soon_threadsafe(block_owner)
    assert blocked.wait(2)
    try:
        with pytest.raises(TimeoutError):
            with probe():
                diagnostics.run_engine_snapshot(
                    lambda: None, loop=owner_loop, timeout_s=0.02
                )
    finally:
        release.set()
    result = summary(caplog)
    assert result["counts"]["engine_snapshot_timeouts"] == 1
    assert result["phases"]["engine_snapshot_unresolved_wait"]["errors"] == 1
    # Let the scheduled cancellation settle before stopping the fixture's loop.
    asyncio.run_coroutine_threadsafe(asyncio.sleep(0.01), owner_loop).result(2)


def test_unmeasured_snapshot_keeps_the_same_result(owner_loop, caplog):
    value = object()
    assert (
        diagnostics.run_engine_snapshot(lambda: value, loop=owner_loop, timeout_s=2)
        is value
    )
    assert not caplog.records


def test_snapshot_distinguishes_cut_thread_wakeup_from_callback(
    owner_loop, caplog, monkeypatch
):
    caplog.set_level(logging.INFO)
    submit = asyncio.run_coroutine_threadsafe

    def delayed_result(coroutine, loop):
        future = submit(coroutine, loop)
        result = future.result

        def wait(timeout=None):
            value = result(timeout=timeout)
            time.sleep(0.03)
            return value

        monkeypatch.setattr(future, "result", wait)
        return future

    monkeypatch.setattr(diagnostics.asyncio, "run_coroutine_threadsafe", delayed_result)
    with probe():
        assert (
            diagnostics.run_engine_snapshot(lambda: 42, loop=owner_loop, timeout_s=2)
            == 42
        )
    phases = summary(caplog)["phases"]
    assert phases["engine_snapshot_return_wakeup"]["wall_s"] >= 0.02
    assert phases["engine_snapshot_callback"]["calls"] == 1


def test_actual_vllm_adapter_uses_the_split_snapshot_lookup(owner_loop, caplog):
    caplog.set_level(logging.INFO)
    source = (
        Path(__file__).resolve().parents[3]
        / "nemo_rl/models/generation/vllm/vllm_worker_async.py"
    )
    owner = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "_AsyncLLMHTTPClient"
    )
    method = next(
        node
        for node in owner.body
        if getattr(node, "name", None) == "snapshot_routed_experts"
    )
    namespace = dict(Any=Any, run_engine_snapshot=diagnostics.run_engine_snapshot)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            str(source),
            "exec",
        ),
        namespace,
    )
    chunks = [object(), object()]
    client = SimpleNamespace(
        _engine_loop=owner_loop,
        _direct_route_snapshot_enabled=lambda: False,
        _routed_expert_chunks=lambda request_id: chunks,
    )
    with probe():
        assert (
            namespace["snapshot_routed_experts"](client, "request", timeout_s=2)
            is chunks
        )
    assert summary(caplog)["phases"]["engine_snapshot_callback"]["calls"] == 1


def test_generator_completion_is_not_an_acknowledgement_error(caplog):
    caplog.set_level(logging.INFO)
    with probe():
        with pytest.raises(StopIteration):
            with diagnostics.prefix_cut_timing("seal", expected_stop_iteration=True):
                raise StopIteration
    assert summary(caplog)["phases"]["seal"]["errors"] == 0
