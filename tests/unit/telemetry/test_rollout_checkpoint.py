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

"""Live checkpoint diagnostics against real Lens/OTel exporters and asyncio."""

from __future__ import annotations

import asyncio
import inspect
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

import nemo_rl.telemetry.rollout_checkpoint as checkpoint


def spans(exporter):
    from opentelemetry import trace

    trace.get_tracer_provider().force_flush()
    return {span.name: span for span in exporter.get_finished_spans()}


def points(reader):
    data = reader.get_metrics_data()
    return (
        [
            (metric.name, point)
            for resource in data.resource_metrics
            for scope in resource.scope_metrics
            for metric in scope.metrics
            for point in metric.data.data_points
        ]
        if data is not None
        else []
    )


@pytest.mark.parametrize(
    "saved,reason,outcome",
    [(True, "completed", "completed"), (False, "missing_trainer_anchor", "skipped")],
)
def test_attempt_spans_success_and_skip_and_preserves_return(
    checkpoint_exporting, saved, reason, outcome
):
    exporter, reader = checkpoint_exporting
    result = SimpleNamespace(saved=saved, reason=reason)

    @checkpoint.trace_checkpoint_save
    async def save(*, force=False):
        assert force is True
        with checkpoint.checkpoint_phase("dispatch_pause"):
            with checkpoint.checkpoint_phase("gym_prepare"):
                assert checkpoint.checkpoint_identity()
                await asyncio.sleep(0)
        return result

    assert asyncio.run(save(force=True)) is result
    emitted = spans(exporter)
    parent = emitted["rl.rollout_checkpoint.save"]
    pause = emitted["rl.rollout_checkpoint.dispatch_pause"]
    prepare = emitted["rl.rollout_checkpoint.gym_prepare"]
    assert pause.parent.span_id == parent.context.span_id
    assert prepare.parent.span_id == pause.context.span_id
    assert parent.attributes["rl.checkpoint.outcome"] == outcome
    assert parent.attributes["rl.checkpoint.reason"] == reason
    assert (
        prepare.attributes["rl.checkpoint.id"] == parent.attributes["rl.checkpoint.id"]
    )
    assert all("rl.bucket" not in span.attributes for span in emitted.values())
    assert checkpoint.checkpoint_identity() is None
    measured = points(reader)
    assert any(
        name == "rl.rollout_checkpoint.live_phase.duration" and point.count == 1
        for name, point in measured
    )
    assert all(
        set(point.attributes) <= {"rl.checkpoint.operation", "rl.checkpoint.phase"}
        for _, point in measured
    )


def test_failed_phase_identifies_pending_instances_and_preserves_exception(
    checkpoint_exporting, caplog
):
    exporter, _ = checkpoint_exporting
    failure = TimeoutError("test deadline")

    async def run():
        async with checkpoint.checkpoint_attempt("save"):
            with checkpoint.checkpoint_phase("gym_prepare") as progress:
                progress.pending(f"instance-{i}" for i in range(40))
                raise failure

    with pytest.raises(TimeoutError) as caught:
        asyncio.run(run())
    assert caught.value is failure
    emitted = spans(exporter)
    phase = emitted["rl.rollout_checkpoint.gym_prepare"]
    assert phase.attributes["rl.checkpoint.outcome"] == "failed"
    assert phase.attributes["rl.checkpoint.pending_deployments"] == 40
    assert len(phase.attributes["rl.checkpoint.pending_instances"]) == 16
    assert (
        emitted["rl.rollout_checkpoint.save"].attributes["rl.checkpoint.outcome"]
        == "failed"
    )
    line = next(
        r.message
        for r in caplog.records
        if r.message.startswith("rollout_checkpoint_phase_interrupted ")
    )
    diagnostic = json.loads(line.split(" ", 1)[1])
    assert diagnostic["checkpoint_id"] and diagnostic["phase"] == "gym_prepare"
    assert diagnostic["pending_deployments"] == 40
    assert checkpoint.checkpoint_identity() is None


def test_observer_reports_ongoing_work_then_retires_gauges(
    checkpoint_exporting, monkeypatch, caplog
):
    exporter, reader = checkpoint_exporting
    monkeypatch.setattr(checkpoint, "_PROGRESS_INTERVAL_S", 0.001)

    async def run():
        existing = asyncio.all_tasks()
        async with checkpoint.checkpoint_attempt("save"):
            with checkpoint.checkpoint_phase("gym_prepare") as progress:
                progress.pending(["slow-instance"])
                for _ in range(200):
                    if any(
                        r.message.startswith("rollout_checkpoint_progress ")
                        for r in caplog.records
                    ):
                        break
                    await asyncio.sleep(0.001)
                else:
                    pytest.fail("no progress diagnostic while phase is still pending")
                assert not exporter.get_finished_spans()
                live = points(reader)
                assert any(
                    name.endswith("pending_deployments") and point.value == 1
                    for name, point in live
                )
                assert any(
                    name.endswith("phase_elapsed") and point.value > 0
                    for name, point in live
                )
        assert asyncio.all_tasks() == existing

    asyncio.run(run())
    final = points(reader)
    assert all(
        point.value == 0
        for name, point in final
        if name.endswith(("phase_elapsed", "pending_deployments", "progress_age"))
    )


def test_concurrent_phases_keep_independent_progress(checkpoint_exporting):
    async def run():
        async with checkpoint.checkpoint_attempt("save") as attempt:
            ready = asyncio.Event()
            release = asyncio.Event()

            async def renewal():
                with checkpoint.checkpoint_phase("gym_renew") as progress:
                    progress.pending(["renewing"])
                    ready.set()
                    await release.wait()

            task = asyncio.create_task(renewal())
            with checkpoint.checkpoint_phase("tq_save"):
                await ready.wait()
                snapshot = attempt.snapshot()
                assert {phase["phase"] for phase in snapshot["phases"]} == {
                    "gym_renew",
                    "tq_save",
                }
                release.set()
                await task
                assert [phase["phase"] for phase in attempt.snapshot()["phases"]] == [
                    "tq_save"
                ]

    asyncio.run(run())


def test_cancellation_preserves_cancel_and_stops_observer(checkpoint_exporting):
    exporter, _ = checkpoint_exporting

    async def run():
        existing = asyncio.all_tasks()
        entered = asyncio.Event()

        async def blocked():
            async with checkpoint.checkpoint_attempt("save"):
                with checkpoint.checkpoint_phase("generation_fence"):
                    entered.set()
                    await asyncio.Event().wait()

        task = asyncio.create_task(blocked())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert asyncio.all_tasks() == existing

    asyncio.run(run())
    assert (
        spans(exporter)["rl.rollout_checkpoint.save"].attributes[
            "rl.checkpoint.outcome"
        ]
        == "cancelled"
    )


def test_disabled_telemetry_does_not_start_observer_or_register_metrics(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("disabled diagnostics started work")

    monkeypatch.setattr(checkpoint, "get_telemetry_handle", lambda: None)
    monkeypatch.setattr(checkpoint.asyncio, "create_task", forbidden)
    monkeypatch.setattr(checkpoint, "register_metric_group", forbidden)

    async def run():
        async with checkpoint.checkpoint_attempt("save") as attempt:
            assert attempt is None
            with checkpoint.checkpoint_phase("gym_prepare") as phase:
                assert phase is None

    asyncio.run(run())


def test_export_failure_does_not_change_save_result(checkpoint_exporting, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("export unavailable")

    monkeypatch.setattr(checkpoint, "record_metrics", fail)
    result = SimpleNamespace(saved=True, reason="completed")

    @checkpoint.trace_checkpoint_save
    async def save():
        with checkpoint.checkpoint_phase("dispatch_pause"):
            return result

    assert asyncio.run(save()) is result


def test_observer_failure_does_not_change_save_result(
    checkpoint_exporting, monkeypatch
):
    async def fail(_attempt):
        raise RuntimeError("diagnostic failed")

    monkeypatch.setattr(checkpoint, "_observe", fail)
    result = SimpleNamespace(saved=True, reason="completed")

    @checkpoint.trace_checkpoint_save
    async def save():
        await asyncio.sleep(0)
        return result

    assert asyncio.run(save()) is result


def test_metrics_still_record_when_checkpoint_spans_are_disabled(checkpoint_exporting):
    from nemo.lens.state import set_enabled_span_groups

    exporter, reader = checkpoint_exporting
    set_enabled_span_groups(frozenset({"job"}))

    async def run():
        async with checkpoint.checkpoint_attempt("save"):
            with checkpoint.checkpoint_phase("dispatch_pause"):
                pass

    asyncio.run(run())
    assert not exporter.get_finished_spans()
    assert any(name.endswith("live_phase.duration") for name, _ in points(reader))


@pytest.mark.parametrize("detailed", [False, True])
def test_individual_prefix_restore_spans_are_opt_in(checkpoint_exporting, detailed):
    from nemo.lens.state import set_enabled_span_groups

    exporter, _ = checkpoint_exporting
    set_enabled_span_groups(
        frozenset({"checkpoint", "per_prompt"} if detailed else {"checkpoint"})
    )

    @checkpoint.checkpoint_phase_sync("prefix_restore_reconstruct", detail=True)
    def reconstruct(value):
        return value

    assert reconstruct(7) == 7
    assert bool(spans(exporter)) is detailed


def test_barrier_acquisition_and_hold_are_distinct_and_forward_errors(
    checkpoint_exporting,
):
    exporter, _ = checkpoint_exporting
    failure = RuntimeError("cut rejected")
    seen = []

    class Barrier:
        @asynccontextmanager
        async def checkpoint(self):
            await asyncio.sleep(0)
            try:
                yield "cut"
            except BaseException as error:
                seen.append(error)
                raise

    async def run():
        async with checkpoint.checkpoint_attempt("save"):
            async with checkpoint.checkpoint_barrier(Barrier()) as cut:
                assert cut == "cut"
                raise failure

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(run())
    assert caught.value is failure and seen == [failure]
    emitted = spans(exporter)
    wait = emitted["rl.rollout_checkpoint.dp_barrier_wait"]
    hold = emitted["rl.rollout_checkpoint.dp_barrier_hold"]
    assert wait.end_time <= hold.start_time
    assert wait.attributes["rl.checkpoint.outcome"] == "completed"
    assert hold.attributes["rl.checkpoint.outcome"] == "failed"


def test_lock_wait_cancellation_does_not_release_another_save_lock(
    checkpoint_exporting,
):
    async def run():
        lock = asyncio.Lock()
        await lock.acquire()
        entered = asyncio.Event()

        async def save():
            async with checkpoint.checkpoint_attempt("save"):
                entered.set()
                async with checkpoint.checkpoint_lock(lock):
                    pytest.fail("acquired held lock")

        task = asyncio.create_task(save())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert lock.locked()
        lock.release()

    asyncio.run(run())


def test_phase_decorators_preserve_signature_context_and_return(checkpoint_exporting):
    @checkpoint.checkpoint_phase_async("prefix_control")
    async def run(value, *, factor=2):
        return await asyncio.to_thread(
            lambda: (checkpoint.checkpoint_identity(), value * factor)
        )

    assert str(inspect.signature(run)) == "(value, *, factor=2)"

    async def body():
        async with checkpoint.checkpoint_attempt("save") as attempt:
            assert await run(3, factor=4) == (attempt.checkpoint_id, 12)

    asyncio.run(body())


def test_cancellation_during_observer_teardown_is_not_swallowed(
    checkpoint_exporting, monkeypatch
):
    exporter, _ = checkpoint_exporting

    async def run():
        entered = asyncio.Event()
        cleanup = asyncio.Event()

        async def observer(_attempt):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup.set()
                await asyncio.Event().wait()

        monkeypatch.setattr(checkpoint, "_observe", observer)

        @checkpoint.trace_checkpoint_save
        async def save():
            await entered.wait()
            return SimpleNamespace(saved=True, reason="completed")

        task = asyncio.create_task(save())
        await cleanup.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert (
        spans(exporter)["rl.rollout_checkpoint.save"].attributes[
            "rl.checkpoint.outcome"
        ]
        == "cancelled"
    )


def test_stale_observation_cannot_revive_a_finished_phase_gauge(checkpoint_exporting):
    _, reader = checkpoint_exporting

    async def run():
        async with checkpoint.checkpoint_attempt("save"):
            with checkpoint.checkpoint_phase("gym_prepare") as phase:
                phase.pending(["waiting"])
                phase.record_live("save")
            # An observer may capture a phase just before a worker finishes it.
            # Its later sample must leave the exit zeros in place.
            phase.record_live("save")

    asyncio.run(run())
    assert all(
        point.value == 0
        for name, point in points(reader)
        if name.endswith(("phase_elapsed", "pending_deployments", "progress_age"))
    )
