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

"""The rollout trace: its file, the ledger's events, and the controller's."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from nemo_rl.algorithms.metric_utils import SetupTimingMetrics
from nemo_rl.algorithms.rollout_checkpointing.manifest import (
    bootstrap_compatibility_identity,
)
from nemo_rl.algorithms.rollout_checkpointing.trace import (
    BATCH_AGE_TOKENS_METRIC,
    BATCH_AGE_VERSION_SUM_METRIC,
    ROLLOUT_TRACE_FILENAME,
    RefitCuts,
    RolloutTrace,
    batch_age_by_step,
    batch_age_group_metrics,
    pop_batch_age,
)
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils import AsyncRLConfig
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import ROLLOUT_METRICS
from nemo_rl.experience.rollout_reassembler import FinalizedGroup
from nemo_rl.experience.rollout_recovery import (
    RecoveryTargetLevel,
    RolloutRecoveryLedger,
)
from nemo_rl.utils.timer import Timer
from tests.unit.experience.test_rollout_manager import (
    _FakeCaptureBuffer,
    _make_capture_manager,
)
from tests.unit.experience.test_rollout_recovery import (
    _dispatched_turn_group,
    _mutate,
    _prompt,
    _reserve,
)
from tests.unit.single_controller.test_checkpointing import (
    _ACTOR_CLS,
    _actor_master_config,
    _make_actor_args,
    _run_train_pump,
)
from tests.unit.single_controller.test_finalizer_lifecycle import (
    _controller as _finalizer_controller,
)
from tests.unit.single_controller.test_finalizer_lifecycle import (
    _RemoteFinalize,
    _request,
)


def _events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _recording_ledger(
    from_ledger: RolloutRecoveryLedger | None = None,
) -> tuple[RolloutRecoveryLedger, list[dict[str, Any]]]:
    """A ledger recording its events, new or restored from ``from_ledger``."""
    ledger = (
        RolloutRecoveryLedger()
        if from_ledger is None
        else RolloutRecoveryLedger.from_state_dict(from_ledger.state_dict())
    )
    events: list[dict[str, Any]] = []
    ledger.set_event_sink(
        lambda event, **fields: events.append(dict(event=event, **fields))
    )
    return ledger, events


def _seal(ledger: RolloutRecoveryLedger, generation_index: int) -> None:
    gate_id = ledger.get_group("g7").gate_rollout_id(generation_index)
    _mutate(
        lambda cut: ledger.mark_sibling_sealed(
            cut,
            "g7",
            generation_index=generation_index,
            gate_rollout_id=gate_id,
            receipt=None,
            reward=1.0,
            mask_sample=False,
        )
    )


def test_each_run_appends_its_own_events(tmp_path: Path) -> None:
    path = tmp_path / "logs" / ROLLOUT_TRACE_FILENAME
    first = RolloutTrace(path)
    first.emit("step_trained", step=1, group_ids=["g1"])
    first.close()
    second = RolloutTrace(path)
    second.emit("restore_started", path="ckpt")
    # Written in the background, before close.
    second.flush()
    events = _events(path)
    second.close()

    assert [event["event"] for event in events] == [
        "trace_opened",
        "step_trained",
        "trace_opened",
        "restore_started",
    ]
    assert events[1]["step"] == 1 and events[1]["group_ids"] == ["g1"]
    assert events[0]["run"] == events[1]["run"] != events[2]["run"]
    assert all(isinstance(event["t"], float) for event in events)


def test_the_ledger_reports_a_group_from_reservation_to_release() -> None:
    ledger, events = _recording_ledger()
    _dispatched_turn_group(ledger)
    _seal(ledger, 0)
    _seal(ledger, 1)
    _mutate(lambda cut: ledger.discard_group(cut, "g7"))

    assert [event["event"] for event in events] == [
        "group_reserved",
        "attempt_dispatched",
        "attempt_dispatched",
        "attempt_sealed",
        "attempt_sealed",
        "group_sealed",
        "group_discarded",
    ]
    assert all(
        event["group_id"] == "g7" and event["prompt_id"] == "7" for event in events
    )
    assert events[0]["restore_level"] == "turn"
    assert events[0]["admission_id"] == "batch-7"
    dispatched = events[1:3]
    assert [event["generation_index"] for event in dispatched] == [0, 1]
    assert all(event["attempt"] == 0 for event in dispatched)
    assert all(
        event["environment_instance_id"] == "tools/replica-0" for event in dispatched
    )


def test_a_dropped_rollout_is_reported_and_its_retry_is_a_new_attempt() -> None:
    ledger, events = _recording_ledger()
    _dispatched_turn_group(ledger, generations=1)
    _mutate(
        lambda cut: ledger.replace_dropped_dispatch(cut, "g7", generation_indices=[0])
    )
    _mutate(
        lambda cut: ledger.mark_group_dispatched(
            cut, "g7", environment_instance_id="tools/replica-0"
        )
    )

    abandoned, redispatched = events[-2:]
    assert abandoned["event"] == "attempt_abandoned"
    assert abandoned["reason"] == "dropped"
    assert abandoned["attempt"] == 0
    assert redispatched["event"] == "attempt_dispatched"
    assert redispatched["attempt"] == 1


def test_a_restart_reports_which_interrupted_rollouts_resume() -> None:
    ledger = RolloutRecoveryLedger()
    resumed = _dispatched_turn_group(ledger).environment_episode(0)
    restored, events = _recording_ledger(from_ledger=ledger)

    _mutate(
        lambda cut: restored.prepare_for_restart(
            cut,
            continued_attempts={
                ("tools/replica-0", resumed.rollout_id, resumed.attempt)
            },
        )
    )

    assert [(event["event"], event["generation_index"]) for event in events] == [
        ("attempt_resumed", 0),
        ("attempt_abandoned", 1),
    ]
    # The resumed rollout keeps its RL attempt and moves to Gym's next one.
    assert events[0]["attempt"] == 0 and events[0]["environment_attempt"] == 1
    assert events[1]["reason"] == "restart"


def _prompt_group(ledger: RolloutRecoveryLedger) -> None:
    _reserve(
        ledger,
        group_id="g7",
        prompt_id="7",
        prompt_payload=_prompt(),
        expected_generations=2,
        target_step=7,
        start_weight_version=6,
        restore_level=RecoveryTargetLevel.PROMPT_GROUP,
    )
    _mutate(lambda cut: ledger.mark_group_dispatched(cut, "g7"))


def test_a_prompt_group_restart_reports_the_whole_group_thrown_away() -> None:
    ledger = RolloutRecoveryLedger()
    _prompt_group(ledger)
    restored, events = _recording_ledger(from_ledger=ledger)

    _mutate(lambda cut: restored.prepare_for_restart(cut))

    assert [
        (event["event"], event["generation_index"], event["reason"]) for event in events
    ] == [("attempt_abandoned", 0, "restart"), ("attempt_abandoned", 1, "restart")]
    assert all(event["whole_group"] for event in events)


@pytest.mark.parametrize("build", [_dispatched_turn_group, _prompt_group])
def test_a_restart_does_not_rereport_an_already_abandoned_attempt(build) -> None:
    ledger = RolloutRecoveryLedger()
    build(ledger)
    _mutate(lambda cut: ledger.abandon_unsealed(cut, "g7"))
    restored, events = _recording_ledger(from_ledger=ledger)

    _mutate(lambda cut: restored.prepare_for_restart(cut))

    assert [e for e in events if e["event"] == "attempt_abandoned"] == []


def test_the_controller_traces_steps_weight_syncs_and_checkpoints(
    tmp_path: Path,
) -> None:
    mc = _actor_master_config(tmp_path, max_num_steps=4, save_period=2)
    mc.rollout_checkpointing.trace = True

    _run_train_pump(mc, _make_actor_args())._rollout_trace.close()

    (path,) = (tmp_path / "logs").rglob(ROLLOUT_TRACE_FILENAME)
    events = _events(path)
    steps = [event for event in events if event["event"] == "step_trained"]
    assert [event["step"] for event in steps] == [1, 2, 3, 4]
    assert all(isinstance(event["group_ids"], list) for event in steps)
    synced = [event for event in events if event["event"] == "weights_synced"]
    assert [event["weight_version"] for event in synced] == [1, 2, 3, 4]
    assert [
        event["step"]
        for event in events
        if event["event"] == "trainer_checkpoint_published"
    ] == [2, 4]


def test_tracing_is_off_by_default(tmp_path: Path) -> None:
    mc = _actor_master_config(tmp_path, max_num_steps=2, save_period=2)

    _run_train_pump(mc, _make_actor_args())

    assert not list((tmp_path / "logs").rglob(ROLLOUT_TRACE_FILENAME))


def test_a_published_snapshot_and_a_restore_are_traced(tmp_path: Path) -> None:
    mc = _actor_master_config(
        tmp_path,
        buffer_checkpoint=True,
        rollout_checkpoint_attempt_interval_s=120.0,
        token_capture_enabled=True,
    )
    mc.rollout_checkpointing.trace = True

    async def exercise() -> None:
        actor = _ACTOR_CLS(
            mc,
            _make_actor_args(bootstrap_identity=bootstrap_compatibility_identity(mc)),
            SetupTimingMetrics(),
        )
        try:
            # A fresh run restores nothing, but still reports the restore.
            await actor._rollout_restorer.restore()
            assert (await actor._rollout_snapshots.save_snapshot(force=True)).saved
        finally:
            actor._checkpointer.shutdown()
            actor._rollout_trace.close()

    asyncio.run(exercise())

    (path,) = (tmp_path / "logs").rglob(ROLLOUT_TRACE_FILENAME)
    events = {event["event"]: event for event in _events(path)}
    assert events["restore_started"]["path"] is None
    assert events["restore_finished"]["unfinished_groups"] == []
    published = events["snapshot_published"]
    assert published["path"].endswith("snapshot_000001")
    assert published["step"] == 0
    assert published["t"] >= published["cut_time"]


def test_a_call_counts_its_tokens_under_each_version_a_refit_split_it_at() -> None:
    cuts = RefitCuts()
    # Call c1 began on v3 and had 200 tokens when the v4 refit paused it, and
    # 260 when the v5 refit did. c2 began on v4 and ran past no refit.
    cuts.record(3, 4, [["c1", 200]])
    cuts.record(4, 5, [["c1", 260]])

    assert cuts.tokens_by_version([("c1", 3, 300), ("c2", 4, 50)]) == {
        3: 200,
        4: 60 + 50,
        5: 40,
    }
    # Publishing forgets the call: counted again, it stays on its first version.
    assert cuts.tokens_by_version([("c1", 3, 300)]) == {3: 300}


def test_refit_cuts_forget_calls_too_old_to_train() -> None:
    cuts = RefitCuts()
    cuts.record(3, 4, [["old", 10]])
    cuts.record(5, 6, [["new", 10]])

    cuts.forget_before(4)

    assert cuts.tokens_by_version([("old", 3, 20), ("new", 5, 20)]) == {
        3: 20,
        5: 10,
        6: 10,
    }


def test_batch_age_weights_each_version_by_its_tokens() -> None:
    events = [
        {
            "event": "group_published",
            "group_id": "g1",
            "tokens_by_version": {"3": 1200},
        },
        {"event": "group_published", "group_id": "g2", "tokens_by_version": {"4": 150}},
        {
            "event": "step_trained",
            "step": 6,
            "trainer_version": 7,
            "trained_version": 6,
            "group_ids": ["g1", "g2"],
        },
    ]
    expected = (1200 * 3 + 150 * 2) / 1350

    assert batch_age_by_step(events) == {6: expected}
    # The step metric computes the same number from the per-group sums.
    per_group = {"reward": [1.0, 0.0]}
    for counts in ({3: 1200}, {4: 150}):
        for name, value in batch_age_group_metrics(counts).items():
            per_group.setdefault(name, []).append(value)
    assert pop_batch_age(per_group, trained_version=6) == expected
    assert per_group == {"reward": [1.0, 0.0]}
    assert pop_batch_age({}, trained_version=6) is None


def _sync_weights_controller(trace: RolloutTrace, progress: list[dict]) -> Any:
    """A bare controller that can run one _sync_weights with a pausing engine."""
    ctrl = object.__new__(SingleControllerActor.__ray_metadata__.modified_class)
    ctrl._rollout_trace = trace
    ctrl._async_cfg = AsyncRLConfig()
    ctrl._rollout_permitted = asyncio.Event()
    ctrl._gen_fleet = None
    ctrl._weight_synchronizer = SimpleNamespace(sync_weights=MagicMock())
    ctrl._trainer = SimpleNamespace(sync_params_before_refit=MagicMock())
    ctrl._timer = Timer()
    ctrl._rollout_manager = SimpleNamespace(resume_request_deadlines=MagicMock())
    ctrl._gen = SimpleNamespace(
        pause_generation_for_refit=MagicMock(return_value=True),
        captured_call_progress=MagicMock(return_value=progress),
        requires_kv_scale_sync=False,
        resume_generation_after_refit=MagicMock(return_value=True),
        set_rollout_weight_version=MagicMock(),
    )
    ctrl._rollout_recovery_enabled = False
    ctrl._master_config = SimpleNamespace(
        env={"should_use_nemo_gym": True},
        policy={
            "generation": {
                "backend": "vllm",
                "vllm_cfg": {"async_engine": True, "expose_http_server": True},
            }
        },
        token_capture=SimpleNamespace(enabled=True),
    )
    ctrl._inflight_by_group_id = {}
    ctrl._trainer_version = 4
    ctrl._refit_cuts = RefitCuts()
    ctrl._sampler = SimpleNamespace()
    return ctrl


@pytest.mark.parametrize("trace_on", [True, False])
def test_a_refit_records_where_each_running_call_switches_weights(
    tmp_path: Path, trace_on: bool
) -> None:
    """Batch age is logged without the trace, so the refit read does not need it."""
    trace = RolloutTrace(tmp_path / ROLLOUT_TRACE_FILENAME if trace_on else None)
    progress = [{"weight_version": 3, "calls": [["c1", 200]]}]
    ctrl = _sync_weights_controller(trace, progress)

    asyncio.run(ctrl._sync_weights())
    trace.close()

    # c1's first 200 tokens came from v3, the rest from v4, the synced version.
    assert ctrl._refit_cuts.tokens_by_version([("c1", 3, 250)]) == {3: 200, 4: 50}


def test_the_refit_read_runs_after_the_weights_land() -> None:
    """A call that arrives during the transfer reports 0 tokens: all count as new."""
    order: list[str] = []
    ctrl = _sync_weights_controller(RolloutTrace(None), [])
    ctrl._weight_synchronizer.sync_weights.side_effect = lambda **_: order.append(
        "transfer"
    )
    ctrl._gen.captured_call_progress.side_effect = (
        lambda **_: order.append("read") or []
    )
    ctrl._gen.resume_generation_after_refit.side_effect = lambda **_: (
        order.append("resume") or True
    )

    asyncio.run(ctrl._sync_weights())

    assert order == ["transfer", "read", "resume"]


def test_a_failed_refit_read_still_resumes_generation() -> None:
    ctrl = _sync_weights_controller(RolloutTrace(None), [])
    ctrl._gen.captured_call_progress.side_effect = AttributeError("renamed field")

    with pytest.warns(UserWarning, match="AttributeError"):
        asyncio.run(ctrl._sync_weights())

    ctrl._gen.resume_generation_after_refit.assert_called_once()
    assert ctrl._rollout_permitted.is_set()
    assert ctrl._refit_cuts.tokens_by_version([("c1", 3, 10)]) == {3: 10}


def test_without_token_capture_a_refit_asks_for_no_call_progress() -> None:
    ctrl = _sync_weights_controller(RolloutTrace(None), [])
    ctrl._master_config.token_capture.enabled = False

    asyncio.run(ctrl._sync_weights())

    ctrl._gen.captured_call_progress.assert_not_called()


def test_a_published_group_traces_each_call_and_its_version(tmp_path: Path) -> None:
    meta = KVBatchMeta(
        partition_id="canonical",
        task_name="train",
        sample_ids=["group_g0"],
        fields=["input_ids"],
        sequence_lengths=[3],
        tags=[{"weight_version": 3}],
    )
    result = FinalizedGroup(
        meta=meta,
        group_min_wv=3,
        group_max_wv=4,
        staging_keys=[],
        call_versions=[("c1", 3, 10), ("c2", 4, 5)],
    )
    actor = SimpleNamespace(finalize=_RemoteFinalize(result=result))
    ctrl = _finalizer_controller(actor)
    ctrl._rollout_trace = RolloutTrace(tmp_path / ROLLOUT_TRACE_FILENAME)
    ctrl._refit_cuts = RefitCuts()
    # The v4 refit paused c1 after 6 of its 10 tokens.
    ctrl._refit_cuts.record(3, 4, [["c1", 6]])

    assert asyncio.run(ctrl._finalize_with_actor(_request(), {})) is result
    ctrl._rollout_trace.close()

    (published,) = [
        event
        for event in _events(tmp_path / ROLLOUT_TRACE_FILENAME)
        if event["event"] == "group_published"
    ]
    assert published["group_id"] == "group"
    assert published["tokens_by_version"] == {"3": 6, "4": 4 + 5}
    # The same counts reach the step's batch age through the rollout metrics.
    committed_meta = ctrl._buffer.commit_finalized.await_args.args[2]
    (group_metrics,) = committed_meta.extra_info[ROLLOUT_METRICS]
    assert group_metrics[BATCH_AGE_TOKENS_METRIC] == 15
    assert group_metrics[BATCH_AGE_VERSION_SUM_METRIC] == 6 * 3 + 9 * 4


def test_a_rollout_cancelled_mid_run_is_traced_as_cancelled() -> None:
    """The run ending cancels rollouts still in flight; they did not fail."""

    async def exercise() -> list[dict[str, Any]]:
        started = asyncio.Event()

        async def block(_sample: object) -> None:
            started.set()
            await asyncio.Event().wait()

        manager = _make_capture_manager(_FakeCaptureBuffer(), on_run=block)
        events: list[dict[str, Any]] = []
        manager.recovery_ledger.set_event_sink(
            lambda event, **fields: events.append(dict(event=event, **fields))
        )
        task = asyncio.create_task(
            manager.generate_for_finalization({"prompt": "p", "idx": 0})
        )
        await asyncio.wait_for(started.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return events

    events = asyncio.run(exercise())

    abandoned = [event for event in events if event["event"] == "attempt_abandoned"]
    assert abandoned
    assert {event["reason"] for event in abandoned} == {"cancelled"}


def test_a_failed_write_stops_tracing_without_failing_the_caller(
    tmp_path: Path,
) -> None:
    trace = RolloutTrace(tmp_path / ROLLOUT_TRACE_FILENAME)
    trace.flush()

    def fail(_text: str) -> int:
        raise OSError("disk full")

    trace._file.write = fail  # type: ignore[method-assign,union-attr]
    with pytest.warns(UserWarning, match="rollout trace stopped writing"):
        trace.emit("step_trained", step=1)
        trace.flush()
    # Later events are dropped, and closing still succeeds.
    trace.emit("step_trained", step=2)
    trace.close()


def test_the_controller_class_with_its_disabled_trace_pickles() -> None:
    """Ray pickles the controller class, whose default trace is disabled."""
    import ray.cloudpickle

    ray.cloudpickle.dumps(RolloutTrace(None))
    ray.cloudpickle.dumps(SingleControllerActor.__ray_metadata__.modified_class)
