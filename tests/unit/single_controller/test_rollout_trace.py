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
from typing import Any

import pytest

from nemo_rl.algorithms.metric_utils import SetupTimingMetrics
from nemo_rl.algorithms.rollout_checkpointing.manifest import (
    bootstrap_compatibility_identity,
)
from nemo_rl.algorithms.rollout_checkpointing.trace import (
    ROLLOUT_TRACE_FILENAME,
    RolloutTrace,
)
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.experience.rollout_recovery import (
    RecoveryTargetLevel,
    RolloutRecoveryLedger,
)
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
