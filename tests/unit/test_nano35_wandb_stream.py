# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve real TensorBoard values and axes across incremental imports."""

import json
from pathlib import Path

import pytest
from nano35.stream_wandb_results import load_state, scalar_rows, tensorboard_dir
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def test_incremental_events_preserve_repeats_late_tags_and_separate_axes(
    tmp_path: Path,
) -> None:
    writer = EventFileWriter(str(tmp_path))
    accumulator = EventAccumulator(str(tmp_path), size_guidance={"scalars": 0})

    def write(tag: str, step: int, value: float, wall_time: float) -> None:
        writer.add_event(
            Event(
                wall_time=wall_time,
                step=step,
                summary=Summary(value=[Summary.Value(tag=tag, simple_value=value)]),
            )
        )

    try:
        write("train/reward", 10, 0.25, 100)
        write("train/reward", 11, 0.25, 101)
        write("train/reward", 11, 0.5, 102)
        write("ray/gpu_utilization", 10, 90, 103)
        writer.flush()
        rows, cursors = scalar_rows(accumulator.Reload(), {})
        assert [
            (r["global_step"], r["train/reward"]) for r in rows if "global_step" in r
        ] == [
            (10, 0.25),
            (11, 0.25),
            (11, 0.5),
        ]
        assert rows[-1] == {
            "ray_step": 10,
            "_timestamp": 103,
            "ray/gpu_utilization": 90,
        }
        assert scalar_rows(accumulator.Reload(), cursors) == ([], cursors)

        write("train/loss", 10, 0.125, 104)
        write("train/reward", 12, 0.75, 105)
        writer.flush()
        rows, next_cursors = scalar_rows(accumulator.Reload(), cursors)
        assert rows == [
            {"global_step": 10, "_timestamp": 104, "train/loss": 0.125},
            {"global_step": 12, "_timestamp": 105, "train/reward": 0.75},
        ]
        assert sum(next_cursors.values()) == 6
    finally:
        writer.close()


def test_shortened_event_history_does_not_silently_skip_metrics(tmp_path: Path) -> None:
    accumulator = EventAccumulator(str(tmp_path), size_guidance={"scalars": 0})
    writer = EventFileWriter(str(tmp_path))
    try:
        writer.add_event(
            Event(
                wall_time=100,
                step=1,
                summary=Summary(
                    value=[Summary.Value(tag="train/reward", simple_value=0.5)]
                ),
            )
        )
        writer.flush()
        with pytest.raises(ValueError, match="history shrank"):
            scalar_rows(accumulator.Reload(), {"train/reward": 2})
    finally:
        writer.close()


def test_saved_cursors_cannot_be_reused_for_a_different_destination(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.json"
    state = load_state(path, entity="team", project="nano3.5-e2e")
    state["jobs"]["123"] = {"cursors": {"train/reward": 10}}
    path.write_text(json.dumps(state))
    assert load_state(path, entity="team", project="nano3.5-e2e") == state
    with pytest.raises(ValueError, match="destination"):
        load_state(path, entity="other-team", project="nano3.5-e2e")


def test_event_discovery_follows_driver_and_rejects_unrelated_run(
    tmp_path: Path,
) -> None:
    log = tmp_path / "123-logs/ray-driver.log"
    log.parent.mkdir()
    source = tmp_path / "nemo/experiment-123"
    log.write_text(f"Using log directory: {source}\n")
    assert tensorboard_dir(tmp_path, "123") == source / "tensorboard"
    log.write_text("Using log directory: /tmp/unrelated-experiment\n")
    with pytest.raises(ValueError, match="outside this run"):
        tensorboard_dir(tmp_path, "123")
