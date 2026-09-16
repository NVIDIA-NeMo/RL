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

import json
from pathlib import Path

import pytest

from nemo_rl.utils.timer import Timer
from nemo_rl.utils.trace import Tracer, resolve_trace_config, save_trace


def test_tracer_disabled_by_default():
    tracer = Tracer("driver")

    with tracer.span("ignored"):
        pass

    assert tracer.events() == []


def test_disabled_tracer_async_spans_are_noop():
    tracer = Tracer("driver", virtual_process_name="rollouts")

    tracer.start_async_span("rollout", "sample", track_name="sample 0")
    tracer.end_async_span("sample", args={"total_reward": 1.0})
    tracer.finalize_open_spans()

    assert tracer.events() == []


def test_timer_and_async_sample_spans():
    tracer = Tracer("driver", virtual_process_name="ppo_rollouts", enabled=True)
    timer = Timer(trace=tracer)

    with timer.time("total_step_time"):
        with timer.time("generation"):
            pass
    tracer.start_async_span(
        "sample_rollout",
        "step_0_sample_0",
        track_name="step 0 / sample 0",
        args={"sample_idx": 0},
    )
    tracer.end_async_span("step_0_sample_0", args={"total_reward": 1.0})

    events = tracer.events()
    assert [event["name"] for event in events if event["ph"] == "B"] == [
        "total_step_time",
        "generation",
    ]
    complete = [event for event in events if event["ph"] == "X"]
    assert len(complete) == 1
    assert complete[0]["dur"] >= 0
    assert complete[0]["args"] == {"sample_idx": 0, "total_reward": 1.0}
    assert any(
        event.get("name") == "process_name"
        and event.get("args", {}).get("name") == "ppo_rollouts"
        for event in events
    )


def test_finalize_marks_incomplete_async_spans():
    tracer = Tracer("driver", virtual_process_name="rollouts", enabled=True)
    tracer.start_async_span("rollout", "unfinished", track_name="sample 0")

    tracer.finalize_open_spans()

    complete = [event for event in tracer.events() if event["ph"] == "X"]
    assert complete[0]["args"]["incomplete"] is True


def test_duplicate_and_missing_async_span_ids_raise():
    tracer = Tracer("driver", virtual_process_name="rollouts", enabled=True)
    tracer.start_async_span("rollout", "sample", track_name="sample 0")

    with pytest.raises(ValueError, match="already open"):
        tracer.start_async_span("rollout", "sample", track_name="sample 0")
    tracer.end_async_span("sample")
    with pytest.raises(ValueError, match="not open"):
        tracer.end_async_span("sample")


def test_save_trace_writes_perfetto_json(tmp_path: Path):
    output_path = tmp_path / "trace.json"
    tracer = Tracer("driver", enabled=True)
    tracer.instant("training_started", args={"algorithm": "ppo"})

    result = save_trace(tracer.events(), output_path=output_path)

    assert result == output_path
    events = json.loads(output_path.read_text())
    assert any(event.get("name") == "training_started" for event in events)
    assert all(isinstance(event, dict) for event in events)


def test_save_trace_preserves_equal_timestamp_order(tmp_path: Path):
    output_path = tmp_path / "trace.json"
    events = [
        {"ph": "E", "ts": 1, "pid": 1, "tid": 1},
        {"name": "next", "ph": "B", "ts": 1, "pid": 1, "tid": 1},
    ]

    save_trace(events, output_path=output_path)

    written_events = json.loads(output_path.read_text())
    assert [event["ph"] for event in written_events] == ["E", "B"]


def test_save_trace_skips_empty_event_list(tmp_path: Path):
    output_path = tmp_path / "trace.json"

    assert save_trace([], output_path=output_path) is None
    assert not output_path.exists()


def test_resolve_trace_config_uses_run_log_dir(tmp_path: Path):
    enabled, output_path = resolve_trace_config(
        {
            "log_dir": str(tmp_path / "exp_001"),
            "perfetto": {"enable": True, "name": "ppo_trace.json"},
        }
    )

    assert enabled is True
    assert output_path == tmp_path / "exp_001" / "ppo_trace.json"


def test_resolve_trace_config_preserves_absolute_name(tmp_path: Path):
    absolute_name = tmp_path / "shared" / "trace.json"
    enabled, output_path = resolve_trace_config(
        {
            "log_dir": str(tmp_path / "exp_001"),
            "perfetto": {"enable": True, "name": str(absolute_name)},
        }
    )

    assert enabled is True
    assert output_path == absolute_name


def test_resolve_trace_config_rejects_empty_name_when_enabled():
    with pytest.raises(ValueError, match="non-empty"):
        resolve_trace_config(
            {"log_dir": "logs", "perfetto": {"enable": True, "name": ""}}
        )


def test_resolve_trace_config_disabled_when_block_is_missing(tmp_path: Path):
    enabled, output_path = resolve_trace_config({"log_dir": str(tmp_path)})

    assert enabled is False
    assert output_path == tmp_path / "nemo_rl_perfetto_trace.json"
