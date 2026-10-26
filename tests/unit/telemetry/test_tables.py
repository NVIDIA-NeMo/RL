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

import pytest
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace import TracerProvider

import nemo_rl.telemetry.tables as tables_mod
from nemo_rl.telemetry.tables import emit_table, stop_table_emitter

_COLUMNS = ["sample_id", "rollout_id", "rewards"]
_ROWS = [["g_g0", "r0", 1.0], ["g_g1", None, 0.0]]


@pytest.fixture
def exported(monkeypatch):
    """Lens exporting logs: an SDK LoggerProvider writing to memory."""
    exporter = InMemoryLogRecordExporter()
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    monkeypatch.setattr(tables_mod, "get_logger", provider.get_logger)
    yield exporter
    stop_table_emitter(5.0)


def test_each_row_is_one_log_record_with_non_null_cells(exported) -> None:
    emit_table(_COLUMNS, _ROWS, step=3, name="train/samples")
    stop_table_emitter(5.0)  # emission runs on the emitter thread

    records = [log.log_record for log in exported.get_finished_logs()]
    assert [r.event_name for r in records] == ["train/samples"] * 2
    assert dict(records[0].attributes) == {
        "rl.table": "train/samples",
        "rl.step": 3,
        "rl.row.sample_id": "g_g0",
        "rl.row.rollout_id": "r0",
        "rl.row.rewards": 1.0,
    }
    # None cells are left out rather than exported as a null attribute.
    assert "rl.row.rollout_id" not in dict(records[1].attributes)


def test_records_keep_the_callers_span(exported) -> None:
    tracer = TracerProvider().get_tracer("test")
    with tracer.start_as_current_span("rl.sc.step") as span:
        emit_table(_COLUMNS, _ROWS[:1], step=1, name="t")
    stop_table_emitter(5.0)
    (log,) = exported.get_finished_logs()
    assert log.log_record.span_id == span.get_span_context().span_id


def test_a_failing_emit_never_raises_and_the_emitter_survives(exported, monkeypatch):
    calls = []

    def boom(*args):
        calls.append(args)
        raise RuntimeError("collector down")

    monkeypatch.setattr(tables_mod, "_emit", boom)
    emit_table(_COLUMNS, _ROWS, step=1, name="t")
    emit_table(_COLUMNS, _ROWS, step=2, name="t")
    stop_table_emitter(5.0)
    assert len(calls) == 2


def test_stop_without_a_table_is_a_no_op() -> None:
    stop_table_emitter(0.1)
