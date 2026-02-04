# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import os
import tempfile
import time

import pytest

from nemo_rl.utils.trace import new_tracer, save_trace


class TestNemoRLTracer:
    @pytest.fixture(autouse=True)
    def mock_tracing_env(self, monkeypatch):
        monkeypatch.setenv("NEMORL_TRACE_ENABLED", "1")

    def test_tracer_disabled(self, monkeypatch):
        """Test that tracer is disabled when not explicitly enabled."""
        monkeypatch.delenv("NEMORL_TRACE_ENABLED", raising=False)

        tracer = new_tracer()
        assert not tracer.enabled

        # Operations should be no-ops
        with tracer.span("test"):
            pass

        assert len(tracer.get_events()) == 0

    def test_tracer_basic(self):
        """Test that tracer captures events when enabled."""
        tracer = new_tracer()
        assert tracer.enabled

        with tracer.span("test_span"):
            pass

        events = tracer.get_events()
        assert len(events) == 2  # Begin and end events
        assert events[0]["ph"] == "B"
        assert events[1]["ph"] == "E"
        assert events[0]["name"] == "test_span"
        assert events[1]["name"] == "test_span"

    def test_nested_spans(self):
        """Test that nested spans are properly tracked."""
        tracer = new_tracer()

        with tracer.span("outer"):
            with tracer.span("inner"):
                pass

        events = tracer.get_events()
        assert len(events) == 4  # 2 begin + 2 end events
        assert events[0]["name"] == "outer"
        assert events[1]["name"] == "inner"
        assert events[2]["name"] == "inner"
        assert events[3]["name"] == "outer"

    def test_span_with_metadata(self):
        """Test that metadata is properly attached to spans."""
        tracer = new_tracer(name="foo")

        with tracer.span("test", metadata={"step": 5, "batch_size": 32}):
            pass

        events = tracer.get_events()
        begin_event = events[0]
        assert "args" in begin_event
        assert begin_event["args"]["tracer_name"] == "foo"
        assert begin_event["args"]["step"] == 5
        assert begin_event["args"]["batch_size"] == 32

    def test_explicit_start_end_span(self):
        """Test explicit start_span/end_span calls."""
        tracer = new_tracer()

        tracer.start_span("phase1", metadata={"id": 1})
        # ...
        tracer.end_span("phase1")

        events = tracer.get_events()
        assert len(events) == 2
        assert events[0]["name"] == "phase1"
        assert events[0]["args"]["id"] == 1

    def test_mismatched_end_span_raises_error(self):
        """Test that ending a span with wrong name raises error."""
        tracer = new_tracer()

        tracer.start_span("span1")
        with pytest.raises(ValueError, match="Span name mismatch"):
            tracer.end_span("span2")

    def test_end_span_without_start_raises_error(self):
        """Test that ending a non-existent span raises error."""
        tracer = new_tracer()

        with pytest.raises(ValueError, match="No active span"):
            tracer.end_span("nonexistent")

    def test_instant_event(self):
        """Test instant event creation."""
        tracer = new_tracer()

        tracer.add_instant_event("checkpoint_saved", metadata={"step": 100})

        events = tracer.get_events()
        assert len(events) == 1
        assert events[0]["ph"] == "i"
        assert events[0]["name"] == "checkpoint_saved"
        assert events[0]["args"]["step"] == 100

    def test_counter_event(self):
        """Test counter event creation."""
        tracer = new_tracer()

        tracer.add_counter("reward", 0.85, metadata={"step": 1})

        events = tracer.get_events()
        assert len(events) == 1
        assert events[0]["ph"] == "C"
        assert events[0]["name"] == "reward"
        assert events[0]["args"]["value"] == 0.85
        assert events[0]["args"]["step"] == 1

    def test_save_trace_file(self, monkeypatch):
        """Test saving trace to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_trace.json")
            monkeypatch.setenv("NEMORL_TRACE_FILE", output_path)
            tracer = new_tracer()

            with tracer.span("test"):
                pass

            save_trace(tracer.get_events(), actors=())
            assert os.path.exists(output_path)

            # Verify JSON is valid and contains events
            with open(output_path, "r") as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == 2  # Begin and end events
