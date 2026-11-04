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
"""The NeMo Gym model-call hook: engine milestones to segments, with cross-host clock mapping."""

import json
import subprocess
import sys

import pytest

from nemo_rl.telemetry import llm_span_breakdown_hook
from nemo_rl.telemetry.llm_span_breakdown_hook import (
    CALIBRATION_PATH_ENV,
    llm_span_breakdown,
)

#: Epoch microseconds; the values only need to be ordered.
T0 = 1_700_000_000_000_000
GYM_OFFSET_US = 200
ENGINE_OFFSET_US = 1_000


@pytest.fixture(autouse=True)
def _fresh_calibration_cache(monkeypatch):
    monkeypatch.delenv(CALIBRATION_PATH_ENV, raising=False)
    monkeypatch.setattr(llm_span_breakdown_hook, "_calibration_cache", {})


def _response(**overrides):
    response = {
        "nemo_llm_arrival_ts_us": T0 + 10,
        "nemo_llm_queued_ts_us": T0 + 12,
        "nemo_llm_first_scheduled_ts_us": T0 + 20,
        "nemo_llm_first_token_ts_us": T0 + 50,
        "nemo_llm_last_token_ts_us": T0 + 150,
        "nemo_llm_response_ready_ts_us": T0 + 155,
    }
    response.update(overrides)
    return response


class _Span:
    def __init__(self):
        self.attributes = {}

    def set_attributes(self, attributes):
        self.attributes.update(attributes)


def _call(response, **kwargs):
    """Run the hook against a fake span; returns the attributes it set (prefix removed) and the spans it drew."""
    kwargs.setdefault("sent_us", T0)
    kwargs.setdefault("received_us", T0 + 160)
    span = _Span()
    drawn = []
    llm_span_breakdown(
        response,
        span=span,
        emit_span=lambda name, start, end, attributes=None: drawn.append(
            (name, start, end, attributes)
        ),
        **kwargs,
    )
    prefix = llm_span_breakdown_hook.ATTRIBUTE_PREFIX
    return {
        "attributes": {
            key.removeprefix(prefix): value for key, value in span.attributes.items()
        },
        "segments": [
            (
                name,
                start,
                end,
                attributes[llm_span_breakdown_hook.SEGMENT_CLASS_ATTRIBUTE],
            )
            for name, start, end, attributes in drawn
        ],
    }


def _segments(result):
    return {name: (start, end, cls) for name, start, end, cls in result["segments"]}


def _write_calibration(path, offsets, reference="driver"):
    with open(path, "w") as handle:
        for host, offset in offsets.items():
            handle.write(
                json.dumps(
                    {
                        "host_clock_id": host,
                        "reference_clock_id": reference,
                        "offset_us": offset,
                    }
                )
                + "\n"
            )


def test_attributes_carry_tokens_hit_rate_replica():
    result = _call(
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 21,
                "prompt_tokens_details": {"cached_tokens": 40},
            }
        },
        replica_idx=3,
    )

    assert result["attributes"] == {
        "prompt_tokens": 100,
        "decode_tokens": 21,
        "cached_prompt_tokens": 40,
        "kv_cache_hit_rate": 0.4,
        "generation_replica_idx": 3,
        "timeline_skew_segments": 0,
    }


def test_a_zero_hit_rate_is_kept_and_an_unreported_one_is_left_out():
    miss = _call(
        {
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 0},
            }
        }
    )
    unreported = _call({"usage": {"prompt_tokens": 50, "completion_tokens": 1}})

    assert miss["attributes"]["kv_cache_hit_rate"] == 0.0
    assert "kv_cache_hit_rate" not in unreported["attributes"]
    assert "generation_replica_idx" not in unreported["attributes"]


def test_aggregated_segments_are_cut_at_the_milestones_and_classed():
    segments = _segments(_call(_response()))

    assert segments == {
        "proxy_to_engine": (T0, T0 + 10, "plumbing"),
        "gen_submit": (T0 + 10, T0 + 12, "plumbing"),
        "gen_sched_queue": (T0 + 12, T0 + 20, "queue"),
        "gen_first_token": (T0 + 20, T0 + 50, "compute"),
        "decode": (T0 + 50, T0 + 150, "compute"),
        "response_build": (T0 + 150, T0 + 155, "plumbing"),
        "llm_to_proxy": (T0 + 155, T0 + 160, "plumbing"),
    }


def test_disaggregation_adds_the_frontend_and_context_legs_and_drops_proxy_to_engine():
    response = _response(
        nemo_fe_recv_ts_us=T0 + 2,
        nemo_ctx_arrival_ts_us=T0 + 3,
        nemo_ctx_queued_ts_us=T0 + 4,
        nemo_ctx_first_scheduled_ts_us=T0 + 6,
        nemo_ctx_done_ts_us=T0 + 9,
        nemo_ctx_cached_tokens=30,
        nemo_ctx_computed_tokens=70,
        nemo_ctx_num_chunks=2,
    )
    result = _call(response)
    segments = _segments(result)

    assert "proxy_to_engine" not in segments
    assert segments["proxy_to_frontend"] == (T0, T0 + 2, "plumbing")
    assert segments["ctx_prefill"] == (T0 + 6, T0 + 9, "compute")
    assert segments["kv_handoff"] == (T0 + 9, T0 + 10, "plumbing")
    assert {k: v for k, v in result["attributes"].items() if k.startswith("ctx_")} == {
        "ctx_cached_tokens": 30,
        "ctx_computed_tokens": 70,
        "ctx_num_chunks": 2,
    }


def test_a_response_with_no_engine_fields_gives_no_engine_segments():
    result = _call({})

    assert _segments(result) == {}
    assert result["attributes"] == {"timeline_skew_segments": 0}


def test_without_a_calibration_timestamps_are_used_as_stamped():
    result = _call(_response(nemo_llm_host="engine-host"))

    assert "clock_corrected" not in result["attributes"]
    assert _segments(result)["proxy_to_engine"] == (T0, T0 + 10, "plumbing")


def test_a_calibration_maps_engine_timestamps_onto_the_model_servers_clock(
    tmp_path, monkeypatch
):
    path = tmp_path / "clock_calibration.jsonl"
    _write_calibration(
        path, {"driver": 0, "gym-host": GYM_OFFSET_US, "engine-host": ENGINE_OFFSET_US}
    )
    monkeypatch.setattr(llm_span_breakdown_hook, "_LOCAL_HOST", "gym-host")
    monkeypatch.setenv(CALIBRATION_PATH_ENV, str(path))

    # The engine's clock runs this far ahead of the model server's.
    ahead = ENGINE_OFFSET_US - GYM_OFFSET_US
    engine = {key: value + ahead for key, value in _response().items()}
    result = _call({**engine, "nemo_llm_host": "engine-host"})

    assert result["attributes"]["clock_corrected"] is True
    assert result["attributes"]["timeline_skew_segments"] == 0
    assert _segments(result) == _segments(_call(_response()))


def test_without_the_calibration_the_same_response_looks_skewed():
    ahead = ENGINE_OFFSET_US - GYM_OFFSET_US
    engine = {key: value + ahead for key, value in _response().items()}
    result = _call({**engine, "nemo_llm_host": "engine-host"})

    assert "llm_to_proxy" not in _segments(result)
    assert result["attributes"]["timeline_skew_segments"] == 1


@pytest.mark.parametrize(
    "response_host, calibrated",
    [
        ("unlisted-host", {"driver": 0, "gym-host": 200}),
        (None, {"driver": 0, "gym-host": 200}),
    ],
)
def test_an_unknown_or_missing_host_is_left_as_stamped_and_flagged(
    tmp_path, monkeypatch, response_host, calibrated
):
    path = tmp_path / "clock_calibration.jsonl"
    _write_calibration(path, calibrated)
    monkeypatch.setattr(llm_span_breakdown_hook, "_LOCAL_HOST", "gym-host")
    monkeypatch.setenv(CALIBRATION_PATH_ENV, str(path))

    result = _call(
        _response(**({"nemo_llm_host": response_host} if response_host else {}))
    )

    assert result["attributes"]["clock_corrected"] is False
    assert _segments(result)["proxy_to_engine"] == (T0, T0 + 10, "plumbing")


@pytest.mark.parametrize(
    "content", [None, "", "not json\n", '{"host_clock_id": "a", "offset_us": 1}\n']
)
def test_an_unusable_calibration_file_means_nothing_is_corrected(
    tmp_path, monkeypatch, content
):
    path = tmp_path / "clock_calibration.jsonl"
    if content is not None:
        path.write_text(content)
    monkeypatch.setenv(CALIBRATION_PATH_ENV, str(path))

    result = _call(_response(nemo_llm_host="engine-host"))

    assert result["attributes"]["clock_corrected"] is False


def test_the_calibration_is_read_once_and_an_empty_result_is_retried_later(
    tmp_path, monkeypatch
):
    path = tmp_path / "clock_calibration.jsonl"
    reads = []
    real = llm_span_breakdown_hook._read_clock_offsets
    monkeypatch.setattr(
        llm_span_breakdown_hook,
        "_read_clock_offsets",
        lambda p: reads.append(p) or real(p),
    )
    monkeypatch.setenv(CALIBRATION_PATH_ENV, str(path))
    clock = [1_000.0]
    monkeypatch.setattr(llm_span_breakdown_hook, "monotonic", lambda: clock[0])

    _call(_response())
    _call(_response())
    assert len(reads) == 1  # the file did not exist yet, and a retry is not due

    _write_calibration(path, {"driver": 0})
    clock[0] += llm_span_breakdown_hook._CALIBRATION_RETRY_S + 1
    _call(_response())
    _call(_response())
    assert len(reads) == 2  # read again once it is due, then cached


def test_the_module_imports_the_standard_library_only():
    """Gym loads this file by path in a venv that has no nemo_rl, so nothing else may be imported."""
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('hook', {llm_span_breakdown_hook.__file__!r})\n"
        "before = set(sys.modules)\n"
        "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
        "extra = {n.split('.')[0] for n in set(sys.modules) - before} - set(sys.stdlib_module_names) - {'hook'}\n"
        "assert not extra, extra\n"
        "assert 'nemo_rl' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_a_segment_that_runs_backwards_is_not_drawn_and_is_counted():
    result = _call(_response(nemo_llm_arrival_ts_us=T0 - 500))

    assert "proxy_to_engine" not in _segments(result)
    assert result["attributes"]["timeline_skew_segments"] == 1
