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
import asyncio
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.utils import r3_trace


@pytest.fixture
def refit_records(monkeypatch):
    monkeypatch.setenv("NRL_R3_TRACE", "1")
    monkeypatch.setenv("NRL_R3_TRACE_STEPS", "10")
    monkeypatch.setenv("NRL_R3_TRACE_SAMPLES", "1")
    monkeypatch.setattr(r3_trace, "_refit_boundary_count", 0)
    monkeypatch.setattr(r3_trace, "_active_refit_rpcs", 0)
    monkeypatch.setattr(r3_trace, "_event_counts", defaultdict(int))
    records = []
    monkeypatch.setattr(r3_trace, "_write_record", records.append)
    return records


def test_same_version_refit_spans_survive_tensor_sample_budget(refit_records):
    before = r3_trace.capture_refit_snapshot()

    def completed(call_id, admission, response):
        r3_trace.trace_capture_weight_span(
            rollout_id="r",
            model_call_id=call_id,
            admission_weight_version=7,
            response_weight_version=7,
            admission_refit=admission,
            response_refit=response,
        )

    completed("baseline", before, before)
    with r3_trace.trace_weight_refit_rpc(transport="collective"):
        during = r3_trace.capture_refit_snapshot()
        completed("inside", during, during)
    after = r3_trace.capture_refit_snapshot()
    completed("spanning", before, after)
    completed("after", after, after)

    spans = [r for r in refit_records if r["event"] == "captured_call_weight_span"]
    assert [r["model_call_id"] for r in spans] == [
        "baseline",
        "inside",
        "spanning",
        "after",
    ]
    assert [r["observed_refit_rpc_overlap"] for r in spans] == [
        False,
        True,
        True,
        False,
    ]
    assert all(r["observed_version_change"] is False for r in spans)
    assert all(r["refit_free_generation_verified"] is False for r in spans)
    assert spans[-1]["refit_rpc_overlap_calls_observed"] == 2
    assert before == r3_trace.RefitTraceSnapshot(0, 0)
    assert during == r3_trace.RefitTraceSnapshot(1, 1)
    assert after == r3_trace.RefitTraceSnapshot(2, 0)


def test_refit_trace_preserves_exception_and_closes_interval(refit_records):
    failure = RuntimeError("real RPC failure")
    with pytest.raises(RuntimeError) as caught:
        with r3_trace.trace_weight_refit_rpc(transport="collective"):
            raise failure

    assert caught.value is failure
    assert r3_trace.capture_refit_snapshot() == r3_trace.RefitTraceSnapshot(2, 0)
    assert [r["phase"] for r in refit_records] == ["begin", "end"]
    assert refit_records[-1]["rpc_returned_without_exception"] is False


def test_refit_trace_preserves_async_cancellation(refit_records):
    async def run():
        entered = asyncio.Event()

        async def waiting_rpc():
            with r3_trace.trace_weight_refit_rpc(transport="nccl_reshard"):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(waiting_rpc())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert r3_trace.capture_refit_snapshot() == r3_trace.RefitTraceSnapshot(2, 0)
    assert refit_records[-1]["rpc_returned_without_exception"] is False


def test_nested_refit_rpc_intervals_keep_active_count(refit_records):
    with r3_trace.trace_weight_refit_rpc(transport="collective"):
        with r3_trace.trace_weight_refit_rpc(transport="nccl_reshard"):
            assert r3_trace.capture_refit_snapshot() == r3_trace.RefitTraceSnapshot(
                2, 2
            )
        assert r3_trace.capture_refit_snapshot() == r3_trace.RefitTraceSnapshot(3, 1)
    assert r3_trace.capture_refit_snapshot() == r3_trace.RefitTraceSnapshot(4, 0)
    assert [r["rpc_id"] for r in refit_records] == [1, 2, 2, 1]


@pytest.mark.parametrize("enabled,steps", [("0", "10"), ("1", "0")])
def test_capture_version_trace_disabled_means_no_records_or_counters(
    monkeypatch, enabled, steps
):
    monkeypatch.setenv("NRL_R3_TRACE", enabled)
    monkeypatch.setenv("NRL_R3_TRACE_STEPS", steps)
    monkeypatch.setattr(r3_trace, "_event_counts", defaultdict(int))
    records = []
    monkeypatch.setattr(r3_trace, "_write_record", records.append)

    r3_trace.trace_capture_weight_span(
        rollout_id="r",
        model_call_id="c",
        admission_weight_version=2,
        response_weight_version=3,
    )
    r3_trace.trace_serving_weight_version(previous_version=2, version=3)
    with r3_trace.trace_weight_refit_rpc(transport="collective"):
        assert r3_trace.capture_refit_snapshot() is None

    assert records == []
    assert not r3_trace._event_counts


def test_capture_version_trace_keeps_normal_calls_and_crossings(monkeypatch):
    monkeypatch.setenv("NRL_R3_TRACE", "1")
    monkeypatch.setenv("NRL_R3_TRACE_STEPS", "10")
    monkeypatch.setenv("NRL_R3_TRACE_SAMPLES", "1")
    monkeypatch.setattr(r3_trace, "_event_counts", defaultdict(int))
    records = []
    monkeypatch.setattr(r3_trace, "_write_record", records.append)

    for call_id, begin, end in (("a", 7, 7), ("b", 7, 7), ("c", 7, 8), ("d", 8, 8)):
        r3_trace.trace_capture_weight_span(
            rollout_id="r",
            model_call_id=call_id,
            admission_weight_version=begin,
            response_weight_version=end,
        )
    r3_trace.trace_serving_weight_version(previous_version=7, version=8)

    assert [record.get("model_call_id") for record in records] == [
        "a",
        "b",
        "c",
        "d",
        None,
    ]
    assert records[0]["observed_version_change"] is False
    assert records[1]["observed_version_change"] is False
    assert records[2]["observed_version_change"] is True
    assert all(
        record["refit_free_generation_verified"] is False for record in records[:-1]
    )
    assert records[-1]["completed_captured_calls_observed"] == 4
    assert records[-1]["cross_version_calls_observed"] == 1


@pytest.mark.parametrize("tensor_samples", ["0", "1", "2"])
def test_late_normal_call_persists_exact_rpc_snapshot(
    tmp_path, monkeypatch, tensor_samples
):
    monkeypatch.setenv("NRL_R3_TRACE", "1")
    monkeypatch.setenv("NRL_R3_TRACE_STEPS", "10")
    monkeypatch.setenv("NRL_R3_TRACE_SAMPLES", tensor_samples)
    monkeypatch.setenv("NRL_R3_TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(r3_trace, "_trace_write_failed", False)
    monkeypatch.setattr(r3_trace, "_event_counts", defaultdict(int))
    snapshot = r3_trace.RefitTraceSnapshot(4, 0)
    for index in range(4):
        r3_trace.trace_capture_weight_span(
            rollout_id="owner-child",
            model_call_id=f"call-{index}",
            admission_weight_version=7,
            response_weight_version=7,
            admission_refit=snapshot,
            response_refit=snapshot,
        )
    records = [
        json.loads(line)
        for path in tmp_path.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert [record["model_call_id"] for record in records] == [
        "call-0",
        "call-1",
        "call-2",
        "call-3",
    ]
    late = records[-1]
    assert (
        late["admission_refit_rpc"]
        == late["response_refit_rpc"]
        == {"boundary": 4, "active_rpcs": 0}
    )
    assert late["observed_refit_rpc_overlap"] is False
    assert late["refit_free_generation_verified"] is False


def test_capture_version_trace_respects_diagnostic_horizon(monkeypatch):
    monkeypatch.setenv("NRL_R3_TRACE", "1")
    monkeypatch.setenv("NRL_R3_TRACE_STEPS", "2")
    monkeypatch.setattr(r3_trace, "_event_counts", defaultdict(int))
    records = []
    monkeypatch.setattr(r3_trace, "_write_record", records.append)

    r3_trace.trace_capture_weight_span(
        rollout_id="r",
        model_call_id="c",
        admission_weight_version=2,
        response_weight_version=3,
    )
    r3_trace.trace_serving_weight_version(previous_version=2, version=3)

    assert records == []


@pytest.mark.parametrize("groups,minimum,expected", [(8, 1, 16), (4, 4, 2), (7, 2, 8)])
def test_streaming_trace_budget_counts_chunks(
    monkeypatch: pytest.MonkeyPatch, groups: int, minimum: int, expected: int
) -> None:
    monkeypatch.setenv("NRL_R3_TRACE", "1")
    monkeypatch.setenv("NRL_R3_TRACE_STEPS", "2")
    monkeypatch.delenv("NRL_R3_TRACE_CONSUMER_CALLS", raising=False)

    r3_trace.configure_r3_trace_for_streaming(
        num_prompt_groups=groups, min_groups_per_chunk=minimum
    )

    assert os.environ["NRL_R3_TRACE_CONSUMER_CALLS"] == str(expected)
    assert os.environ["NRL_R3_TRACE_STEPS"] == "2"


def test_streaming_trace_budget_respects_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NRL_R3_TRACE", "1")
    monkeypatch.setenv("NRL_R3_TRACE_CONSUMER_CALLS", "3")
    r3_trace.configure_r3_trace_for_streaming(
        num_prompt_groups=8, min_groups_per_chunk=1
    )
    assert os.environ["NRL_R3_TRACE_CONSUMER_CALLS"] == "3"


def test_disabled_trace_does_not_configure_consumer_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NRL_R3_TRACE", "0")
    monkeypatch.delenv("NRL_R3_TRACE_CONSUMER_CALLS", raising=False)
    r3_trace.configure_r3_trace_for_streaming(
        num_prompt_groups=8, min_groups_per_chunk=1
    )
    assert "NRL_R3_TRACE_CONSUMER_CALLS" not in os.environ


@pytest.mark.parametrize("groups,minimum", [(0, 1), (8, 0)])
def test_streaming_trace_requires_positive_group_sizes(
    monkeypatch: pytest.MonkeyPatch, groups: int, minimum: int
) -> None:
    monkeypatch.setenv("NRL_R3_TRACE", "1")
    with pytest.raises(ValueError, match="must be positive"):
        r3_trace.configure_r3_trace_for_streaming(
            num_prompt_groups=groups, min_groups_per_chunk=minimum
        )


def test_streaming_consumer_trace_covers_chunks_without_expanding_producers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NRL_R3_TRACE", "1")
    monkeypatch.setenv("NRL_R3_TRACE_STEPS", "2")
    monkeypatch.setenv("NRL_R3_TRACE_DIR", str(tmp_path))
    monkeypatch.delenv("NRL_R3_TRACE_CONSUMER_CALLS", raising=False)
    monkeypatch.setattr(r3_trace, "_event_counts", defaultdict(int))
    r3_trace.configure_r3_trace_for_streaming(
        num_prompt_groups=3, min_groups_per_chunk=1
    )
    data = {
        "input_ids": torch.tensor([[11, 22]]),
        "input_lengths": torch.tensor([2]),
        "routed_experts": torch.tensor([[[[1, 2]], [[2, 3]]]], dtype=torch.int16),
    }

    for index in range(7):
        keys = [f"group-{index}"]
        r3_trace.trace_rollout_payload(keys=keys, data=data)
        for stage in ("prev_lp", "train"):
            r3_trace.trace_tq_fetch_payload(stage=stage, keys=keys, data=data)
        for stage in ("prev-logprob", "train"):
            with r3_trace.r3_trace_stage(stage):
                assert (r3_trace._current_context() is not None) == (index < 6)

    records = [
        json.loads(line)
        for path in tmp_path.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    counts = Counter((r["event"], r.get("stage")) for r in records)
    assert counts == {
        ("rollout_payload_sample", None): 2,
        ("tq_fetch_sample", "prev_lp"): 6,
        ("tq_fetch_sample", "train"): 6,
    }
    for producer in (r for r in records if r["event"] == "rollout_payload_sample"):
        for stage in ("prev_lp", "train"):
            fetched = next(
                r
                for r in records
                if r["event"] == "tq_fetch_sample"
                and r["stage"] == stage
                and r["key"] == producer["key"]
            )
            assert producer["routed_experts"] == fetched["routed_experts"]
            assert producer["input_ids"] == fetched["input_ids"]


def test_disk_error_cannot_abort_forward_or_disable_strict_route_verification(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("NRL_R3_TRACE", "1")
    monkeypatch.setenv("NRL_R3_TRACE_VERIFY_FORWARD", "1")
    monkeypatch.setattr(r3_trace, "_trace_write_failed", False)
    calls = []

    def directory_instead_of_file():
        calls.append(1)
        return tmp_path

    monkeypatch.setattr(r3_trace, "_trace_path", directory_instead_of_file)
    with caplog.at_level(logging.WARNING):
        r3_trace._write_record({"event": "first"})
        r3_trace._write_record({"event": "second"})
    assert len(calls) == 1
    assert r3_trace._trace_write_failed
    assert r3_trace.r3_trace_verify_forward_enabled()
    assert len(caplog.records) == 1
    token = r3_trace._context.set(
        {"active": True, "stage": "prev-logprob", "trace_step": 1}
    )
    try:
        with pytest.raises(RuntimeError, match="do not match"):
            r3_trace._trace_router_replay_topk_use(
                replay_instance=SimpleNamespace(_nrl_layer_number=7),
                action="forward",
                scores=torch.zeros(1, 4),
                topk=2,
                expected=torch.tensor([[1, 2]]),
                actual=torch.tensor([[2, 3]]),
                backward_list_len_before=None,
                backward_list_len_after=None,
            )
    finally:
        r3_trace._context.reset(token)
    assert len(calls) == 1
