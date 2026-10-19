# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU prediction fingerprints retain provenance without keeping route tensors."""

import hashlib
import struct

import pytest
import torch

from nemo_rl.utils import r3_trace


@pytest.fixture
def prediction_records(monkeypatch):
    monkeypatch.setenv("NRL_R3_TRACE", "1")
    monkeypatch.setenv("NRL_R3_TRACE_STEPS", "10")
    monkeypatch.setenv("NRL_R3_TRACE_SAMPLES", "1")
    records = []
    monkeypatch.setattr(r3_trace, "_write_record", records.append)
    return records


def prediction(routes, **overrides):
    fields = {
        "context": r3_trace.CapturedPredictionContext("rollout", "call", 4, 5),
        "choice_index": 0,
        "prompt_token_ids": [101, 102, 103],
        "generation_token_ids": [201, 202],
        "generation_logprobs": [-0.5, -1.25],
        "routed_experts": routes,
    }
    fields.update(overrides)
    r3_trace.trace_captured_prediction_evidence(**fields)


def routes():
    return torch.arange(20, dtype=torch.int16).reshape(5, 2, 2)


def test_prediction_hashes_exact_original_predictors(prediction_records):
    original = routes()
    before = original.clone()
    prediction(original)
    record = prediction_records[0]
    assert record["status"] == "available"
    assert record["model_call_id"] == "call"
    assert record["admission_weight_version"] == 4
    assert record["response_weight_version"] == 5
    assert record["predictor_start"] == 2
    assert record["predictor_end"] == 4
    assert record["predictor_shape"] == [2, 2, 2]
    assert (
        record["generation_predictor_routes_sha256"]
        == hashlib.sha256(struct.pack("<8h", *range(8, 16))).hexdigest()
    )
    assert (
        record["generation_token_ids_sha256"]
        == hashlib.sha256(struct.pack("<2q", 201, 202)).hexdigest()
    )
    assert (
        record["generation_logprobs_sha256"]
        == hashlib.sha256(struct.pack("<2f", -0.5, -1.25)).hexdigest()
    )
    assert torch.equal(original, before)
    assert all(not isinstance(v, torch.Tensor) for v in record.values())


def test_predictor_digest_excludes_dummy_tail_but_detects_first_predictor_change(
    prediction_records,
):
    original = routes()
    prediction(original)
    conditioning_and_tail = original.clone()
    conditioning_and_tail[0] = 100
    conditioning_and_tail[-1] = 200
    prediction(conditioning_and_tail)
    changed = original.clone()
    changed[2, 0, 0] += 1
    prediction(changed)
    digests = [r["generation_predictor_routes_sha256"] for r in prediction_records]
    assert digests[0] == digests[1]
    assert digests[0] != digests[2]
    # Scalar provenance survives the unrelated first-tensor sampling budget.
    assert len(prediction_records) == 3


@pytest.mark.parametrize("trace,steps", [("0", "10"), ("1", "0"), ("1", "3")])
def test_prediction_trace_gate_never_reads_inputs(
    monkeypatch, prediction_records, trace, steps
):
    monkeypatch.setenv("NRL_R3_TRACE", trace)
    monkeypatch.setenv("NRL_R3_TRACE_STEPS", steps)
    prediction(object(), prompt_token_ids=None, generation_token_ids=None)
    assert prediction_records == []


def test_prediction_trace_never_copies_non_cpu_routes(prediction_records):
    prediction(torch.empty((5, 2, 2), device="meta"))
    assert prediction_records[0]["status"] == "unavailable"
    assert prediction_records[0]["reason"] == "routes_not_cpu_tensor"
    assert "generation_predictor_routes_sha256" not in prediction_records[0]


@pytest.mark.parametrize(
    "overrides",
    [
        {"routed_experts": torch.zeros(5, 2)},
        {"routed_experts": torch.zeros(4, 2, 2)},
        {"generation_logprobs": None},
        {"generation_logprobs": [-0.5]},
    ],
)
def test_prediction_trace_marks_unaligned_data_unavailable(
    prediction_records, overrides
):
    prediction(routes(), **overrides)
    assert prediction_records[0]["status"] == "unavailable"
    assert prediction_records[0]["reason"] == "unaligned_prediction_fields"


def test_empty_generation_has_empty_predictor_digest(prediction_records):
    prediction(routes()[:3], generation_token_ids=[], generation_logprobs=[])
    assert prediction_records[0]["status"] == "available"
    assert prediction_records[0]["predictor_shape"] == [0, 2, 2]
    assert (
        prediction_records[0]["generation_predictor_routes_sha256"]
        == hashlib.sha256(b"").hexdigest()
    )
