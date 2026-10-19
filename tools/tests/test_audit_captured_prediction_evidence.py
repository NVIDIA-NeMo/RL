# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch

from nemo_rl.utils import r3_trace
from tools.audit_captured_prediction_evidence import compare_predictions


@pytest.fixture
def evidence(monkeypatch):
    monkeypatch.setenv("NRL_R3_TRACE", "1")
    monkeypatch.setenv("NRL_R3_TRACE_STEPS", "10")
    records = []
    monkeypatch.setattr(r3_trace, "_write_record", records.append)
    raw = {
        "input_ids": torch.tensor([101, 102, 103, 201, 202]),
        "input_lengths": torch.tensor(5),
        "sequence_length": 5,
        "generation_logprobs": torch.tensor([0, 0, 0, -0.5, -1.25]),
        "routed_experts": torch.arange(20, dtype=torch.int16).reshape(5, 2, 2),
    }
    r3_trace.trace_captured_prediction_evidence(
        context=r3_trace.CapturedPredictionContext("rollout", "call", 4, 5),
        choice_index=0,
        prompt_token_ids=[101, 102, 103],
        generation_token_ids=[201, 202],
        generation_logprobs=[-0.5, -1.25],
        routed_experts=raw["routed_experts"],
    )
    calls = [
        {
            "model_call_id": "call",
            "prev_len": 0,
            "cum_len": 5,
            "token_prefix_sha_matches": True,
            "chain_hash_matches": True,
        }
    ]
    return raw, calls, records


def test_original_prediction_fingerprints_match_canonical_dump(evidence):
    raw, calls, records = evidence
    report = compare_predictions(raw, calls, records)
    assert report["complete_call_evidence"]
    assert report["all_original_token_ids_and_logprobs_match"]
    assert report["all_generation_predictor_routes_match"]
    raw["routed_experts"][-1] = 123
    assert compare_predictions(raw, calls, records)[
        "all_generation_predictor_routes_match"
    ]


def test_route_change_is_distinguished_from_logprob_change(evidence):
    raw, calls, records = evidence
    raw["routed_experts"][2, 0, 0] += 1
    report = compare_predictions(raw, calls, records)
    assert report["all_original_token_ids_and_logprobs_match"]
    assert not report["all_generation_predictor_routes_match"]
    raw["generation_logprobs"][3] -= 1
    report = compare_predictions(raw, calls, records)
    assert not report["all_original_token_ids_and_logprobs_match"]


def test_missing_or_wrong_call_evidence_never_passes(evidence):
    raw, calls, records = evidence
    for incoming in ([], [{**records[0], "model_call_id": "unrelated"}]):
        report = compare_predictions(raw, calls, incoming)
        assert not report["complete_call_evidence"]
        assert report["all_original_token_ids_and_logprobs_match"] is None
        assert report["all_generation_predictor_routes_match"] is None


def test_ambiguous_or_misaligned_trace_never_passes(evidence):
    raw, calls, records = evidence
    duplicate = copy.deepcopy(records[0])
    duplicate["generation_predictor_routes_sha256"] = "different"
    report = compare_predictions(raw, calls, [*records, duplicate])
    assert report["calls"][0]["status"] == "ambiguous_trace"
    misaligned = {**records[0], "prompt_tokens": 4}
    assert (
        compare_predictions(raw, calls, [misaligned])["calls"][0]["status"]
        == "unaligned_trace"
    )


def test_unauthenticated_lineage_is_rejected(evidence):
    raw, calls, records = evidence
    calls[0]["chain_hash_matches"] = False
    with pytest.raises(ValueError, match="not authenticated"):
        compare_predictions(raw, calls, records)
