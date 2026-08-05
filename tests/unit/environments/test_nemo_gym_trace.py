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

from copy import deepcopy
import hashlib
import json

import pytest

from nemo_rl.environments.nemo_gym_trace import (
    build_rollout_trace_bundle,
    validate_rollout_trace_bundle,
    validate_rollout_trace_group,
)


def _call(
    turn_id,
    prompt,
    sampled,
    *,
    media_ids=(),
    completion_id=None,
):
    return {
        "turn_id": turn_id,
        "prompt_token_ids": prompt,
        "sampled_token_ids": sampled,
        "sampled_logprobs": [-0.1] * len(sampled),
        "media_ids": media_ids,
        "completion_id": completion_id or f"completion-{turn_id}",
    }


_GENERATION_CONTRACT = {
    "generation_contract_id": "generation-contract-1",
    "training_eligible": False,
    "incomplete_reasons": ["processor-fingerprint-unavailable"],
}
_FINAL_POLICY_DECISION = {
    "policy_name": "recency",
    "policy_version": "1",
    "config_digest": "policy-config",
    "retained_part_count": 1,
    "omitted_part_count": 0,
    "lineage": {
        "transformation_id": "transform-final",
        "transformation_type": "visual_recency",
        "transformation_version": "1",
        "configuration_digest": "policy-config",
        "deterministic": True,
        "lossy": False,
        "generator_contract_id": None,
        "unit_records": [
            {
                "source_unit_id": "part-final",
                "source_digest": "digest-final",
                "disposition": "kept",
                "output_unit_ids": ["part-final"],
                "output_digests": ["digest-final"],
            }
        ],
        "validator_result": "passed",
    },
}


def _lineage_deltas(calls):
    final_record = _FINAL_POLICY_DECISION["lineage"]["unit_records"][0]
    payload = json.dumps(
        [final_record],
        sort_keys=True,
        separators=(",", ":"),
    )
    state_digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    parent = None
    deltas = []
    for call_index, call in enumerate(calls):
        transformation_id = call["policy_decision"]["transformation_id"]
        deltas.append(
            {
                "transformation_id": transformation_id,
                "parent_transformation_id": parent,
                "transformation_type": "visual_recency",
                "transformation_version": "1",
                "configuration_digest": "policy-config",
                "deterministic": True,
                "lossy": False,
                "generator_contract_id": None,
                "unit_upserts": [final_record] if call_index == 0 else [],
                "source_unit_count": 1,
                "state_digest": state_digest,
                "validator_result": "passed",
            }
        )
        parent = transformation_id
    return deltas


def _strict_call(
    turn_id,
    prompt,
    sampled,
    *,
    media_ids=(),
    eligible=True,
    segment_index=0,
    expected_append_compatible=False,
    compaction_event_id=None,
    rollout_id="rollout-strict",
    generation_contract_id="generation-contract-1",
):
    call = _call(
        turn_id,
        prompt,
        sampled,
        media_ids=media_ids,
        completion_id=f"completion-{turn_id}",
    )
    call.update(
        {
            "rollout_id": rollout_id,
            "action_id": f"action-{turn_id}",
            "prepared_request_id": f"prepared-{turn_id}",
            "request_id": f"request-{turn_id}",
            "context_epoch": segment_index,
            "segment_index": segment_index,
            "segment_id": f"segment-{segment_index}",
            "expected_append_compatible": expected_append_compatible,
            "compaction_event_id": compaction_event_id,
            "finish_reason": "stop",
            "eligible": eligible,
            "evidence_source": "generation_response",
            "policy_decision": {
                "policy_name": "recency",
                "policy_version": "1",
                "config_digest": "policy-config",
                "decision_turn": turn_id,
                "selection_digest": f"selection-{turn_id}",
                "transformation_id": f"transform-{turn_id}",
            },
            "generation_contract_id": generation_contract_id,
            "policy_output_spans": [
                {
                    "policy_output_span_id": f"span-{turn_id}",
                    "model_call_id": f"model-call-{turn_id}",
                    "action_ids": [f"action-{turn_id}"],
                    "start": 0,
                    "end": len(sampled),
                    "eligible": eligible,
                    "old_logprobs_alignment": "sampled_tokens",
                }
            ],
            "media_occurrences": [
                {
                    "media_id": media_id,
                    "occurrence_ordinal": ordinal,
                    "model_call_id": f"model-call-{turn_id}",
                    "placeholder_span_or_position": None,
                    "processed_dimensions": None,
                    "model_specific_sidecars": {},
                }
                for ordinal, media_id in enumerate(media_ids)
            ],
        }
    )
    return call


def test_no_compaction_is_one_tensor_equivalent_physical_trace():
    bundle = build_rollout_trace_bundle(
        rollout_id="rollout-no-cc",
        calls=[
            _call(1, [10, 11], [12], media_ids=["screen-a"]),
            _call(
                2,
                [10, 11, 12, 13],
                [14, 15],
                media_ids=["screen-a", "screen-b"],
            ),
        ],
    )

    assert bundle["checks"]["ok"]
    assert bundle["checks"]["physical_trace_count"] == 1
    trace = bundle["physical_traces"][0]
    assert trace["source_turn_ids"] == [1, 2]
    assert [segment["token_ids"] for segment in trace["segments"]] == [
        [10, 11],
        [12],
        [13],
        [14, 15],
    ]
    assert [segment["loss_mask"] for segment in trace["segments"]] == [
        [0, 0],
        [1],
        [0],
        [1, 1],
    ]
    assert trace["ordered_media_ids"] == ["screen-a", "screen-b"]


def test_material_rewrite_starts_new_physical_trace():
    boundary = {
        "event_id": "boundary-turn-3",
        "applies_to_step": 3,
        "reason": "history_policy_rewrite",
    }
    bundle = build_rollout_trace_bundle(
        rollout_id="rollout-k2",
        calls=[
            _call(1, [10], [11], media_ids=["screen-a"]),
            _call(
                2,
                [10, 11, 12],
                [13],
                media_ids=["screen-a", "screen-b"],
            ),
            _call(
                3,
                [20, 21],
                [22],
                media_ids=["screen-b", "screen-c"],
            ),
            _call(
                4,
                [20, 21, 22, 23],
                [24],
                media_ids=["screen-b", "screen-c", "screen-d"],
            ),
        ],
        boundary_events=[boundary],
        policy_name="recency",
    )

    assert bundle["checks"]["ok"]
    assert bundle["checks"]["physical_trace_count"] == 2
    assert [trace["source_turn_ids"] for trace in bundle["physical_traces"]] == [
        [1, 2],
        [3, 4],
    ]
    assert bundle["model_calls"][2]["starts_physical_trace"]
    assert bundle["model_calls"][2]["new_media_ids"] == [
        "screen-b",
        "screen-c",
    ]
    assert bundle["physical_traces"][1]["boundary_before"] == boundary


def test_nominal_chunk_boundary_without_rewrite_does_not_split_trace():
    bundle = build_rollout_trace_bundle(
        rollout_id="rollout-no-op-boundary",
        calls=[
            _call(1, [1], [2]),
            _call(2, [1, 2, 3], [4]),
        ],
        boundary_events=[
            {
                "event_id": "nominal-only",
                "applies_to_step": 2,
                "reason": "chunk_start",
            }
        ],
    )

    assert bundle["checks"]["physical_trace_count"] == 1
    assert not bundle["model_calls"][1]["starts_physical_trace"]


def test_repeated_media_occurrences_and_order_are_preserved():
    bundle = build_rollout_trace_bundle(
        rollout_id="rollout-repeat",
        calls=[
            _call(1, [1], [2], media_ids=["same"]),
            _call(2, [1, 2, 3], [4], media_ids=["same", "same"]),
        ],
    )

    assert bundle["physical_traces"][0]["ordered_media_ids"] == [
        "same",
        "same",
    ]
    assert bundle["model_calls"][1]["new_media_ids"] == ["same"]


def test_media_only_rewrite_starts_a_new_physical_trace():
    boundary = {
        "event_id": "boundary-media-2",
        "applies_to_step": 2,
        "reason": "history_policy_rewrite",
    }
    bundle = build_rollout_trace_bundle(
        rollout_id="rollout-media-only-rewrite",
        calls=[
            _call(1, [1], [2], media_ids=["image-a"]),
            _call(2, [1, 2, 3], [4], media_ids=["image-b"]),
        ],
        boundary_events=[boundary],
    )

    assert bundle["checks"]["ok"]
    assert bundle["checks"]["physical_trace_count"] == 2
    assert bundle["model_calls"][1]["token_append_compatible"]
    assert not bundle["model_calls"][1]["media_append_compatible"]
    assert bundle["model_calls"][1]["starts_physical_trace"]
    assert bundle["physical_traces"][1]["ordered_media_ids"] == ["image-b"]


def test_rewrite_without_boundary_is_visible_in_failed_checks():
    bundle = build_rollout_trace_bundle(
        rollout_id="rollout-unexplained",
        calls=[
            _call(1, [1], [2]),
            _call(2, [9], [10]),
        ],
    )

    assert not bundle["checks"]["boundary_records_cover_rewrites"]
    assert not bundle["checks"]["ok"]


def test_completion_token_logprob_mismatch_fails_closed():
    with pytest.raises(ValueError, match="token/logprob mismatch"):
        build_rollout_trace_bundle(
            rollout_id="rollout-bad",
            calls=[
                {
                    "turn_id": 1,
                    "prompt_token_ids": [1],
                    "sampled_token_ids": [2, 3],
                    "sampled_logprobs": [-0.1],
                    "media_ids": [],
                }
            ],
        )


def test_strict_bundle_preserves_provenance_and_eligibility():
    boundary = {
        "event_id": "boundary-2",
        "applies_to_step": 2,
        "policy_name": "recency",
        "policy_version": "1",
        "config_digest": "policy-config",
    }
    calls = [
        _strict_call(1, [1], [2], media_ids=["image-a"]),
        _strict_call(
            2,
            [8],
            [9, 10],
            media_ids=["image-b"],
            eligible=False,
            segment_index=1,
            compaction_event_id="boundary-2",
        ),
    ]
    bundle = build_rollout_trace_bundle(
        rollout_id="rollout-strict",
        group_id="group-1",
        source_row_index=7,
        reward=0.5,
        calls=calls,
        boundary_events=[boundary],
        policy_name="recency",
        media_assets={"image-a": {}, "image-b": {}},
        generation_contract=_GENERATION_CONTRACT,
        final_policy_decision=_FINAL_POLICY_DECISION,
        lineage_deltas=_lineage_deltas(calls),
        strict=True,
    )

    assert bundle["schema_version"] == 3
    assert bundle["group_id"] == "group-1"
    assert bundle["source_row_index"] == 7
    assert bundle["reward"] == 0.5
    assert bundle["model_calls"][1]["action_id"] == "action-2"
    assert bundle["model_calls"][1]["finish_reason"] == "stop"
    assert not bundle["model_calls"][1]["eligible"]
    assert bundle["physical_traces"][1]["segments"][1]["loss_mask"] == [0, 0]
    assert bundle["checks"]["eligible_trainable_token_count"] == 1
    assert validate_rollout_trace_bundle(
        bundle,
        media_assets={"image-a": {}, "image-b": {}},
    ) == {
        "model_call_count": 2,
        "physical_trace_count": 2,
        "sampled_token_count": 3,
        "eligible_trainable_token_count": 1,
        "media_occurrence_count": 2,
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda bundle: bundle["physical_traces"][1]["completion_spans"][
                0
            ].__setitem__("start", 0),
            "completion span is corrupted",
        ),
        (
            lambda bundle: bundle["physical_traces"][1]["segments"][1].__setitem__(
                "loss_mask", [0]
            ),
            "completion loss mask is corrupted",
        ),
        (
            lambda bundle: bundle["physical_traces"][1].__setitem__(
                "ordered_media_ids", ["image-a"]
            ),
            "ordered media is corrupted",
        ),
        (
            lambda bundle: bundle["model_calls"][1].__setitem__(
                "sampled_logprobs", [float("nan")]
            ),
            "non-finite logprob",
        ),
        (
            lambda bundle: bundle["lineage_deltas"][0].__setitem__(
                "state_digest", "corrupted"
            ),
            "Lineage delta state digest is corrupted",
        ),
        (
            lambda bundle: bundle["model_calls"][0]["policy_output_spans"][
                0
            ].__setitem__("end", 99),
            "policy-output span is corrupted",
        ),
        (
            lambda bundle: bundle["physical_traces"][1]["boundary_before"].__setitem__(
                "config_digest", "corrupted"
            ),
            "boundary config_digest is corrupted",
        ),
        (
            lambda bundle: bundle["model_calls"][0].__setitem__(
                "source_rollout_id", "another-rollout"
            ),
            "source rollout is corrupted",
        ),
        (
            lambda bundle: bundle["model_calls"][1].__setitem__(
                "action_id", bundle["model_calls"][0]["action_id"]
            ),
            "Duplicate action ID",
        ),
        (
            lambda bundle: bundle["model_calls"][1].__setitem__("segment_index", 99),
            "segment identity is corrupted",
        ),
        (
            lambda bundle: bundle["checks"].__setitem__("ok", False),
            "Trace check ok is corrupted",
        ),
    ],
)
def test_independent_validator_rejects_corrupted_serialized_bundles(mutation, match):
    calls = [
        _strict_call(1, [1], [2], media_ids=["image-a"]),
        _strict_call(
            2,
            [8],
            [9],
            media_ids=["image-b"],
            segment_index=1,
            compaction_event_id="boundary-2",
        ),
    ]
    bundle = build_rollout_trace_bundle(
        rollout_id="rollout-strict",
        calls=calls,
        boundary_events=[
            {
                "event_id": "boundary-2",
                "applies_to_step": 2,
                "policy_name": "recency",
                "policy_version": "1",
                "config_digest": "policy-config",
            }
        ],
        policy_name="recency",
        media_assets={"image-a": {}, "image-b": {}},
        generation_contract=_GENERATION_CONTRACT,
        final_policy_decision=_FINAL_POLICY_DECISION,
        lineage_deltas=_lineage_deltas(calls),
        strict=True,
    )
    corrupted = deepcopy(bundle)
    mutation(corrupted)

    with pytest.raises(ValueError, match=match):
        validate_rollout_trace_bundle(
            corrupted,
            media_assets={"image-a": {}, "image-b": {}},
        )


@pytest.mark.parametrize(
    ("calls", "boundaries", "media_assets", "match"),
    [
        (
            [
                _strict_call(1, [1], [2]),
                _strict_call(2, [9], [10], segment_index=1),
            ],
            [],
            {},
            "has no boundary record",
        ),
        (
            [
                _strict_call(1, [1], [2]),
                _strict_call(
                    2,
                    [1, 2, 3],
                    [4],
                    expected_append_compatible=True,
                ),
            ],
            [
                {
                    "event_id": "orphan",
                    "applies_to_step": 2,
                    "policy_name": "recency",
                    "policy_version": "1",
                    "config_digest": "policy-config",
                }
            ],
            {},
            "does not correspond to a rewrite",
        ),
        (
            [_strict_call(1, [1], [2], media_ids=["missing"])],
            [],
            {},
            "unknown media IDs",
        ),
    ],
)
def test_strict_builder_fails_closed_on_contract_corruption(
    calls, boundaries, media_assets, match
):
    with pytest.raises(ValueError, match=match):
        build_rollout_trace_bundle(
            rollout_id="rollout-strict",
            calls=calls,
            boundary_events=boundaries,
            policy_name="recency",
            media_assets=media_assets,
            generation_contract=_GENERATION_CONTRACT,
            final_policy_decision=_FINAL_POLICY_DECISION,
            lineage_deltas=_lineage_deltas(calls),
            strict=True,
        )


def _group_bundle(
    rollout_id: str,
    *,
    generation_contract_id: str = "generation-contract-1",
):
    generation_contract = {
        **_GENERATION_CONTRACT,
        "generation_contract_id": generation_contract_id,
    }
    calls = [
        _strict_call(
            1,
            [1],
            [2],
            rollout_id=rollout_id,
            generation_contract_id=generation_contract_id,
        )
    ]
    return build_rollout_trace_bundle(
        rollout_id=rollout_id,
        group_id="group-1",
        reward=1.0,
        calls=calls,
        policy_name="recency",
        media_assets={},
        generation_contract=generation_contract,
        final_policy_decision=_FINAL_POLICY_DECISION,
        lineage_deltas=_lineage_deltas(calls),
        strict=True,
    )


def test_group_validator_deduplicates_identical_retry_replays():
    bundle = _group_bundle("rollout-a")

    summary = validate_rollout_trace_group(
        [bundle, deepcopy(bundle)],
        expected_group_id="group-1",
    )

    assert summary["unique_rollout_count"] == 1
    assert summary["duplicate_retry_count"] == 1


def test_group_validator_rejects_rollout_id_reuse_with_different_content():
    bundle = _group_bundle("rollout-a")
    corrupted_retry = deepcopy(bundle)
    corrupted_retry["reward"] = 0.0

    with pytest.raises(ValueError, match="reused for different content"):
        validate_rollout_trace_group(
            [bundle, corrupted_retry],
            expected_group_id="group-1",
        )


def test_group_validator_rejects_mixed_generation_contracts():
    with pytest.raises(ValueError, match="mixed generation contracts"):
        validate_rollout_trace_group(
            [
                _group_bundle("rollout-a"),
                _group_bundle(
                    "rollout-b",
                    generation_contract_id="generation-contract-2",
                ),
            ],
            expected_group_id="group-1",
        )


def test_group_validator_fails_closed_before_training_admission():
    with pytest.raises(ValueError, match="no NeMo-RL training admission"):
        validate_rollout_trace_group(
            [_group_bundle("rollout-a")],
            expected_group_id="group-1",
            training_admission=True,
        )
