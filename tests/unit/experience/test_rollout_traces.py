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
from pathlib import Path

import pytest

from nemo_rl.environments.generation_contract import (
    build_training_admission_contract,
    canonical_digest,
    stable_id,
)
from nemo_rl.experience.rollout_traces import (
    build_trace_batch_plan,
    validate_trace_batch_plan,
)


_FIXTURE_DIR = Path(__file__).parents[2] / "fixtures" / "context_compaction_traces"


def _fixture(name: str) -> dict:
    return json.loads((_FIXTURE_DIR / name).read_text())


def _advantages(*bundles: dict, value: float = 2.5) -> dict[str, float]:
    return {bundle["rollout_id"]: value for bundle in bundles}


def _training_admitted(bundle: dict, *, policy_version: str = "sync-step-0") -> dict:
    result = deepcopy(bundle)
    generation_contract = {
        "generation_contract_id": "gym-generation-contract",
        "sampling_contract_id": "gym-sampling-contract",
        "compaction_policy_id": "gym-compaction-policy",
        "loss_normalization": "global_action_token_mean",
        "training_eligible": False,
        "incomplete_reasons": [
            "exact_tokenizer_identity_not_reported_by_generation_server",
            "exact_chat_template_identity_not_reported_by_generation_server",
            "exact_multimodal_processor_fingerprint_not_reported_by_generation_server",
        ],
    }
    result["generation_contract"] = generation_contract
    for call in result["model_calls"]:
        call["generation_contract_id"] = generation_contract["generation_contract_id"]
    definitions = {
        "model": {"generation_policy_version": policy_version},
        "tokenizer": {"vocab": "test"},
        "template": {"template": "test"},
        "processor": {"processor": "test"},
    }
    component_ids = {
        "model_contract_id": stable_id("model-contract", definitions["model"]),
        "tokenizer_contract_id": stable_id(
            "tokenizer-contract", definitions["tokenizer"]
        ),
        "template_contract_id": stable_id("template-contract", definitions["template"]),
        "processor_contract_id": stable_id(
            "processor-contract", definitions["processor"]
        ),
    }
    runtime = {
        "schema_version": 1,
        **component_ids,
        "runtime_contract_id": stable_id(
            "generation-runtime-contract",
            canonical_digest(component_ids),
        ),
        "component_definitions": definitions,
        "training_eligible": True,
        "incomplete_reasons": [],
    }
    result["training_admission"] = build_training_admission_contract(
        generation_contract,
        runtime,
    )
    return result


def _rekey_rollout(
    bundle: dict,
    *,
    rollout_id: str,
    group_id: str,
    source_row_index: int,
    reward: float,
) -> dict:
    result = deepcopy(bundle)
    result["rollout_id"] = rollout_id
    result["group_id"] = group_id
    result["source_row_index"] = source_row_index
    result["reward"] = reward
    trace_ids = {}
    for trace in result["physical_traces"]:
        old_trace_id = trace["trace_id"]
        new_trace_id = f"{rollout_id}:trace-{trace['trace_index']:06d}"
        trace["trace_id"] = new_trace_id
        trace_ids[old_trace_id] = new_trace_id
    for call in result["model_calls"]:
        call["source_rollout_id"] = rollout_id
        call["trace_id"] = trace_ids[call["trace_id"]]
    return result


def _refresh_plan_id(plan: dict) -> None:
    payload = json.dumps(
        {key: value for key, value in plan.items() if key != "plan_id"},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    plan["plan_id"] = hashlib.sha256(payload.encode()).hexdigest()


def _eligible_weighted_advantage(plan: dict) -> float:
    numerator = sum(
        row["advantage"] * row["eligible_token_count"] * row["sample_mask"]
        for row in plan["rows"]
    )
    return numerator / plan["eligible_action_token_count"]


def test_identity_plan_separates_logical_rollouts_from_padding_rows():
    bundle = _fixture("without_compaction.json")

    plan = build_trace_batch_plan(
        [bundle],
        rollout_advantages=_advantages(bundle),
        expected_rollouts_per_group=1,
        batch_quantum=4,
        optimizer_step_id="step-000001",
    )

    assert plan["logical_rollout_count"] == 1
    assert plan["physical_trace_count"] == 1
    assert plan["padding_row_count"] == 3
    assert plan["total_row_count"] == 4
    assert plan["rollout_to_rows"] == [[0]]
    assert plan["parent_indices"] == [0, -1, -1, -1]
    assert plan["eligible_action_token_count"] == 5
    assert plan["loss_normalization"] == "global_action_token_mean"
    assert not plan["training_admitted"]
    assert all(
        row["sample_mask"] == 0.0
        and row["eligible_token_count"] == 0
        and row["completion_ids"] == []
        for row in plan["rows"][1:]
    )


def test_k2_plan_maps_one_rollout_advantage_to_every_physical_trace():
    bundle = _fixture("k2_compaction.json")

    plan = build_trace_batch_plan(
        [bundle],
        rollout_advantages=_advantages(bundle, value=-1.25),
        expected_rollouts_per_group=1,
        batch_quantum=4,
        optimizer_step_id="step-k2",
    )

    assert plan["rollout_to_rows"] == [[0, 1, 2]]
    assert plan["parent_indices"] == [0, 0, 0, -1]
    assert [row["eligible_token_count"] for row in plan["rows"]] == [2, 2, 1, 0]
    assert [row["advantage"] for row in plan["rows"]] == [-1.25, -1.25, -1.25, 0.0]
    assert [row["completion_ids"] for row in plan["rows"][:3]] == [
        ["completion-1", "completion-2"],
        ["completion-3", "completion-4"],
        ["completion-5"],
    ]


def test_plan_preserves_multiple_rollouts_and_group_ownership():
    identity = _fixture("without_compaction.json")
    compacted = _fixture("k2_compaction.json")

    plan = build_trace_batch_plan(
        [identity, compacted],
        rollout_advantages={
            identity["rollout_id"]: 2.0,
            compacted["rollout_id"]: -1.0,
        },
        expected_rollouts_per_group=1,
        batch_quantum=4,
        optimizer_step_id="step-two-groups",
    )

    assert plan["comparison_group_count"] == 2
    assert plan["logical_rollout_count"] == 2
    assert plan["physical_trace_count"] == 4
    assert plan["padding_row_count"] == 0
    assert plan["rollout_to_rows"] == [[0], [1, 2, 3]]
    assert plan["parent_indices"] == [0, 1, 1, 1]
    assert [row["advantage"] for row in plan["rows"]] == [2.0, -1.0, -1.0, -1.0]


def test_complete_two_rollout_comparison_group_is_preserved():
    first = _rekey_rollout(
        _fixture("without_compaction.json"),
        rollout_id="group-rollout-a",
        group_id="complete-group",
        source_row_index=0,
        reward=1.0,
    )
    second = _rekey_rollout(
        _fixture("without_compaction.json"),
        rollout_id="group-rollout-b",
        group_id="complete-group",
        source_row_index=1,
        reward=0.0,
    )

    plan = build_trace_batch_plan(
        [first, second],
        rollout_advantages={
            first["rollout_id"]: 1.0,
            second["rollout_id"]: -1.0,
        },
        expected_rollouts_per_group=2,
        batch_quantum=2,
        optimizer_step_id="step-complete-group",
    )

    assert plan["comparison_group_count"] == 1
    assert plan["logical_rollout_count"] == 2
    assert plan["rollout_to_rows"] == [[0], [1]]
    assert [row["group_id"] for row in plan["rows"]] == [
        "complete-group",
        "complete-group",
    ]
    assert [row["reward"] for row in plan["rows"]] == [1.0, 0.0]
    assert [row["advantage"] for row in plan["rows"]] == [1.0, -1.0]


def test_identical_retry_delivery_is_deduplicated_before_expansion():
    bundle = _fixture("k2_compaction.json")

    plan = build_trace_batch_plan(
        [bundle, deepcopy(bundle)],
        rollout_advantages=_advantages(bundle),
        expected_rollouts_per_group=1,
        batch_quantum=1,
        optimizer_step_id="step-retry",
    )

    assert plan["logical_rollout_count"] == 1
    assert plan["physical_trace_count"] == 3
    assert plan["duplicate_retry_count"] == 1


def test_artificial_segmentation_preserves_global_action_token_reduction():
    identity = _fixture("without_compaction.json")
    compacted = _fixture("k2_compaction.json")
    plans = [
        build_trace_batch_plan(
            [bundle],
            rollout_advantages=_advantages(bundle, value=3.25),
            expected_rollouts_per_group=1,
            batch_quantum=1,
            optimizer_step_id=f"step-{index}",
        )
        for index, bundle in enumerate((identity, compacted))
    ]

    assert [plan["physical_trace_count"] for plan in plans] == [1, 3]
    assert [plan["eligible_action_token_count"] for plan in plans] == [5, 5]
    assert [_eligible_weighted_advantage(plan) for plan in plans] == [3.25, 3.25]


def test_json_round_trip_passes_independent_plan_validation():
    bundle = _fixture("k2_compaction.json")
    plan = build_trace_batch_plan(
        [bundle],
        rollout_advantages=_advantages(bundle),
        expected_rollouts_per_group=1,
        batch_quantum=4,
        optimizer_step_id="step-json",
    )

    reloaded = json.loads(json.dumps(plan))

    assert validate_trace_batch_plan(reloaded, bundles=[bundle]) == {
        "logical_rollout_count": 1,
        "physical_trace_count": 3,
        "padding_row_count": 1,
        "eligible_action_token_count": 5,
        "duplicate_retry_count": 0,
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda plan: plan["rows"][1].__setitem__("parent_rollout_index", -1),
            "parent rollout",
        ),
        (
            lambda plan: plan["rows"][-1].__setitem__("sample_mask", 1.0),
            "not fully masked",
        ),
        (
            lambda plan: plan["rollout_to_rows"][0].pop(),
            "rollout-to-row mapping is corrupted",
        ),
        (
            lambda plan: plan.__setitem__("eligible_action_token_count", 99),
            "eligible-token count is corrupted",
        ),
        (
            lambda plan: plan["rows"][1].__setitem__("advantage", 99.0),
            "inconsistent advantages",
        ),
        (
            lambda plan: plan["rows"][1].__setitem__("trace_index", 99),
            "trace indices are not rollout-local",
        ),
        (
            lambda plan: plan.__setitem__("training_admitted", True),
            "has no admission identity",
        ),
    ],
)
def test_independent_validator_rejects_corrupted_serialized_plans(
    mutation,
    match,
):
    bundle = _fixture("k2_compaction.json")
    plan = build_trace_batch_plan(
        [bundle],
        rollout_advantages=_advantages(bundle),
        expected_rollouts_per_group=1,
        batch_quantum=4,
        optimizer_step_id="step-corruption",
    )
    mutation(plan)
    _refresh_plan_id(plan)

    with pytest.raises(ValueError, match=match):
        validate_trace_batch_plan(plan, bundles=[bundle])


def test_incomplete_comparison_group_fails_before_expansion():
    bundle = _fixture("k2_compaction.json")

    with pytest.raises(ValueError, match="is incomplete"):
        build_trace_batch_plan(
            [bundle],
            rollout_advantages=_advantages(bundle),
            expected_rollouts_per_group=2,
            batch_quantum=1,
            optimizer_step_id="step-incomplete",
        )


def test_optimizer_step_plan_rejects_mixed_generation_contracts():
    identity = _fixture("without_compaction.json")
    compacted = _fixture("k2_compaction.json")
    compacted["generation_contract"]["generation_contract_id"] = "another-contract"
    for call in compacted["model_calls"]:
        call["generation_contract_id"] = "another-contract"

    with pytest.raises(ValueError, match="cannot mix generation contracts"):
        build_trace_batch_plan(
            [identity, compacted],
            rollout_advantages={
                identity["rollout_id"]: 1.0,
                compacted["rollout_id"]: -1.0,
            },
            expected_rollouts_per_group=1,
            batch_quantum=1,
            optimizer_step_id="step-mixed-contract",
        )


@pytest.mark.parametrize(
    "advantages",
    [
        {},
        {"unknown-rollout": 1.0},
        {"computer-use-k2": float("nan")},
    ],
)
def test_advantage_ownership_must_be_exact_and_finite(advantages):
    bundle = _fixture("k2_compaction.json")

    with pytest.raises(ValueError, match="advantage|advantages"):
        build_trace_batch_plan(
            [bundle],
            rollout_advantages=advantages,
            expected_rollouts_per_group=1,
            batch_quantum=1,
            optimizer_step_id="step-advantage",
        )


def test_training_admission_remains_fail_closed_for_generation_only_contract():
    bundle = _fixture("k2_compaction.json")

    with pytest.raises(ValueError, match="no NeMo-RL training admission"):
        build_trace_batch_plan(
            [bundle],
            rollout_advantages=_advantages(bundle),
            expected_rollouts_per_group=1,
            batch_quantum=1,
            optimizer_step_id="step-training-admission",
            training_admission=True,
        )


def test_training_admission_is_owned_by_nemo_rl_not_gym():
    bundle = _training_admitted(_fixture("k2_compaction.json"))

    plan = build_trace_batch_plan(
        [bundle],
        rollout_advantages=_advantages(bundle),
        expected_rollouts_per_group=1,
        batch_quantum=1,
        optimizer_step_id="step-training-admitted",
        training_admission=True,
    )

    assert plan["training_admitted"]
    assert (
        plan["training_admission_contract_id"]
        == bundle["training_admission"]["admission_contract_id"]
    )
    assert bundle["generation_contract"]["training_eligible"] is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"advantage_estimator_name": "reinforce_plus_plus"},
        {"sequence_level_ratios_enabled": True},
        {"sequence_level_clipping_enabled": True},
    ],
)
def test_unsupported_multitrace_semantics_fail_closed(kwargs):
    bundle = _fixture("k2_compaction.json")

    with pytest.raises(ValueError, match="does not support|Sequence-level"):
        build_trace_batch_plan(
            [bundle],
            rollout_advantages=_advantages(bundle),
            expected_rollouts_per_group=1,
            batch_quantum=1,
            optimizer_step_id="step-unsupported",
            **kwargs,
        )
