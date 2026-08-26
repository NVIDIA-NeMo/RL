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
import json
from pathlib import Path

import pytest
import torch

from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    GRPOAdvantageEstimator,
    ReinforceBaselineAdvantageEstimator,
)
from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.trace_batch_scoring import (
    prepare_trace_batch_for_scoring,
    score_prepared_trace_batch,
)


_FIXTURE_DIR = Path(__file__).parents[2] / "fixtures" / "context_compaction_traces"


def _fixture(name: str) -> dict:
    return json.loads((_FIXTURE_DIR / name).read_text())


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


def _message_logs(bundle: dict) -> list[list[dict]]:
    result = []
    for trace in bundle["physical_traces"]:
        messages = []
        for segment in trace["segments"]:
            message = {
                "role": ("user" if segment["kind"] == "prompt" else "assistant"),
                "content": "",
                "token_ids": torch.tensor(
                    segment["token_ids"],
                    dtype=torch.int64,
                ),
            }
            if segment["kind"] == "completion":
                message["generation_logprobs"] = torch.tensor(
                    segment["generation_logprobs"],
                    dtype=torch.float32,
                )
            messages.append(message)
        result.append(messages)
    return result


def _estimator() -> GRPOAdvantageEstimator:
    return GRPOAdvantageEstimator(
        AdvEstimatorConfig(
            use_leave_one_out_baseline=False,
            normalize_rewards=True,
        ),
        ClippedPGLossConfig(),
    )


def _reinforce_baseline_estimator() -> ReinforceBaselineAdvantageEstimator:
    return ReinforceBaselineAdvantageEstimator({}, ClippedPGLossConfig())


def _rollout_batch(bundles: list[dict]) -> dict:
    return {
        "rollout_trace_bundle": bundles,
        "physical_message_logs": [_message_logs(bundle) for bundle in bundles],
        "total_reward": torch.tensor(
            [bundle["reward"] for bundle in bundles],
            dtype=torch.float32,
        ),
        "loss_multiplier": torch.ones(len(bundles)),
        "mask_sample": torch.zeros(len(bundles), dtype=torch.bool),
        "truncated": torch.zeros(len(bundles), dtype=torch.bool),
    }


def test_logical_grpo_advantages_are_fixed_before_physical_expansion():
    first = _rekey_rollout(
        _fixture("without_compaction.json"),
        rollout_id="group-rollout-a",
        group_id="shared-group",
        source_row_index=0,
        reward=1.0,
    )
    second = _rekey_rollout(
        _fixture("without_compaction.json"),
        rollout_id="group-rollout-b",
        group_id="shared-group",
        source_row_index=1,
        reward=3.0,
    )

    prepared = prepare_trace_batch_for_scoring(
        _rollout_batch([first, second]),
        prompt_ids=torch.tensor([[42], [42]]),
        advantage_estimator=_estimator(),
        expected_rollouts_per_group=2,
        batch_quantum=2,
        optimizer_step_id="step-two-rollouts",
        pad_token_id=999,
    )

    expected = 1.0 / (2.0**0.5)
    assert prepared["rollout_advantages"] == pytest.approx(
        {
            "group-rollout-a": -expected,
            "group-rollout-b": expected,
        }
    )
    assert [row["advantage"] for row in prepared["plan"]["rows"]] == pytest.approx(
        [-expected, expected]
    )
    assert prepared["plan"]["logical_rollout_count"] == 2
    assert prepared["plan"]["physical_trace_count"] == 2


def test_one_compacted_rollout_expands_after_its_scalar_advantage_is_fixed():
    bundle = _fixture("k2_compaction.json")

    prepared = prepare_trace_batch_for_scoring(
        _rollout_batch([bundle]),
        prompt_ids=torch.tensor([[101, 102]]),
        advantage_estimator=_estimator(),
        expected_rollouts_per_group=1,
        batch_quantum=4,
        optimizer_step_id="step-k2",
        pad_token_id=999,
        make_sequence_length_divisible_by=8,
    )

    assert prepared["rollout_advantages"] == {
        bundle["rollout_id"]: 0.0,
    }
    assert prepared["plan"]["rollout_to_rows"] == [[0, 1, 2]]
    assert prepared["plan"]["parent_indices"] == [0, 0, 0, -1]
    assert prepared["logprob_data"]["input_ids"].shape == (4, 8)
    assert prepared["logprob_data"]["sample_mask"].tolist() == [
        1.0,
        1.0,
        1.0,
        0.0,
    ]


def test_reinforce_baseline_is_invariant_when_one_rollout_splits_into_traces():
    split_rollout = _rekey_rollout(
        _fixture("k2_compaction.json"),
        rollout_id="split-rollout",
        group_id="shared-group",
        source_row_index=0,
        reward=1.0,
    )
    unsplit_rollout = _rekey_rollout(
        _fixture("without_compaction.json"),
        rollout_id="unsplit-rollout",
        group_id="shared-group",
        source_row_index=1,
        reward=3.0,
    )

    prepared = prepare_trace_batch_for_scoring(
        _rollout_batch([split_rollout, unsplit_rollout]),
        prompt_ids=torch.tensor([[42], [42]]),
        advantage_estimator=_reinforce_baseline_estimator(),
        expected_rollouts_per_group=2,
        batch_quantum=4,
        optimizer_step_id="step-reinforce-baseline",
        pad_token_id=999,
    )

    # Both logical rollouts contain five eligible action tokens. The split
    # rollout must therefore have the same statistical weight as the unsplit
    # rollout, rather than three times the weight because it occupies 3 rows.
    assert prepared["rollout_advantages"] == pytest.approx(
        {"split-rollout": -1.0, "unsplit-rollout": 1.0}
    )
    assert prepared["plan"]["advantage_estimator_name"] == "reinforce_baseline"
    assert [row["advantage"] for row in prepared["plan"]["rows"]] == pytest.approx(
        [-1.0, -1.0, -1.0, 1.0]
    )
    train_data = prepared["materialization"]["train_data"]
    eligible = train_data["token_mask"].bool()
    assert train_data["advantages"][eligible].tolist() == pytest.approx(
        [-1.0] * 5 + [1.0] * 5
    )


def test_logprob_data_matches_existing_worker_input_contract():
    bundle = _fixture("without_compaction.json")
    prepared = prepare_trace_batch_for_scoring(
        _rollout_batch([bundle]),
        prompt_ids=torch.tensor([[17]]),
        advantage_estimator=_estimator(),
        expected_rollouts_per_group=1,
        batch_quantum=1,
        optimizer_step_id="step-worker-contract",
        pad_token_id=999,
    )

    class _FakePolicy:
        def get_logprobs(self, data):
            assert set(data) == {
                "input_ids",
                "input_lengths",
                "token_mask",
                "sample_mask",
            }
            assert data["input_ids"].shape == data["token_mask"].shape
            return {"logprobs": torch.zeros_like(data["input_ids"]).float()}

    output = _FakePolicy().get_logprobs(prepared["logprob_data"])

    assert output["logprobs"].shape == prepared["logprob_data"]["input_ids"].shape
    assert "generation_logprobs" not in prepared["logprob_data"]
    assert "advantages" not in prepared["logprob_data"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("loss_multiplier", torch.tensor([0.0])),
        ("mask_sample", torch.tensor([True])),
        ("truncated", torch.tensor([True])),
    ],
)
def test_masked_or_truncated_logical_rollouts_fail_before_expansion(key, value):
    bundle = _fixture("without_compaction.json")
    rollout_batch = _rollout_batch([bundle])
    rollout_batch[key] = value

    with pytest.raises(ValueError, match="masked or truncated"):
        prepare_trace_batch_for_scoring(
            rollout_batch,
            prompt_ids=torch.tensor([[1]]),
            advantage_estimator=_estimator(),
            expected_rollouts_per_group=1,
            batch_quantum=1,
            optimizer_step_id="step-rejected",
            pad_token_id=999,
        )


def test_prompt_tokens_and_declared_groups_must_define_the_same_partition():
    first = _rekey_rollout(
        _fixture("without_compaction.json"),
        rollout_id="group-rollout-a",
        group_id="shared-group",
        source_row_index=0,
        reward=1.0,
    )
    second = _rekey_rollout(
        _fixture("without_compaction.json"),
        rollout_id="group-rollout-b",
        group_id="shared-group",
        source_row_index=1,
        reward=2.0,
    )

    with pytest.raises(ValueError, match="comparison-group ownership disagree"):
        prepare_trace_batch_for_scoring(
            _rollout_batch([first, second]),
            prompt_ids=torch.tensor([[1], [2]]),
            advantage_estimator=_estimator(),
            expected_rollouts_per_group=2,
            batch_quantum=2,
            optimizer_step_id="step-group-mismatch",
            pad_token_id=999,
        )


def test_post_rollout_reward_rewriting_fails_closed():
    bundle = _fixture("without_compaction.json")
    rollout_batch = _rollout_batch([bundle])
    rollout_batch["total_reward"][0] += 1

    with pytest.raises(ValueError, match="reward disagrees"):
        prepare_trace_batch_for_scoring(
            rollout_batch,
            prompt_ids=torch.tensor([[1]]),
            advantage_estimator=_estimator(),
            expected_rollouts_per_group=1,
            batch_quantum=1,
            optimizer_step_id="step-reward-mismatch",
            pad_token_id=999,
        )


def test_unsupported_advantage_estimator_fails_closed():
    bundle = _fixture("without_compaction.json")

    with pytest.raises(TypeError, match="GRPOAdvantageEstimator"):
        prepare_trace_batch_for_scoring(
            _rollout_batch([bundle]),
            prompt_ids=torch.tensor([[1]]),
            advantage_estimator=object(),
            expected_rollouts_per_group=1,
            batch_quantum=1,
            optimizer_step_id="step-wrong-estimator",
            pad_token_id=999,
        )


def _prepared_compacted_batch():
    bundle = _fixture("k2_compaction.json")
    return prepare_trace_batch_for_scoring(
        _rollout_batch([bundle]),
        prompt_ids=torch.tensor([[101, 102]]),
        advantage_estimator=_estimator(),
        expected_rollouts_per_group=1,
        batch_quantum=4,
        optimizer_step_id="step-score-workers",
        pad_token_id=999,
    )


def test_policy_and_reference_outputs_are_attached_to_exact_physical_rows():
    prepared = _prepared_compacted_batch()
    expected_input_ids = prepared["logprob_data"]["input_ids"].clone()
    calls = []

    class _FakePolicy:
        def get_logprobs(self, data, timer=None):
            calls.append(("policy", timer, data["input_ids"].clone()))
            result = torch.full(data["input_ids"].shape, -1.0)
            result[~data["token_mask"].bool()] = float("nan")
            return {"logprobs": result}

        def get_reference_policy_logprobs(self, data, timer=None):
            calls.append(("reference", timer, data["input_ids"].clone()))
            result = torch.full(data["input_ids"].shape, -2.0)
            result[~data["token_mask"].bool()] = float("inf")
            return {"reference_logprobs": result}

    scored = score_prepared_trace_batch(
        prepared,
        policy=_FakePolicy(),
        timer="timer-sentinel",
    )

    assert [call[:2] for call in calls] == [
        ("policy", "timer-sentinel"),
        ("reference", "timer-sentinel"),
    ]
    assert all(torch.equal(call[2], expected_input_ids) for call in calls)
    train_data = scored["train_data"]
    effective_mask = train_data["token_mask"].bool() & (
        train_data["sample_mask"].bool().unsqueeze(-1)
    )
    assert torch.all(train_data["prev_logprobs"][effective_mask] == -1.0)
    assert torch.all(train_data["reference_policy_logprobs"][effective_mask] == -2.0)
    assert torch.count_nonzero(train_data["prev_logprobs"][~effective_mask]) == 0
    assert (
        torch.count_nonzero(train_data["reference_policy_logprobs"][~effective_mask])
        == 0
    )


def test_skipped_workers_produce_zero_placeholders_without_calls():
    prepared = _prepared_compacted_batch()

    class _NoCallPolicy:
        def get_logprobs(self, data, timer=None):
            raise AssertionError("policy worker must be skipped")

        def get_reference_policy_logprobs(self, data, timer=None):
            raise AssertionError("reference worker must be skipped")

    scored = score_prepared_trace_batch(
        prepared,
        policy=_NoCallPolicy(),
        skip_policy_logprobs=True,
        skip_reference_logprobs=True,
    )

    assert torch.count_nonzero(scored["train_data"]["prev_logprobs"]) == 0
    assert torch.count_nonzero(scored["train_data"]["reference_policy_logprobs"]) == 0


@pytest.mark.parametrize(
    ("method", "key", "bad_output", "match"),
    [
        (
            "get_logprobs",
            "logprobs",
            lambda shape: torch.zeros(shape[0], shape[1] + 1),
            "floating tensor",
        ),
        (
            "get_logprobs",
            "logprobs",
            lambda shape: torch.zeros(shape, dtype=torch.int64),
            "floating tensor",
        ),
        (
            "get_logprobs",
            "logprobs",
            lambda shape: None,
            "floating tensor",
        ),
        (
            "get_reference_policy_logprobs",
            "reference_logprobs",
            lambda shape: torch.full(shape, float("nan")),
            "non-finite",
        ),
    ],
)
def test_malformed_worker_outputs_fail_closed(method, key, bad_output, match):
    prepared = _prepared_compacted_batch()
    shape = prepared["logprob_data"]["input_ids"].shape

    class _FakePolicy:
        def get_logprobs(self, data, timer=None):
            if method == "get_logprobs":
                return {key: bad_output(shape)}
            return {"logprobs": torch.zeros(shape)}

        def get_reference_policy_logprobs(self, data, timer=None):
            if method == "get_reference_policy_logprobs":
                return {key: bad_output(shape)}
            return {"reference_logprobs": torch.zeros(shape)}

    with pytest.raises(ValueError, match=match):
        score_prepared_trace_batch(
            prepared,
            policy=_FakePolicy(),
            skip_reference_logprobs=method == "get_logprobs",
        )


def test_non_mapping_worker_output_fails_closed():
    prepared = _prepared_compacted_batch()

    class _FakePolicy:
        def get_logprobs(self, data, timer=None):
            return torch.zeros_like(data["input_ids"]).float()

        def get_reference_policy_logprobs(self, data, timer=None):
            raise AssertionError("reference worker must not run")

    with pytest.raises(TypeError, match="must be a mapping"):
        score_prepared_trace_batch(
            prepared,
            policy=_FakePolicy(),
            skip_reference_logprobs=True,
        )


@pytest.mark.parametrize(
    "loss_overrides",
    [
        {},
        {
            "use_on_policy_kl_approximation": True,
            "use_importance_sampling_correction": True,
        },
        {"reference_policy_kl_penalty": 0.01},
        {
            "use_importance_sampling_correction": True,
            "truncated_importance_sampling_type": "tis",
            "truncated_importance_sampling_ratio": 1.1,
            "truncated_importance_sampling_ratio_min": 0.9,
        },
        {
            "use_importance_sampling_correction": True,
            "truncated_importance_sampling_type": "icepop",
            "truncated_importance_sampling_ratio": 1.1,
            "truncated_importance_sampling_ratio_min": 0.9,
        },
        {"force_on_policy_ratio": True},
        {
            "use_cispo": True,
            "use_importance_sampling_correction": True,
        },
    ],
)
def test_token_level_loss_is_invariant_to_artificial_physical_segmentation(
    loss_overrides,
):
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            **{
                "reference_policy_kl_penalty": 0.0,
                "use_importance_sampling_correction": False,
                "sequence_level_importance_ratios": False,
                "token_level_loss": True,
                **loss_overrides,
            }
        )
    )
    current = torch.tensor([[-0.2, -0.4, -0.6]], requires_grad=True)
    unsplit = BatchedDataDict(
        {
            "token_mask": torch.tensor([[0, 1, 1, 1]]),
            "sample_mask": torch.tensor([1.0]),
            "advantages": torch.tensor([[0.0, 2.0, 2.0, 2.0]]),
            "prev_logprobs": torch.tensor([[0.0, -0.1, -0.3, -0.5]]),
            "generation_logprobs": torch.tensor([[0.0, -0.05, -0.35, -0.7]]),
            "reference_policy_logprobs": torch.zeros(1, 4),
        }
    )
    unsplit_loss, _ = loss_fn(
        current,
        unsplit,
        global_valid_seqs=torch.tensor(1.0),
        global_valid_toks=torch.tensor(3.0),
    )

    split_current = current.detach().reshape(3, 1).clone().requires_grad_(True)
    split = BatchedDataDict(
        {
            "token_mask": torch.tensor([[0, 1], [0, 1], [0, 1]]),
            "sample_mask": torch.ones(3),
            "advantages": torch.tensor([[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]]),
            "prev_logprobs": torch.tensor([[0.0, -0.1], [0.0, -0.3], [0.0, -0.5]]),
            "generation_logprobs": torch.tensor(
                [[0.0, -0.05], [0.0, -0.35], [0.0, -0.7]]
            ),
            "reference_policy_logprobs": torch.zeros(3, 2),
        }
    )
    split_loss, _ = loss_fn(
        split_current,
        split,
        global_valid_seqs=torch.tensor(3.0),
        global_valid_toks=torch.tensor(3.0),
    )

    assert split_loss.item() == pytest.approx(unsplit_loss.item())
    unsplit_loss.backward()
    split_loss.backward()
    torch.testing.assert_close(
        split_current.grad.reshape_as(current),
        current.grad,
    )


def test_supported_token_objective_is_invariant_to_uneven_traces_and_padding():
    """CC row boundaries must not affect the supported token-level objective."""

    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            token_level_loss=True,
            sequence_level_importance_ratios=False,
            ratio_clip_min=0.2,
            ratio_clip_max=0.2,
            reference_policy_kl_penalty=0.2,
            use_importance_sampling_correction=True,
            truncated_importance_sampling_type="tis",
            truncated_importance_sampling_ratio=1.1,
            truncated_importance_sampling_ratio_min=0.9,
        )
    )
    advantages = torch.tensor([2.0, 2.0, 2.0, -1.5, -1.5, -1.5, -1.5])
    prev_logprobs = torch.tensor([-1.4, -1.2, -1.6, -1.1, -1.5, -1.3, -1.7])
    ppo_ratios = torch.tensor([1.5, 0.6, 1.1, 1.3, 0.7, 1.0, 0.9])
    current_values = prev_logprobs + torch.log(ppo_ratios)
    sampling_ratios = torch.tensor([1.3, 0.7, 1.0, 1.2, 0.8, 1.05, 0.95])
    generation_logprobs = prev_logprobs - torch.log(sampling_ratios)
    reference_logprobs = torch.tensor([-1.8, -1.7, -1.9, -1.6, -1.8, -1.5, -2.0])

    def evaluate(layout: list[list[int]]):
        current = current_values.clone().requires_grad_(True)
        row_width = max(len(row) for row in layout)

        def aligned_row(source: torch.Tensor, indexes: list[int]) -> torch.Tensor:
            values = [source[index] for index in indexes]
            values.extend(source.new_zeros(row_width - len(indexes)))
            return torch.stack(values)

        def shifted_row(source: torch.Tensor, indexes: list[int]) -> torch.Tensor:
            return torch.cat((source.new_zeros(1), aligned_row(source, indexes)))

        data = BatchedDataDict(
            {
                "token_mask": torch.tensor(
                    [
                        [0.0]
                        + [1.0] * len(indexes)
                        + [0.0] * (row_width - len(indexes))
                        for indexes in layout
                    ]
                ),
                "sample_mask": torch.tensor(
                    [1.0 if indexes else 0.0 for indexes in layout]
                ),
                "advantages": torch.stack(
                    [shifted_row(advantages, indexes) for indexes in layout]
                ),
                "prev_logprobs": torch.stack(
                    [shifted_row(prev_logprobs, indexes) for indexes in layout]
                ),
                "generation_logprobs": torch.stack(
                    [shifted_row(generation_logprobs, indexes) for indexes in layout]
                ),
                "reference_policy_logprobs": torch.stack(
                    [shifted_row(reference_logprobs, indexes) for indexes in layout]
                ),
            }
        )
        loss, metrics = loss_fn(
            torch.stack([aligned_row(current, indexes) for indexes in layout]),
            data,
            global_valid_seqs=data["sample_mask"].sum(),
            global_valid_toks=data["token_mask"].sum(),
        )
        loss.backward()
        return loss.detach(), current.grad.detach(), metrics, data

    unsplit = evaluate([[0, 1, 2], [3, 4, 5, 6], [], []])
    split = evaluate([[0], [1, 2], [3, 4, 5], [6], [], []])

    unsplit_loss, unsplit_grad, unsplit_metrics, unsplit_data = unsplit
    split_loss, split_grad, split_metrics, split_data = split
    torch.testing.assert_close(split_loss, unsplit_loss)
    torch.testing.assert_close(split_grad, unsplit_grad)

    assert unsplit_data["token_mask"].sum(dim=1).tolist() == [3.0, 4.0, 0.0, 0.0]
    assert split_data["token_mask"].sum(dim=1).tolist() == [
        1.0,
        2.0,
        3.0,
        1.0,
        0.0,
        0.0,
    ]
    assert unsplit_data["sample_mask"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert split_data["sample_mask"].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]

    # num_valid_samples intentionally counts physical rows. Every metric that
    # participates in the token-level objective must remain segmentation-invariant.
    for name in unsplit_metrics:
        if name != "num_valid_samples":
            assert split_metrics[name] == pytest.approx(unsplit_metrics[name])

    assert split_metrics["probs_ratio_min"] < split_metrics["probs_ratio_clamped_min"]
    assert split_metrics["probs_ratio_max"] > split_metrics["probs_ratio_clamped_max"]
    assert split_metrics["is_oob_ratio"] > 0
    assert split_metrics["kl_penalty"] > 0
    assert torch.isfinite(split_loss)
    assert torch.all(torch.isfinite(split_grad))
