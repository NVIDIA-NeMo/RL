"""Differential reward semantics and evidence compatibility without token decoding."""

from dataclasses import replace
from itertools import product

import pytest

from nemo_rl.experience.reward_penalties import (
    CaptureRewardPenaltyConfig,
    RolloutTextPenaltyEvidence,
    aggregate_capture_reward_metrics,
    compute_text_penalty_evidence,
    finalize_reward_penalties,
    has_duplicated_reasoning,
    has_empty_final_answer,
)
from nemo_rl.experience.rollouts import (
    apply_reward_penalties,
    resolve_reward_penalty_config,
)

OUTPUTS = [
    ([], False, True),
    ([{"type": "message", "content": "  answer "}], False, False),
    ([{"type": "message", "content": [{"text": "  "}]}], False, True),
    (
        [{"type": "reasoning", "summary": [{"text": " a "}]}, {"content": "a"}],
        True,
        False,
    ),
    (
        [
            {"type": "reasoning", "summary": [{"text": " a "}]},
            {"content": [{"text": " a "}]},
        ],
        True,
        False,
    ),
    (
        [{"type": "reasoning", "summary": [{"text": " "}]}, {"content": " "}],
        False,
        True,
    ),
    ([{"type": "reasoning", "summary": []}, {"content": "a"}], False, False),
    (
        [
            {"type": "reasoning", "summary": [{"text": "a"}]},
            {"type": "function_call"},
            {"content": "a"},
        ],
        False,
        False,
    ),
    ([{"content": "a"}, {"content": []}], False, False),
    ([{"content": "a"}, {"content": [{}]}], False, False),
    ([{"content": {"text": "a"}}], False, True),
    (
        [{"type": "reasoning", "summary": [{"text": "a"}]}, {"type": "function_call"}],
        False,
        False,
    ),
]


@pytest.mark.parametrize("output,duplicated,empty", OUTPUTS)
def test_text_detector_contract(output, duplicated, empty):
    assert has_duplicated_reasoning(output) is duplicated
    assert has_empty_final_answer(output) is empty


@pytest.mark.parametrize("flags", list(product([False, True], repeat=3)))
@pytest.mark.parametrize("reward", [-2.0, 0.0, 1.5])
@pytest.mark.parametrize("output,duplicated,empty", OUTPUTS)
def test_inline_and_capture_reward_parity(flags, reward, output, duplicated, empty):
    resolved = resolve_reward_penalty_config(
        {
            "penalize_duplicated_reasoning": flags[0],
            "penalize_empty_final_answer": flags[1],
            "penalize_unwanted_tokens": flags[2],
            "token_ids": {"unwanted": [99]},
        },
        None,
    )
    config = CaptureRewardPenaltyConfig.from_resolved(resolved)
    result = {
        "full_result": {"reward": reward, "response": {"output": output}},
        "message_log": [
            {"role": "user", "token_ids": [99, 10]},
            {"role": "assistant", "token_ids": [11]},
            {"role": "user", "token_ids": [99]},
            {"role": "assistant", "token_ids": [12, 99]},
        ],
    }
    expected = apply_reward_penalties([result], resolved)
    evidence = compute_text_penalty_evidence("r", output, config)
    evidence = RolloutTextPenaltyEvidence.from_state_dict(evidence.state_dict())
    actual_reward, counts = finalize_reward_penalties(
        "r",
        reward,
        evidence,
        config,
        [99, 10, 11, 99, 12, 99],
        [("c1", 2, 1), ("c2", 1, 2)],
    )
    assert actual_reward == result["full_result"]["reward"]
    assert counts == {k: expected[k] for k in counts}


@pytest.mark.parametrize(
    "ids,expected",
    [
        ([99, 10, 11, 99, 12, 13], False),
        ([10, 11, 99, 20, 12, 13], True),
        ([10, 11, 12, 20, 13, 99], True),
    ],
)
def test_generated_spans_only_including_each_terminal(ids, expected):
    reward, counts = finalize_reward_penalties(
        "r",
        -1.0,
        None,
        CaptureRewardPenaltyConfig(False, False, (99,)),
        ids,
        [("c1", 2, 1), ("c2", 1, 2)],
    )
    assert reward == (0.0 if expected else -1.0)
    assert counts["unwanted_token"] == int(expected)


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"schema_version": 2}, "incompatible"),
        ({"semantics_fingerprint": "future"}, "incompatible"),
        ({"rollout_id": "other"}, "identity"),
        ({"duplicated_reasoning": None}, "unevaluated"),
        ({"empty_final_answer": None}, "unevaluated"),
    ],
)
def test_incompatible_evidence_rejected(change, reason):
    config = CaptureRewardPenaltyConfig(True, True, ())
    evidence = replace(compute_text_penalty_evidence("r", [], config), **change)
    with pytest.raises(ValueError, match=reason):
        finalize_reward_penalties("r", 1.0, evidence, config, [], [])


def test_old_checkpoints_require_evidence_only_when_text_checks_enabled():
    with pytest.raises(ValueError, match="missing_text_evidence"):
        finalize_reward_penalties(
            "r", 1.0, None, CaptureRewardPenaltyConfig(True, False, ()), [], []
        )
    assert (
        finalize_reward_penalties(
            "r",
            1.0,
            None,
            CaptureRewardPenaltyConfig(False, False, (99,)),
            [99],
            [("c", 0, 1)],
        )[0]
        == 0.0
    )


@pytest.mark.parametrize("spans", [[("c", -1, 2)], [("c", 0, 2)], [("c", 0, 0)], []])
def test_span_coverage_is_validated(spans):
    with pytest.raises(ValueError, match="span"):
        finalize_reward_penalties(
            "r", 1.0, None, CaptureRewardPenaltyConfig(False, False, (99,)), [99], spans
        )


def test_metric_pooling_uses_valid_rollouts_and_legacy_names():
    metrics = aggregate_capture_reward_metrics(
        {
            "finalize/reward_count": [1, 3, 0],
            "finalize/reward_sum": [0, 6],
            "finalize/reward_sumsq": [0, 12],
            "finalize/reward_min": [0, 2],
            "finalize/reward_max": [0, 2],
            "finalize/penalty_count/empty_final_answer": [1, 0, 0],
        }
    )
    assert metrics == {
        "finalize/reward_count": 4,
        "finalize/penalty_count/empty_final_answer": 1,
        "empty_final_answer_rate": 0.25,
        "total_reward/mean": 1.5,
        "total_reward/stddev": 1.0,
        "total_reward/min": 0,
        "total_reward/max": 2,
    }
    assert aggregate_capture_reward_metrics({"finalize/reward_count": [0]}) == {}


@pytest.mark.parametrize(
    "config,reason",
    [
        ({"penalize_malformed_think_tag": True}, "does not support"),
        ({"penalize_unwanted_tokens": True}, "must be set"),
    ],
)
def test_capture_config_rejects_unsupported_or_unresolved_options(config, reason):
    with pytest.raises(ValueError, match=reason):
        CaptureRewardPenaltyConfig.from_resolved(config)
