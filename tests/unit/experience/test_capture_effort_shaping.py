"""Effort parity, exact metric pooling, and durable context validation."""

from dataclasses import replace

import pytest

from nemo_rl.experience.effort_shaping import (
    EffortLevelsConfig,
    EffortShapingMetrics,
    RolloutEffortContext,
    aggregate_capture_effort_metrics,
    capture_effort_statistics,
    compute_effort_context,
    effort_shaping_metrics,
    finalize_effort_reward,
)
from nemo_rl.experience.rollouts import _apply_effort_shaping


def _input(*messages):
    return {"responses_create_params": {"input": list(messages)}}


LOW = _input({"role": "user", "content": "budget"})
CONFIG = EffortLevelsConfig(
    low_weight=1, low_penalty=1, low_ub=1000, low_string="budget"
)


@pytest.mark.parametrize("reward", [-2.0, 0.0, 2.0])
@pytest.mark.parametrize("length", [1, 100, 900, 1000, 1200, 2500])
@pytest.mark.parametrize("weight", [0.0, -1.0, 1.0, 3.0])
@pytest.mark.parametrize("low", [True, False])
def test_finalized_effort_matches_inline_formula_and_metrics(
    reward, length, weight, low
):
    config = CONFIG.model_copy(update={"low_weight": weight})
    prompt = LOW if low else _input({"role": "user", "content": "explain fully"})
    inline = {
        "message_log": [
            {"role": "assistant", "token_ids": [9] * 900},
            {"role": "user", "token_ids": [4] * 17},
            {"role": "assistant", "token_ids": [8] * length},
        ],
        "full_result": {"reward": reward},
    }
    expected_metrics = _apply_effort_shaping([inline], [prompt], config)
    context = compute_effort_context("r", prompt, config)
    if context is not None:
        context = RolloutEffortContext.from_state_dict(context.state_dict())
    actual, metrics = finalize_effort_reward(
        "r", reward, terminal_length=length, context=context, config=config
    )
    assert actual == inline["full_result"]["reward"]
    assert metrics == expected_metrics
    if weight > 0 and low:
        term = min(1, weight * (1 - length / 1000))
        assert actual == reward + reward * max(term, 0) + min(term, 0)


@pytest.mark.parametrize(
    "messages,expected",
    [
        (
            [
                {"role": "user", "content": "budget"},
                {"role": "assistant", "content": "tools"},
            ],
            True,
        ),
        (
            [
                {"role": "user", "content": "budget"},
                {"role": "user", "content": "long"},
            ],
            False,
        ),
        ([{"role": "user", "content": "budget"}, {"role": "user"}], True),
        ([{"role": "system", "content": "budget"}], False),
        (
            [{"role": "user", "content": [{"type": "input_text", "text": "budget"}]}],
            False,
        ),
    ],
)
def test_classification_retains_original_last_user_membership_rule(messages, expected):
    assert (
        compute_effort_context("r", _input(*messages), CONFIG).is_low_effort is expected
    )


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 99},
        {"rollout_id": "other"},
        {"semantics_fingerprint": "wrong"},
    ],
)
def test_incompatible_context_rejected(change):
    context = replace(compute_effort_context("r", LOW, CONFIG), **change)
    with pytest.raises(ValueError):
        finalize_effort_reward(
            "r", 1, terminal_length=100, context=context, config=CONFIG
        )


def test_missing_context_and_changed_config_do_not_silently_change_rewards():
    with pytest.raises(ValueError, match="missing_effort_context"):
        finalize_effort_reward("r", 1, terminal_length=100, context=None, config=CONFIG)
    context = compute_effort_context("r", LOW, CONFIG)
    for config in (None, CONFIG.model_copy(update={"low_penalty": 2})):
        with pytest.raises(ValueError, match="context"):
            finalize_effort_reward(
                "r", 1, terminal_length=100, context=context, config=config
            )


@pytest.mark.parametrize("bound", [0, -1])
def test_active_effort_requires_positive_bound(bound):
    with pytest.raises(ValueError, match="low_ub"):
        EffortLevelsConfig(low_weight=1, low_string="budget", low_ub=bound)
    assert (
        EffortLevelsConfig(low_weight=0, low_string="budget", low_ub=bound).low_ub
        == bound
    )


def test_exact_effort_metrics_pool_unequal_groups_through_checkpoint():
    from nemo_rl.experience.rollout_recovery import (
        RolloutRecoveryLedger,
        build_rollout_recovery_state,
        parse_rollout_recovery_state,
    )

    groups = [
        EffortShapingMetrics([0.9], [1.9], [100], [200]),
        EffortShapingMetrics(
            [0.1, 0, -0.2], [1.1, 1, 0.8], [900, 1000, 1200], [100, 300, 900]
        ),
    ]
    state = build_rollout_recovery_state(
        RolloutRecoveryLedger(),
        batch_shortfall={},
        sampler_stamps_target_steps=True,
        finalizer_metrics_by_group={
            str(i): capture_effort_statistics(group) for i, group in enumerate(groups)
        },
    )
    restored = parse_rollout_recovery_state(state)
    pooled = {}
    for metrics in restored.finalizer_metrics_by_group.values():
        for name, value in metrics.items():
            pooled.setdefault(name, []).append(value)
    actual = aggregate_capture_effort_metrics(pooled)
    expected = effort_shaping_metrics(
        EffortShapingMetrics(
            [0.9, 0.1, 0, -0.2],
            [1.9, 1.1, 1, 0.8],
            [100, 900, 1000, 1200],
            [200, 100, 300, 900],
        )
    )
    assert actual == pytest.approx(expected)
    assert actual["median_length_low"] == 950
    assert actual["median_length_high"] == 250
