"""Tests for turn-level GRPO advantage composition."""

import pytest
import torch
from turn_level_credit.advantage import TurnLevelGRPOAdvantageEstimator
from turn_level_credit.config import TurnCreditConfig
from turn_level_credit.trace import TurnBatch, attach_turn_batch


class _FixedBaseEstimator:
    def __init__(self, output):
        self.output = output

    def compute_advantage(self, **_kwargs):
        return self.output


def _repeated_batch():
    batch = {}
    attach_turn_batch(
        batch,
        TurnBatch(
            rewards=torch.tensor([[0.5, -0.25]]),
            mask=torch.tensor([[True, True]]),
            trainable_mask=torch.tensor([[True, True]]),
            assistant_spans=torch.tensor([[[1, 3], [4, 5]]]),
            terminateds=torch.tensor([[False, True]]),
        ),
    )
    return batch


def test_zero_turn_weight_is_bitwise_base_equivalent():
    base_output = torch.tensor([[7.0, 7.0, 7.0, 7.0, 7.0]])
    estimator = TurnLevelGRPOAdvantageEstimator(
        base_estimator=_FixedBaseEstimator(base_output),
        config=TurnCreditConfig(enabled=True, turn_weight=0.0),
    )

    actual = estimator.compute_advantage(
        prompt_ids=torch.tensor([[1]]),
        rewards=torch.tensor([1.0]),
        mask=torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0]]),
        repeated_batch=_repeated_batch(),
    )

    assert actual.data_ptr() == base_output.data_ptr()
    assert torch.equal(actual, base_output)


def test_zero_turn_weight_still_validates_trace_transport():
    estimator = TurnLevelGRPOAdvantageEstimator(
        base_estimator=_FixedBaseEstimator(torch.ones((1, 5))),
        config=TurnCreditConfig(enabled=True, turn_weight=0.0),
    )

    with pytest.raises(ValueError, match="missing fields"):
        estimator.compute_advantage(
            prompt_ids=torch.tensor([[1]]),
            rewards=torch.tensor([1.0]),
            mask=torch.ones((1, 5)),
            repeated_batch={},
        )


def test_composes_macro_and_turn_credit_on_generated_spans():
    estimator = TurnLevelGRPOAdvantageEstimator(
        base_estimator=_FixedBaseEstimator(torch.ones((1, 5))),
        config=TurnCreditConfig(
            enabled=True,
            macro_weight=1.0,
            turn_weight=2.0,
        ),
    )

    actual = estimator.compute_advantage(
        prompt_ids=torch.tensor([[1]]),
        rewards=torch.tensor([1.0]),
        mask=torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0]]),
        repeated_batch=_repeated_batch(),
    )

    assert actual.tolist() == [[0.0, 2.0, 2.0, 0.0, 0.5]]


def test_fractional_sample_multiplier_is_applied_only_by_the_loss():
    estimator = TurnLevelGRPOAdvantageEstimator(
        base_estimator=_FixedBaseEstimator(torch.ones((1, 5))),
        config=TurnCreditConfig(
            enabled=True,
            macro_weight=1.0,
            turn_weight=2.0,
        ),
    )

    actual = estimator.compute_advantage(
        prompt_ids=torch.tensor([[1]]),
        rewards=torch.tensor([1.0]),
        mask=torch.tensor([[0.0, 0.5, 0.5, 0.0, 0.5]]),
        repeated_batch=_repeated_batch(),
    )

    assert actual.tolist() == [[0.0, 2.0, 2.0, 0.0, 0.5]]


def test_return_to_go_changes_earlier_turn_only():
    estimator = TurnLevelGRPOAdvantageEstimator(
        base_estimator=_FixedBaseEstimator(torch.zeros((1, 5))),
        config=TurnCreditConfig(
            enabled=True,
            environment_mode="return_to_go",
            discount=0.5,
            turn_weight=1.0,
        ),
    )

    actual = estimator.compute_advantage(
        prompt_ids=torch.tensor([[1]]),
        rewards=torch.tensor([1.0]),
        mask=torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0]]),
        repeated_batch=_repeated_batch(),
    )

    assert actual.tolist() == [[0.0, 0.375, 0.375, 0.0, -0.25]]
