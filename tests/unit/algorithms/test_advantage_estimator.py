# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import inspect

import pytest
import torch

from nemo_rl.algorithms.advantage_estimator import (
    AdvantageEstimator,
    GDPOAdvantageEstimator,
    GeneralizedAdvantageEstimator,
    GRPOAdvantageEstimator,
    OPDAdvantageEstimator,
    RawRewardAdvantageEstimator,
    ReinforcePlusPlusAdvantageEstimator,
)


def _make_estimator():
    return OPDAdvantageEstimator({"name": "opd"}, {})


# Every estimator either loop can construct. The loops pass the union of what
# all of them need and rely on ``**kwargs`` to absorb the rest, so an estimator
# that *requires* anything outside the contract cannot be swapped in.
ALL_ESTIMATORS = [
    GRPOAdvantageEstimator,
    GDPOAdvantageEstimator,
    OPDAdvantageEstimator,
    ReinforcePlusPlusAdvantageEstimator,
    GeneralizedAdvantageEstimator,
    RawRewardAdvantageEstimator,
]
CONTRACT_ARGS = ("prompt_ids", "rewards", "mask")


@pytest.mark.parametrize("estimator_cls", ALL_ESTIMATORS, ids=lambda c: c.__name__)
def test_estimator_requires_nothing_beyond_the_contract(estimator_cls):
    """An estimator may accept extra arguments, but may not require them.

    ``runtime_checkable`` only checks that ``compute_advantage`` exists, which
    every one of these passes even with an incompatible signature -- so check
    the signature directly.
    """
    signature = inspect.signature(estimator_cls.compute_advantage)

    extra_required = sorted(
        name
        for name, parameter in signature.parameters.items()
        if name not in CONTRACT_ARGS
        and name != "self"
        and parameter.default is inspect.Parameter.empty
        and parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
    )
    assert not extra_required, (
        f"{estimator_cls.__name__}.compute_advantage requires {extra_required}, "
        "which the algorithm loops do not promise to every estimator"
    )

    # And the contract arguments must bind by keyword, which is how it is called.
    signature.bind(object(), **dict.fromkeys(CONTRACT_ARGS, object()))


@pytest.mark.parametrize("estimator_cls", ALL_ESTIMATORS, ids=lambda c: c.__name__)
def test_estimator_is_structurally_an_advantage_estimator(estimator_cls):
    assert issubclass(estimator_cls, AdvantageEstimator)


def test_opd_basic_positive_distill_advantage():
    """teacher_lp > student_lp => positive advantages."""
    estimator = _make_estimator()
    B, S = 2, 4
    teacher_lp = torch.zeros(B, S)  # log(1) = 0
    student_lp = torch.full((B, S), -1.0)  # lower logprob
    mask = torch.ones(B, S)
    prompt_ids = torch.arange(B)
    rewards = torch.zeros(B)

    adv = estimator.compute_advantage(
        prompt_ids, rewards, mask, teacher_logprobs=teacher_lp, prev_logprobs=student_lp
    ).advantages

    assert adv.shape == (B, S)
    assert (adv > 0).all(), "teacher_lp > student_lp should yield positive advantages"


def test_opd_teacher_equals_student():
    """Same logprobs => zero advantages."""
    estimator = _make_estimator()
    B, S = 2, 4
    logprobs = torch.randn(B, S)
    mask = torch.ones(B, S)
    prompt_ids = torch.arange(B)
    rewards = torch.zeros(B)

    adv = estimator.compute_advantage(
        prompt_ids, rewards, mask, teacher_logprobs=logprobs, prev_logprobs=logprobs
    ).advantages

    torch.testing.assert_close(adv, torch.zeros(B, S))


def test_opd_mask_applied():
    """Masked tokens should have zero advantage."""
    estimator = _make_estimator()
    B, S = 1, 6
    teacher_lp = torch.zeros(B, S)
    student_lp = torch.full((B, S), -1.0)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.float32)
    prompt_ids = torch.arange(B)
    rewards = torch.zeros(B)

    adv = estimator.compute_advantage(
        prompt_ids, rewards, mask, teacher_logprobs=teacher_lp, prev_logprobs=student_lp
    ).advantages

    # Masked positions must be zero
    assert (adv[:, 3:] == 0).all(), "Masked positions should be zero"
    # Unmasked positions should be positive (teacher > student)
    assert (adv[:, :3] > 0).all(), "Unmasked positions should be positive"


def test_opd_metrics_returned():
    """Metrics travel on the returned AdvantageResult."""
    estimator = _make_estimator()
    B, S = 2, 4
    teacher_lp = torch.zeros(B, S)
    student_lp = torch.full((B, S), -1.0)
    mask = torch.ones(B, S)
    prompt_ids = torch.arange(B)
    rewards = torch.zeros(B)

    result = estimator.compute_advantage(
        prompt_ids, rewards, mask, teacher_logprobs=teacher_lp, prev_logprobs=student_lp
    )

    assert "on_policy_distillation/teacher_student_logprob_gap_mean" in result.metrics
    assert "on_policy_distillation/adv_mean" in result.metrics
    assert "on_policy_distillation/adv_std" in result.metrics
    # teacher - student = 0 - (-1) = 1.0
    assert (
        abs(
            result.metrics["on_policy_distillation/teacher_student_logprob_gap_mean"]
            - 1.0
        )
        < 1e-5
    )
    assert abs(result.metrics["on_policy_distillation/adv_mean"] - 1.0) < 1e-5
    assert abs(result.metrics["on_policy_distillation/adv_std"]) < 1e-5
