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
"""The reward-side KL must honour the same clamp config the loss-side KL does.

Both estimators read ``reference_policy_kl_penalty`` and
``reference_policy_kl_type`` off the very ``ClippedPGLossConfig`` that carries
``kl_input_clamp_value`` / ``kl_output_clamp_value``, but called
``calculate_kl`` without them -- so with ``use_kl_in_reward`` the configured
values were silently replaced by the function defaults (20.0 / 10.0).

CPU-only: the estimators are plain tensor math.
"""

from __future__ import annotations

import torch

from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    GAEConfig,
    GeneralizedAdvantageEstimator,
    ReinforcePlusPlusAdvantageEstimator,
)
from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig

B, S = 2, 4
# The reference must diverge by a DIFFERENT amount at each position. Both
# estimators normalize the advantage globally at the end, so a KL that is
# constant across the batch normalizes away to zeros whatever the clamp does --
# only a clamp that reshapes the per-position KL is observable.
_POLICY = torch.full((B, S), -0.5)
_REFERENCE = torch.tensor([[-0.5, -3.0, -30.0, -60.0], [-1.0, -2.0, -15.0, -45.0]])
_MASK = torch.ones(B, S)
_PROMPT_IDS = torch.tensor([0, 0])
_REWARDS = torch.zeros(B)


def _loss_cfg(**over) -> ClippedPGLossConfig:
    return ClippedPGLossConfig(
        use_kl_in_reward=True,
        reference_policy_kl_penalty=1.0,
        reference_policy_kl_type="k1",
        **over,
    )


def _reinforce(loss_cfg):
    return ReinforcePlusPlusAdvantageEstimator(
        AdvEstimatorConfig(name="reinforce_plus_plus", minus_baseline=False),
        loss_cfg,
    )


def _gae(loss_cfg):
    return GeneralizedAdvantageEstimator(GAEConfig(name="gae"), loss_cfg)


def _advantages(result):
    """``compute_advantage`` returns a bare tensor (Reinforce++), a
    ``(advantages, returns)`` tuple (GAE), or an ``AdvantageResult`` once #3512
    lands. Accept all three so this survives that rebase."""
    if hasattr(result, "advantages"):
        return result.advantages
    if isinstance(result, tuple):
        return result[0]
    return result


def _reinforce_advantages(loss_cfg):
    return _reinforce(loss_cfg).compute_advantage(
        _PROMPT_IDS,
        _REWARDS,
        _MASK,
        logprobs_policy=_POLICY,
        logprobs_reference=_REFERENCE,
    )


def _gae_advantages(loss_cfg):
    return _gae(loss_cfg).compute_advantage(
        _PROMPT_IDS,
        _REWARDS,
        _MASK,
        values=torch.zeros(B, S),
        logprobs_policy=_POLICY,
        logprobs_reference=_REFERENCE,
    )


class TestReinforcePlusPlus:
    def test_the_configured_output_clamp_changes_the_reward_kl(self):
        loose = _reinforce_advantages(_loss_cfg(kl_output_clamp_value=None))
        tight = _reinforce_advantages(_loss_cfg(kl_output_clamp_value=1.0))
        assert not torch.allclose(_advantages(loose), _advantages(tight))

    def test_the_configured_input_clamp_changes_the_reward_kl(self):
        wide = _reinforce_advantages(
            _loss_cfg(kl_input_clamp_value=None, kl_output_clamp_value=None)
        )
        narrow = _reinforce_advantages(
            _loss_cfg(kl_input_clamp_value=1.0, kl_output_clamp_value=None)
        )
        assert not torch.allclose(_advantages(wide), _advantages(narrow))

    def test_the_estimator_carries_the_values_off_the_loss_config(self):
        est = _reinforce(_loss_cfg(kl_input_clamp_value=7.0, kl_output_clamp_value=3.0))
        assert est.kl_input_clamp_value == 7.0
        assert est.kl_output_clamp_value == 3.0


class TestGeneralizedAdvantageEstimator:
    def test_the_configured_output_clamp_changes_the_reward_kl(self):
        loose = _gae_advantages(_loss_cfg(kl_output_clamp_value=None))
        tight = _gae_advantages(_loss_cfg(kl_output_clamp_value=1.0))
        assert not torch.allclose(_advantages(loose), _advantages(tight))

    def test_the_estimator_carries_the_values_off_the_loss_config(self):
        est = _gae(_loss_cfg(kl_input_clamp_value=7.0, kl_output_clamp_value=3.0))
        assert est.kl_input_clamp_value == 7.0
        assert est.kl_output_clamp_value == 3.0


def test_both_estimators_agree_with_the_loss_side_defaults():
    """The defaults must keep matching ClippedPGLossConfig's, or the reward KL
    and the loss KL disagree out of the box."""
    cfg = _loss_cfg()
    for est in (_reinforce(cfg), _gae(cfg)):
        assert est.kl_input_clamp_value == cfg.kl_input_clamp_value
        assert est.kl_output_clamp_value == cfg.kl_output_clamp_value
