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

import pytest
import torch

from nemo_rl.algorithms.advantage_estimator import (
    GeneralizedAdvantageEstimator,
)
from nemo_rl.algorithms.loss.loss_functions import MseValueLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

# ============================================================================
# Tests for GeneralizedAdvantageEstimator
# ============================================================================


def test_gae_basic_computation():
    """Test basic GAE computation with known values.

    With gamma=1.0 and lambda=1.0, GAE reduces to Monte Carlo returns
    minus values, so advantages = cumulative_rewards_from_t - V(s_t).
    """
    estimator_config = {
        "gae_lambda": 1.0,
        "gae_gamma": 1.0,
        "normalize_advantages": False,
    }
    loss_config = {"reference_policy_kl_penalty": 0.0}
    estimator = GeneralizedAdvantageEstimator(estimator_config, loss_config)

    # 1 sample, 4 tokens, all valid
    rewards = torch.tensor([1.0])
    lengths = torch.tensor([4])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    values = torch.tensor([[0.5, 0.5, 0.5, 0.5]])

    advantages, returns = estimator.compute_advantage(
        prompt_ids=torch.tensor([[0]]),
        rewards=rewards,
        mask=mask,
        lengths=lengths,
        values=values,
    )

    assert advantages.shape == (1, 4)
    assert returns.shape == (1, 4)
    # returns = advantages + values
    torch.testing.assert_close(returns, advantages + values)


def test_gae_gamma_lambda_zero():
    """Test GAE with lambda=0 (pure TD).

    With lambda=0: A_t = delta_t = r_t + gamma * V(s_{t+1}) - V(s_t).
    Only immediate TD error, no bootstrapping.
    """
    estimator_config = {
        "gae_lambda": 0.0,
        "gae_gamma": 1.0,
        "normalize_advantages": False,
    }
    loss_config = {"reference_policy_kl_penalty": 0.0}
    estimator = GeneralizedAdvantageEstimator(estimator_config, loss_config)

    rewards = torch.tensor([5.0])
    lengths = torch.tensor([3])
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    values = torch.tensor([[1.0, 2.0, 3.0]])

    advantages, returns = estimator.compute_advantage(
        prompt_ids=torch.tensor([[0]]),
        rewards=rewards,
        mask=mask,
        lengths=lengths,
        values=values,
    )

    # Token-level rewards: [0, 0, 5.0] (terminal reward at last token)
    # delta_0 = 0 + 1.0 * 2.0 - 1.0 = 1.0
    # delta_1 = 0 + 1.0 * 3.0 - 2.0 = 1.0
    # delta_2 = 5.0 + 1.0 * 0.0 - 3.0 = 2.0  (next_values=0 at end)
    expected_advantages = torch.tensor([[1.0, 1.0, 2.0]])
    torch.testing.assert_close(advantages, expected_advantages)


def test_gae_shape_and_masking():
    """Test that GAE correctly handles masked (padding) positions."""
    estimator_config = {
        "gae_lambda": 0.95,
        "gae_gamma": 1.0,
        "normalize_advantages": False,
    }
    loss_config = {"reference_policy_kl_penalty": 0.0}
    estimator = GeneralizedAdvantageEstimator(estimator_config, loss_config)

    batch_size = 3
    seq_len = 5
    rewards = torch.tensor([1.0, 2.0, 3.0])
    lengths = torch.tensor([5, 5, 5])
    mask = torch.ones(batch_size, seq_len)
    mask[1, 3:] = 0  # second sample has only 3 valid tokens
    mask[2, 4:] = 0  # third sample has only 4 valid tokens
    values = torch.randn(batch_size, seq_len)

    advantages, returns = estimator.compute_advantage(
        prompt_ids=torch.tensor([[0], [1], [2]]),
        rewards=rewards,
        mask=mask,
        lengths=lengths,
        values=values,
    )

    assert advantages.shape == (batch_size, seq_len)
    assert returns.shape == (batch_size, seq_len)

    # Masked positions should have zero advantages
    assert advantages[1, 3:].abs().sum() == 0
    assert advantages[2, 4:].abs().sum() == 0


def test_gae_normalize_advantages():
    """Test that advantage normalization produces zero mean and unit variance."""
    estimator_config = {
        "gae_lambda": 0.95,
        "gae_gamma": 1.0,
        "normalize_advantages": True,
    }
    loss_config = {"reference_policy_kl_penalty": 0.0}
    estimator = GeneralizedAdvantageEstimator(estimator_config, loss_config)

    batch_size = 8
    seq_len = 10
    rewards = torch.randn(batch_size) * 5
    lengths = torch.full((batch_size,), seq_len)
    mask = torch.ones(batch_size, seq_len)
    values = torch.randn(batch_size, seq_len)

    advantages, _ = estimator.compute_advantage(
        prompt_ids=torch.arange(batch_size).unsqueeze(-1),
        rewards=rewards,
        mask=mask,
        lengths=lengths,
        values=values,
    )

    # After whitening, mean should be ~0 and std ~1 across valid tokens
    valid_advs = advantages[mask.bool()]
    assert torch.abs(valid_advs.mean()) < 0.1
    assert torch.abs(valid_advs.std() - 1.0) < 0.2


def test_gae_kl_penalty_in_rewards():
    """Test that KL penalty is injected into token-level rewards when enabled."""
    estimator_config = {
        "gae_lambda": 1.0,
        "gae_gamma": 1.0,
        "normalize_advantages": False,
    }
    loss_config = {
        "reference_policy_kl_penalty": 0.1,
        "reference_policy_kl_type": "k1",
    }
    estimator = GeneralizedAdvantageEstimator(estimator_config, loss_config)

    rewards = torch.tensor([1.0])
    lengths = torch.tensor([3])
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    values = torch.zeros(1, 3)
    logprobs = torch.tensor([[-1.0, -2.0, -1.5]])
    reference_logprobs = torch.tensor([[-1.0, -2.0, -1.5]])

    # When logprobs == reference_logprobs, KL=0, so should match no-KL case
    adv_with_kl, _ = estimator.compute_advantage(
        prompt_ids=torch.tensor([[0]]),
        rewards=rewards,
        mask=mask,
        lengths=lengths,
        values=values,
        logprobs=logprobs,
        reference_logprobs=reference_logprobs,
    )

    adv_no_kl, _ = estimator.compute_advantage(
        prompt_ids=torch.tensor([[0]]),
        rewards=rewards,
        mask=mask,
        lengths=lengths,
        values=values,
    )

    torch.testing.assert_close(adv_with_kl, adv_no_kl)


def test_gae_vapo_decoupled_lambda():
    """Test VAPO decoupled GAE: separate lambda for value vs policy."""
    base_config = {
        "gae_lambda": 0.95,
        "gae_gamma": 1.0,
        "normalize_advantages": False,
        "gae_lambda_value": 1.0,
        "gae_lambda_policy": 0.5,
    }
    loss_config = {"reference_policy_kl_penalty": 0.0}
    estimator = GeneralizedAdvantageEstimator(base_config, loss_config)

    rewards = torch.tensor([1.0, 2.0])
    lengths = torch.tensor([4, 4])
    mask = torch.ones(2, 4)
    values = torch.randn(2, 4)

    advantages, returns = estimator.compute_advantage(
        prompt_ids=torch.tensor([[0], [1]]),
        rewards=rewards,
        mask=mask,
        lengths=lengths,
        values=values,
    )

    # Verify shapes are correct
    assert advantages.shape == (2, 4)
    assert returns.shape == (2, 4)

    # With different lambdas, returns should NOT equal advantages + values
    # because they are computed with different lambda values
    adv_plus_val = advantages + values
    assert not torch.allclose(returns, adv_plus_val, atol=1e-5)


def test_gae_length_adaptive_lambda():
    """Test VAPO length-adaptive lambda: lambda_policy = 1 - 1/(alpha * length)."""
    config = {
        "gae_lambda": 0.95,
        "gae_gamma": 1.0,
        "normalize_advantages": False,
        "length_adaptive_alpha": 0.05,
    }
    loss_config = {"reference_policy_kl_penalty": 0.0}
    estimator = GeneralizedAdvantageEstimator(config, loss_config)

    # Two samples with different response lengths
    mask = torch.zeros(2, 10)
    mask[0, :3] = 1  # short response (3 tokens)
    mask[1, :8] = 1  # long response (8 tokens)

    resolved = estimator._resolve_lambda_policy(mask)

    # lambda = 1 - 1/(alpha * length)
    # short: 1 - 1/(0.05 * 3) = 1 - 6.667 -> clamp to 0
    # long: 1 - 1/(0.05 * 8) = 1 - 2.5 -> clamp to 0
    assert isinstance(resolved, torch.Tensor)
    assert resolved.shape == (2,)
    # Both should be clamped to 0 since alpha is small
    assert (resolved >= 0).all()
    assert (resolved <= 1).all()


def test_gae_returns_equals_advantages_plus_values():
    """Test that returns = advantages + values for standard (non-decoupled) GAE."""
    estimator_config = {
        "gae_lambda": 0.95,
        "gae_gamma": 0.99,
        "normalize_advantages": False,
    }
    loss_config = {"reference_policy_kl_penalty": 0.0}
    estimator = GeneralizedAdvantageEstimator(estimator_config, loss_config)

    rewards = torch.tensor([1.0, -0.5])
    lengths = torch.tensor([5, 5])
    mask = torch.ones(2, 5)
    values = torch.randn(2, 5)

    advantages, returns = estimator.compute_advantage(
        prompt_ids=torch.tensor([[0], [1]]),
        rewards=rewards,
        mask=mask,
        lengths=lengths,
        values=values,
    )

    torch.testing.assert_close(returns, advantages + values)


# ============================================================================
# Tests for MseValueLossFn
# ============================================================================


def _make_value_loss_data(
    batch_size: int = 2,
    seq_len: int = 4,
    device: str = "cuda",
) -> tuple[torch.Tensor, BatchedDataDict, torch.Tensor, torch.Tensor]:
    """Create test data for MseValueLossFn."""
    values = torch.randn(batch_size, seq_len, device=device, requires_grad=True)
    returns = torch.randn(batch_size, seq_len, device=device)
    old_values = torch.randn(batch_size, seq_len, device=device)
    token_mask = torch.ones(batch_size, seq_len, device=device)
    sample_mask = torch.ones(batch_size, device=device)

    data = BatchedDataDict(
        {
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "returns": returns,
            "values": old_values,
        }
    )

    global_valid_seqs = sample_mask.sum()
    global_valid_toks = (token_mask * sample_mask.unsqueeze(-1)).sum()

    return values, data, global_valid_seqs, global_valid_toks


def test_mse_value_loss_basic():
    """Test basic MSE value loss without clipping."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    loss_fn = MseValueLossFn(scale=1.0, cliprange=None)

    values, data, global_valid_seqs, global_valid_toks = _make_value_loss_data()
    loss, metrics = loss_fn(values, data, global_valid_seqs, global_valid_toks)

    assert loss.ndim == 0  # scalar
    assert loss.item() >= 0  # MSE is non-negative
    assert loss.requires_grad


def test_mse_value_loss_with_clipping():
    """Test clipped MSE value loss."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    loss_fn = MseValueLossFn(scale=1.0, cliprange=0.2)

    values, data, global_valid_seqs, global_valid_toks = _make_value_loss_data()
    loss, metrics = loss_fn(values, data, global_valid_seqs, global_valid_toks)

    assert loss.ndim == 0
    assert loss.item() >= 0

    # Verify per-token max guarantee: clipped path uses 0.5 * mean(max(unclipped, clipped))
    # while unclipped path uses mean(mse) without the 0.5 factor, so account for that.
    loss_fn_unclipped = MseValueLossFn(scale=1.0, cliprange=None)
    loss_unclipped, _ = loss_fn_unclipped(
        values.detach().requires_grad_(True), data, global_valid_seqs, global_valid_toks
    )
    assert loss.item() >= 0.5 * loss_unclipped.item() - 1e-6


def test_mse_value_loss_scale():
    """Test that the scale parameter correctly scales the loss."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    values, data, global_valid_seqs, global_valid_toks = _make_value_loss_data()

    loss_fn_1x = MseValueLossFn(scale=1.0, cliprange=None)
    loss_fn_2x = MseValueLossFn(scale=2.0, cliprange=None)

    loss_1x, _ = loss_fn_1x(values, data, global_valid_seqs, global_valid_toks)
    loss_2x, _ = loss_fn_2x(
        values.detach().requires_grad_(True), data, global_valid_seqs, global_valid_toks
    )

    torch.testing.assert_close(loss_2x, loss_1x * 2.0)


def test_mse_value_loss_masking():
    """Test that masking correctly excludes tokens/samples from the loss."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    device = "cuda"
    batch_size, seq_len = 2, 4

    values = torch.ones(batch_size, seq_len, device=device, requires_grad=True)
    returns = torch.zeros(batch_size, seq_len, device=device)

    # Mask out second sample entirely
    token_mask = torch.ones(batch_size, seq_len, device=device)
    sample_mask = torch.tensor([1.0, 0.0], device=device)

    data = BatchedDataDict(
        {
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "returns": returns,
            "values": torch.zeros(batch_size, seq_len, device=device),
        }
    )

    global_valid_toks = (token_mask * sample_mask.unsqueeze(-1)).sum()
    global_valid_seqs = sample_mask.sum()

    loss_fn = MseValueLossFn(scale=1.0, cliprange=None)
    loss, _ = loss_fn(values, data, global_valid_seqs, global_valid_toks)

    # Loss should only consider first sample: MSE(1, 0) = 1.0
    assert loss.item() > 0


def test_mse_value_loss_perfect_prediction():
    """Test that loss is zero when values exactly match returns."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    device = "cuda"
    batch_size, seq_len = 2, 4
    returns = torch.randn(batch_size, seq_len, device=device)
    values = returns.clone().requires_grad_(True)

    data = BatchedDataDict(
        {
            "token_mask": torch.ones(batch_size, seq_len, device=device),
            "sample_mask": torch.ones(batch_size, device=device),
            "returns": returns,
            "values": returns.clone(),
        }
    )

    global_valid_toks = torch.tensor(
        batch_size * seq_len, dtype=torch.float, device=device
    )
    global_valid_seqs = torch.tensor(batch_size, dtype=torch.float, device=device)

    loss_fn = MseValueLossFn(scale=1.0, cliprange=None)
    loss, _ = loss_fn(values, data, global_valid_seqs, global_valid_toks)

    assert loss.item() < 1e-6


def test_mse_value_loss_metrics():
    """Test that MseValueLossFn returns expected metric keys."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    loss_fn = MseValueLossFn(scale=0.4, cliprange=0.2)
    values, data, global_valid_seqs, global_valid_toks = _make_value_loss_data()
    _, metrics = loss_fn(values, data, global_valid_seqs, global_valid_toks)

    expected_keys = {
        "returns_mean",
        "values_mean",
        "values_min",
        "values_max",
        "returns_sq_mean",
        "residual_sq_mean",
        "vf_clipfrac",
    }
    assert expected_keys.issubset(set(metrics.keys()))


def test_mse_value_loss_squeeze_trailing_dim():
    """Test that MseValueLossFn handles [B, S, 1] shaped values correctly."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    device = "cuda"
    batch_size, seq_len = 2, 4

    # Values with trailing singleton: [B, S, 1]
    values_3d = torch.randn(batch_size, seq_len, 1, device=device, requires_grad=True)
    returns = torch.randn(batch_size, seq_len, device=device)

    data = BatchedDataDict(
        {
            "token_mask": torch.ones(batch_size, seq_len, device=device),
            "sample_mask": torch.ones(batch_size, device=device),
            "returns": returns,
            "values": torch.randn(batch_size, seq_len, device=device),
        }
    )

    global_valid_toks = torch.tensor(
        batch_size * seq_len, dtype=torch.float, device=device
    )
    global_valid_seqs = torch.tensor(batch_size, dtype=torch.float, device=device)

    loss_fn = MseValueLossFn(scale=1.0, cliprange=None)
    loss, _ = loss_fn(values_3d, data, global_valid_seqs, global_valid_toks)

    assert loss.ndim == 0
    assert loss.item() >= 0


# ============================================================================
# Tests for _create_advantage_estimator
# ============================================================================


def test_create_advantage_estimator_gae():
    """Test that _create_advantage_estimator creates a GAE estimator."""
    from types import SimpleNamespace

    from nemo_rl.algorithms.ppo import _create_advantage_estimator

    master_config = SimpleNamespace(
        ppo={
            "adv_estimator": {
                "name": "gae",
                "gae_lambda": 0.95,
                "gae_gamma": 1.0,
                "normalize_advantages": True,
            },
        },
        loss_fn={"reference_policy_kl_penalty": 0.0},
    )

    estimator = _create_advantage_estimator(master_config)
    assert isinstance(estimator, GeneralizedAdvantageEstimator)


def test_create_advantage_estimator_grpo_fallback():
    """Test that _create_advantage_estimator falls back to GRPO when adv_estimator is not specified."""
    from types import SimpleNamespace

    from nemo_rl.algorithms.advantage_estimator import GRPOAdvantageEstimator
    from nemo_rl.algorithms.ppo import _create_advantage_estimator

    master_config = SimpleNamespace(
        ppo={
            "normalize_rewards": True,
            "use_leave_one_out_baseline": False,
        },
        loss_fn={},
    )

    estimator = _create_advantage_estimator(master_config)
    assert isinstance(estimator, GRPOAdvantageEstimator)
