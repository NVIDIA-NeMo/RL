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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.algorithms.advantage_estimator import (
    GeneralizedAdvantageEstimator,
    RawRewardAdvantageEstimator,
)
from nemo_rl.algorithms.loss.loss_functions import (
    ClippedPGLossConfig,
    MseValueLossConfig,
    MseValueLossFn,
)
from nemo_rl.data import DataConfig
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _make_loss_config(
    kl_penalty: float = 0.0,
    kl_type: str = "k1",
    use_kl_in_reward: bool = False,
) -> ClippedPGLossConfig:
    """Construct a minimal ClippedPGLossConfig for advantage-estimator tests.

    The estimator only reads ``use_kl_in_reward``, ``reference_policy_kl_penalty``
    and ``reference_policy_kl_type``; other fields fall back to model defaults.
    """
    return ClippedPGLossConfig(
        reference_policy_kl_penalty=kl_penalty,
        reference_policy_kl_type=kl_type,
        use_kl_in_reward=use_kl_in_reward,
    )


def _make_gae_config(
    gae_lambda: float = 0.95,
    gae_gamma: float = 1.0,
    normalize_advantages: bool = False,
    length_adaptive_alpha: float = 0.0,
    gae_lambda_value: float | None = None,
    gae_lambda_policy: float | None = None,
    **overrides,
) -> dict:
    """Build an estimator_config dict with all GAE-required keys populated.

    ``GeneralizedAdvantageEstimator.__init__`` requires every field to be
    present (no hidden ``.get()`` defaults). VAPO fields default to ``None``
    (standard GAE, no decoupling) and can be overridden via kwargs.
    """
    return {
        "gae_lambda": gae_lambda,
        "gae_gamma": gae_gamma,
        "normalize_advantages": normalize_advantages,
        "length_adaptive_alpha": length_adaptive_alpha,
        "gae_lambda_value": gae_lambda_value,
        "gae_lambda_policy": gae_lambda_policy,
        **overrides,
    }


# ============================================================================
# Tests for GeneralizedAdvantageEstimator
# ============================================================================


def test_gae_basic_computation():
    """Test basic GAE computation with known values.

    With gamma=1.0 and lambda=1.0, GAE reduces to Monte Carlo returns
    minus values, so advantages = cumulative_rewards_from_t - V(s_t).
    """
    estimator_config = _make_gae_config(gae_lambda=1.0, gae_gamma=1.0)
    loss_config = _make_loss_config(kl_penalty=0.0)
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
    estimator_config = _make_gae_config(gae_lambda=0.0, gae_gamma=1.0)
    loss_config = _make_loss_config(kl_penalty=0.0)
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
    estimator_config = _make_gae_config(gae_lambda=0.95, gae_gamma=1.0)
    loss_config = _make_loss_config(kl_penalty=0.0)
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
    estimator_config = _make_gae_config(
        gae_lambda=0.95, gae_gamma=1.0, normalize_advantages=True
    )
    loss_config = _make_loss_config(kl_penalty=0.0)
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
    """Test KL penalty injection into token-level rewards (gated + applied)."""
    estimator_config = _make_gae_config(gae_lambda=1.0, gae_gamma=1.0)
    loss_config = _make_loss_config(kl_penalty=0.1, kl_type="k1", use_kl_in_reward=True)
    estimator = GeneralizedAdvantageEstimator(estimator_config, loss_config)

    rewards = torch.tensor([1.0])
    lengths = torch.tensor([3])
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    values = torch.zeros(1, 3)
    logprobs = torch.tensor([[-1.0, -2.0, -1.5]])
    reference_logprobs_same = torch.tensor([[-1.0, -2.0, -1.5]])

    # Case 1: logprobs == reference_logprobs → KL=0, advantages should match
    # the no-reference (no-KL) call.
    adv_kl_zero, _ = estimator.compute_advantage(
        prompt_ids=torch.tensor([[0]]),
        rewards=rewards,
        mask=mask,
        lengths=lengths,
        values=values,
        logprobs=logprobs,
        reference_logprobs=reference_logprobs_same,
    )
    adv_no_kl, _ = estimator.compute_advantage(
        prompt_ids=torch.tensor([[0]]),
        rewards=rewards,
        mask=mask,
        lengths=lengths,
        values=values,
    )
    torch.testing.assert_close(adv_kl_zero, adv_no_kl)

    # Case 2: divergent reference_logprobs → KL>0, advantages should differ
    # from the no-KL baseline (the KL penalty subtracts from token rewards).
    reference_logprobs_divergent = torch.tensor([[-3.0, -4.0, -3.5]])
    adv_kl_positive, _ = estimator.compute_advantage(
        prompt_ids=torch.tensor([[0]]),
        rewards=rewards,
        mask=mask,
        lengths=lengths,
        values=values,
        logprobs=logprobs,
        reference_logprobs=reference_logprobs_divergent,
    )
    assert not torch.allclose(adv_kl_positive, adv_no_kl), (
        "Non-zero KL should change advantages relative to the no-KL baseline"
    )

    # Case 3: use_kl_in_reward=False gate → even with divergent reference
    # logprobs, KL must not be applied; advantages should match the no-KL case.
    loss_config_gated_off = _make_loss_config(
        kl_penalty=0.1, kl_type="k1", use_kl_in_reward=False
    )
    estimator_gated_off = GeneralizedAdvantageEstimator(
        estimator_config, loss_config_gated_off
    )
    adv_gate_off, _ = estimator_gated_off.compute_advantage(
        prompt_ids=torch.tensor([[0]]),
        rewards=rewards,
        mask=mask,
        lengths=lengths,
        values=values,
        logprobs=logprobs,
        reference_logprobs=reference_logprobs_divergent,
    )
    torch.testing.assert_close(adv_gate_off, adv_no_kl)


def test_gae_vapo_decoupled_lambda():
    """Test VAPO decoupled GAE: separate lambda for value vs policy."""
    base_config = _make_gae_config(
        gae_lambda=0.95,
        gae_gamma=1.0,
        gae_lambda_value=1.0,
        gae_lambda_policy=0.5,
    )
    loss_config = _make_loss_config(kl_penalty=0.0)
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
    config = _make_gae_config(
        gae_lambda=0.95, gae_gamma=1.0, length_adaptive_alpha=0.05
    )
    loss_config = _make_loss_config(kl_penalty=0.0)
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


def test_gae_carry_forward_interior_gap():
    """Carry-forward masking with interior (non-trailing) gaps.

    Verifies that a masked position in the middle of a response does not
    corrupt the GAE accumulators: the advantages at the valid tokens must
    match the case where the gap is simply removed from the sequence.
    """
    estimator_config = _make_gae_config(gae_lambda=0.95, gae_gamma=0.99)
    loss_config = _make_loss_config(kl_penalty=0.0)
    estimator = GeneralizedAdvantageEstimator(estimator_config, loss_config)

    # 4 valid response tokens, with a gap at position 2 (e.g. a non-assistant
    # turn or a separator). The value at the gap is deliberately huge (999)
    # to make any leakage into the accumulators obvious.
    rewards = torch.tensor([1.0])
    mask_with_gap = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0]])
    values_with_gap = torch.tensor([[0.1, 0.2, 999.0, 0.4, 0.5]])

    adv_with_gap, _ = estimator.compute_advantage(
        prompt_ids=torch.tensor([[0]]),
        rewards=rewards,
        mask=mask_with_gap,
        values=values_with_gap,
    )

    # Same sequence with the gap collapsed out (4 contiguous valid tokens).
    mask_no_gap = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    values_no_gap = torch.tensor([[0.1, 0.2, 0.4, 0.5]])
    adv_no_gap, _ = estimator.compute_advantage(
        prompt_ids=torch.tensor([[0]]),
        rewards=rewards,
        mask=mask_no_gap,
        values=values_no_gap,
    )

    # Valid-token advantages must match: the carry-forward must skip over the
    # masked position rather than propagate the 999 placeholder.
    valid_positions = mask_with_gap[0].bool()
    torch.testing.assert_close(adv_with_gap[0][valid_positions], adv_no_gap[0])
    # Masked position itself should be zero (final mask multiply).
    assert adv_with_gap[0, 2].abs() == 0


# ============================================================================
# Tests for RawRewardAdvantageEstimator
# ============================================================================


def test_raw_reward_basic_broadcast_and_masking():
    """RawRewardAdvantageEstimator broadcasts scalar reward across the response.

    Validates the no-value-model path: advantages are the per-sample reward
    expanded to ``mask.shape`` (broadcast at every position, including
    masked-out trailing pads — the downstream loss masking is responsible
    for zeroing those out). ``returns`` is None since there is no value head.
    """
    estimator_config = {"normalize_advantages": False}
    loss_config = _make_loss_config(kl_penalty=0.0)
    estimator = RawRewardAdvantageEstimator(estimator_config, loss_config)

    rewards = torch.tensor([2.0, -1.0])
    # Sample 0 has 3 valid tokens, sample 1 has 2 valid tokens.
    mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )

    advantages, returns = estimator.compute_advantage(
        prompt_ids=torch.tensor([[0], [1]]),
        rewards=rewards,
        mask=mask,
    )

    expected = torch.tensor(
        [
            [2.0, 2.0, 2.0, 2.0],
            [-1.0, -1.0, -1.0, -1.0],
        ]
    )
    torch.testing.assert_close(advantages, expected)
    # RawReward has no value model, so there are no returns to compute.
    assert returns is None


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

    loss_fn = MseValueLossFn(MseValueLossConfig(scale=1.0, cliprange=None))

    values, data, global_valid_seqs, global_valid_toks = _make_value_loss_data()
    loss, metrics = loss_fn(values, data, global_valid_seqs, global_valid_toks)

    assert loss.ndim == 0  # scalar
    assert loss.item() >= 0  # MSE is non-negative
    assert loss.requires_grad


def test_mse_value_loss_with_clipping():
    """Test clipped MSE value loss."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    loss_fn = MseValueLossFn(MseValueLossConfig(scale=1.0, cliprange=0.2))

    values, data, global_valid_seqs, global_valid_toks = _make_value_loss_data()
    loss, metrics = loss_fn(values, data, global_valid_seqs, global_valid_toks)

    assert loss.ndim == 0
    assert loss.item() >= 0

    # Per-token max guarantee: both branches apply the same 0.5*scale prefactor,
    # so clipped loss must be >= unclipped loss because of torch.max. This
    # would fail if someone refactored the clipped branch to drop torch.max.
    loss_fn_unclipped = MseValueLossFn(MseValueLossConfig(scale=1.0, cliprange=None))
    loss_unclipped, _ = loss_fn_unclipped(
        values.detach().requires_grad_(True), data, global_valid_seqs, global_valid_toks
    )
    assert loss.item() >= loss_unclipped.item() - 1e-6


def test_mse_value_loss_scale():
    """Test that the scale parameter correctly scales the loss."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    values, data, global_valid_seqs, global_valid_toks = _make_value_loss_data()

    loss_fn_1x = MseValueLossFn(MseValueLossConfig(scale=1.0, cliprange=None))
    loss_fn_2x = MseValueLossFn(MseValueLossConfig(scale=2.0, cliprange=None))

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

    loss_fn = MseValueLossFn(MseValueLossConfig(scale=1.0, cliprange=None))
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

    loss_fn = MseValueLossFn(MseValueLossConfig(scale=1.0, cliprange=None))
    loss, _ = loss_fn(values, data, global_valid_seqs, global_valid_toks)

    assert loss.item() < 1e-6


def test_mse_value_loss_metrics():
    """Test that MseValueLossFn returns expected metric keys."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    loss_fn = MseValueLossFn(MseValueLossConfig(scale=0.4, cliprange=0.2))
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

    loss_fn = MseValueLossFn(MseValueLossConfig(scale=1.0, cliprange=None))
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

    # adv_estimator dict needs every GAE-required key (no hidden .get() defaults
    # in the estimator __init__); loss_fn must be a real ClippedPGLossConfig
    # because the estimator accesses .use_kl_in_reward / .reference_policy_kl_*
    # as attributes, not dict keys.
    master_config = SimpleNamespace(
        ppo={
            "adv_estimator": {
                "name": "gae",
                **_make_gae_config(
                    gae_lambda=0.95, gae_gamma=1.0, normalize_advantages=True
                ),
            },
        },
        loss_fn=_make_loss_config(kl_penalty=0.0),
    )

    estimator = _create_advantage_estimator(master_config)
    assert isinstance(estimator, GeneralizedAdvantageEstimator)


def test_create_advantage_estimator_raw_reward():
    """PPO factory accepts raw_reward (other group-relative estimators are rejected)."""
    from types import SimpleNamespace

    from nemo_rl.algorithms.advantage_estimator import RawRewardAdvantageEstimator
    from nemo_rl.algorithms.ppo import _create_advantage_estimator

    master_config = SimpleNamespace(
        ppo={
            "adv_estimator": {"name": "raw_reward", "normalize_advantages": True},
        },
        loss_fn={"reference_policy_kl_penalty": 0.0},
    )

    estimator = _create_advantage_estimator(master_config)
    assert isinstance(estimator, RawRewardAdvantageEstimator)


def test_create_advantage_estimator_rejects_unsupported_name():
    """PPO loop only consumes (advantages, returns) from GAE/raw_reward — others must error."""
    from types import SimpleNamespace

    from nemo_rl.algorithms.ppo import _create_advantage_estimator

    master_config = SimpleNamespace(
        ppo={"adv_estimator": {"name": "grpo"}},
        loss_fn={"reference_policy_kl_penalty": 0.0},
    )

    with pytest.raises(ValueError, match="only supports 'gae' or 'raw_reward'"):
        _create_advantage_estimator(master_config)


def test_create_advantage_estimator_requires_adv_estimator_key():
    """No more silent default — missing `adv_estimator` should KeyError."""
    from types import SimpleNamespace

    from nemo_rl.algorithms.ppo import _create_advantage_estimator

    master_config = SimpleNamespace(
        ppo={},
        loss_fn={},
    )

    with pytest.raises(KeyError):
        _create_advantage_estimator(master_config)


# ============================================================================
# Tests for PPO train/generation logprob mismatch handling
# ============================================================================


def _make_logprob_mismatch_data() -> BatchedDataDict:
    return BatchedDataDict(
        {
            "token_mask": torch.ones(3, 4),
            "sample_mask": torch.tensor([1.0, 1.0, 0.0]),
            "prev_logprobs": torch.zeros(3, 4),
            "generation_logprobs": torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.6931472, 0.6931472, 0.6931472],
                    [0.0, 1.0986123, 1.0986123, 1.0986123],
                ]
            ),
        }
    )


def test_ppo_seq_logprob_error_metrics_do_not_mask_without_threshold():
    from nemo_rl.algorithms.ppo import _apply_ppo_seq_logprob_error_masking

    train_data = _make_logprob_mismatch_data()
    advantage_mask, metrics = _apply_ppo_seq_logprob_error_masking(
        train_data=train_data,
        rewards=torch.tensor([0.0, 1.0, 1.0]),
        seq_logprob_error_threshold=None,
    )

    expected_sample_mask = torch.tensor([1.0, 1.0, 0.0])
    torch.testing.assert_close(train_data["sample_mask"], expected_sample_mask)
    torch.testing.assert_close(
        advantage_mask,
        train_data["token_mask"] * expected_sample_mask.unsqueeze(-1),
    )
    assert metrics["num_masked_seqs_by_logprob_error"] == 0
    assert "num_masked_seqs" not in metrics
    assert metrics["max_seq_mult_prob_error"] == pytest.approx(2.0)


def test_ppo_seq_logprob_error_mask_returns_combined_advantage_mask():
    from nemo_rl.algorithms.ppo import _apply_ppo_seq_logprob_error_masking

    train_data = _make_logprob_mismatch_data()
    advantage_mask, metrics = _apply_ppo_seq_logprob_error_masking(
        train_data=train_data,
        rewards=torch.tensor([0.0, 1.0, 1.0]),
        seq_logprob_error_threshold=1.5,
    )

    expected_sample_mask = torch.tensor([1.0, 0.0, 0.0])
    torch.testing.assert_close(train_data["sample_mask"], expected_sample_mask)
    torch.testing.assert_close(
        advantage_mask,
        train_data["token_mask"] * expected_sample_mask.unsqueeze(-1),
    )
    assert metrics["num_masked_seqs_by_logprob_error"] == 1


def test_ppo_seq_logprob_error_mask_rejects_all_masked_batch():
    from nemo_rl.algorithms.ppo import _apply_ppo_seq_logprob_error_masking

    train_data = _make_logprob_mismatch_data()
    with pytest.raises(
        RuntimeError,
        match="no valid response tokens after filtering",
    ):
        _apply_ppo_seq_logprob_error_masking(
            train_data=train_data,
            rewards=torch.tensor([0.0, 1.0, 1.0]),
            seq_logprob_error_threshold=0.5,
        )


def _make_ppo_loop_batch() -> BatchedDataDict:
    return BatchedDataDict(
        {
            "message_log": [
                [
                    {
                        "role": "user",
                        "content": "prompt-0",
                        "token_ids": torch.tensor([1]),
                    },
                    {
                        "role": "assistant",
                        "content": "answer-0",
                        "token_ids": torch.tensor([2, 3]),
                    },
                ],
                [
                    {
                        "role": "user",
                        "content": "prompt-1",
                        "token_ids": torch.tensor([1]),
                    },
                    {
                        "role": "assistant",
                        "content": "answer-1",
                        "token_ids": torch.tensor([4, 5]),
                    },
                ],
            ],
            "total_reward": torch.tensor([0.0, 1.0]),
            "loss_multiplier": torch.ones(2),
            "truncated": torch.zeros(2, dtype=torch.bool),
            "length": torch.ones(2, dtype=torch.int32),
        }
    )


def _run_mock_ppo_train(
    monkeypatch,
    *,
    max_num_steps: int,
    ppo_epochs: int,
    seq_logprob_error_threshold: float | None,
    policy_training_start_step: int = 0,
):
    """Run the real PPO loop with deterministic in-process collaborators."""
    from nemo_rl.algorithms import ppo as ppo_mod

    events: list[str] = []
    generation_logprobs = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.6931472, 0.6931472],
        ]
    )

    def fake_flatten(message_logs, *_args, **_kwargs):
        batch_size = len(message_logs)
        return (
            BatchedDataDict(
                {
                    "token_ids": torch.tensor([[1, 2, 3], [1, 4, 5]])[:batch_size],
                    "generation_logprobs": generation_logprobs[:batch_size].clone(),
                    "token_loss_mask": torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])[
                        :batch_size
                    ],
                    "content": [["prompt", "answer"] for _ in range(batch_size)],
                }
            ),
            torch.full((batch_size,), 3, dtype=torch.int32),
        )

    class DummyAdvantageEstimator:
        def __init__(self):
            self.masks = []

        def compute_advantage(self, **kwargs):
            mask = kwargs["mask"].clone()
            self.masks.append(mask)
            return mask.clone(), mask.clone()

    class DummyTimer:
        def time(self, *_args, **_kwargs):
            return nullcontext()

        def get_timing_metrics(self, **_kwargs):
            return {"total_step_time": 1.0}

        def reset(self):
            pass

    class DummyTimeoutChecker:
        def __init__(self, *_args, **_kwargs):
            pass

        def start_iterations(self):
            pass

        def mark_iteration(self):
            pass

        def check_save(self):
            return False

    class DummyMemoryTracker:
        def snapshot_start_of_stage(self, *_args, **_kwargs):
            pass

    class DummyLoader:
        def __init__(self, batches):
            self.batches = batches

        def __iter__(self):
            return iter(self.batches)

        def __len__(self):
            return len(self.batches)

    train_result = {
        "loss": torch.tensor([0.1]),
        "grad_norm": torch.tensor([1.0]),
        "all_mb_metrics": {},
    }
    value_result = {
        "loss": torch.tensor([0.2]),
        "grad_norm": torch.tensor([2.0]),
        "all_mb_metrics": {},
    }

    policy = MagicMock()
    policy.prepare_for_lp_inference.side_effect = lambda: events.append("policy_lp")
    policy.offload_to_cpu.side_effect = lambda: events.append("policy_offload")
    policy.prepare_for_training.side_effect = lambda: events.append("policy_train_prep")
    policy.train.side_effect = lambda *_args, **_kwargs: (
        events.append("policy_train") or train_result
    )
    policy.get_logprobs.return_value = {"logprobs": torch.zeros(2, 3)}
    policy.get_reference_policy_logprobs.return_value = {
        "reference_logprobs": torch.zeros(2, 3)
    }

    value_model = MagicMock()
    value_model.finish_training.side_effect = lambda: events.append("value_finish")
    value_model.get_values.return_value = {"values": torch.zeros(2, 3, 1)}
    value_model.train.side_effect = lambda *_args, **_kwargs: (
        events.append("value_train") or value_result
    )

    policy_generation = MagicMock()
    policy_generation.requires_kv_scale_sync = False
    policy_generation.prepare_for_generation.side_effect = lambda: events.append(
        "generation_prepare"
    )
    policy_generation.get_logger_metrics.return_value = {}
    policy_generation.get_step_metrics.return_value = {}

    def fake_rollout(*_args, input_batch, **_kwargs):
        events.append("rollout")
        return input_batch, {"mean_gen_tokens_per_sample": 2.0}

    refit = MagicMock(side_effect=lambda *_args, **_kwargs: events.append("refit"))
    advantage_estimator = DummyAdvantageEstimator()

    monkeypatch.setattr(ppo_mod, "Timer", DummyTimer)
    monkeypatch.setattr(ppo_mod, "TimeoutChecker", DummyTimeoutChecker)
    monkeypatch.setattr(ppo_mod, "MemoryTracker", DummyMemoryTracker)
    monkeypatch.setattr(ppo_mod, "maybe_gpu_profile_step", lambda *_args: None)
    monkeypatch.setattr(ppo_mod, "print_performance_metrics", lambda *_args: {})
    monkeypatch.setattr(ppo_mod, "scale_rewards", lambda batch, _config: batch)
    monkeypatch.setattr(ppo_mod, "_should_use_nemo_gym", lambda _config: False)
    monkeypatch.setattr(ppo_mod, "_should_use_async_rollouts", lambda _config: False)
    monkeypatch.setattr(ppo_mod, "run_multi_turn_rollout", fake_rollout)
    monkeypatch.setattr(ppo_mod, "refit_policy_generation", refit)
    monkeypatch.setattr(ppo_mod, "batched_message_log_to_flat_message", fake_flatten)
    monkeypatch.setattr(
        ppo_mod,
        "extract_initial_prompt_messages",
        lambda message_logs, _lengths: message_logs,
    )
    monkeypatch.setattr(
        ppo_mod,
        "_create_advantage_estimator",
        lambda _config: advantage_estimator,
    )

    master_config = SimpleNamespace(
        ppo={
            "max_num_steps": max_num_steps,
            "max_num_epochs": 1,
            "max_rollout_turns": 1,
            "num_prompts_per_step": 2,
            "num_generations_per_prompt": 1,
            "overlong_filtering": False,
            "policy_training_start_step": policy_training_start_step,
            "ppo_epochs": ppo_epochs,
            "reward_scaling": {"enabled": False},
            "reward_shaping": {"enabled": False},
            "seq_logprob_error_threshold": seq_logprob_error_threshold,
            "val_at_start": False,
            "val_at_end": False,
            "val_period": 0,
        },
        policy={
            "generation": {
                "backend": "vllm",
                "colocated": {"enabled": False},
                "vllm_cfg": {"async_engine": False},
            },
            "max_total_sequence_length": 3,
            "make_sequence_length_divisible_by": 1,
        },
        loss_fn=_make_loss_config(),
        checkpointing={
            "enabled": False,
            "checkpoint_must_save_by": None,
            "save_period": 100,
            "metric_name": None,
        },
        cluster={"num_nodes": 1, "gpus_per_node": 2},
    )

    logger = MagicMock()
    checkpointer = MagicMock()
    checkpointer.save_optimizer = False
    dataloader = DummyLoader([_make_ppo_loop_batch() for _ in range(max_num_steps)])
    tokenizer = SimpleNamespace(pad_token_id=0)

    ppo_mod.ppo_train(
        policy,
        policy_generation,
        value_model,
        dataloader,
        None,
        tokenizer,
        MagicMock(),
        MagicMock(),
        {},
        None,
        logger,
        checkpointer,
        ppo_mod._default_ppo_save_state(),
        master_config,
    )

    return SimpleNamespace(
        policy=policy,
        policy_generation=policy_generation,
        value_model=value_model,
        logger=logger,
        checkpointer=checkpointer,
        advantage_estimator=advantage_estimator,
        refit=refit,
        events=events,
    )


def test_ppo_train_noncolocated_refit_offload_lifecycle(monkeypatch):
    harness = _run_mock_ppo_train(
        monkeypatch,
        max_num_steps=2,
        ppo_epochs=2,
        seq_logprob_error_threshold=None,
    )

    assert harness.refit.call_count == 2
    assert harness.policy.train.call_count == 4
    assert harness.value_model.train.call_count == 4
    assert harness.policy.offload_to_cpu.call_count == 4
    harness.policy_generation.prepare_for_generation.assert_not_called()
    assert harness.policy_generation.finish_generation.call_count == 2

    for call in harness.refit.call_args_list:
        assert call.args[0] is harness.policy
        assert call.args[1] is harness.policy_generation
        assert call.args[2] is False

    refit_indices = [
        index for index, event in enumerate(harness.events) if event == "refit"
    ]
    for index in refit_indices:
        assert harness.events[index - 2 : index + 3] == [
            "value_finish",
            "policy_lp",
            "refit",
            "policy_offload",
            "rollout",
        ]


def test_ppo_train_critic_warmup_reuses_generation_until_policy_update(monkeypatch):
    harness = _run_mock_ppo_train(
        monkeypatch,
        max_num_steps=2,
        ppo_epochs=1,
        seq_logprob_error_threshold=None,
        policy_training_start_step=1,
    )

    assert harness.refit.call_count == 1
    assert harness.policy_generation.prepare_for_generation.call_count == 1
    assert harness.policy.train.call_count == 1
    assert harness.value_model.train.call_count == 2
    assert harness.policy_generation.finish_generation.call_count == 2

    rollout_indices = [
        index for index, event in enumerate(harness.events) if event == "rollout"
    ]
    assert harness.events[rollout_indices[1] - 1 : rollout_indices[1] + 1] == [
        "generation_prepare",
        "rollout",
    ]


def test_ppo_train_wires_logprob_mask_to_advantage_training_and_metrics(monkeypatch):
    harness = _run_mock_ppo_train(
        monkeypatch,
        max_num_steps=1,
        ppo_epochs=1,
        seq_logprob_error_threshold=1.5,
    )

    expected_sample_mask = torch.tensor([1.0, 0.0])
    expected_advantage_mask = torch.tensor([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    torch.testing.assert_close(
        harness.advantage_estimator.masks[0], expected_advantage_mask
    )

    policy_train_data = harness.policy.train.call_args.args[0]
    value_train_data = harness.value_model.train.call_args.args[0]
    torch.testing.assert_close(policy_train_data["sample_mask"], expected_sample_mask)
    torch.testing.assert_close(value_train_data["sample_mask"], expected_sample_mask)

    final_train_metrics = [
        call.args[0]
        for call in harness.logger.log_metrics.call_args_list
        if call.kwargs.get("prefix") == "train"
        and "num_masked_seqs_by_logprob_error" in call.args[0]
    ]
    assert len(final_train_metrics) == 1
    assert final_train_metrics[0]["num_masked_seqs_by_logprob_error"] == 1
    assert final_train_metrics[0]["max_seq_mult_prob_error"] == pytest.approx(2.0)
    assert final_train_metrics[0]["advantages/mean"] == pytest.approx(1.0)
    assert final_train_metrics[0]["advantages/min"] == pytest.approx(1.0)
    assert final_train_metrics[0]["advantages/max"] == pytest.approx(1.0)


# ============================================================================
# Tests for non-colocated setup
# ============================================================================


def _make_noncolocated_setup_config(
    *,
    backend: str = "vllm",
    total_nodes: int = 1,
    total_gpus_per_node: int = 8,
    inference_nodes: int | None = None,
    inference_gpus_per_node: int | None = 2,
    reward_model_gpus_per_node: int | None = None,
    segment_size: int | None = None,
    tensor_parallel_size: int = 1,
):
    """Build the minimal config needed to exercise PPO cluster setup."""
    from nemo_rl.algorithms.ppo import MasterConfig

    data_config: DataConfig = {
        "max_input_seq_length": 1,
        "shuffle": False,
        "num_workers": 0,
        "train": {"dataset_name": "fake-dataset"},
    }
    env_config = {}
    if reward_model_gpus_per_node is not None:
        data_config["train"] = [
            {"dataset_name": "fake-dataset", "env_name": "reward_model"}
        ]
        env_config["reward_model"] = {
            "resources": {
                "num_nodes": 1,
                "gpus_per_node": reward_model_gpus_per_node,
            }
        }

    return MasterConfig.model_construct(
        policy={
            "model_name": "fake-model",
            "train_global_batch_size": 1,
            "train_micro_batch_size": 1,
            "dtensor_cfg": {"enabled": True},
            "megatron_cfg": {"enabled": False},
            "generation": {
                "backend": backend,
                "colocated": {
                    "enabled": False,
                    "resources": {
                        "num_nodes": inference_nodes,
                        "gpus_per_node": inference_gpus_per_node,
                    },
                },
                "vllm_cfg": {
                    "precision": "bf16",
                    "kv_cache_dtype": "auto",
                    "tensor_parallel_size": tensor_parallel_size,
                    "pipeline_parallel_size": 1,
                },
                "vllm_kwargs": {},
                "sglang_cfg": {},
            },
        },
        value={
            "megatron_cfg": {
                "enabled": True,
                "context_parallel_size": 1,
            },
            "sequence_packing": {"enabled": False},
        },
        loss_fn=ClippedPGLossConfig(),
        value_loss_fn=MseValueLossConfig(),
        env=env_config,
        data=data_config,
        ppo={
            "max_num_steps": 1,
            "max_num_epochs": 1,
            "num_prompts_per_step": 1,
            "num_generations_per_prompt": 1,
            "max_rollout_turns": 1,
            "val_period": 0,
            "val_batch_size": 1,
            "val_at_start": False,
            "val_at_end": False,
            "max_val_samples": 1,
            "seed": 42,
            "overlong_filtering": False,
            "use_dynamic_sampling": False,
            "batch_multiplier": 1,
            "ppo_epochs": 1,
            "policy_training_start_step": 0,
            "reward_shaping": {"enabled": False},
            "reward_scaling": {"enabled": False},
            "adv_estimator": {"name": "raw_reward"},
        },
        logger={"num_val_samples_to_print": 0},
        cluster={
            "num_nodes": total_nodes,
            "gpus_per_node": total_gpus_per_node,
            "segment_size": segment_size,
        },
        checkpointing={
            "enabled": False,
            "save_optimizer": False,
        },
    )


def _patch_ppo_setup_prerequisites(monkeypatch):
    """Replace setup dependencies that are unrelated to resource validation."""
    from nemo_rl.algorithms import ppo as ppo_mod

    class DummyLogger:
        def log_hyperparams(self, *_args, **_kwargs):
            pass

        def log_metrics(self, *_args, **_kwargs):
            pass

    class DummyCheckpointer:
        def get_latest_checkpoint_path(self):
            return None

        def load_training_info(self, _path):
            return None

        def get_resume_paths(self, _path, *, model_component="policy"):
            return None, None

    class DummyLoader:
        def __init__(self, *_args, **_kwargs):
            pass

        def __len__(self):
            return 1

    monkeypatch.setattr(ppo_mod, "Logger", lambda *_args, **_kwargs: DummyLogger())
    monkeypatch.setattr(
        ppo_mod,
        "CheckpointManager",
        lambda *_args, **_kwargs: DummyCheckpointer(),
    )
    monkeypatch.setattr(ppo_mod, "StatefulDataLoader", DummyLoader)
    return ppo_mod


def _setup_dataset():
    from unittest.mock import MagicMock

    dataset = MagicMock()
    dataset.__len__.return_value = 1
    return dataset


def _run_noncolocated_setup(monkeypatch, config):
    """Run setup with lightweight workers and return the observable topology."""
    from unittest.mock import MagicMock

    ppo_mod = _patch_ppo_setup_prerequisites(monkeypatch)
    cluster_calls = []

    class DummyCluster:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.get_placement_groups_called = False
            cluster_calls.append(self)

        def world_size(self):
            return sum(self.kwargs["bundle_ct_per_node_list"])

        def get_master_address_and_port(self):
            return "127.0.0.1", 1234

        def get_placement_groups(self):
            self.get_placement_groups_called = True
            return []

    policy = MagicMock()
    policy.prepare_refit_info.return_value = {"state": "dict"}
    policy.init_collective.return_value = ["policy-future"]
    value_model = MagicMock()
    generation = MagicMock()
    generation.init_collective.return_value = ["generation-future"]
    policy_factory = MagicMock(return_value=policy)
    value_factory = MagicMock(return_value=value_model)
    generation_factory = MagicMock(return_value=generation)
    ray_get = MagicMock(side_effect=lambda futures: futures)

    monkeypatch.setattr(ppo_mod, "RayVirtualCluster", DummyCluster)
    monkeypatch.setattr(ppo_mod, "Policy", policy_factory)
    monkeypatch.setattr(ppo_mod, "Value", value_factory)
    monkeypatch.setattr(ppo_mod, "VllmGeneration", generation_factory)
    monkeypatch.setattr(ppo_mod.ray, "get", ray_get)

    result = ppo_mod.setup(config, MagicMock(), _setup_dataset(), None)
    return (
        result,
        cluster_calls,
        policy,
        generation,
        policy_factory,
        value_factory,
        generation_factory,
        ray_get,
    )


def test_noncolocated_sglang_is_rejected_before_cluster_creation(monkeypatch):
    """SGLang has no cross-cluster refit path, so setup must reject it early."""
    from unittest.mock import MagicMock

    ppo_mod = _patch_ppo_setup_prerequisites(monkeypatch)
    cluster_cls = MagicMock()
    monkeypatch.setattr(ppo_mod, "RayVirtualCluster", cluster_cls)
    config = _make_noncolocated_setup_config(backend="sglang")

    with pytest.raises(
        AssertionError,
        match="Non-colocated PPO generation currently supports only vLLM",
    ):
        ppo_mod.setup(config, MagicMock(), _setup_dataset(), None)

    cluster_cls.assert_not_called()


def test_noncolocated_inference_requires_explicit_gpus_per_node_single_node(
    monkeypatch,
):
    """A single-node split cannot infer how many GPUs belong to rollout."""
    from unittest.mock import MagicMock

    ppo_mod = _patch_ppo_setup_prerequisites(monkeypatch)
    cluster_cls = MagicMock()
    monkeypatch.setattr(ppo_mod, "RayVirtualCluster", cluster_cls)
    config = _make_noncolocated_setup_config(inference_gpus_per_node=None)

    with pytest.raises(
        AssertionError,
        match=(
            "policy.generation.colocated.resources.gpus_per_node must be explicitly set"
        ),
    ):
        ppo_mod.setup(config, MagicMock(), _setup_dataset(), None)

    cluster_cls.assert_not_called()


def test_noncolocated_inference_requires_explicit_gpus_per_node_multi_node(
    monkeypatch,
):
    """A multi-node split requires full-node GPU allocation for rollout."""
    from unittest.mock import MagicMock

    ppo_mod = _patch_ppo_setup_prerequisites(monkeypatch)
    cluster_cls = MagicMock()
    monkeypatch.setattr(ppo_mod, "RayVirtualCluster", cluster_cls)
    config = _make_noncolocated_setup_config(
        total_nodes=2,
        inference_nodes=1,
        inference_gpus_per_node=None,
    )

    with pytest.raises(
        AssertionError,
        match=(
            "policy.generation.colocated.resources.gpus_per_node must be "
            "explicitly set and equal to cluster.gpus_per_node"
        ),
    ):
        ppo_mod.setup(config, MagicMock(), _setup_dataset(), None)

    cluster_cls.assert_not_called()


def test_noncolocated_topology_requires_enough_alive_nodes(monkeypatch):
    """Topology placement fails before creating partially schedulable clusters."""
    from unittest.mock import MagicMock

    ppo_mod = _patch_ppo_setup_prerequisites(monkeypatch)
    cluster_cls = MagicMock()
    monkeypatch.setattr(ppo_mod, "RayVirtualCluster", cluster_cls)
    monkeypatch.setattr(
        ppo_mod,
        "get_ray_cluster_topology",
        lambda: {
            "node-0": ("domain-0", 0),
            "node-1": ("domain-0", 1),
        },
    )
    config = _make_noncolocated_setup_config(
        total_nodes=3,
        inference_nodes=1,
        inference_gpus_per_node=8,
        segment_size=1,
    )

    with pytest.raises(
        AssertionError,
        match="Not enough alive Ray nodes for all PPO roles",
    ):
        ppo_mod.setup(config, MagicMock(), _setup_dataset(), None)

    cluster_cls.assert_not_called()


def test_noncolocated_topology_counts_shared_node_once(monkeypatch):
    """A one-node GPU split must not require two physical Ray nodes."""
    from nemo_rl.algorithms import ppo as ppo_mod

    monkeypatch.setattr(
        ppo_mod,
        "get_ray_cluster_topology",
        lambda: {"node-0": ("domain-0", 0)},
    )
    config = _make_noncolocated_setup_config(
        total_nodes=1,
        total_gpus_per_node=8,
        inference_gpus_per_node=2,
        segment_size=1,
    )

    result, cluster_calls, *_ = _run_noncolocated_setup(monkeypatch, config)

    train_cluster, inference_cluster = result[3]
    assert train_cluster.kwargs["node_resource_constraints"] == [{"domain-0": 0.001}]
    assert inference_cluster.kwargs["node_resource_constraints"] is None
    assert train_cluster.get_placement_groups_called
    assert len(cluster_calls) == 2


def test_noncolocated_multi_node_topology_constraints(monkeypatch):
    """Topology constraints are applied to a successful multi-node train split."""
    from nemo_rl.algorithms import ppo as ppo_mod

    monkeypatch.setattr(
        ppo_mod,
        "get_ray_cluster_topology",
        lambda: {
            "node-0": ("domain-0", 0),
            "node-1": ("domain-0", 1),
            "node-2": ("domain-0", 2),
        },
    )
    config = _make_noncolocated_setup_config(
        total_nodes=3,
        total_gpus_per_node=8,
        inference_nodes=1,
        inference_gpus_per_node=8,
        segment_size=1,
    )

    result, _, *_ = _run_noncolocated_setup(monkeypatch, config)

    train_cluster, inference_cluster = result[3]
    assert train_cluster.kwargs["node_resource_constraints"] == [
        {"domain-0": 0.001},
        {"domain-0": 0.001},
    ]
    assert inference_cluster.kwargs["node_resource_constraints"] is None
    assert train_cluster.get_placement_groups_called


def test_noncolocated_multi_node_inference_topology_constraints(monkeypatch):
    """Multi-node vLLM instances are pinned and initialized eagerly."""
    from nemo_rl.algorithms import ppo as ppo_mod

    monkeypatch.setattr(
        ppo_mod,
        "get_ray_cluster_topology",
        lambda: {
            "node-0": ("domain-0", 0),
            "node-1": ("domain-0", 1),
            "node-2": ("domain-1", 0),
            "node-3": ("domain-1", 1),
        },
    )
    config = _make_noncolocated_setup_config(
        total_nodes=4,
        total_gpus_per_node=8,
        inference_nodes=2,
        inference_gpus_per_node=8,
        segment_size=2,
        tensor_parallel_size=16,
    )

    result, _, _, _, _, _, generation_factory, _ = _run_noncolocated_setup(
        monkeypatch, config
    )

    train_cluster, inference_cluster = result[3]
    assert train_cluster.kwargs["node_resource_constraints"] == [
        {"domain-0": 0.001},
        {"domain-0": 0.001},
    ]
    assert inference_cluster.kwargs["node_resource_constraints"] == [
        {"domain-1": 0.001},
        {"domain-1": 0.001},
    ]
    assert inference_cluster.kwargs["segment_size"] == 2
    generation_factory.init_cluster_placement_groups.assert_called_once_with(
        inference_cluster, config.policy["generation"]
    )


def test_noncolocated_vllm_builds_separate_clusters_and_collective(monkeypatch):
    """Policy/value share two slots while rollout gets a separate one-slot cluster."""
    config = _make_noncolocated_setup_config(
        total_gpus_per_node=8,
        inference_gpus_per_node=2,
    )
    (
        result,
        cluster_calls,
        policy,
        generation,
        policy_factory,
        value_factory,
        generation_factory,
        ray_get,
    ) = _run_noncolocated_setup(monkeypatch, config)

    train_cluster, inference_cluster = result[3]
    assert [cluster.kwargs["name"] for cluster in cluster_calls] == [
        "ppo_train_cluster",
        "ppo_inference_cluster",
    ]
    assert train_cluster.kwargs["bundle_ct_per_node_list"] == [6]
    assert train_cluster.kwargs["max_colocated_worker_groups"] == 2
    assert inference_cluster.kwargs["bundle_ct_per_node_list"] == [2]
    assert inference_cluster.kwargs["max_colocated_worker_groups"] == 1
    assert policy_factory.call_args.kwargs["cluster"] is train_cluster
    assert value_factory.call_args.kwargs["cluster"] is train_cluster
    assert generation_factory.call_args.kwargs["cluster"] is inference_cluster

    policy.init_collective.assert_called_once_with(
        "127.0.0.1", 1234, 8, train_world_size=6
    )
    generation.init_collective.assert_called_once_with(
        "127.0.0.1", 1234, 8, train_world_size=6
    )
    ray_get.assert_called_once_with(["policy-future", "generation-future"])

    policy.offload_to_cpu.assert_called_once_with()
    value_model = result[2]
    value_model.finish_training.assert_called_once_with()
    policy.prepare_for_training.assert_called_once_with()
    policy.prepare_refit_info.assert_called_once_with()
    generation.prepare_refit_info.assert_called_once_with({"state": "dict"})


def test_colocated_setup_keeps_single_cluster_and_skips_collective(monkeypatch):
    """The default colocated setup remains unchanged by the cluster split."""
    config = _make_noncolocated_setup_config()
    config.policy["generation"]["colocated"] = {
        "enabled": True,
        "resources": {"num_nodes": None, "gpus_per_node": None},
    }

    result, cluster_calls, policy, generation, *_, ray_get = _run_noncolocated_setup(
        monkeypatch, config
    )

    train_cluster, inference_cluster = result[3]
    assert train_cluster is inference_cluster
    assert len(cluster_calls) == 1
    assert train_cluster.kwargs["name"] == "ppo_policy_cluster"
    assert train_cluster.kwargs["max_colocated_worker_groups"] == 3
    policy.init_collective.assert_not_called()
    generation.init_collective.assert_not_called()
    ray_get.assert_not_called()


def test_noncolocated_vllm_multi_node_cluster_and_collective_sizes(monkeypatch):
    """A full inference node is carved out of a three-node PPO allocation."""
    config = _make_noncolocated_setup_config(
        total_nodes=3,
        total_gpus_per_node=8,
        inference_nodes=1,
        inference_gpus_per_node=8,
    )
    result, _, policy, generation, *_ = _run_noncolocated_setup(monkeypatch, config)

    train_cluster, inference_cluster = result[3]
    assert train_cluster.kwargs["bundle_ct_per_node_list"] == [8, 8]
    assert train_cluster.kwargs["max_colocated_worker_groups"] == 2
    assert inference_cluster.kwargs["bundle_ct_per_node_list"] == [8]
    assert inference_cluster.kwargs["max_colocated_worker_groups"] == 1
    policy.init_collective.assert_called_once_with(
        "127.0.0.1", 1234, 24, train_world_size=16
    )
    generation.init_collective.assert_called_once_with(
        "127.0.0.1", 1234, 24, train_world_size=16
    )


def test_noncolocated_vllm_single_node_reserves_reward_model_gpu(monkeypatch):
    """Training receives GPUs left after rollout and reward-model reservations."""
    config = _make_noncolocated_setup_config(
        total_gpus_per_node=8,
        inference_gpus_per_node=2,
        reward_model_gpus_per_node=1,
    )
    result, *_ = _run_noncolocated_setup(monkeypatch, config)

    train_cluster, inference_cluster = result[3]
    assert train_cluster.kwargs["bundle_ct_per_node_list"] == [5]
    assert train_cluster.kwargs["max_colocated_worker_groups"] == 2
    assert inference_cluster.kwargs["bundle_ct_per_node_list"] == [2]
    assert inference_cluster.kwargs["max_colocated_worker_groups"] == 1


def test_noncolocated_reward_model_node_leaves_shared_train_inference_node(
    monkeypatch,
):
    """A dedicated RM node does not consume GPUs on the shared PPO node."""
    config = _make_noncolocated_setup_config(
        total_nodes=2,
        total_gpus_per_node=8,
        inference_gpus_per_node=2,
        reward_model_gpus_per_node=1,
    )

    result, _, policy, generation, *_ = _run_noncolocated_setup(monkeypatch, config)

    train_cluster, inference_cluster = result[3]
    assert train_cluster.kwargs["bundle_ct_per_node_list"] == [6]
    assert inference_cluster.kwargs["bundle_ct_per_node_list"] == [2]
    policy.init_collective.assert_called_once_with(
        "127.0.0.1", 1234, 8, train_world_size=6
    )
    generation.init_collective.assert_called_once_with(
        "127.0.0.1", 1234, 8, train_world_size=6
    )
