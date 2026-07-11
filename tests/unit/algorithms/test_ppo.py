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
    RawRewardAdvantageEstimator,
)
from nemo_rl.algorithms.loss.loss_functions import (
    ClippedPGLossConfig,
    MseValueLossConfig,
    MseValueLossFn,
)
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


# ===============================================================================
# Non-colocated generation (setup())
# ===============================================================================


def _build_ppo_master_config():
    """Minimal PPO MasterConfig sufficient to drive setup() end to end.

    Uses model_construct (like test_grpo.py's mock_grpo_components) so nested
    sub-configs can stay plain dicts instead of satisfying full pydantic/TypedDict
    validation. Individual tests mutate fields in place before calling setup().
    """
    from nemo_rl.algorithms.ppo import MasterConfig

    return MasterConfig.model_construct(
        **{
            "policy": {
                "model_name": "fake-model",
                "train_global_batch_size": 1,
                "train_micro_batch_size": 1,
                "max_total_sequence_length": 128,
                "generation": {
                    "backend": "vllm",
                    "model_name": "fake-model",
                    "colocated": {
                        "enabled": True,
                        "resources": {"gpus_per_node": None, "num_nodes": None},
                    },
                    "vllm_cfg": {
                        "precision": "bfloat16",
                        "kv_cache_dtype": "auto",
                    },
                    "vllm_kwargs": {},
                },
            },
            "value": {
                "megatron_cfg": {"enabled": False},
                "dtensor_cfg": {"enabled": True, "context_parallel_size": 1},
                "sequence_packing": {"enabled": False},
                "dynamic_batching": {"enabled": False},
            },
            "loss_fn": ClippedPGLossConfig(),
            "value_loss_fn": MseValueLossConfig(),
            "env": {},
            "data": {"shuffle": False, "num_workers": 0},
            "ppo": {
                "seed": 42,
                "batch_multiplier": 1,
                "num_prompts_per_step": 1,
                "use_dynamic_sampling": False,
                "val_period": 0,
                "val_at_start": False,
                "val_at_end": False,
                "ppo_epochs": 1,
                "max_num_steps": 1,
                "max_num_epochs": 1,
            },
            "logger": {},
            "cluster": {"num_nodes": 1, "gpus_per_node": 2},
            "checkpointing": {},
        }
    )


@pytest.mark.parametrize(
    "num_nodes,inference_num_nodes",
    [
        pytest.param(1, None, id="single_node"),
        pytest.param(2, 1, id="multi_node"),
    ],
)
def test_ppo_noncolocated_requires_explicit_gpus_per_node(
    num_nodes, inference_num_nodes
):
    """Non-colocated PPO must set an explicit inference GPU count, whether
    train and inference share a single node or each get dedicated nodes."""
    from unittest.mock import MagicMock, patch

    from nemo_rl.algorithms.ppo import setup

    master_config = _build_ppo_master_config()
    master_config.policy["generation"]["colocated"] = {
        "enabled": False,
        "resources": {"gpus_per_node": None, "num_nodes": inference_num_nodes},
    }
    master_config.cluster["num_nodes"] = num_nodes
    master_config.cluster["gpus_per_node"] = 8

    tokenizer = MagicMock()
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=10)

    with (
        patch("nemo_rl.algorithms.ppo.Logger"),
        patch("nemo_rl.algorithms.ppo.CheckpointManager") as mock_checkpointer,
        patch("nemo_rl.algorithms.ppo.StatefulDataLoader"),
        pytest.raises(
            AssertionError,
            match="policy.generation.colocated.resources.gpus_per_node must be explicitly set",
        ),
    ):
        mock_checkpointer.return_value.get_latest_checkpoint_path.return_value = None
        mock_checkpointer.return_value.load_training_info.return_value = None
        setup(master_config, tokenizer, dataset, None)


def test_ppo_noncolocated_rejects_sglang_backend():
    """SGLangGeneration.init_collective() is a no-op, so non-colocated PPO must
    fail loudly at setup() rather than hang forever waiting for the training
    side's NCCL collective to be joined by peers that never connect."""
    from unittest.mock import MagicMock, patch

    from nemo_rl.algorithms.ppo import setup

    master_config = _build_ppo_master_config()
    master_config.policy["generation"]["backend"] = "sglang"
    master_config.policy["generation"]["colocated"] = {
        "enabled": False,
        "resources": {"gpus_per_node": 1, "num_nodes": None},
    }
    master_config.cluster["num_nodes"] = 1
    master_config.cluster["gpus_per_node"] = 2

    tokenizer = MagicMock()
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=10)

    with (
        patch("nemo_rl.algorithms.ppo.Logger"),
        patch("nemo_rl.algorithms.ppo.CheckpointManager") as mock_checkpointer,
        patch("nemo_rl.algorithms.ppo.StatefulDataLoader"),
        pytest.raises(
            AssertionError,
            match="Non-colocated PPO currently requires the vLLM generation backend",
        ),
    ):
        mock_checkpointer.return_value.get_latest_checkpoint_path.return_value = None
        mock_checkpointer.return_value.load_training_info.return_value = None
        setup(master_config, tokenizer, dataset, None)


@pytest.mark.parametrize("colocated", [True, False])
def test_ppo_setup_cluster_split_matches_colocation_mode(monkeypatch, colocated):
    """train_cluster/inference_cluster identity and worker-group budget by mode.

    Colocated: train_cluster is inference_cluster (single shared pool of
    generation+policy+value). Non-colocated: they are distinct clusters, and
    train_cluster must budget for 2 co-timesharing worker groups (policy +
    value) since generation now lives on its own inference_cluster.
    """
    from unittest.mock import MagicMock

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

    class DummyLoader:
        def __init__(self, *_args, **_kwargs):
            pass

        def __len__(self):
            return 1

    class DummyCluster:
        instances = []

        def __init__(self, *_args, max_colocated_worker_groups=1, **_kwargs):
            self.max_colocated_worker_groups = max_colocated_worker_groups
            DummyCluster.instances.append(self)

        def world_size(self):
            return 1

        def get_master_address_and_port(self):
            return "127.0.0.1", 1234

        def get_placement_groups(self):
            return []

    class DummyPolicy:
        def offload_to_cpu(self):
            pass

        def print_node_ip_and_gpu_id(self):
            pass

        def init_collective(self, *_args, **_kwargs):
            return []

        def prepare_for_training(self):
            pass

        def prepare_refit_info(self):
            return {}

    class DummyValue:
        def __init__(self, *_args, **_kwargs):
            pass

        def finish_training(self):
            pass

    class DummyVllmGeneration:
        def __init__(self, *_args, **_kwargs):
            pass

        def finish_generation(self):
            pass

        def prepare_refit_info(self, _state):
            pass

        def init_collective(self, *_args, **_kwargs):
            return []

    DummyCluster.instances = []
    monkeypatch.setattr(ppo_mod, "Logger", lambda *_a, **_k: DummyLogger())
    monkeypatch.setattr(
        ppo_mod, "CheckpointManager", lambda *_a, **_k: DummyCheckpointer()
    )
    monkeypatch.setattr(ppo_mod, "StatefulDataLoader", DummyLoader)
    monkeypatch.setattr(ppo_mod, "RayVirtualCluster", DummyCluster)
    monkeypatch.setattr(ppo_mod, "Policy", lambda *_a, **_k: DummyPolicy())
    monkeypatch.setattr(ppo_mod, "Value", lambda *_a, **_k: DummyValue())
    monkeypatch.setattr(
        ppo_mod, "VllmGeneration", lambda *_a, **_k: DummyVllmGeneration()
    )
    monkeypatch.setattr(ppo_mod.ray, "get", lambda x: x)

    master_config = _build_ppo_master_config()
    if colocated:
        master_config.policy["generation"]["colocated"] = {
            "enabled": True,
            "resources": {"gpus_per_node": None, "num_nodes": None},
        }
    else:
        master_config.policy["generation"]["colocated"] = {
            "enabled": False,
            "resources": {"gpus_per_node": 1, "num_nodes": None},
        }
    master_config.cluster["num_nodes"] = 1
    master_config.cluster["gpus_per_node"] = 2

    tokenizer = MagicMock()
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=1)

    _, _, _, (train_cluster, inference_cluster), *_ = ppo_mod.setup(
        master_config, tokenizer, dataset, None
    )

    if colocated:
        assert train_cluster is inference_cluster
        assert train_cluster.max_colocated_worker_groups == 3
    else:
        assert train_cluster is not inference_cluster
        assert train_cluster.max_colocated_worker_groups == 2
        assert inference_cluster.max_colocated_worker_groups == 1


# ===============================================================================
# Async PPO entry guards (async_ppo_train)
# ===============================================================================


def _build_async_ppo_master_config():
    """PPO MasterConfig pre-configured for a valid async run.

    Individual guard tests mutate one field to trip a specific assertion. All
    guards fire before any Ray actor is created, so the other async_ppo_train
    arguments can be plain mocks.
    """
    master_config = _build_ppo_master_config()
    master_config.policy["generation"]["backend"] = "vllm"
    master_config.policy["generation"]["colocated"] = {
        "enabled": False,
        "resources": {"gpus_per_node": 1, "num_nodes": 1},
    }
    master_config.policy["generation"]["vllm_cfg"]["async_engine"] = True
    master_config.loss_fn = ClippedPGLossConfig(use_importance_sampling_correction=True)
    master_config.ppo["policy_training_start_step"] = 0
    master_config.ppo["num_generations_per_prompt"] = 1
    master_config.ppo["overlong_filtering"] = False
    master_config.ppo["async_ppo"] = {
        "enabled": True,
        "max_trajectory_age_steps": 1,
        "in_flight_weight_updates": False,
    }
    return master_config


def _call_async_ppo_train(master_config):
    from unittest.mock import MagicMock

    from nemo_rl.algorithms.ppo import async_ppo_train

    async_ppo_train(
        policy=MagicMock(),
        policy_generation=MagicMock(),
        value_model=MagicMock(),
        dataloader=MagicMock(),
        val_dataloader=None,
        tokenizer=MagicMock(),
        loss_fn=MagicMock(),
        value_loss_fn=MagicMock(),
        task_to_env={},
        val_task_to_env=None,
        logger=MagicMock(),
        checkpointer=MagicMock(),
        ppo_save_state=MagicMock(),
        master_config=master_config,
        max_trajectory_age_steps=master_config.ppo["async_ppo"][
            "max_trajectory_age_steps"
        ],
    )


def test_async_ppo_rejects_non_vllm_backend():
    master_config = _build_async_ppo_master_config()
    master_config.policy["generation"]["backend"] = "sglang"
    with pytest.raises(AssertionError, match="async vLLM generation engine"):
        _call_async_ppo_train(master_config)


def test_async_ppo_requires_async_engine():
    master_config = _build_async_ppo_master_config()
    master_config.policy["generation"]["vllm_cfg"]["async_engine"] = False
    with pytest.raises(AssertionError, match="async vLLM generation engine"):
        _call_async_ppo_train(master_config)


def test_async_ppo_requires_importance_sampling_correction():
    master_config = _build_async_ppo_master_config()
    master_config.loss_fn = ClippedPGLossConfig(
        use_importance_sampling_correction=False
    )
    with pytest.raises(AssertionError, match="Importance sampling correction"):
        _call_async_ppo_train(master_config)


def test_async_ppo_rejects_colocated_inference():
    master_config = _build_async_ppo_master_config()
    master_config.policy["generation"]["colocated"]["enabled"] = True
    with pytest.raises(AssertionError, match="Colocated inference is not supported"):
        _call_async_ppo_train(master_config)


def test_async_ppo_requires_positive_ppo_epochs():
    """ppo_epochs == 0 would leave train_results unset; guard rejects it."""
    master_config = _build_async_ppo_master_config()
    master_config.ppo["ppo_epochs"] = 0
    with pytest.raises(AssertionError, match="ppo_epochs must be >= 1"):
        _call_async_ppo_train(master_config)


def test_async_ppo_rejects_nemo_gym():
    """NeMo-Gym rollout is not wired for async PPO; guard fails loud at startup."""
    master_config = _build_async_ppo_master_config()
    master_config.env["should_use_nemo_gym"] = True
    # _should_use_nemo_gym also requires the http server to be exposed.
    master_config.policy["generation"]["vllm_cfg"]["expose_http_server"] = True
    with pytest.raises(AssertionError, match="NeMo-Gym rollout is not yet supported"):
        _call_async_ppo_train(master_config)


# ---------------------------------------------------------------------------
# Warmup trajectory-age boundaries (the two-boundary fix for the frozen-actor
# critic-warmup phase). See async_ppo_train's _collector_lead_age /
# _sample_max_age and the pi_0 policy-version analysis.
# ---------------------------------------------------------------------------
def test_async_warmup_age_boundaries_no_elevation():
    """warmup_age == train_age (the default) => constant age at every step, so
    behaviour is identical to plain async PPO regardless of the boundary math."""
    from nemo_rl.algorithms.ppo import (
        _async_warmup_collector_lead_age,
        _async_warmup_sample_max_age,
    )

    W, train_age = 20, 1
    for s in range(0, W + 10):
        assert _async_warmup_collector_lead_age(s, W, train_age, train_age) == train_age
        assert _async_warmup_sample_max_age(s, W, train_age, train_age) == train_age


def test_async_warmup_collector_lead_age_boundary_at_W():
    """Collector generation-lead is elevated through step W (frozen actor), then
    drops to the training age from W+1 so it regenerates against the trained policy."""
    from nemo_rl.algorithms.ppo import _async_warmup_collector_lead_age

    W, train_age, warmup_age = 20, 1, 8
    assert _async_warmup_collector_lead_age(W - 1, W, train_age, warmup_age) == warmup_age
    assert _async_warmup_collector_lead_age(W, W, train_age, warmup_age) == warmup_age
    assert _async_warmup_collector_lead_age(W + 1, W, train_age, warmup_age) == train_age


def test_async_warmup_sample_max_age_boundary_at_W_plus_train_age():
    """Driver eviction age stays elevated through W + train_age: a frozen (pi_0)
    rollout is within train_age POLICY-steps of the actor there, so it is admitted
    as valid lag-<=train_age data instead of being wrongly evicted (which would
    deadlock, since the collector never regenerates that target)."""
    from nemo_rl.algorithms.ppo import _async_warmup_sample_max_age

    W, warmup_age = 20, 8
    # train_age = 1: elevated through W+1 (policy-age 1), drops at W+2.
    assert _async_warmup_sample_max_age(W, W, 1, warmup_age) == warmup_age
    assert _async_warmup_sample_max_age(W + 1, W, 1, warmup_age) == warmup_age
    assert _async_warmup_sample_max_age(W + 2, W, 1, warmup_age) == 1
    # train_age = 2: pi_0 stays within 2 policy-steps through W+2, drops at W+3.
    assert _async_warmup_sample_max_age(W + 2, W, 2, warmup_age) == warmup_age
    assert _async_warmup_sample_max_age(W + 3, W, 2, warmup_age) == 2


def test_async_trajectory_policy_age_is_freeze_aware():
    """The gen-version age overcounts staleness during warmup; policy-age is the
    true off-policy distance and must stay <= max_trajectory_age_steps.

    Reproduces the smoke-test boundary (W=3, A_w=5, A_t=1): the frozen pi_0 banked
    at gen-version 0 is consumed at steps 0..4, then a fresh pi_1 (gen-version 4) at
    step 5. Its gen-version age spikes to 4 at step 4, but its POLICY-age is only 1.
    """
    from nemo_rl.algorithms.ppo import _async_trajectory_policy_age

    W, train_age = 3, 1
    # (weight_version s, gen_version g) actually consumed each step of the smoke run.
    consumed = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 4)]
    expected_gen_age = [0, 1, 2, 3, 4, 1]
    expected_policy_age = [0, 0, 0, 0, 1, 1]
    for (s, g), gen_age, pol_age in zip(
        consumed, expected_gen_age, expected_policy_age
    ):
        assert (s - g) == gen_age  # what avg_trajectory_age reports
        got = _async_trajectory_policy_age(g, s, W)
        assert got == pol_age, (s, g, got, pol_age)
        assert got <= train_age  # the invariant that actually matters

    # No warmup (W == 0): policy-age reduces to the plain gen-version age.
    for s in range(6):
        for g in range(s + 1):
            assert _async_trajectory_policy_age(g, s, 0) == s - g


def test_replay_buffer_admits_banked_frozen_rollout_at_boundary():
    """End-to-end buffer check of the boundary: a frozen rollout banked deep during
    warmup (low gen-version) targeting step W+1 must be ADMITTED when sampled with
    the elevated (warmup) age at step W+1, and EVICTED once the age snaps back.

    This is the exact scenario that deadlocked with a single W boundary: the
    collector had already advanced its lead past target W+1, so an eviction there
    is unrecoverable.
    """
    from nemo_rl.algorithms.async_utils.replay_buffer import ReplayBufferImpl

    W, train_age, warmup_age = 20, 1, 8
    num_prompts = 2

    def _fresh_buffer():
        buf = ReplayBufferImpl(max_size=64)
        # Two groups for target W+1, generated during warmup at gen-version W+1-A_w
        # (the frozen actor pi_0). gen-version age is warmup_age, policy-age is 1.
        for _ in range(num_prompts):
            buf.add({"batch": None}, W + 1 - warmup_age, W + 1)
        return buf

    # At step W+1 with the elevated (warmup) age -> admitted (valid lag-1 pi_0).
    buf = _fresh_buffer()
    result = buf.sample(
        num_prompt_groups=num_prompts,
        current_weight_version=W + 1,
        max_age_steps=warmup_age,
    )
    assert result is not None
    assert len(result["trajectories"]) == num_prompts

    # Same rollout at step W+2 with the training age -> gen-version W+1-A_w is now
    # older than (W+2 - train_age), so it is correctly evicted (pi_0 genuinely stale).
    buf = _fresh_buffer()
    # relabel the (identical) banked groups to target W+2 to model the surplus that
    # the driver would try to consume one step later.
    buf.target_weight_versions = [W + 2, W + 2]
    result = buf.sample(
        num_prompt_groups=num_prompts,
        current_weight_version=W + 2,
        max_age_steps=train_age,
    )
    assert result is None  # evicted as stale; collector must regenerate fresh


# ---------------------------------------------------------------------------
# validate() rollout dispatch. Async PPO runs the vLLM async engine, whose
# generation worker has no sync `generate` method — validate() must take the
# async rollout path or it raises AttributeError ('ActorHandle' has no
# attribute 'generate') at the first validation step.
# ---------------------------------------------------------------------------
def _run_validate_with_mocked_rollouts(master_config):
    """Drive validate() through one batch with both rollout fns mocked.

    Returns (async_rollout_mock, sync_rollout_mock) for call assertions.
    """
    from unittest.mock import MagicMock, patch

    from nemo_rl.algorithms.ppo import validate

    # Fields validate() reads to run one batch and summarize.
    master_config.ppo["max_val_samples"] = 1
    master_config.ppo["val_batch_size"] = 1
    master_config.ppo["max_rollout_turns"] = 1
    master_config.logger = {"num_val_samples_to_print": 0}

    rollout_return = (
        {"total_reward": torch.tensor([1.0]), "message_log": []},
        {"mean_gen_tokens_per_sample": 5.0},
    )
    with (
        patch(
            "nemo_rl.algorithms.ppo.run_async_multi_turn_rollout",
            return_value=rollout_return,
        ) as async_mock,
        patch(
            "nemo_rl.algorithms.ppo.run_multi_turn_rollout",
            return_value=rollout_return,
        ) as sync_mock,
    ):
        validate(
            policy_generation=MagicMock(),
            val_dataloader=[MagicMock()],  # one validation batch
            tokenizer=MagicMock(),
            val_task_to_env={},
            step=5,
            master_config=master_config,
            logger=MagicMock(),
        )
    return async_mock, sync_mock


def test_validate_dispatches_to_async_rollout_for_async_engine():
    """Regression: async PPO (vLLM async_engine) validation must use the ASYNC
    rollout path. The sync path calls policy_generation.generate(), which the
    async worker lacks -> AttributeError at the first validation step."""
    master_config = _build_async_ppo_master_config()  # vllm async_engine=True
    async_mock, sync_mock = _run_validate_with_mocked_rollouts(master_config)
    async_mock.assert_called_once()
    sync_mock.assert_not_called()


def test_validate_dispatches_to_sync_rollout_when_not_async():
    """Non-async (colocated/sync) PPO validation uses the synchronous rollout."""
    master_config = _build_async_ppo_master_config()
    master_config.policy["generation"]["vllm_cfg"]["async_engine"] = False
    async_mock, sync_mock = _run_validate_with_mocked_rollouts(master_config)
    sync_mock.assert_called_once()
    async_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Resume optimizer-path resolution. Megatron bundles the optimizer + LR
# scheduler inside weights/iter_*/*.distcp (no separate optimizer/ dir), so a
# resume must point optimizer_path at the weights or load_optim stays False and
# the LR-warmup scheduler restarts on every resume (V-shaped critic/lr).
# ---------------------------------------------------------------------------
def test_resolve_resume_optimizer_path(tmp_path):
    from nemo_rl.algorithms.ppo import _resolve_resume_optimizer_path

    weights = tmp_path / "weights"
    weights.mkdir()
    optim = tmp_path / "optimizer"
    missing_optim = tmp_path / "optimizer_absent"
    megatron = {"megatron_cfg": {"enabled": True}}
    dtensor = {"megatron_cfg": {"enabled": False}}

    # DTensor writes a separate optimizer/ dir -> use it when present.
    optim.mkdir()
    assert _resolve_resume_optimizer_path(optim, weights, dtensor) == optim
    # A present separate dir wins even on megatron.
    assert _resolve_resume_optimizer_path(optim, weights, megatron) == optim

    # Megatron: no separate optimizer/ dir (bundled in weights) -> weights path,
    # so load_optim=True and the optimizer + scheduler actually resume.
    assert _resolve_resume_optimizer_path(missing_optim, weights, megatron) == weights
    # Megatron during warmup: policy weights not saved -> nothing to resume.
    assert _resolve_resume_optimizer_path(missing_optim, None, megatron) is None
    # DTensor without a separate optimizer/ dir -> None (don't misread weights as
    # an optimizer path for a backend that stores it separately).
    assert _resolve_resume_optimizer_path(missing_optim, weights, dtensor) is None
