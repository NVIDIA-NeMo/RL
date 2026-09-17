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

"""Regression tests for the undiscounted, lambda-one GAE fast path."""

import pytest
import torch

from nemo_rl.algorithms.advantage_estimator import (
    GAEConfig,
    GeneralizedAdvantageEstimator,
)
from nemo_rl.algorithms.loss import ClippedPGLossConfig


def _reference_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    gamma: float = 1.0,
    lam: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Original backward recurrence, including carry across masked positions."""
    next_value = torch.zeros_like(values[:, 0])
    last_advantage = torch.zeros_like(next_value)
    reversed_advantages = []
    for index in reversed(range(rewards.shape[1])):
        delta = rewards[:, index] + gamma * next_value - values[:, index]
        advantage = delta + gamma * lam * last_advantage
        valid = mask[:, index]
        next_value = values[:, index] * valid + (1 - valid) * next_value
        last_advantage = advantage * valid + (1 - valid) * last_advantage
        reversed_advantages.append(last_advantage)
    advantages = torch.stack(reversed_advantages[::-1], dim=1)
    return advantages, advantages + values


@pytest.fixture(params=["cpu", "cuda"])
def device(request: pytest.FixtureRequest) -> torch.device:
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch.device(request.param)


@pytest.mark.parametrize("seq_len", [1, 8, 257, 1024])
@pytest.mark.parametrize(
    "reward_dtype,value_dtype,mask_dtype",
    [
        (torch.float32, torch.float32, torch.float32),
        (torch.float64, torch.float64, torch.float64),
        (torch.float32, torch.float64, torch.float32),
        (torch.float64, torch.float32, torch.int64),
    ],
)
def test_gae_fast_path_matches_recurrence(
    device: torch.device,
    seq_len: int,
    reward_dtype: torch.dtype,
    value_dtype: torch.dtype,
    mask_dtype: torch.dtype,
) -> None:
    generator = torch.Generator().manual_seed(42)
    rewards = torch.randn(5, seq_len, generator=generator, dtype=torch.float64).to(
        device=device, dtype=reward_dtype
    )
    values = torch.randn(5, seq_len, generator=generator, dtype=torch.float64).to(
        device=device, dtype=value_dtype
    )
    mask = torch.ones(5, seq_len, device=device, dtype=mask_dtype)
    mask[1, : seq_len // 2] = 0  # Prefix padding.
    mask[2, ::3] = 0  # Interior gaps followed by trailing padding.
    mask[2, -2:] = 0
    mask[3:] = 0
    mask[4, seq_len // 2] = 1  # A single valid token; row 3 stays empty.
    rewards.masked_fill_(~mask.bool(), 123.0)
    values.masked_fill_(~mask.bool(), 999.0)
    estimator = GeneralizedAdvantageEstimator(
        GAEConfig(gae_lambda=1.0), ClippedPGLossConfig()
    )

    actual = estimator._compute_gae(rewards, values, mask)
    expected = _reference_gae(rewards, values, mask)

    # Compare all positions: the private helper carries advantages through gaps
    # and returns = advantages + values even outside the response mask.
    for actual_tensor, expected_tensor in zip(actual, expected):
        # Mixed inputs can promote the result to float64; choose the tolerance
        # from the effective output dtype, not just the rewards' dtype.
        tolerance = 1e-13 if actual_tensor.dtype == torch.float64 else 1e-5
        torch.testing.assert_close(
            actual_tensor, expected_tensor, rtol=tolerance, atol=tolerance
        )
    torch.testing.assert_close(actual[0][3], torch.zeros_like(actual[0][3]))
    torch.testing.assert_close(actual[1][3], values[3].to(actual[1].dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gae_unit_discount_matches_closed_form_with_masked_gaps(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Check both production paths against hand-calculated, exact binary values."""
    rewards = torch.tensor(
        [[123, 0.5, 123, -0.25, 0.75, 123], [123] * 6],
        device=device,
        dtype=dtype,
    )
    values = torch.tensor(
        [[999, 0.125, 999, 0.25, -0.5, 999], [999] * 6],
        device=device,
        dtype=dtype,
    )
    mask = torch.tensor([[0, 1, 0, 1, 1, 0], [0] * 6], device=device, dtype=dtype)
    # At valid positions, R = [0.5 - 0.25 + 0.75, -0.25 + 0.75, 0.75]
    # and A = R - V = [0.875, 0.25, 1.25]. Masked positions carry the next
    # valid A (or zero after the final valid token), but returns still add V.
    expected_advantages = torch.tensor(
        [[0.875, 0.875, 0.25, 0.25, 1.25, 0], [0] * 6],
        device=device,
        dtype=dtype,
    )
    expected_returns = torch.tensor(
        [[999.875, 1.0, 999.25, 0.5, 0.75, 999], [999] * 6],
        device=device,
        dtype=dtype,
    )
    estimator = GeneralizedAdvantageEstimator(
        GAEConfig(gae_lambda=1.0), ClippedPGLossConfig()
    )

    # A tensor-valued lambda deliberately selects the actual production
    # recurrence, independently of the copied helper used in other tests.
    for override in (None, torch.ones(2, device=device, dtype=dtype)):
        advantages, returns = estimator._compute_gae(
            rewards, values, mask, gae_lambda=override
        )
        torch.testing.assert_close(advantages, expected_advantages, rtol=0, atol=0)
        torch.testing.assert_close(returns, expected_returns, rtol=0, atol=0)


@pytest.mark.parametrize("seq_len", [196_608, 1_048_576])
def test_gae_fast_path_long_sequence_matches_analytic_returns(seq_len: int) -> None:
    """Bound FP32 error against a closed form without a long Python recurrence."""
    # Keep this CPU-only: the intended driver-side path must handle long
    # sequences, without adding hundreds of thousands of CUDA launches in CI.
    reward_per_token = 2.0**-20
    rewards = torch.full((1, seq_len), reward_per_token, dtype=torch.float32)
    rewards[:, -1] = 1.0
    values = torch.full_like(rewards, 32.0)
    mask = torch.ones_like(rewards)
    estimator = GeneralizedAdvantageEstimator(
        GAEConfig(gae_lambda=1.0), ClippedPGLossConfig()
    )

    advantages, returns = estimator._compute_gae(rewards, values, mask)

    # Count the remaining nonterminal rewards analytically, rather than using
    # either a reverse cumsum or the GAE recurrence as the oracle. Each reward
    # and suffix sum is exactly representable in FP32 at these sequence lengths.
    remaining = torch.arange(seq_len - 1, -1, -1, dtype=torch.float64).unsqueeze(0)
    expected_returns = 1.0 + remaining * reward_per_token
    expected_advantages = expected_returns - values.double()
    # Here 30 <= |A| < 32, so subtracting V rounds by at most half an FP32 ULP.
    # The old FP32 recurrence loses the small reward in (reward + 32) - 32;
    # matching that accumulated error is not the mathematical correctness goal.
    tolerance = 2.0**-20
    torch.testing.assert_close(
        advantages.double(), expected_advantages, rtol=0, atol=tolerance
    )
    torch.testing.assert_close(
        returns.double(), expected_returns, rtol=0, atol=tolerance
    )


@pytest.mark.parametrize(
    "gamma,configured_lambda,override",
    [
        (1.0, 1.0, None),
        (1.0, 0.95, 1.0),
        (1.0, 1.0, 0.8),
        (0.99, 1.0, None),
        (1.0 - 1e-7, 1.0, None),
        (1.0, 1.0 - 1e-7, None),
        (1.0, 1.0, [1.0, 1.0]),
        (1.0, 1.0, [0.8, 0.5]),
    ],
)
def test_gae_effective_lambda_and_fallback(
    device: torch.device,
    gamma: float,
    configured_lambda: float,
    override: float | list[float] | None,
) -> None:
    estimator = GeneralizedAdvantageEstimator(
        GAEConfig(gae_gamma=gamma, gae_lambda=configured_lambda), ClippedPGLossConfig()
    )
    rewards = torch.tensor(
        [[0.3, 0.0, -0.2, 1.0]] * 2, device=device, dtype=torch.float64
    )
    values = torch.tensor(
        [[0.2, 999.0, 0.5, 0.8]] * 2, device=device, dtype=torch.float64
    )
    mask = torch.tensor([[1.0, 0.0, 1.0, 1.0]] * 2, device=device, dtype=torch.float64)
    effective_override = (
        torch.tensor(override, device=device, dtype=torch.float64)
        if isinstance(override, list)
        else override
    )
    effective_lambda = (
        configured_lambda if effective_override is None else effective_override
    )

    actual = estimator._compute_gae(
        rewards, values, mask, gae_lambda=effective_override
    )
    expected = _reference_gae(rewards, values, mask, gamma=gamma, lam=effective_lambda)

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor, expected_tensor, rtol=1e-13, atol=1e-13
        )


def test_gae_fractional_sample_weights_preserve_recurrence(
    device: torch.device,
) -> None:
    """PPO's advantage mask can include a fractional sample loss multiplier."""
    estimator = GeneralizedAdvantageEstimator(
        GAEConfig(gae_lambda=1.0), ClippedPGLossConfig()
    )
    rewards = torch.tensor(
        [[0.3, 0.0, -0.2, 1.0]] * 2, device=device, dtype=torch.float64
    )
    values = torch.tensor(
        [[0.2, 999.0, 0.5, 0.8]] * 2, device=device, dtype=torch.float64
    )
    mask = torch.tensor(
        [[0.25, 0.0, 0.25, 0.25], [0.5, 0.0, 1.0, 0.5]],
        device=device,
        dtype=torch.float64,
    )

    actual = estimator._compute_gae(rewards, values, mask)
    expected = _reference_gae(rewards, values, mask)

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


@pytest.mark.parametrize("configured_lambda,override", [(1.0, None), (0.95, 1.0)])
def test_gae_fast_path_logs_activation_every_call(
    capsys: pytest.CaptureFixture[str],
    configured_lambda: float,
    override: float | None,
) -> None:
    estimator = GeneralizedAdvantageEstimator(
        GAEConfig(gae_gamma=1.0, gae_lambda=configured_lambda), ClippedPGLossConfig()
    )
    rewards = torch.tensor([[0.0, 1.0]])
    values = torch.tensor([[0.2, 0.5]])
    mask = torch.ones_like(rewards)
    assert capsys.readouterr().out == ""

    estimator._compute_gae(rewards, values, mask, gae_lambda=override)
    assert capsys.readouterr().out == (
        "Fast GAE compute activated for lambda=1.0, gamma=1.0\n"
    )

    estimator._compute_gae(rewards, values, mask, gae_lambda=override)
    assert capsys.readouterr().out == (
        "Fast GAE compute activated for lambda=1.0, gamma=1.0\n"
    )


@pytest.mark.parametrize("fallback", ["gamma", "lambda", "tensor_lambda", "fractional"])
def test_gae_fallback_does_not_log_fast_path_activation(
    capsys: pytest.CaptureFixture[str],
    fallback: str,
) -> None:
    estimator = GeneralizedAdvantageEstimator(
        GAEConfig(
            gae_gamma=0.99 if fallback == "gamma" else 1.0,
            gae_lambda=0.95 if fallback == "lambda" else 1.0,
        ),
        ClippedPGLossConfig(),
    )
    rewards = torch.tensor([[0.0, 1.0]])
    values = torch.tensor([[0.2, 0.5]])
    mask = (
        torch.full_like(rewards, 0.5)
        if fallback == "fractional"
        else torch.ones_like(rewards)
    )
    override = torch.ones(1) if fallback == "tensor_lambda" else None

    estimator._compute_gae(rewards, values, mask, gae_lambda=override)
    assert capsys.readouterr().out == ""

    # A subsequent fast-path call still logs after a silent fallback call.
    estimator.gae_gamma = 1.0
    estimator._compute_gae(rewards, values, torch.ones_like(mask), gae_lambda=1.0)
    assert capsys.readouterr().out == (
        "Fast GAE compute activated for lambda=1.0, gamma=1.0\n"
    )


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("use_kl", [False, True])
@pytest.mark.parametrize("decoupling", ["none", "fixed", "adaptive"])
def test_gae_public_path_preserves_rewards_whitening_and_decoupling(
    device: torch.device,
    normalize: bool,
    use_kl: bool,
    decoupling: str,
) -> None:
    config = GAEConfig(
        gae_lambda=1.0,
        normalize_advantages=normalize,
        gae_lambda_value=1.0 if decoupling != "none" else None,
        gae_lambda_policy=0.6 if decoupling == "fixed" else None,
        length_adaptive_alpha=1.0 if decoupling == "adaptive" else 0.0,
    )
    estimator = GeneralizedAdvantageEstimator(
        config,
        ClippedPGLossConfig(
            use_kl_in_reward=use_kl,
            reference_policy_kl_penalty=0.1,
            reference_policy_kl_type="k1",
        ),
    )
    mask = torch.tensor(
        [[0, 1, 0, 1, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0]],
        device=device,
        dtype=torch.float64,
    )
    values = torch.tensor(
        [[8, 0.2, 9, 0.7, 8], [0.1, 0.3, 0.5, 8, 8], [8, 8, 8, 8, 8]],
        device=device,
        dtype=torch.float64,
    )
    rewards = torch.tensor([1.0, -0.5, 3.0], device=device, dtype=torch.float64)
    policy_logprobs = -torch.ones_like(values)
    reference_logprobs = policy_logprobs - 0.2
    token_rewards = -0.02 * mask if use_kl else torch.zeros_like(values)
    token_rewards[0, 3] += rewards[0]
    token_rewards[1, 2] += rewards[1]
    policy_lambda = estimator._resolve_lambda_policy(mask)
    expected_advantages, _ = _reference_gae(
        token_rewards, values, mask, lam=policy_lambda
    )
    _, expected_returns = _reference_gae(token_rewards, values, mask)
    if normalize:
        expected_advantages = estimator._reward_whiten(expected_advantages, mask)
    expected_advantages = expected_advantages.masked_fill(~mask.bool(), 0)

    advantages, returns = estimator.compute_advantage(
        prompt_ids=torch.arange(3, device=device),
        rewards=rewards,
        mask=mask,
        values=values,
        logprobs_policy=policy_logprobs,
        logprobs_reference=reference_logprobs,
    )

    torch.testing.assert_close(advantages, expected_advantages)
    torch.testing.assert_close(returns, expected_returns)
    assert torch.count_nonzero(advantages[~mask.bool()]) == 0
