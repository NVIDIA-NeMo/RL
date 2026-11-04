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

import pytest
import torch

from nemo_rl.algorithms.loss.omega_loss import (
    H_FUNCTIONS,
    OmegaLossConfig,
    OmegaLossFn,
    OmegaPerSequenceLossFn,
    compute_omega_coefficients,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

B, K, T = 3, 4, 9


def _make(seed=0, invalid_rows=()):
    g = torch.Generator().manual_seed(seed)
    logp = -torch.rand(B * K, T - 1, generator=g) * 2
    logp.requires_grad_(True)
    token_mask = torch.zeros(B * K, T)
    lens = torch.randint(2, T - 1, (B * K,), generator=g)
    for n, L in enumerate(lens):
        token_mask[n, T - L :] = 1
    rewards = torch.randint(0, 2, (B * K,), generator=g).float()
    adv = (rewards.view(B, K) - rewards.view(B, K).mean(1, keepdim=True)).view(-1, 1)
    sample_mask = torch.ones(B * K)
    for r in invalid_rows:
        sample_mask[r] = 0
    gen_lp = torch.cat([torch.zeros(B * K, 1), logp.detach()], 1)
    gen_lp = gen_lp + 0.05 * torch.randn(gen_lp.shape, generator=g)  # vLLM drift
    data = BatchedDataDict(
        {
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "advantages": adv.expand(-1, T).clone(),
            "generation_logprobs": gen_lp,
            "prev_logprobs": torch.cat([torch.zeros(B * K, 1), logp.detach()], 1),
        }
    )
    return logp, data, lens, rewards


def _reference(logp, token_mask, rewards):
    """Direct transcription of the formula, loop by loop."""
    total = 0.0
    for i in range(B):
        rows = range(i * K, (i + 1) * K)
        b = sum(rewards[j] for j in rows) / K
        scores = []
        for j in rows:
            m = token_mask[j, 1:]
            L = m.sum()
            scores.append((logp[j] * m).sum() / L)  # log pi^h with h = 1/|y|
        w = torch.softmax(torch.stack(scores), 0)
        total = total + sum(w[n] * (rewards[j] - b) for n, j in enumerate(rows)) / K
    return -total / B


def test_matches_reference_loss_and_grad():
    logp, data, _, rewards = _make()
    fn = OmegaLossFn(OmegaLossConfig(), num_generations_per_prompt=K)
    loss, metrics = fn(logp, data, global_valid_seqs=torch.tensor(float(B * K)))
    ref = _reference(logp, data["token_mask"], rewards)
    torch.testing.assert_close(loss, ref)
    g = torch.autograd.grad(loss, logp)[0]
    g_ref = torch.autograd.grad(ref, logp)[0]
    torch.testing.assert_close(g, g_ref)
    assert g.abs().sum() > 0  # gradient flows through the weights
    assert 1 / K <= metrics["weight_concentration"] <= 1


def test_uniform_reward_group_gives_zero_loss():
    logp, data, _, _ = _make()
    data["advantages"] = torch.zeros_like(data["advantages"])
    fn = OmegaLossFn(OmegaLossConfig(), num_generations_per_prompt=K)
    loss, _ = fn(logp, data, global_valid_seqs=torch.tensor(float(B * K)))
    assert loss.item() == 0.0


def test_invalid_rows_get_no_weight_or_grad():
    logp, data, _, _ = _make(invalid_rows=(1, 5))
    fn = OmegaLossFn(OmegaLossConfig(), num_generations_per_prompt=K)
    loss, _ = fn(logp, data, global_valid_seqs=torch.tensor(float(B * K - 2)))
    assert torch.isfinite(loss)
    g = torch.autograd.grad(loss, logp)[0]
    assert g[1].abs().sum() == 0 and g[5].abs().sum() == 0


def test_split_groups_rejected():
    logp, data, _, _ = _make()
    fn = OmegaLossFn(OmegaLossConfig(), num_generations_per_prompt=5)
    with pytest.raises(AssertionError):
        fn(logp, data, global_valid_seqs=torch.tensor(float(B * K)))


def test_unknown_h_rejected():
    with pytest.raises(ValueError):
        OmegaLossFn(OmegaLossConfig(h="nope"), num_generations_per_prompt=K)


def test_empty_response_is_dropped_not_dominant():
    logp, data, _, _ = _make()
    data["token_mask"][2] = 0  # row 2 generated nothing: |y| = 0
    fn = OmegaLossFn(OmegaLossConfig(), num_generations_per_prompt=K)
    loss, metrics = fn(logp, data, global_valid_seqs=torch.tensor(float(B * K - 1)))
    assert torch.isfinite(loss)
    assert metrics["num_empty_responses"] == 1
    g = torch.autograd.grad(loss, logp)[0]
    assert g[2].abs().sum() == 0


def _per_sequence_grad(logp, data, prompt_ids, n_valid, rows_separately=False):
    """Gradient of the per-sequence form, with coefficients from logp's values."""
    prev = torch.cat([torch.zeros(logp.shape[0], 1), logp.detach()], dim=1)
    coef, metrics = compute_omega_coefficients(
        prompt_ids=prompt_ids,
        prev_logprobs=prev,
        token_mask=data["token_mask"],
        sample_mask=data["sample_mask"],
        advantages=data["advantages"],
        h_fn=H_FUNCTIONS["inverse_length"],
        data=data,
    )
    data = BatchedDataDict({**data, "omega_coef": coef})
    fn = OmegaPerSequenceLossFn()
    gvs = torch.tensor(float(n_valid))
    if not rows_separately:
        loss, _ = fn(logp, data, gvs)
        return torch.autograd.grad(loss, logp)[0], metrics
    # Mimic the non-fused packing wrapper: one sequence per call, losses summed.
    total = 0
    for i in range(logp.shape[0]):
        row = BatchedDataDict({k: v[i : i + 1] for k, v in data.items()})
        loss, _ = fn(logp[i : i + 1], row, gvs)
        total = total + loss
    return torch.autograd.grad(total, logp)[0], metrics


def _prompt_ids():
    # Distinct prompt tokens per group; identical within a group.
    return torch.arange(B).repeat_interleave(K).unsqueeze(-1).expand(-1, 5).clone()


def test_per_sequence_gradient_matches_exact_group_form():
    logp, data, _, _ = _make()
    exact = OmegaLossFn(OmegaLossConfig(), num_generations_per_prompt=K)
    loss, _ = exact(logp, data, torch.tensor(float(B * K)))
    g_exact = torch.autograd.grad(loss, logp)[0]
    g, metrics = _per_sequence_grad(logp, data, _prompt_ids(), B * K)
    torch.testing.assert_close(g, g_exact, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(  # logged objective equals the exact loss value
        torch.tensor(metrics["objective"]), loss.detach(), rtol=1e-4, atol=1e-6
    )
    g_rows, _ = _per_sequence_grad(
        logp, data, _prompt_ids(), B * K, rows_separately=True
    )
    torch.testing.assert_close(g_rows, g_exact, rtol=1e-4, atol=1e-6)


def test_per_sequence_groups_by_prompt_not_row_order():
    logp, data, _, _ = _make()
    pids = _prompt_ids()
    g_ref, _ = _per_sequence_grad(logp, data, pids, B * K)
    perm = torch.randperm(B * K, generator=torch.Generator().manual_seed(1))
    logp_p = logp.detach()[perm].clone().requires_grad_(True)
    data_p = BatchedDataDict({k: v[perm] for k, v in data.items()})
    g_p, _ = _per_sequence_grad(logp_p, data_p, pids[perm], B * K)
    torch.testing.assert_close(g_p, g_ref[perm], rtol=1e-4, atol=1e-6)


def test_per_sequence_invalid_and_empty_rows():
    logp, data, _, _ = _make(invalid_rows=(1,))
    data["token_mask"][6] = 0  # empty response
    g, metrics = _per_sequence_grad(logp, data, _prompt_ids(), B * K - 2)
    assert torch.isfinite(g).all()
    assert g[1].abs().sum() == 0 and g[6].abs().sum() == 0
    assert metrics["num_empty_responses"] == 1
    # Matches the exact form on the same masked data.
    data_e = BatchedDataDict({**data, "sample_mask": data["sample_mask"].clone()})
    exact = OmegaLossFn(OmegaLossConfig(), num_generations_per_prompt=K)
    loss, _ = exact(logp, data_e, torch.tensor(float(B * K - 2)))
    torch.testing.assert_close(
        g, torch.autograd.grad(loss, logp)[0], rtol=1e-4, atol=1e-6
    )


def _coef(logp, data, prompt_ids):
    prev = torch.cat([torch.zeros(logp.shape[0], 1), logp.detach()], dim=1)
    return compute_omega_coefficients(
        prompt_ids=prompt_ids,
        prev_logprobs=prev,
        token_mask=data["token_mask"],
        sample_mask=data["sample_mask"],
        advantages=data["advantages"],
        h_fn=H_FUNCTIONS["inverse_length"],
        data=data,
    )


def test_per_sequence_fully_invalid_group_and_single_valid_response():
    # Group 0: all invalid. Group 1: only row K valid (w = 1, so A - Abar = 0).
    invalid = tuple(range(K)) + tuple(range(K + 1, 2 * K))
    logp, data, _, _ = _make(invalid_rows=invalid)
    coef, metrics = _coef(logp, data, _prompt_ids())
    assert torch.isfinite(coef).all()
    assert coef[: 2 * K].abs().sum() == 0
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())


def test_per_sequence_extreme_logprobs_stay_finite():
    logp, data, _, _ = _make()
    with torch.no_grad():
        logp[0] -= 5e4  # one response wildly unlikely
        logp[K] = 0.0  # one response certain
    coef, _ = _coef(logp, data, _prompt_ids())
    assert torch.isfinite(coef).all()
    g, _ = _per_sequence_grad(logp, data, _prompt_ids(), B * K)
    assert torch.isfinite(g).all()


def test_per_sequence_accepts_bf16_logprobs():
    logp, data, _, _ = _make()
    prev = torch.cat([torch.zeros(logp.shape[0], 1), logp.detach()], 1).bfloat16()
    coef, _ = compute_omega_coefficients(
        prompt_ids=_prompt_ids(),
        prev_logprobs=prev,
        token_mask=data["token_mask"],
        sample_mask=data["sample_mask"],
        advantages=data["advantages"],
        h_fn=H_FUNCTIONS["inverse_length"],
    )
    assert coef.dtype == torch.float32 and torch.isfinite(coef).all()


def _clipped_pg_mult_prob_error(data, global_valid_toks):
    """Reference: the ClippedPGLossFn computation of token_mult_prob_error."""
    token_mask = data["token_mask"][:, 1:]
    mask = token_mask * data["sample_mask"].unsqueeze(-1)
    gen = torch.where(mask.bool(), data["generation_logprobs"][:, 1:], 0.0)
    prev = torch.where(mask.bool(), data["prev_logprobs"][:, 1:], 0.0)
    return (torch.exp(torch.abs(gen - prev) * mask) * mask).sum() / (
        global_valid_toks + 1e-8
    )


def test_both_forms_report_token_mult_prob_error_like_grpo():
    """grpo_train reads metrics["token_mult_prob_error"] after every step."""
    logp, data, _, _ = _make(invalid_rows=(3,))
    gvt = (data["token_mask"][:, 1:] * data["sample_mask"].unsqueeze(-1)).sum()
    ref = _clipped_pg_mult_prob_error(data, gvt).item()
    assert ref > 1.0  # the injected drift is visible

    group = OmegaLossFn(OmegaLossConfig(), num_generations_per_prompt=K)
    _, m = group(logp, data, torch.tensor(float(B * K - 1)), gvt)
    assert abs(m["token_mult_prob_error"] - ref) < 1e-5

    data["omega_coef"] = torch.zeros(B * K)
    _, m = OmegaPerSequenceLossFn()(logp, data, torch.tensor(float(B * K - 1)), gvt)
    assert abs(m["token_mult_prob_error"] - ref) < 1e-5

    # Packed, non-fused path: one row per call, metrics summed by the wrapper.
    total = sum(
        OmegaPerSequenceLossFn()(
            logp[i : i + 1],
            BatchedDataDict({k: v[i : i + 1] for k, v in data.items()}),
            torch.tensor(float(B * K - 1)),
            gvt,
        )[1]["token_mult_prob_error"]
        for i in range(B * K)
    )
    assert abs(total - ref) < 1e-4
