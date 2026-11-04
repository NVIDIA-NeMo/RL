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

from nemo_rl.algorithms.loss.omega_loss import OmegaLossConfig, OmegaLossFn
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
    data = BatchedDataDict(
        {
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "advantages": adv.expand(-1, T).clone(),
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
