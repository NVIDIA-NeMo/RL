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

"""Score-centering weights and masking against full-distribution references."""

import itertools
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.loss.score_centering import compute_score_centering_term
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def test_score_centering_prepare_loss_input_values_and_gradients():
    from nemo_rl.algorithms.loss.interfaces import LossInputType
    from nemo_rl.algorithms.loss.loss_input import prepare_loss_input

    torch.manual_seed(19)
    logits = torch.randn(2, 5, 7, requires_grad=True)
    ids = torch.randint(0, 7, (2, 5, 3))
    data = BatchedDataDict(
        {
            "input_ids": ids[..., 0],
            "sampler_topk_token_ids": ids,
            "sampler_topk_logprobs": torch.zeros_like(ids, dtype=torch.float32),
        }
    )
    loss_fn = SimpleNamespace(input_type=LossInputType.LOGPROB, score_centering=True)
    prepared, _ = prepare_loss_input(logits, data, loss_fn, chunk_size=2)
    reference = logits[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:])
    torch.testing.assert_close(prepared["policy_logprobs_at_sampler_topk"], reference)
    torch.testing.assert_close(prepared["next_token_logprobs"], reference[..., 0])
    actual_grad = torch.autograd.grad(
        prepared["policy_logprobs_at_sampler_topk"].sum(), logits, retain_graph=True
    )[0]
    torch.testing.assert_close(
        actual_grad, torch.autograd.grad(reference.sum(), logits)[0]
    )


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("missing", ["sampler_topk_token_ids", "sampler_topk_logprobs"])
def test_score_centering_preparation_rejects_missing_sampler_data(packed, missing):
    from nemo_rl.algorithms.loss.interfaces import LossInputType
    from nemo_rl.algorithms.loss.loss_input import (
        prepare_loss_input,
        prepare_packed_loss_input,
    )

    data = {
        "input_ids": torch.zeros(1, 4, dtype=torch.long),
        "sampler_topk_token_ids": torch.zeros(1, 4, 2, dtype=torch.int32),
        "sampler_topk_logprobs": torch.zeros(1, 4, 2),
    }
    del data[missing]
    loss_fn = SimpleNamespace(input_type=LossInputType.LOGPROB, score_centering=True)
    with pytest.raises(
        ValueError, match=f"{missing}.*SingleController.*policy.generation.top_logprobs"
    ):
        if packed:
            prepare_packed_loss_input(
                torch.randn(1, 4, 7),
                data,
                loss_fn,
                (0, 4),
                (0, 4),
                vocab_parallel_rank=0,
                vocab_parallel_group=object(),
            )
        else:
            prepare_loss_input(torch.randn(1, 4, 7), data, loss_fn)


def _adaptive_ppo_clip_reference_weights(ratios, advantage):
    # adaptive-ppo-clip at lo = 0.5, hi = 2 and outer bound 5, written out
    # independently of adaptive_ppo_clip_kept_mask: a positive advantage drops
    # ratios above 2, a negative one ratios below 0.5, and any sign ratios
    # above 5.
    if advantage > 0:
        sign_bound = ratios <= 2.0
    elif advantage < 0:
        sign_bound = ratios >= 0.5
    else:
        sign_bound = torch.ones_like(ratios, dtype=torch.bool)
    return torch.where((ratios <= 5.0) & sign_bound, ratios, 0.0)


@pytest.mark.parametrize(
    ("mode", "advantage"),
    [
        ("off", 1.0),
        ("is", 1.0),
        ("tis", 1.0),
        ("icepop", 1.0),
        # adaptive-ppo-clip picks its bounds by the advantage's sign.
        ("adaptive-ppo-clip", 1.0),
        ("adaptive-ppo-clip", -1.0),
    ],
)
@pytest.mark.parametrize("head_probs", [(0.1, 0.6), (0.3, 0.65)])
def test_score_centering_matches_full_distribution_gradient(
    mode, advantage, head_probs
):
    logits = torch.tensor([0.6, -0.2, 0.1, -0.7], requires_grad=True)
    logp = logits.log_softmax(-1)
    q_head = torch.tensor(head_probs, requires_grad=True)
    q = torch.cat(
        [q_head.detach(), (1 - q_head.detach().sum()) * logp[2:].softmax(-1).detach()]
    )
    truncated = mode in ("tis", "icepop", "adaptive-ppo-clip")
    correction, _ = compute_score_centering_term(
        logp[:2].reshape(1, 1, 2),
        sampler_topk_logprobs=q_head.log().reshape(1, 1, 2),
        importance_sampling_mode=mode,
        is_bounds=(0.5, 2.0) if truncated else None,
        token_mask=torch.ones(1, 1),
        adaptive_ppo_clip_advantages=torch.full((1, 1), advantage),
        adaptive_ppo_clip_outer_bound=5.0,
    )
    ratios = logp.detach().exp() / q
    if mode == "off":
        weights = torch.ones_like(ratios)
    elif mode == "is":
        weights = ratios
    elif mode == "tis":
        weights = ratios.clamp(0.5, 2.0)
    elif mode == "icepop":
        weights = torch.where((ratios >= 0.5) & (ratios <= 2.0), ratios, 0.0)
    else:
        # Head ratios (4.3, 0.32) with tail 1.26 for the first head_probs,
        # (1.43, 0.30) with tail 7.55 for the second: each sign drops some, and
        # the outer bound drops the second tail for a negative advantage.
        weights = _adaptive_ppo_clip_reference_weights(ratios, advantage)
    expected = (q * weights * logp).sum()
    torch.testing.assert_close(
        torch.autograd.grad(correction.sum(), logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
        atol=1e-6,
        rtol=1e-5,
    )
    assert q_head.grad is None
    if mode == "is":
        torch.testing.assert_close(correction, torch.zeros_like(correction))


def test_score_centering_adaptive_ppo_clip_filters_by_each_positions_advantage():
    logits = torch.tensor([0.6, -0.2, 0.1, -0.7], requires_grad=True)
    logp = logits.log_softmax(-1)
    q_head = torch.tensor([0.1, 0.6])
    q = torch.cat([q_head, (1 - q_head.sum()) * logp[2:].softmax(-1).detach()])
    # Head ratios (4.3, 0.32) with tail 1.26: a positive advantage drops 4.3, a
    # negative one 0.32, and a zero one neither, so each sign has its own
    # gradient. Rows and positions mix signs; position (1, 1) is masked.
    advantages = torch.tensor([[1.0, -1.0, 0.0], [-1.0, 1.0, 1.0]])
    token_mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    correction, _ = compute_score_centering_term(
        logp[:2].expand(2, 3, 2),
        sampler_topk_logprobs=q_head.log().expand(2, 3, 2),
        importance_sampling_mode="adaptive-ppo-clip",
        is_bounds=(0.5, 2.0),
        token_mask=token_mask,
        adaptive_ppo_clip_advantages=advantages,
        adaptive_ppo_clip_outer_bound=5.0,
    )
    assert correction[1, 1] == 0
    ratios = logp.detach().exp() / q
    for row, position in token_mask.nonzero().tolist():
        weights = _adaptive_ppo_clip_reference_weights(
            ratios, advantages[row, position].item()
        )
        expected = (q * weights * logp).sum()
        (actual_grad,) = torch.autograd.grad(
            correction[row, position], logits, retain_graph=True
        )
        (expected_grad,) = torch.autograd.grad(expected, logits, retain_graph=True)
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    ("mode", "force_on_policy"),
    [
        *itertools.product(("off", "is", "tis", "icepop"), (True, False)),
        # adaptive-ppo-clip requires force_on_policy_ratio.
        ("adaptive-ppo-clip", True),
    ],
)
def test_score_centering_masks_nonfinite_positions_and_keeps_filtered_correction(
    mode, force_on_policy
):
    lp = torch.tensor([0.6, -0.2, 0.1, -0.7]).log_softmax(-1)
    token_lp = torch.tensor([[lp[0], float("nan")]], requires_grad=True)
    head_lp = torch.tensor(
        [[[lp[0], lp[1]], [float("nan"), float("inf")]]], requires_grad=True
    )
    advantage = torch.tensor([[0.0, 1.0, float("nan")]], requires_grad=True)
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[0, 0, 0]]),
            "token_mask": torch.tensor([[0.0, 1.0, 0.0]]),
            "sample_mask": torch.ones(1),
            "advantages": advantage,
            # Ratio 2 lies beyond the PPO clip range. The sampled-token gradient
            # must stay clipped when force_on_policy_ratio is disabled.
            "prev_logprobs": torch.tensor(
                [[0.0, lp[0] - torch.log(torch.tensor(2.0)), float("nan")]]
            ),
            "generation_logprobs": torch.tensor([[0.0, -2.3025851, float("nan")]]),
            "sampler_topk_logprobs": torch.tensor(
                [[[0.0, 0.0], [0.1, 0.6], [float("nan"), float("nan")]]]
            ).log(),
        }
    )
    truncated = mode in ("tis", "icepop", "adaptive-ppo-clip")
    cfg = ClippedPGLossConfig(
        score_centering=True,
        reference_policy_kl_penalty=0,
        force_on_policy_ratio=force_on_policy,
        use_importance_sampling_correction=mode != "off",
        truncated_importance_sampling_type=mode if truncated else None,
        truncated_importance_sampling_ratio=2.0 if truncated else None,
        truncated_importance_sampling_ratio_min=0.5 if truncated else None,
    )
    loss, _ = ClippedPGLossFn(cfg)(
        next_token_logprobs=token_lp,
        policy_logprobs_at_sampler_topk=head_lp,
        data=data,
        global_valid_seqs=torch.tensor(1.0),
        global_valid_toks=torch.tensor(1.0),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(token_lp.grad).all() and torch.isfinite(head_lp.grad).all()
    assert token_lp.grad[0, 1] == 0 and torch.all(head_lp.grad[0, 1] == 0)
    assert advantage.grad is None
    if not force_on_policy:
        assert token_lp.grad[0, 0] == 0
    if mode in ("icepop", "adaptive-ppo-clip"):
        # p(sampled)/q(sampled) exceeds the upper bound (for adaptive-ppo-clip,
        # the positive-advantage one): the sampled loss is zero, but the
        # head/tail correction still supplies a gradient.
        assert token_lp.grad[0, 0] == 0
        assert head_lp.grad[0, 0].abs().sum() > 0
