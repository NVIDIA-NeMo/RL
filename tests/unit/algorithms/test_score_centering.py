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
import math
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.loss.score_centering import compute_score_centering_term
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

_REPLAY = TrainingSamplingParams(top_k=4, top_p=0.9, sampler_replay=True)


def _captured_head(batch_size, seq_len, vocab_size, k, filtered_slot=None):
    """Distinct sampler head ids; ``-inf`` sampler log-probs on ``filtered_slot``.

    The finite entries carry all of the sampler's mass, as vLLM reports them.
    """
    ids = torch.stack(
        [torch.randperm(vocab_size)[:k] for _ in range(batch_size * seq_len)]
    ).view(batch_size, seq_len, k)
    kept = k if filtered_slot is None else k - 1
    logprobs = torch.full((batch_size, seq_len, k), -math.log(kept))
    if filtered_slot is not None:
        logprobs[..., filtered_slot] = -float("inf")
    return ids, logprobs


@pytest.mark.parametrize("sampling_params", [None, _REPLAY])
def test_score_centering_prepare_loss_input_values_and_gradients(sampling_params):
    from nemo_rl.algorithms.loss.interfaces import LossInputType
    from nemo_rl.algorithms.loss.loss_input import prepare_loss_input

    torch.manual_seed(19)
    logits = torch.randn(2, 5, 7, requires_grad=True)
    filtered_slot = None if sampling_params is None else 2
    ids, sampler_logprobs = _captured_head(2, 5, 7, 3, filtered_slot)
    data = BatchedDataDict(
        {
            "input_ids": ids[..., 0],
            "token_mask": torch.ones(2, 5),
            "sample_mask": torch.ones(2),
            "sampler_topk_token_ids": ids,
            "sampler_topk_logprobs": sampler_logprobs,
        }
    )
    loss_fn = SimpleNamespace(input_type=LossInputType.LOGPROB, score_centering=True)
    prepared, _ = prepare_loss_input(
        logits, data, loss_fn, sampling_params=sampling_params, chunk_size=2
    )
    if sampling_params is None:
        reference = logits[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:])
    else:
        # Replay: renormalize over the sampler's realized support S_q.
        support = ids[:, 1:].masked_fill(torch.isneginf(sampler_logprobs[:, 1:]), 0)
        z = logits[:, :-1].gather(-1, support)
        z = z.masked_fill(torch.isneginf(sampler_logprobs[:, 1:]), -float("inf"))
        reference = z - z.logsumexp(-1, keepdim=True)
        assert torch.isneginf(reference[..., 2]).all()
    torch.testing.assert_close(prepared["policy_logprobs_at_sampler_topk"], reference)
    torch.testing.assert_close(prepared["next_token_logprobs"], reference[..., 0])
    finite = torch.isfinite(reference)
    actual_grad = torch.autograd.grad(
        prepared["policy_logprobs_at_sampler_topk"][finite].sum()
        + prepared["next_token_logprobs"].sum(),
        logits,
        retain_graph=True,
    )[0]
    torch.testing.assert_close(
        actual_grad,
        torch.autograd.grad(reference[finite].sum() + reference[..., 0].sum(), logits)[
            0
        ],
    )


def test_score_centering_replay_needs_no_vocabulary_top_k(monkeypatch):
    """Replaying S_q gathers the head logits; it never takes a vocabulary top-k."""
    from nemo_rl.algorithms.loss.interfaces import LossInputType
    from nemo_rl.algorithms.loss.loss_input import prepare_loss_input
    from nemo_rl.distributed import model_utils

    def no_top_k(*args, **kwargs):
        raise AssertionError("vocabulary top-k under sampler replay")

    monkeypatch.setattr(model_utils, "apply_top_k_top_p", no_top_k)
    ids, sampler_logprobs = _captured_head(2, 5, 7, 3, filtered_slot=1)
    data = BatchedDataDict(
        {
            "input_ids": ids[..., 0],
            "token_mask": torch.ones(2, 5),
            "sample_mask": torch.ones(2),
            "sampler_topk_token_ids": ids,
            "sampler_topk_logprobs": sampler_logprobs,
        }
    )
    loss_fn = SimpleNamespace(input_type=LossInputType.LOGPROB, score_centering=True)
    prepared, _ = prepare_loss_input(
        torch.randn(2, 5, 7), data, loss_fn, sampling_params=_REPLAY, chunk_size=2
    )
    assert torch.isfinite(prepared["next_token_logprobs"]).all()


def test_sampler_support_replay_without_score_centering(monkeypatch):
    """The actor loss renormalizes over S_q without score centering's head."""
    from nemo_rl.algorithms.loss.interfaces import LossInputType
    from nemo_rl.algorithms.loss.loss_input import prepare_loss_input
    from nemo_rl.distributed import model_utils

    def no_top_k(*args, **kwargs):
        raise AssertionError("vocabulary top-k under sampler replay")

    monkeypatch.setattr(model_utils, "apply_top_k_top_p", no_top_k)
    torch.manual_seed(23)
    logits = torch.randn(2, 5, 7, requires_grad=True)
    ids, sampler_logprobs = _captured_head(2, 5, 7, 3, filtered_slot=1)
    data = BatchedDataDict(
        {
            "input_ids": ids[..., 0],
            "token_mask": torch.ones(2, 5),
            "sample_mask": torch.ones(2),
            "sampler_topk_token_ids": ids,
            "sampler_topk_logprobs": sampler_logprobs,
        }
    )
    loss_fn = SimpleNamespace(input_type=LossInputType.LOGPROB, score_centering=False)
    prepared, _ = prepare_loss_input(
        logits, data, loss_fn, sampling_params=_REPLAY, chunk_size=2
    )

    assert "policy_logprobs_at_sampler_topk" not in prepared
    assert prepared["policy_support_mask"].all()
    # The sampled token (slot 0) renormalized over the finite slots (0 and 2).
    support = ids[:, 1:][..., [0, 2]]
    expected = logits[:, :-1].gather(-1, support).log_softmax(-1)[..., 0]
    torch.testing.assert_close(prepared["next_token_logprobs"], expected)
    weights = torch.randn(expected.shape)
    grad = torch.autograd.grad(
        (prepared["next_token_logprobs"] * weights).sum(), logits, retain_graph=True
    )[0]
    torch.testing.assert_close(
        grad, torch.autograd.grad((expected * weights).sum(), logits)[0]
    )


def test_sampler_support_replay_needs_the_captured_head():
    """Replay without score centering's captured head fails with a clear message."""
    from nemo_rl.algorithms.loss.loss_input import sampler_support_logprobs

    data = BatchedDataDict(
        {
            "input_ids": torch.zeros(1, 4, dtype=torch.long),
            "token_mask": torch.ones(1, 4),
            "sample_mask": torch.ones(1),
        }
    )
    with pytest.raises(ValueError, match="sampler_topk_token_ids"):
        sampler_support_logprobs(torch.randn(1, 4, 7), data, _REPLAY)


def test_sampler_head_under_sampling_filters_requires_replay():
    """Centering under filters is only defined over the sampler's support (no S_p head)."""
    from nemo_rl.algorithms.loss.interfaces import LossInputType
    from nemo_rl.algorithms.loss.loss_input import prepare_loss_input
    from nemo_rl.distributed.model_utils import get_policy_logprobs_at_sampler_topk

    recompute = TrainingSamplingParams(top_k=4, top_p=0.9)
    with pytest.raises(ValueError, match="sampler_replay"):
        get_policy_logprobs_at_sampler_topk(
            sampler_topk_token_ids=torch.zeros(1, 4, 2, dtype=torch.long),
            next_token_logits=torch.randn(1, 4, 7),
            sampling_params=recompute,
        )
    ids, sampler_logprobs = _captured_head(1, 4, 7, 3, filtered_slot=2)
    data = BatchedDataDict(
        {
            "input_ids": ids[..., 0],
            "token_mask": torch.ones(1, 4),
            "sample_mask": torch.ones(1),
            "sampler_topk_token_ids": ids,
            "sampler_topk_logprobs": sampler_logprobs,
        }
    )
    loss_fn = SimpleNamespace(input_type=LossInputType.LOGPROB, score_centering=True)
    with pytest.raises(ValueError, match="sampler_replay.enabled=true"):
        prepare_loss_input(
            torch.randn(1, 4, 7), data, loss_fn, sampling_params=recompute
        )


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("corruption", ["sampled_outside", "no_finite_head"])
def test_score_centering_replay_rejects_a_corrupt_capture(packed, corruption):
    """A sampled token outside its captured S_q, or an empty S_q, is a capture bug."""
    from nemo_rl.algorithms.loss.interfaces import LossInputType
    from nemo_rl.algorithms.loss.loss_input import (
        prepare_loss_input,
        prepare_packed_loss_input,
    )

    ids, sampler_logprobs = _captured_head(1, 4, 7, 3, filtered_slot=2)
    input_ids = ids[..., 0].clone()
    if corruption == "sampled_outside":
        input_ids[0, 2] = ids[0, 2, 2]  # the filtered slot's id
        match = "not in its captured sampler support"
    else:
        sampler_logprobs[0, 2] = -float("inf")
        match = "no finite sampler candidate"
    token_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "token_mask": token_mask,
            "sample_mask": torch.ones(1),
            "sampler_topk_token_ids": ids,
            "sampler_topk_logprobs": sampler_logprobs,
        }
    )
    loss_fn = SimpleNamespace(input_type=LossInputType.LOGPROB, score_centering=True)
    with pytest.raises(ValueError, match=match):
        if packed:
            prepare_packed_loss_input(
                torch.randn(1, 4, 7),
                data,
                loss_fn,
                (0, 4),
                (0, 4),
                vocab_parallel_rank=0,
                vocab_parallel_group=object(),
                sampling_params=_REPLAY,
            )
        else:
            prepare_loss_input(
                torch.randn(1, 4, 7), data, loss_fn, sampling_params=_REPLAY
            )
    # Prompt positions (token_mask 0) are not checked.
    token_mask[0, 2] = 0.0
    prepare_loss_input(torch.randn(1, 4, 7), data, loss_fn, sampling_params=_REPLAY)


@pytest.mark.parametrize("score_centering", [False, True])
@pytest.mark.parametrize(
    ("missing_mass", "sampled_in_head"),
    [
        # The kept ties the head left out hold 0.1 of the sampler's mass.
        (0.1, True),
        (0.1, False),
        # They hold 5e-5, under the mass tolerance, and the sampler drew one.
        (5e-5, False),
    ],
)
def test_replay_drops_tokens_whose_head_misses_support(
    score_centering, missing_mass, sampled_in_head
):
    """A full head the sampler outgrew (ties at the k-th logit) drops that token, not training."""
    from nemo_rl.algorithms.loss.interfaces import LossInputType
    from nemo_rl.algorithms.loss.loss_input import prepare_loss_input

    ids, sampler_logprobs = _captured_head(1, 4, 7, 3)
    sampler_logprobs[0, 2] = torch.tensor([0.5, 0.3, 0.2 - missing_mass]).log()
    input_ids = ids[..., 0].clone()
    if not sampled_in_head:
        # A tied token the head did not capture.
        input_ids[0, 2] = next(i for i in range(7) if i not in ids[0, 2].tolist())
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "token_mask": torch.tensor([[0.0, 1.0, 1.0, 1.0]]),
            "sample_mask": torch.ones(1),
            "sampler_topk_token_ids": ids,
            "sampler_topk_logprobs": sampler_logprobs,
        }
    )
    loss_fn = SimpleNamespace(
        input_type=LossInputType.LOGPROB, score_centering=score_centering
    )
    prepared, _ = prepare_loss_input(
        torch.randn(1, 4, 7), data, loss_fn, sampling_params=_REPLAY
    )

    # Next-token position 1 predicts token 2.
    logprobs = prepared["next_token_logprobs"]
    assert torch.isneginf(logprobs[0, 1])
    assert torch.isfinite(logprobs[0, [0, 2]]).all()
    assert prepared["policy_support_mask"].tolist() == [[True, False, True]]


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
        sampler_replay=False,
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


@pytest.mark.parametrize("mode", ["off", "is", "tis", "icepop", "adaptive-ppo-clip"])
def test_score_centering_is_exact_over_the_sampler_support(mode):
    """Replay: p and q share S_q and the head covers it, so there is no tail (rho = 0)."""
    logits = torch.tensor([0.6, -0.2, 0.1, -0.7, -1.5], requires_grad=True)
    # Head ids 0..4; the sampler filtered ids 3 and 4 (-inf), so S_q = {0, 1, 2}.
    kept = torch.tensor([True, True, True, False, False])
    logp = logits.masked_fill(~kept, -float("inf")).log_softmax(-1)
    p = logp.detach().exp()
    q = torch.tensor([0.5, 0.3, 0.2, 0.0, 0.0])
    lo, hi = 0.5, 2.0
    ratio = torch.where(kept, p / q, 0.0)
    if mode == "off":
        mass = q
    elif mode == "is":
        mass = p
    elif mode == "tis":
        mass = q * ratio.clamp(lo, hi)
    elif mode == "icepop":
        mass = torch.where((ratio >= lo) & (ratio <= hi), p, 0.0)
    else:
        mass = q * _adaptive_ppo_clip_reference_weights(ratio, 1.0)
    expected = (mass[kept] * logp[kept]).sum()

    correction, head_mass = compute_score_centering_term(
        logp.reshape(1, 1, -1),
        sampler_topk_logprobs=q.log().reshape(1, 1, -1),
        importance_sampling_mode=mode,
        is_bounds=None if mode in ("off", "is") else (lo, hi),
        token_mask=torch.ones(1, 1),
        sampler_replay=True,
        adaptive_ppo_clip_advantages=torch.ones(1, 1),
        adaptive_ppo_clip_outer_bound=5.0,
    )

    assert torch.isfinite(correction).all() and correction.abs().max() < 10
    torch.testing.assert_close(head_mass, torch.ones(1, 1))
    torch.testing.assert_close(
        torch.autograd.grad(correction.sum(), logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
        atol=1e-6,
        rtol=1e-5,
    )
    if mode == "is":
        torch.testing.assert_close(correction, torch.zeros_like(correction))


@pytest.mark.parametrize("mode", ["off", "is", "tis", "icepop", "adaptive-ppo-clip"])
def test_score_centering_replay_has_no_tail(mode):
    """Under replay the head holds all of p, so rho is zero.

    A complete head may still miss up to 1e-4 of the sampler's mass to rounding.
    Dividing that by the empty trainer tail, clamped to 1e-6, would scale the
    term's value by about 50 (the gradient is unaffected, but the value enters
    the reported loss).
    """
    logits = torch.tensor([0.6, -0.2, 0.1], requires_grad=True)
    logp = logits.log_softmax(-1)
    q = torch.tensor([0.5, 0.3, 0.19995])
    correction, _ = compute_score_centering_term(
        logp.reshape(1, 1, -1),
        sampler_topk_logprobs=q.log().reshape(1, 1, -1),
        importance_sampling_mode=mode,
        is_bounds=None if mode in ("off", "is") else (0.5, 2.0),
        token_mask=torch.ones(1, 1),
        sampler_replay=True,
        adaptive_ppo_clip_advantages=torch.ones(1, 1),
        adaptive_ppo_clip_outer_bound=5.0,
    )
    assert torch.isfinite(correction).all() and correction.abs().max() < 10


def test_score_centering_keeps_a_small_positive_trainer_tail_without_replay():
    """Unfiltered capture models the sampler's tail on the trainer's, however small.

    The top-2 head leaves a trainer tail of 9e-7 but a sampler tail of 0.2. The
    estimate rho = 0.2 / 1e-6 (the clamp) cancels the expected tail-logit
    gradient down to about 0.02; rho = 0 would leave all 0.2 of it.
    """
    p = torch.tensor([0.6, 0.3999991, 9e-7], dtype=torch.float64)
    q = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    logits = p.log().requires_grad_()
    logp = logits.log_softmax(-1)
    correction, _ = compute_score_centering_term(
        logp[:2].reshape(1, 1, 2),
        sampler_topk_logprobs=q[:2].log().reshape(1, 1, 2),
        importance_sampling_mode="off",
        is_bounds=None,
        token_mask=torch.ones(1, 1),
        sampler_replay=False,
    )
    # E_q[grad log p_y] = q - p, minus the gradient of the centering term.
    centered = (q - p) - torch.autograd.grad(correction.sum(), logits)[0]
    rho = 0.2 / 1e-6
    torch.testing.assert_close(
        centered[2],
        torch.tensor(0.2 - rho * 9e-7, dtype=torch.float64),
        atol=1e-6,
        rtol=0,
    )


def _replayed_score_centering_loss(mode, sampled_ids, token_level_loss, sample_weight):
    """Loss for response tokens drawn from S_q = {0, 1, 2, 3}, the trainer replaying it."""
    logits = torch.tensor([0.6, -0.2, 0.1, -0.7, -1.5], requires_grad=True)
    kept = torch.tensor([True, True, True, True, False])
    logp = logits.masked_fill(~kept, -float("inf")).log_softmax(-1)
    q = torch.tensor([0.4, 0.3, 0.2, 0.1, 0.0])
    ids = torch.tensor(sampled_ids)
    n = len(sampled_ids)
    data = BatchedDataDict(
        {
            "token_mask": torch.tensor([[0.0] + [1.0] * n]),
            "sample_mask": torch.tensor([sample_weight]),
            "advantages": torch.tensor([[0.0] + [1.5] * n]),
            "generation_logprobs": torch.cat([torch.zeros(1), q[ids].log()])[None],
            "sampler_topk_logprobs": torch.cat(
                [torch.zeros(1, 5), q.log().expand(n, 5)]
            )[None],
        }
    )
    token_lp = logp[ids][None]
    tis = mode in ("tis", "icepop", "adaptive-ppo-clip")
    cfg = ClippedPGLossConfig(
        score_centering=True,
        reference_policy_kl_penalty=0,
        force_on_policy_ratio=True,
        token_level_loss=token_level_loss,
        use_importance_sampling_correction=mode != "off",
        truncated_importance_sampling_type=mode if tis else None,
        truncated_importance_sampling_ratio=1.3 if tis else None,
        truncated_importance_sampling_ratio_min=0.7 if tis else None,
    )
    loss, _ = ClippedPGLossFn(cfg)(
        next_token_logprobs=token_lp,
        policy_logprobs_at_sampler_topk=logp.expand(n, 5)[None],
        data=data,
        global_valid_seqs=torch.tensor(1.0),
        global_valid_toks=torch.tensor(float(n)),
        policy_support_mask=~torch.isneginf(token_lp),
    )
    return loss, logits, q


@pytest.mark.parametrize("sample_weight", [1.0, 0.5])  # a fractional loss_multiplier
@pytest.mark.parametrize("token_level_loss", [True, False])
@pytest.mark.parametrize("num_tokens", [1, 2])
@pytest.mark.parametrize("mode", ["off", "is", "tis", "icepop", "adaptive-ppo-clip"])
def test_score_centering_cancels_in_expectation_over_the_sampler_support(
    mode, num_tokens, token_level_loss, sample_weight
):
    """With S_q replayed and a constant advantage, the expected gradient is zero."""
    expected_grad = torch.zeros(5)
    for sampled_ids in itertools.product(range(4), repeat=num_tokens):
        loss, logits, q = _replayed_score_centering_loss(
            mode, sampled_ids, token_level_loss, sample_weight
        )
        weight = q[list(sampled_ids)].prod()
        expected_grad += weight * torch.autograd.grad(loss, logits)[0]
    torch.testing.assert_close(expected_grad, torch.zeros(5), atol=1e-6, rtol=0)


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
        sampler_replay=False,
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
