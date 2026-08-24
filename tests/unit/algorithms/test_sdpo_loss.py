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

"""Unit tests for SDPOLossFn.

These tests exercise the loss math directly with hand-built top-k log-prob
tensors so they are CPU-friendly and do not require a real model forward.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from nemo_rl.algorithms.loss import SDPOLossFn


def _build_inputs(
    batch_size: int = 4,
    seq_len: int = 6,
    topk: int = 5,
    vocab_size: int = 32,
    student_logits: torch.Tensor | None = None,
    teacher_logits: torch.Tensor | None = None,
    sdpo_mask: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
):
    """Build a self-consistent set of inputs for SDPOLossFn.

    Returns the student top-k logprobs (gathered at teacher indices), the
    teacher top-k logprobs, full-vocab student entropy H_all, and the data
    dict expected by the loss.
    """
    torch.manual_seed(0)

    if student_logits is None:
        student_logits = torch.randn(batch_size, seq_len - 1, vocab_size)
    if teacher_logits is None:
        teacher_logits = torch.randn(batch_size, seq_len - 1, vocab_size)

    student_logp = F.log_softmax(student_logits, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits, dim=-1)

    teacher_topk_logits, teacher_topk_indices = teacher_logits.topk(topk, dim=-1)
    teacher_topk_logp = teacher_logp.gather(-1, teacher_topk_indices)
    student_topk_logp = student_logp.gather(-1, teacher_topk_indices)

    # Full-vocab student entropy (negative): sum_v p log p
    H_all = (student_logp.exp() * student_logp).sum(-1)

    if token_mask is None:
        # token_mask in data is [B, S] (length seq_len); the loss slices to S-1.
        token_mask = torch.ones(batch_size, seq_len)
    if sdpo_mask is None:
        sdpo_mask = torch.ones(batch_size)

    data = {
        "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "token_mask": token_mask,
        "sample_mask": torch.ones(batch_size),
        "sdpo_mask": sdpo_mask,
    }

    global_valid_toks = (token_mask[:, 1:] * sdpo_mask.unsqueeze(-1)).sum()
    return student_topk_logp, teacher_topk_logp, H_all, data, global_valid_toks


@pytest.mark.parametrize("kl_type", ["forward", "reverse", "mixed", "js"])
def test_sdpo_loss_zero_when_teacher_equals_student(kl_type):
    """When teacher == student, KL is exactly zero at every position."""
    logits = torch.randn(2, 5, 16)
    s, t, H, data, gvt = _build_inputs(
        batch_size=2,
        seq_len=6,
        vocab_size=16,
        student_logits=logits,
        teacher_logits=logits.clone(),
    )
    loss_fn = SDPOLossFn(
        {
            "kl_type": kl_type,
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,  # top-k-only KL; no tail correction
            "success_reward_threshold": 1.0,
        }
    )
    loss, _ = loss_fn(s, t, H, data, torch.tensor(2.0), gvt)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-5), loss.item()


def test_sdpo_loss_positive_when_distributions_differ():
    """A real KL divergence is non-negative and strictly positive when
    teacher ≠ student (here, by construction)."""
    s, t, H, data, gvt = _build_inputs()
    loss_fn = SDPOLossFn(
        {
            "kl_type": "reverse",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,  # top-k-only KL; no tail correction
            "success_reward_threshold": 1.0,
        }
    )
    loss, metrics = loss_fn(s, t, H, data, torch.tensor(4.0), gvt)
    assert loss.item() > 0
    assert metrics["sdpo/per_pos_kl"] > 0


def test_sdpo_loss_zero_when_no_demos():
    """Samples without a demonstration (sdpo_mask=0) contribute zero."""
    s, t, H, data, _ = _build_inputs()
    data["sdpo_mask"] = torch.zeros_like(data["sdpo_mask"])
    loss_fn = SDPOLossFn(
        {
            "kl_type": "reverse",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,  # top-k-only KL; no tail correction
            "success_reward_threshold": 1.0,
        }
    )
    # global_valid_toks=1 to avoid division by zero; mask reduces to all-zeros
    loss, _ = loss_fn(s, t, H, data, torch.tensor(4.0), torch.tensor(1.0))
    assert loss.item() == 0.0


def test_sdpo_loss_detects_disagreement_outside_sampled_token():
    """The whole point of the fix: when student and teacher agree on the
    *sampled* token but disagree on other top-k tokens, the new loss
    detects it (the previous sampled-token-only REINFORCE form would not).

    We construct a vocab of 4 tokens, top-k=4 (full distribution). Student
    and teacher both put 60% mass on token 0; they redistribute the other
    40% differently across tokens 1-3. Reverse KL is strictly positive."""
    batch_size, seq_len, vocab_size, k = 1, 2, 4, 4

    student_p = torch.tensor([[[0.6, 0.2, 0.1, 0.1]]])  # [1, 1, 4]
    teacher_p = torch.tensor([[[0.6, 0.05, 0.05, 0.3]]])  # [1, 1, 4]
    # Same prob on token 0 (the "sampled" token); different elsewhere.

    student_logits = torch.log(student_p)
    teacher_logits = torch.log(teacher_p)
    student_logp = F.log_softmax(student_logits, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits, dim=-1)

    # Top-k = full vocab here, so gather is identity (after sort).
    teacher_topk_logits, teacher_topk_indices = teacher_logits.topk(k, dim=-1)
    teacher_topk_logp = teacher_logp.gather(-1, teacher_topk_indices)
    student_topk_logp = student_logp.gather(-1, teacher_topk_indices)
    H_all = (student_logp.exp() * student_logp).sum(-1)

    data = {
        "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "token_mask": torch.ones(batch_size, seq_len),
        "sample_mask": torch.ones(batch_size),
        "sdpo_mask": torch.ones(batch_size),
    }
    loss_fn = SDPOLossFn(
        {
            "kl_type": "reverse",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,  # full vocab covered → no tail correction needed
            "success_reward_threshold": 1.0,
        }
    )
    gvt = torch.tensor(1.0)  # one valid response position
    loss, _ = loss_fn(student_topk_logp, teacher_topk_logp, None, data, torch.tensor(1.0), gvt)

    # Expected reverse KL: sum_v p_s(v) [log p_s(v) - log p_t(v)]
    expected = (student_p * (student_p.log() - teacher_p.log())).sum().item()
    assert math.isclose(loss.item(), expected, rel_tol=1e-4, abs_tol=1e-5), f"got {loss.item()}, expected {expected}"
    assert loss.item() > 0


def test_sdpo_loss_token_mask_excludes_prompt_positions():
    """token_mask=0 at a position should remove it from the loss average."""
    batch_size, seq_len, topk, vocab = 2, 6, 4, 16
    s, t, H, data, _ = _build_inputs(batch_size=batch_size, seq_len=seq_len, topk=topk, vocab_size=vocab)
    # First half of each sequence is "prompt" (mask=0)
    tm = torch.ones(batch_size, seq_len)
    tm[:, : seq_len // 2] = 0.0
    data["token_mask"] = tm

    loss_fn = SDPOLossFn(
        {
            "kl_type": "reverse",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": True,  # exercise the tail-correction path
            "success_reward_threshold": 1.0,
        }
    )
    gvt = (tm[:, 1:] * data["sdpo_mask"].unsqueeze(-1)).sum()
    loss_masked, _ = loss_fn(s, t, H, data, torch.tensor(2.0), gvt)

    # Same data with full token_mask should give a different (generally larger
    # in absolute mean) loss because the average covers more positions.
    data["token_mask"] = torch.ones(batch_size, seq_len)
    gvt_full = data["token_mask"][:, 1:].sum()
    loss_full, _ = loss_fn(s, t, H, data, torch.tensor(2.0), gvt_full)

    assert loss_masked.item() != loss_full.item()


def test_sdpo_loss_invalid_config_raises():
    with pytest.raises(ValueError, match="kl_type"):
        SDPOLossFn(
            {
                "kl_type": "bogus",
                "mixed_kl_weight": 0.5,
                "zero_outside_topk": False,
                "success_reward_threshold": 1.0,
            }
        )
    with pytest.raises(ValueError, match="mixed_kl_weight"):
        SDPOLossFn(
            {
                "kl_type": "mixed",
                "mixed_kl_weight": 1.5,
                "zero_outside_topk": False,
                "success_reward_threshold": 1.0,
            }
        )


def test_sdpo_loss_js_is_symmetric():
    """JS divergence is symmetric in student/teacher."""
    logits_a = torch.randn(2, 5, 16)
    logits_b = torch.randn(2, 5, 16)
    cfg = {
        "kl_type": "js",
        "mixed_kl_weight": 0.5,
        "zero_outside_topk": False,
        "success_reward_threshold": 1.0,
    }
    loss_fn = SDPOLossFn(cfg)

    # JS uses teacher's top-k indices to gather both student and teacher
    # logprobs; flipping which one is "teacher" picks a different top-k slice
    # of the full distribution. We construct symmetric K=vocab inputs so the
    # gather is identity and the symmetry property is unambiguous.
    student_logp_full = F.log_softmax(logits_a, dim=-1)
    teacher_logp_full = F.log_softmax(logits_b, dim=-1)
    k = logits_a.shape[-1]
    teacher_topk_logits, teacher_topk_idx = logits_b.topk(k, dim=-1)
    teacher_topk_logp_ab = teacher_logp_full.gather(-1, teacher_topk_idx)
    student_topk_logp_ab = student_logp_full.gather(-1, teacher_topk_idx)
    H_ab = (student_logp_full.exp() * student_logp_full).sum(-1)

    data = {
        "input_ids": torch.zeros(2, 6, dtype=torch.long),
        "token_mask": torch.ones(2, 6),
        "sample_mask": torch.ones(2),
        "sdpo_mask": torch.ones(2),
    }
    gvt = (data["token_mask"][:, 1:] * data["sdpo_mask"].unsqueeze(-1)).sum()
    loss_ab, _ = loss_fn(student_topk_logp_ab, teacher_topk_logp_ab, H_ab, data, torch.tensor(2.0), gvt)

    # Swap which side is the "teacher".
    student_logp_full2 = F.log_softmax(logits_b, dim=-1)
    teacher_logp_full2 = F.log_softmax(logits_a, dim=-1)
    teacher_topk_logits2, teacher_topk_idx2 = logits_a.topk(k, dim=-1)
    teacher_topk_logp_ba = teacher_logp_full2.gather(-1, teacher_topk_idx2)
    student_topk_logp_ba = student_logp_full2.gather(-1, teacher_topk_idx2)
    H_ba = (student_logp_full2.exp() * student_logp_full2).sum(-1)
    loss_ba, _ = loss_fn(student_topk_logp_ba, teacher_topk_logp_ba, H_ba, data, torch.tensor(2.0), gvt)

    assert torch.allclose(loss_ab, loss_ba, atol=1e-5), (loss_ab.item(), loss_ba.item())


def test_sdpo_ref_kl_zero_when_student_equals_ref():
    """Trust-region penalty is exactly 0 when current logprobs == reference."""
    s, t, H, data, gvt = _build_inputs()
    # next_token_logprobs is [B, S-1]; data tensors are [B, S] and the loss
    # slices reference_policy_logprobs to [:, 1:] to align them.
    curr_lp = torch.randn(s.shape[0], s.shape[1])
    data["reference_policy_logprobs"] = torch.cat(
        [torch.zeros(s.shape[0], 1), curr_lp], dim=1
    )

    loss_fn = SDPOLossFn(
        {
            "kl_type": "js",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,
            "success_reward_threshold": 1.0,
            "reference_policy_kl_penalty": 1.0,
            "reference_policy_kl_type": "k3",
        }
    )
    _, metrics = loss_fn(
        s, t, H, data, torch.tensor(4.0), gvt, next_token_logprobs=curr_lp
    )
    assert metrics["sdpo/ref_kl"] == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("kl_estimator", ["k1", "k2", "k3"])
def test_sdpo_ref_kl_positive_when_drifted(kl_estimator):
    """Ref-KL is non-zero when student has drifted from the reference, and
    scales with beta. k2 and k3 are always non-negative; k1 is the raw log
    ratio so we check the per-token tensor instead of just sign."""
    s, t, H, data, gvt = _build_inputs()
    torch.manual_seed(123)
    curr_lp = torch.randn(s.shape[0], s.shape[1])
    ref_lp = curr_lp + torch.randn_like(curr_lp) * 0.5  # drifted
    data["reference_policy_logprobs"] = torch.cat(
        [torch.zeros(s.shape[0], 1), ref_lp], dim=1
    )

    cfg_base = {
        "kl_type": "js",
        "mixed_kl_weight": 0.5,
        "zero_outside_topk": False,
        "success_reward_threshold": 1.0,
        "reference_policy_kl_type": kl_estimator,
    }

    loss_low, _ = SDPOLossFn(
        {**cfg_base, "reference_policy_kl_penalty": 0.0}
    )(s, t, H, data, torch.tensor(4.0), gvt)
    loss_high, metrics_high = SDPOLossFn(
        {**cfg_base, "reference_policy_kl_penalty": 1.0}
    )(s, t, H, data, torch.tensor(4.0), gvt, next_token_logprobs=curr_lp)

    # With drift the penalty changes the loss between beta=0 and beta=1.
    assert abs(loss_high.item() - loss_low.item()) > 1e-4
    # The ref_kl metric is logged at beta=1.
    assert "sdpo/ref_kl" in metrics_high
    if kl_estimator in {"k2", "k3"}:
        # k2 and k3 are non-negative by construction.
        assert metrics_high["sdpo/ref_kl"] >= -1e-7


@pytest.mark.parametrize("kl_estimator", ["k1", "k2", "k3"])
def test_sdpo_ref_kl_carries_gradient(kl_estimator):
    """Regression: the ref-KL anchor must contribute gradient through the
    current-policy logprobs. An earlier version built it from two detached
    data-dict tensors, making the penalty a training no-op."""
    s, t, H, data, gvt = _build_inputs()
    torch.manual_seed(321)
    curr_lp = torch.randn(s.shape[0], s.shape[1], requires_grad=True)
    ref_lp = curr_lp.detach() + torch.randn(s.shape[0], s.shape[1]) * 0.5
    data["reference_policy_logprobs"] = torch.cat(
        [torch.zeros(s.shape[0], 1), ref_lp], dim=1
    )

    loss_fn = SDPOLossFn(
        {
            "kl_type": "js",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,
            "success_reward_threshold": 1.0,
            "reference_policy_kl_penalty": 1.0,
            "reference_policy_kl_type": kl_estimator,
        }
    )
    # s/t/H carry no grad, so any gradient on curr_lp comes from the ref-KL.
    loss, _ = loss_fn(
        s, t, H, data, torch.tensor(4.0), gvt, next_token_logprobs=curr_lp
    )
    loss.backward()
    assert curr_lp.grad is not None
    assert curr_lp.grad.abs().sum().item() > 0.0


def test_sdpo_ref_kl_requires_next_token_logprobs():
    """With the penalty on, omitting the grad-carrying logprobs fails loudly."""
    s, t, H, data, gvt = _build_inputs()
    data["reference_policy_logprobs"] = torch.randn(s.shape[0], s.shape[1] + 1)

    loss_fn = SDPOLossFn(
        {
            "kl_type": "js",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,
            "success_reward_threshold": 1.0,
            "reference_policy_kl_penalty": 1.0,
        }
    )
    with pytest.raises(AssertionError, match="next_token_logprobs"):
        loss_fn(s, t, H, data, torch.tensor(4.0), gvt)


def test_sdpo_ref_kl_invalid_config_raises():
    with pytest.raises(ValueError, match="reference_policy_kl_penalty"):
        SDPOLossFn(
            {
                "kl_type": "js",
                "mixed_kl_weight": 0.5,
                "zero_outside_topk": False,
                "success_reward_threshold": 1.0,
                "reference_policy_kl_penalty": -0.1,
            }
        )
    with pytest.raises(ValueError, match="reference_policy_kl_type"):
        SDPOLossFn(
            {
                "kl_type": "js",
                "mixed_kl_weight": 0.5,
                "zero_outside_topk": False,
                "success_reward_threshold": 1.0,
                "reference_policy_kl_penalty": 0.1,
                "reference_policy_kl_type": "k4",
            }
        )


def test_sdpo_loss_js_bounded_by_log2():
    """JS divergence per position is bounded above by log 2."""
    # Construct adversarially-different student and teacher (disjoint supports
    # at the top-k indices). True JS is exactly log 2; the top-k approximation
    # should match it.
    student_p = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    teacher_p = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
    eps = 1e-9
    student_logp = (student_p + eps).log()
    teacher_logp = (teacher_p + eps).log()
    H_all = (student_p * student_logp).sum(-1)

    data = {
        "input_ids": torch.zeros(1, 2, dtype=torch.long),
        "token_mask": torch.ones(1, 2),
        "sample_mask": torch.ones(1),
        "sdpo_mask": torch.ones(1),
    }
    loss_fn = SDPOLossFn(
        {
            "kl_type": "js",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,
            "success_reward_threshold": 1.0,
        }
    )
    gvt = torch.tensor(1.0)
    loss, metrics = loss_fn(student_logp, teacher_logp, H_all, data, torch.tensor(1.0), gvt)
    assert loss.item() <= math.log(2) + 1e-3
    assert metrics["sdpo/per_pos_kl"] <= math.log(2) + 1e-3
    assert loss.item() > 0  # disjoint supports → strictly positive


def _is_clip_loss_fn(clip):
    return SDPOLossFn(
        {
            "kl_type": "reverse",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,  # top-k-only KL; no tail correction
            "success_reward_threshold": 1.0,
            "rollout_importance_sampling_clip": clip,
        }
    )


def test_sdpo_rollout_is_noop_when_on_policy():
    """ratio = pi_theta/pi_rollout = 1 everywhere → loss identical to no-clip."""
    s, t, H, data, gvt = _build_inputs()
    batch_size, seq_len = data["token_mask"].shape
    next_token_logprobs = -torch.rand(batch_size, seq_len - 1)  # arbitrary logprobs
    # generation_logprobs is [B, S]; the loss compares its [:, 1:] slice.
    data["generation_logprobs"] = torch.cat(
        [torch.zeros(batch_size, 1), next_token_logprobs], dim=1
    )

    base_fn = SDPOLossFn(
        {
            "kl_type": "reverse",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,
            "success_reward_threshold": 1.0,
        }
    )
    base_loss, _ = base_fn(s, t, H, data, torch.tensor(4.0), gvt)

    loss_fn = _is_clip_loss_fn(2.0)
    loss, metrics = loss_fn(
        s, t, H, data, torch.tensor(4.0), gvt, next_token_logprobs=next_token_logprobs
    )
    assert torch.allclose(loss, base_loss, atol=1e-6)
    assert metrics["sdpo/rollout_is_ratio_mean"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["sdpo/rollout_is_clip_frac"] == 0.0


def test_sdpo_rollout_is_clips_large_ratios():
    """When pi_theta >> pi_rollout everywhere, every token clips and the loss
    is scaled by exactly the clip value."""
    s, t, H, data, gvt = _build_inputs()
    batch_size, seq_len = data["token_mask"].shape
    next_token_logprobs = -torch.rand(batch_size, seq_len - 1)
    # rollout logprobs 10 nats lower → raw ratio e^10 >> clip
    data["generation_logprobs"] = torch.cat(
        [torch.zeros(batch_size, 1), next_token_logprobs - 10.0], dim=1
    )

    base_fn = SDPOLossFn(
        {
            "kl_type": "reverse",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,
            "success_reward_threshold": 1.0,
        }
    )
    base_loss, _ = base_fn(s, t, H, data, torch.tensor(4.0), gvt)

    clip = 2.0
    loss_fn = _is_clip_loss_fn(clip)
    loss, metrics = loss_fn(
        s, t, H, data, torch.tensor(4.0), gvt, next_token_logprobs=next_token_logprobs
    )
    assert torch.allclose(loss, clip * base_loss, atol=1e-5)
    assert metrics["sdpo/rollout_is_clip_frac"] == pytest.approx(1.0)
    assert metrics["sdpo/rollout_is_ratio_mean"] == pytest.approx(math.exp(10.0), rel=1e-4)


def test_sdpo_rollout_is_requires_inputs():
    """Missing next_token_logprobs or generation_logprobs must fail loudly."""
    s, t, H, data, gvt = _build_inputs()
    loss_fn = _is_clip_loss_fn(2.0)
    with pytest.raises(AssertionError, match="next_token_logprobs"):
        loss_fn(s, t, H, data, torch.tensor(4.0), gvt)

    batch_size, seq_len = data["token_mask"].shape
    with pytest.raises(AssertionError, match="generation_logprobs"):
        loss_fn(
            s,
            t,
            H,
            data,
            torch.tensor(4.0),
            gvt,
            next_token_logprobs=-torch.rand(batch_size, seq_len - 1),
        )


def test_sdpo_rollout_is_invalid_clip_raises():
    with pytest.raises(ValueError, match="rollout_importance_sampling_clip"):
        _is_clip_loss_fn(0.0)
    with pytest.raises(ValueError, match="rollout_importance_sampling_clip"):
        _is_clip_loss_fn(-1.0)


def _bucket_loss_fn(kl_type="reverse", **extra):
    return SDPOLossFn(
        {
            "kl_type": kl_type,
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": True,  # ignored in bucket mode
            "success_reward_threshold": 1.0,
            "tail_mode": "k_plus_one",
            **extra,
        }
    )


def test_sdpo_k_plus_one_zero_when_teacher_equals_student():
    """Identical true log-probs -> identical buckets -> KL exactly 0."""
    logits = torch.randn(2, 5, 16)
    s, t, H, data, gvt = _build_inputs(
        batch_size=2,
        seq_len=6,
        vocab_size=16,
        student_logits=logits,
        teacher_logits=logits.clone(),
    )
    for kl_type in ("forward", "reverse", "mixed", "js"):
        loss_fn = _bucket_loss_fn(kl_type)
        # H_all is not needed in bucket mode
        loss, _ = loss_fn(s, t, None, data, torch.tensor(2.0), gvt)
        assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-5), (kl_type, loss.item())


def test_sdpo_k_plus_one_matches_manual_categorical_kl():
    """Bucket-mode reverse KL == KL of the manually built K+1 categoricals."""
    s, t, H, data, gvt = _build_inputs(batch_size=2, seq_len=4, topk=3, vocab_size=12)
    loss_fn = _bucket_loss_fn("reverse")
    loss, metrics = loss_fn(s, t, None, data, torch.tensor(2.0), gvt)

    # Manual: append log(1 - sum p) buckets to the true log-probs.
    def bucketize(lp):
        rest = torch.log1p(-lp.exp().sum(-1).clamp(max=1 - 1e-8))
        return torch.cat([lp, rest.unsqueeze(-1)], dim=-1)

    sb, tb = bucketize(s), bucketize(t)
    per_token = (sb.exp() * (sb - tb)).sum(-1)  # [B, S-1]
    token_mask = data["token_mask"][:, 1:]
    expected = (per_token * token_mask).sum() / gvt
    assert torch.allclose(loss, expected, atol=1e-6), (loss.item(), expected.item())


def test_sdpo_k_plus_one_nonnegative_and_positive_when_different():
    """Bucket KL is a KL of full categoricals -> guaranteed >= 0, and > 0 for
    differing distributions (unlike the legacy truncated top-k sum, which can
    go negative on partial sums)."""
    torch.manual_seed(3)
    s, t, H, data, gvt = _build_inputs(batch_size=4, seq_len=6, topk=5, vocab_size=32)
    loss_fn = _bucket_loss_fn("reverse")
    loss, metrics = loss_fn(s, t, None, data, torch.tensor(4.0), gvt)
    assert loss.item() > 0
    assert metrics["sdpo/per_pos_kl"] > 0


def test_sdpo_k_plus_one_invalid_tail_mode_raises():
    with pytest.raises(ValueError, match="tail_mode"):
        SDPOLossFn(
            {
                "kl_type": "reverse",
                "mixed_kl_weight": 0.5,
                "zero_outside_topk": True,
                "success_reward_threshold": 1.0,
                "tail_mode": "buckets",
            }
        )


def test_full_vocab_logsumexp_matches_torch():
    """Non-distributed branch of the top-k post-processor's logsumexp helper."""
    from nemo_rl.models.automodel.train import TopkLogitsPostProcessor

    torch.manual_seed(0)
    proc = TopkLogitsPostProcessor(
        cfg={},
        device_mesh=None,
        cp_mesh=None,
        tp_mesh=None,
        cp_size=1,
        k=3,
        enable_seq_packing=False,
        return_logsumexp=True,
    )
    logits = torch.randn(2, 2500, 64)  # > one 1024 seq chunk
    lse = proc._full_vocab_logsumexp(logits)
    assert torch.allclose(lse, torch.logsumexp(logits.to(torch.float32), dim=-1), atol=1e-5)


def test_sdpo_k_plus_one_finite_on_deterministic_tokens():
    """Regression for job 20553: near-deterministic tokens have top-k mass
    numerically equal to 1.0 in fp32; the remainder bucket must stay finite
    (1.0 - 1e-8 rounds to 1.0 in fp32, so a too-small eps makes log1p(-1) =
    -inf and the loss NaN). Indices are student-selected, mirroring
    topk_source="student"."""
    import torch.nn.functional as F

    B, S, K, V = 2, 5, 3, 16
    # One dominant logit per position -> top-1 prob == 1.0 in fp32.
    student_logits = torch.full((B, S - 1, V), -60.0)
    student_logits[..., 0] = 60.0
    teacher_logits = torch.full((B, S - 1, V), -60.0)
    teacher_logits[..., 1] = 60.0  # teacher deterministic on a DIFFERENT token

    student_logp = F.log_softmax(student_logits, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits, dim=-1)
    idx = student_logits.topk(K, dim=-1).indices  # student-selected indices
    s = student_logp.gather(-1, idx)
    t = teacher_logp.gather(-1, idx)

    data = {
        "input_ids": torch.zeros(B, S, dtype=torch.long),
        "token_mask": torch.ones(B, S),
        "sample_mask": torch.ones(B),
        "sdpo_mask": torch.ones(B),
    }
    gvt = torch.tensor(float(B * (S - 1)))

    # Premise: student top-k mass is exactly 1.0 in fp32 -> the clamp must bite.
    assert torch.isclose(s.exp().sum(-1).max(), torch.tensor(1.0))
    # And the teacher has ~zero mass at these indices -> its bucket holds ~all mass.
    assert t.exp().sum(-1).max() < 1e-6

    for kl_type in ("forward", "reverse", "mixed", "js"):
        loss_fn = _bucket_loss_fn(kl_type)
        loss, metrics = loss_fn(s, t, None, data, torch.tensor(2.0), gvt)
        assert torch.isfinite(loss), (kl_type, loss.item())
        assert loss.item() >= 0
        assert math.isfinite(metrics["sdpo/per_pos_kl"])
    # And the gradient path stays finite too.
    s_grad = s.clone().requires_grad_(True)
    loss, _ = _bucket_loss_fn("reverse")(s_grad, t, None, data, torch.tensor(2.0), gvt)
    loss.backward()
    assert torch.isfinite(s_grad.grad).all()
