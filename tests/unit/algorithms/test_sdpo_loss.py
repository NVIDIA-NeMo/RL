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
    """Trust-region penalty is exactly 0 when prev_logprobs == reference."""
    s, t, H, data, gvt = _build_inputs()
    student_lp_at_sampled = torch.randn(s.shape[0], s.shape[1] + 1)
    data["prev_logprobs"] = student_lp_at_sampled
    data["reference_policy_logprobs"] = student_lp_at_sampled.clone()

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
    _, metrics = loss_fn(s, t, H, data, torch.tensor(4.0), gvt)
    assert metrics["sdpo/ref_kl"] == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("kl_estimator", ["k1", "k2", "k3"])
def test_sdpo_ref_kl_positive_when_drifted(kl_estimator):
    """Ref-KL is non-zero when student has drifted from the reference, and
    scales with beta. k2 and k3 are always non-negative; k1 is the raw log
    ratio so we check the per-token tensor instead of just sign."""
    s, t, H, data, gvt = _build_inputs()
    torch.manual_seed(123)
    student_lp = torch.randn(s.shape[0], s.shape[1] + 1)
    ref_lp = student_lp + torch.randn_like(student_lp) * 0.5  # drifted
    data["prev_logprobs"] = student_lp
    data["reference_policy_logprobs"] = ref_lp

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
    )(s, t, H, data, torch.tensor(4.0), gvt)

    # With drift the penalty changes the loss between beta=0 and beta=1.
    assert abs(loss_high.item() - loss_low.item()) > 1e-4
    # The ref_kl metric is logged at beta=1.
    assert "sdpo/ref_kl" in metrics_high
    if kl_estimator in {"k2", "k3"}:
        # k2 and k3 are non-negative by construction.
        assert metrics_high["sdpo/ref_kl"] >= -1e-7


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
