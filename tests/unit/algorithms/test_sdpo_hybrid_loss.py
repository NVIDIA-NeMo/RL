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

"""Unit tests for SDPOHybridLossFn (SDPO+GRPO hybrid, paper §4.5).

The hybrid blends a clipped policy-gradient term with the SDPO logit-level KL
term at the loss level:

    L = grpo_weight · L_GRPO  +  (1 − grpo_weight) · L_SDPO

These tests verify the blend reduces to each component at the extremes and is
the exact weighted sum in between, plus that prepare_loss_input routes the
combined input type to both forward outputs. They run on CPU with hand-built
tensors (no model forward).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from nemo_rl.algorithms.loss import (
    ClippedPGLossFn,
    SDPOHybridLossFn,
    SDPOLossFn,
)
from nemo_rl.algorithms.loss.interfaces import LossInputType


def _grpo_cfg(reference_policy_kl_penalty: float = 0.0) -> dict:
    return {
        "reference_policy_kl_penalty": reference_policy_kl_penalty,
        "reference_policy_kl_type": "k3",
        "kl_input_clamp_value": None,
        "kl_output_clamp_value": None,
        "ratio_clip_min": 0.2,
        "ratio_clip_max": 0.2,
        "ratio_clip_c": None,
        "use_on_policy_kl_approximation": False,
        "use_importance_sampling_correction": False,
        "truncated_importance_sampling_ratio": None,
        "token_level_loss": True,
    }


def _sdpo_cfg() -> dict:
    return {
        "kl_type": "reverse",
        "mixed_kl_weight": 0.5,
        "zero_outside_topk": False,  # top-k-only KL keeps the math simple
        "success_reward_threshold": 1.0,
    }


def _build_inputs(
    batch_size: int = 4,
    seq_len: int = 6,
    topk: int = 5,
    vocab_size: int = 32,
):
    """Build self-consistent inputs for both the SDPO and GRPO components."""
    torch.manual_seed(0)

    # --- SDPO distillation forward outputs ---
    student_logits = torch.randn(batch_size, seq_len - 1, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len - 1, vocab_size)
    student_logp = F.log_softmax(student_logits, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits, dim=-1)
    _, teacher_topk_indices = teacher_logits.topk(topk, dim=-1)
    teacher_topk_logp = teacher_logp.gather(-1, teacher_topk_indices)
    student_topk_logp = student_logp.gather(-1, teacher_topk_indices)
    H_all = (student_logp.exp() * student_logp).sum(-1)

    # --- GRPO logprob forward output + data ---
    next_token_logprobs = torch.randn(batch_size, seq_len - 1)

    token_mask = torch.ones(batch_size, seq_len)
    sample_mask = torch.ones(batch_size)
    sdpo_mask = torch.ones(batch_size)

    data = {
        "input_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "token_mask": token_mask,
        "sample_mask": sample_mask,
        "sdpo_mask": sdpo_mask,
        "advantages": torch.randn(batch_size, seq_len),
        "prev_logprobs": torch.randn(batch_size, seq_len),
        "generation_logprobs": torch.randn(batch_size, seq_len),
    }

    global_valid_seqs = sample_mask.sum()
    global_valid_toks = (token_mask[:, 1:] * sample_mask.unsqueeze(-1)).sum()
    return {
        "student_topk_logprobs": student_topk_logp,
        "teacher_topk_logprobs": teacher_topk_logp,
        "H_all": H_all,
        "next_token_logprobs": next_token_logprobs,
        "data": data,
        "global_valid_seqs": global_valid_seqs,
        "global_valid_toks": global_valid_toks,
    }


def _call_hybrid(grpo_weight: float, inp: dict):
    hybrid = SDPOHybridLossFn({"grpo_weight": grpo_weight, "sdpo": _sdpo_cfg(), "grpo": _grpo_cfg()})
    return hybrid(
        inp["student_topk_logprobs"],
        inp["teacher_topk_logprobs"],
        inp["H_all"],
        inp["next_token_logprobs"],
        inp["data"],
        inp["global_valid_seqs"],
        inp["global_valid_toks"],
    )


def _call_sdpo(inp: dict):
    return SDPOLossFn(_sdpo_cfg())(
        inp["student_topk_logprobs"],
        inp["teacher_topk_logprobs"],
        inp["H_all"],
        inp["data"],
        inp["global_valid_seqs"],
        inp["global_valid_toks"],
    )


def _call_grpo(inp: dict):
    return ClippedPGLossFn(_grpo_cfg())(
        inp["next_token_logprobs"],
        inp["data"],
        inp["global_valid_seqs"],
        inp["global_valid_toks"],
    )


def test_hybrid_lambda_zero_equals_pure_sdpo():
    inp = _build_inputs()
    hybrid_loss, _ = _call_hybrid(0.0, inp)
    sdpo_loss, _ = _call_sdpo(inp)
    assert torch.allclose(hybrid_loss, sdpo_loss, atol=1e-6), (
        hybrid_loss.item(),
        sdpo_loss.item(),
    )


def test_hybrid_lambda_one_equals_pure_grpo():
    inp = _build_inputs()
    hybrid_loss, _ = _call_hybrid(1.0, inp)
    grpo_loss, _ = _call_grpo(inp)
    assert torch.allclose(hybrid_loss, grpo_loss, atol=1e-6), (
        hybrid_loss.item(),
        grpo_loss.item(),
    )


@pytest.mark.parametrize("grpo_weight", [0.25, 0.5, 0.75])
def test_hybrid_intermediate_is_weighted_sum(grpo_weight):
    inp = _build_inputs()
    hybrid_loss, _ = _call_hybrid(grpo_weight, inp)
    sdpo_loss, _ = _call_sdpo(inp)
    grpo_loss, _ = _call_grpo(inp)
    expected = grpo_weight * grpo_loss + (1.0 - grpo_weight) * sdpo_loss
    assert torch.allclose(hybrid_loss, expected, atol=1e-6), (
        hybrid_loss.item(),
        expected.item(),
    )


def test_hybrid_metrics_structure():
    inp = _build_inputs()
    _, metrics = _call_hybrid(0.5, inp)
    assert "loss" in metrics
    assert "num_valid_samples" in metrics
    assert "hybrid/loss_grpo" in metrics
    assert "hybrid/loss_sdpo" in metrics
    # GRPO component metrics are namespaced; SDPO's keep their sdpo/ prefix.
    assert any(k.startswith("grpo/") for k in metrics)
    assert "sdpo/per_pos_kl" in metrics
    # grpo_weight is logged from sdpo_train (not the loss fn) to avoid the
    # worker's per-microbatch metric scaling corrupting a constant.
    assert "hybrid/grpo_weight" not in metrics


@pytest.mark.parametrize("bad_weight", [-0.1, 1.5])
def test_hybrid_rejects_out_of_range_weight(bad_weight):
    with pytest.raises(ValueError, match="grpo_weight"):
        SDPOHybridLossFn({"grpo_weight": bad_weight, "sdpo": _sdpo_cfg(), "grpo": _grpo_cfg()})


def test_prepare_loss_input_routes_combined_type(monkeypatch):
    """The DISTILLATION_AND_LOGPROB branch emits both forward outputs.

    We monkeypatch the two logprob kernels with sentinels so the test verifies
    routing/wiring of the new branch independent of kernel internals.
    """
    import nemo_rl.algorithms.loss.utils as utils

    student_sentinel = torch.tensor([1.0])
    teacher_sentinel = torch.tensor([2.0])
    h_sentinel = torch.tensor([3.0])
    logprob_sentinel = torch.tensor([4.0])

    monkeypatch.setattr(
        utils,
        "get_distillation_topk_logprobs_from_logits",
        lambda **kw: (student_sentinel, teacher_sentinel, h_sentinel),
    )
    monkeypatch.setattr(
        utils,
        "get_next_token_logprobs_from_logits",
        lambda **kw: logprob_sentinel,
    )

    loss_fn = SDPOHybridLossFn({"grpo_weight": 0.5, "sdpo": _sdpo_cfg(), "grpo": _grpo_cfg()})
    assert loss_fn.input_type == LossInputType.DISTILLATION_AND_LOGPROB

    logits = torch.randn(2, 4, 8)
    data = {
        "input_ids": torch.zeros(2, 4, dtype=torch.long),
        "token_mask": torch.ones(2, 4),
        "sample_mask": torch.ones(2),
        "teacher_topk_logits": torch.randn(2, 3, 5),
        "teacher_topk_indices": torch.zeros(2, 3, 5, dtype=torch.long),
    }

    loss_input, _ = utils.prepare_loss_input(logits, data, loss_fn)

    assert set(loss_input) == {
        "student_topk_logprobs",
        "teacher_topk_logprobs",
        "H_all",
        "next_token_logprobs",
    }
    assert torch.equal(loss_input["student_topk_logprobs"], student_sentinel)
    assert torch.equal(loss_input["teacher_topk_logprobs"], teacher_sentinel)
    assert torch.equal(loss_input["H_all"], h_sentinel)
    assert torch.equal(loss_input["next_token_logprobs"], logprob_sentinel)


def test_hybrid_advantage_prompt_ids_grouping():
    """Regression: sdpo_train builds a 2-D [B, 1] prompt-id index for the GRPO
    advantage estimator. A 1-D index crashes calculate_baseline_and_std_per_prompt
    (it groups via torch.unique(dim=0) and reduces with .all(1)).

    Verifies the prompt-major repeat_interleave layout yields correct
    leave-one-out group baselines and the expected [B, S] advantage shape.
    """
    from nemo_rl.algorithms.advantage_estimator import GRPOAdvantageEstimator

    num_prompts, num_generations, seq_len = 2, 2, 5
    batch_size = num_prompts * num_generations  # 4

    # Same construction as sdpo_train (the line that was buggy).
    prompt_ids = torch.arange(num_prompts).repeat_interleave(num_generations).unsqueeze(-1)
    assert prompt_ids.shape == (batch_size, 1)

    rewards = torch.tensor([1.0, 0.0, 1.0, 1.0])  # group0: [1,0], group1: [1,1]
    mask = torch.ones(batch_size, seq_len)

    est = GRPOAdvantageEstimator(
        {"name": "grpo", "use_leave_one_out_baseline": True, "normalize_rewards": False},
        {},
    )
    advantages = est.compute_advantage(prompt_ids=prompt_ids, rewards=rewards, mask=mask)

    assert advantages.shape == (batch_size, seq_len)
    # Advantages are constant across the sequence dimension (expanded from [B, 1]).
    assert torch.allclose(advantages, advantages[:, :1].expand_as(advantages))
    # Leave-one-out baselines: group0 sample0 baseline=reward of sample1 (0.0) -> adv 1.0;
    # sample1 baseline=1.0 -> adv -1.0. group1 both baselines=1.0 -> adv 0.0.
    expected_col = torch.tensor([1.0, -1.0, 0.0, 0.0])
    assert torch.allclose(advantages[:, 0], expected_col), advantages[:, 0]
