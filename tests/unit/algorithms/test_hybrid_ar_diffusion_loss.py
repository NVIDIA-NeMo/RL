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

"""Unit tests for HybridARDiffusionLossFn.

The two terms are exercised in isolation (by zeroing the other's weight or
inputs) and then together, so a regression in one cannot hide behind the other.
"""

import math

import pytest
import torch

from nemo_rl.algorithms.loss import ClippedPGLossConfig, HybridARDiffusionLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy import HybridARDiffusionLogprobEstimationConfig

# Every test uses a 4-position layout: positions 0,1 are the noisy half (CE),
# positions 2,3 the clean half (PG).


def _loss_fn(
    ce_loss_weight: float = 1.0,
    pg_loss_weight: float = 1.0,
    elbo_weight_ce: bool = False,
    **loss_overrides,
) -> HybridARDiffusionLossFn:
    loss_cfg = {
        "ratio_clip_min": 0.2,
        "ratio_clip_max": 0.2,
        "ratio_clip_c": None,
        "token_level_loss": True,
        "reference_policy_kl_penalty": 0.0,
    }
    loss_cfg.update(loss_overrides)
    estimation_cfg = HybridARDiffusionLogprobEstimationConfig(
        type="hybrid_ar_diffusion",
        mask_token_id=100,
        ce_loss_weight=ce_loss_weight,
        pg_loss_weight=pg_loss_weight,
        elbo_weight_ce=elbo_weight_ce,
    )
    return HybridARDiffusionLossFn(ClippedPGLossConfig(**loss_cfg), estimation_cfg)


def _data(
    advantages=(0.0, 0.0, 1.0, 1.0),
    prev=(0.0, 0.0, -1.0, -1.0),
    ce_mask=(1.0, 1.0, 0.0, 0.0),
    pg_mask=(0.0, 0.0, 1.0, 1.0),
    mask_ratio=0.5,
):
    return BatchedDataDict(
        {
            "advantages": torch.tensor([advantages], dtype=torch.float32),
            "prev_logprobs": torch.tensor([prev], dtype=torch.float32),
            "hybrid_ce_mask": torch.tensor([ce_mask], dtype=torch.float32),
            "hybrid_pg_mask": torch.tensor([pg_mask], dtype=torch.float32),
            "hybrid_mask_ratio": torch.tensor([mask_ratio], dtype=torch.float32),
            "sample_mask": torch.ones(1, dtype=torch.float32),
        }
    )


def _run(loss_fn, logprobs, data, global_valid_toks=2.0, global_valid_seqs=1.0):
    return loss_fn(
        torch.tensor([logprobs], dtype=torch.float32),
        data,
        torch.tensor(global_valid_seqs),
        torch.tensor(global_valid_toks),
    )


def test_both_terms_use_the_global_normalizer_not_local_mask_sums():
    # With global_valid_toks deliberately != the local mask sums, a term that
    # normalized by its own mask sum would give a different number. This is what
    # makes the sum-over-microbatches convention reconstruct the true mean.
    _, m = _run(
        _loss_fn(ce_loss_weight=1.0),
        [-2.0, -2.0, -1.0, -1.0],
        _data(),
        global_valid_toks=8.0,
    )
    # PG: 2 positions, A=1, ratio=1 -> sum(-1,-1) = -2, /8 = -0.25
    assert m["pg_loss"] == pytest.approx(-0.25)
    # CE: 2 masked positions at logprob -2 -> sum(2,2) = 4, /8 = 0.5
    assert m["ce_loss"] == pytest.approx(0.5)


def test_ce_and_pg_use_their_own_masks_not_each_others():
    # Unequal mask sizes: swapping the two masks would change both numbers.
    data = _data(
        advantages=(0.0, 0.0, 1.0, 1.0),
        ce_mask=(1.0, 0.0, 0.0, 0.0),  # 1 CE position
        pg_mask=(0.0, 0.0, 1.0, 1.0),  # 2 PG positions
    )
    _, m = _run(
        _loss_fn(ce_loss_weight=1.0),
        [-3.0, -7.0, -1.0, -1.0],
        data,
        global_valid_toks=4.0,
    )
    assert m["num_ce_tokens"] == pytest.approx(1.0)
    assert m["num_pg_tokens"] == pytest.approx(2.0)
    # CE reads only position 0 (-3 -> 3.0), not position 1 (-7).
    assert m["ce_loss"] == pytest.approx(3.0 / 4.0)
    assert m["pg_loss"] == pytest.approx(-2.0 / 4.0)


def test_on_policy_ratio_is_one_and_pg_equals_neg_advantage():
    # curr == prev on the clean half -> ratio 1 -> actor_loss = -A.
    loss, metrics = _run(_loss_fn(ce_loss_weight=0.0), [0.0, 0.0, -1.0, -1.0], _data())
    # Two PG positions, A=1 each, normalized by global_valid_toks=2.
    assert metrics["pg_loss"] == pytest.approx(-1.0)
    assert loss.item() == pytest.approx(-1.0)
    assert metrics["ratio_clipped_fraction"] == pytest.approx(0.0)
    assert metrics["approx_kl"] == pytest.approx(0.0)


def test_ce_term_only_counts_masked_noisy_positions():
    # Zero advantages -> PG term vanishes; only CE remains.
    data = _data(advantages=(0.0, 0.0, 0.0, 0.0))
    # logprobs -2 on the two CE positions, garbage on the clean half.
    loss, metrics = _run(_loss_fn(ce_loss_weight=1.0), [-2.0, -2.0, -99.0, -99.0], data)
    # CE = sum(2.0, 2.0) / global_valid_toks(2) = 2.0
    assert metrics["ce_loss"] == pytest.approx(2.0)
    assert metrics["pg_loss"] == pytest.approx(0.0)
    assert loss.item() == pytest.approx(2.0)
    # The clean-half garbage must not leak into CE.
    assert metrics["num_ce_tokens"] == pytest.approx(2.0)


def test_ce_weight_scales_only_the_ce_term():
    data = _data(advantages=(0.0, 0.0, 0.0, 0.0))
    a, ma = _run(_loss_fn(ce_loss_weight=1.0), [-2.0, -2.0, 0.0, 0.0], data)
    b, mb = _run(_loss_fn(ce_loss_weight=0.25), [-2.0, -2.0, 0.0, 0.0], data)
    assert ma["ce_loss"] == pytest.approx(mb["ce_loss"])
    assert b.item() == pytest.approx(0.25 * a.item())


def test_pg_weight_zero_leaves_only_the_ce_term():
    loss, metrics = _run(
        _loss_fn(ce_loss_weight=0.5, pg_loss_weight=0.0),
        [-2.0, -2.0, -1.0, -1.0],
        _data(),
    )
    assert metrics["pg_loss"] != pytest.approx(0.0)
    assert loss.item() == pytest.approx(0.5 * metrics["ce_loss"])


def test_elbo_weighting_scales_ce_by_inverse_t():
    logprobs = [-2.0, -2.0, 0.0, 0.0]
    data_plain = _data(advantages=(0.0,) * 4, mask_ratio=0.5)
    data_elbo = _data(advantages=(0.0,) * 4, mask_ratio=0.5)
    _, m_plain = _run(_loss_fn(), logprobs, data_plain)
    _, m_elbo = _run(_loss_fn(elbo_weight_ce=True), logprobs, data_elbo)
    # t = 0.5 -> 1/t = 2x
    assert m_elbo["ce_loss"] == pytest.approx(2.0 * m_plain["ce_loss"])


def test_positive_advantage_ratio_is_clipped_above():
    # curr - prev = +1.0 -> ratio e ~= 2.718, clipped to 1.2 for A > 0.
    _, metrics = _run(_loss_fn(ce_loss_weight=0.0), [0.0, 0.0, 0.0, 0.0], _data())
    # max(-A*r, -A*r_clamped) with A=1 picks the *less negative* -> -1.2
    assert metrics["pg_loss"] == pytest.approx(-1.2)
    assert metrics["ratio_clipped_fraction"] == pytest.approx(1.0)


def test_negative_advantage_uses_unclipped_when_larger():
    data = _data(advantages=(0.0, 0.0, -1.0, -1.0))
    # ratio ~= 2.718; -A*r = +2.718 vs -A*r_clamped = +1.2 -> max picks 2.718
    _, metrics = _run(_loss_fn(ce_loss_weight=0.0), [0.0, 0.0, 0.0, 0.0], data)
    assert metrics["pg_loss"] == pytest.approx(math.e, rel=1e-4)


def test_dual_clip_bounds_negative_advantage_branch():
    data = _data(advantages=(0.0, 0.0, -1.0, -1.0))
    # Unclipped branch would give e ~= 2.718; dual clip caps at -A*c = 2.0.
    _, metrics = _run(
        _loss_fn(ce_loss_weight=0.0, ratio_clip_c=2.0), [0.0, 0.0, 0.0, 0.0], data
    )
    assert metrics["pg_loss"] == pytest.approx(2.0)


def test_dual_clip_leaves_positive_advantage_untouched():
    _, ma = _run(_loss_fn(ce_loss_weight=0.0), [0.0, 0.0, 0.0, 0.0], _data())
    _, mb = _run(
        _loss_fn(ce_loss_weight=0.0, ratio_clip_c=2.0), [0.0, 0.0, 0.0, 0.0], _data()
    )
    assert ma["pg_loss"] == pytest.approx(mb["pg_loss"])


def test_both_terms_sum():
    loss, metrics = _run(
        _loss_fn(ce_loss_weight=0.5), [-2.0, -2.0, -1.0, -1.0], _data()
    )
    assert loss.item() == pytest.approx(metrics["pg_loss"] + 0.5 * metrics["ce_loss"])
    assert metrics["weighted_ce_loss"] == pytest.approx(0.5 * metrics["ce_loss"])


def test_sample_mask_zeroes_a_sample():
    data = _data()
    data["sample_mask"] = torch.zeros(1, dtype=torch.float32)
    loss, metrics = _run(_loss_fn(), [-2.0, -2.0, -1.0, -1.0], data)
    assert loss.item() == pytest.approx(0.0)
    assert metrics["num_pg_tokens"] == pytest.approx(0.0)
    assert metrics["num_ce_tokens"] == pytest.approx(0.0)


def test_token_mult_prob_error_reads_generation_logprobs():
    data = _data()
    data["generation_logprobs"] = torch.tensor([[0.0, 0.0, -1.0, -1.0]])
    _, metrics = _run(_loss_fn(), [-2.0, -2.0, -1.0, -1.0], data)
    # Engine and trainer agree on the clean half -> exp(0) averaged over 2/2.
    assert metrics["token_mult_prob_error"] == pytest.approx(1.0)


def test_metrics_expose_both_terms_for_lambda_tuning():
    _, metrics = _run(_loss_fn(), [-2.0, -2.0, -1.0, -1.0], _data())
    for key in (
        "pg_loss",
        "ce_loss",
        "weighted_ce_loss",
        "ratio_clipped_fraction",
        "approx_kl",
        "num_pg_tokens",
        "num_ce_tokens",
        "mean_mask_ratio",
    ):
        assert key in metrics
    assert metrics["mean_mask_ratio"] == pytest.approx(0.5)


def test_metric_normalizations_cover_every_metric():
    # Split-API trainers rescale each metric by its advertised denominator, so an
    # unadvertised metric would silently fall back to the gradient normalizer.
    loss_fn = _loss_fn()
    _, metrics = _run(loss_fn, [-2.0, -2.0, -1.0, -1.0], _data())
    assert set(metrics) == set(loss_fn.metric_normalizations)


def test_sequence_level_loss_is_rejected():
    with pytest.raises(ValueError, match="token_level_loss"):
        _loss_fn(token_level_loss=False)


def test_reference_kl_penalty_is_rejected_unless_routed_through_reward():
    with pytest.raises(ValueError, match="reference_policy_kl_penalty"):
        _loss_fn(reference_policy_kl_penalty=0.01)
    # With use_kl_in_reward the loss-side penalty is not applied, so it is fine.
    _loss_fn(reference_policy_kl_penalty=0.01, use_kl_in_reward=True)


def test_importance_sampling_correction_is_rejected():
    with pytest.raises(ValueError, match="use_importance_sampling_correction"):
        _loss_fn(use_importance_sampling_correction=True)


@pytest.mark.parametrize(
    "option,value",
    [
        ("disable_ppo_ratio", True),
        ("sequence_level_importance_ratios", True),
        ("force_on_policy_ratio", True),
        ("use_cispo", True),
        ("use_on_policy_kl_approximation", True),
        ("seq_logprob_error_in_loss", True),
        ("truncated_importance_sampling_type", "tis"),
        ("positive_example_nll_weight", 0.1),
    ],
)
def test_unsupported_clipped_pg_options_are_rejected(option, value):
    with pytest.raises(ValueError, match=option):
        _loss_fn(**{option: value})


def test_dual_clip_must_exceed_one():
    with pytest.raises(ValueError, match="ratio_clip_c"):
        _loss_fn(ratio_clip_c=1.0)
