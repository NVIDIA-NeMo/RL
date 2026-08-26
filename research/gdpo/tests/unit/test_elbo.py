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

"""Tests for the SDMC ELBO estimator used to score masked diffusion LMs."""

import math

import pytest
import torch
from gdpo import (
    MaskedDiffusionConfig,
    SdmcElboEstimator,
    SdmcLikelihoodConfig,
    accumulate_elbo_logprobs,
    get_quadrature,
    make_dllm_mask_seeds,
    resolve_mask_id,
)

MASK_ID = 126336


def make_cfg(**overrides) -> SdmcLikelihoodConfig:
    return SdmcLikelihoodConfig(**overrides)


def make_batch(
    batch: int = 3, prompt_len: int = 5, completion_len: int = 11, pad: int = 2
):
    """Builds ids plus a completion mask with a prompt prefix and padding suffix."""
    seq_len = prompt_len + completion_len + pad
    torch.manual_seed(0)
    input_ids = torch.randint(0, 1000, (batch, seq_len))
    completion_mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    completion_mask[:, prompt_len : prompt_len + completion_len] = True
    return input_ids, completion_mask


@pytest.mark.parametrize(
    "rule", ["gauss-1", "gauss-2", "gauss-3", "gauss-4", "gauss-5"]
)
def test_quadrature_integrates_polynomials_exactly(rule):
    """Gauss-Legendre with n points is exact for polynomials of degree < 2n."""
    times, weights = get_quadrature(rule)
    degree = 2 * len(times) - 1

    def integrand(t):
        return t**degree

    approx = sum(w * integrand(t) for t, w in zip(times, weights))
    exact = 1.0 / (degree + 1)
    assert approx == pytest.approx(exact, rel=1e-9)


@pytest.mark.parametrize("rule", ["gauss-1", "gauss-2", "gauss-3", "simpson"])
def test_quadrature_weights_sum_to_unit_interval(rule):
    _, weights = get_quadrature(rule)
    assert sum(weights) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize(
    "rule", ["gauss-1", "gauss-2", "gauss-3", "gauss-4", "gauss-5"]
)
def test_quadrature_times_lie_strictly_inside_unit_interval(rule):
    """t=0 would make the 1/t ELBO weight singular."""
    times, _ = get_quadrature(rule)
    assert all(0.0 < t <= 1.0 for t in times)


def test_unknown_quadrature_rule_raises():
    with pytest.raises(ValueError, match="Unknown dllm quadrature rule"):
        get_quadrature("gauss-99")


def test_per_position_contributions_sum_to_scalar_elbo():
    """The per-position tensor must be a faithful decomposition of the ELBO.

    This is the property the whole integration leans on: it lets the ELBO ride in
    the per-token logprob slot that the rest of the stack already consumes.
    """
    cfg = make_cfg(quadrature="gauss-3")
    estimator = SdmcElboEstimator(cfg, MASK_ID)
    input_ids, completion_mask = make_batch()

    torch.manual_seed(7)
    scalar = torch.zeros(input_ids.shape[0])
    per_position = torch.zeros(input_ids.shape, dtype=torch.float32)
    for point in estimator.mask_points(input_ids, completion_mask, seed=1234):
        logprobs = -torch.rand(input_ids.shape)
        per_position += estimator.accumulate(point, logprobs)
        masked_lp = (logprobs * point.masked).sum(-1)
        scalar += point.coefficient * masked_lp

    torch.testing.assert_close(per_position.sum(-1), scalar)


def test_masking_never_touches_prompt_or_padding():
    cfg = make_cfg(quadrature="gauss-3", mc_samples=2)
    estimator = SdmcElboEstimator(cfg, MASK_ID)
    input_ids, completion_mask = make_batch()

    for point in estimator.mask_points(input_ids, completion_mask, seed=0):
        assert not (point.masked & ~completion_mask).any()
        # Positions outside the mask keep their original token.
        untouched = point.input_ids == input_ids
        assert untouched[~point.masked].all()
        # Masked positions carry the mask token.
        assert (point.input_ids[point.masked] == MASK_ID).all()


def test_masked_fraction_tracks_quadrature_time():
    cfg = make_cfg(quadrature="gauss-5")
    estimator = SdmcElboEstimator(cfg, MASK_ID)
    input_ids, completion_mask = make_batch(completion_len=100)
    scorable = completion_mask.sum(-1)[0].item()

    for point in estimator.mask_points(input_ids, completion_mask, seed=3):
        fraction = point.masked.sum(-1).float() / scorable
        assert fraction.allclose(
            torch.full_like(fraction, point.time), atol=1.0 / scorable
        )


def test_same_seed_gives_identical_masks():
    """old/ref/current ELBOs must share masks or the ratio is pure estimator noise."""
    cfg = make_cfg(quadrature="gauss-2")
    estimator = SdmcElboEstimator(cfg, MASK_ID)
    input_ids, completion_mask = make_batch()

    first = [
        p.masked.clone()
        for p in estimator.mask_points(input_ids, completion_mask, seed=99)
    ]
    second = [
        p.masked.clone()
        for p in estimator.mask_points(input_ids, completion_mask, seed=99)
    ]
    third = [
        p.masked.clone()
        for p in estimator.mask_points(input_ids, completion_mask, seed=100)
    ]

    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert not all(torch.equal(a, c) for a, c in zip(first, third))


def test_mask_seeds_are_stable_per_sequence():
    input_ids, _ = make_batch(batch=3)
    input_ids[1] = input_ids[0]

    first = make_dllm_mask_seeds(input_ids)
    second = make_dllm_mask_seeds(input_ids.clone())

    torch.testing.assert_close(first, second)
    assert first.dtype == torch.int64
    assert first[0] == first[1]


@pytest.mark.parametrize("quadrature", ["gauss-3", "mc"])
def test_per_row_seeds_are_independent_and_microbatch_invariant(quadrature):
    cfg = make_cfg(quadrature=quadrature, mc_samples=2)
    estimator = SdmcElboEstimator(cfg, MASK_ID)
    input_ids, completion_mask = make_batch(batch=3)
    seeds = torch.tensor([11, 22, 33])

    combined = list(estimator.mask_points(input_ids, completion_mask, seed=seeds))
    per_row = [
        list(
            estimator.mask_points(
                input_ids[row : row + 1],
                completion_mask[row : row + 1],
                seed=seeds[row : row + 1],
            )
        )
        for row in range(input_ids.shape[0])
    ]

    for point_index, point in enumerate(combined):
        for row in range(input_ids.shape[0]):
            torch.testing.assert_close(
                point.masked[row], per_row[row][point_index].masked[0]
            )
    assert not torch.equal(combined[0].masked[0], combined[0].masked[1])


def test_at_least_one_position_masked_at_small_t():
    """A point that masks nothing would be scaled by 1/t into a NaN."""
    cfg = make_cfg(quadrature="gauss-5")
    estimator = SdmcElboEstimator(cfg, MASK_ID)
    # 4 scorable positions and t as low as ~0.046 rounds to zero masked tokens.
    input_ids, completion_mask = make_batch(completion_len=4)

    for point in estimator.mask_points(input_ids, completion_mask, seed=5):
        assert (point.masked.sum(-1) >= 1).all()
        contribution = estimator.accumulate(point, -torch.rand(input_ids.shape))
        assert torch.isfinite(contribution).all()


def test_num_forwards_matches_points_times_samples():
    assert SdmcElboEstimator(make_cfg(quadrature="gauss-3"), MASK_ID).num_forwards == 3
    assert (
        SdmcElboEstimator(
            make_cfg(quadrature="gauss-3", mc_samples=4), MASK_ID
        ).num_forwards
        == 12
    )
    assert (
        SdmcElboEstimator(make_cfg(quadrature="mc", mc_samples=8), MASK_ID).num_forwards
        == 8
    )


def test_prompt_masking_corrupts_input_but_is_not_scored():
    cfg = make_cfg(quadrature="gauss-1", p_mask_prompt=1.0)
    estimator = SdmcElboEstimator(cfg, MASK_ID)
    input_ids, completion_mask = make_batch()

    (point,) = list(estimator.mask_points(input_ids, completion_mask, seed=11))
    prompt_positions = ~completion_mask
    # Every prompt token is corrupted at p=1.0 ...
    assert (point.input_ids[prompt_positions] == MASK_ID).all()
    # ... but none of them contribute to the likelihood.
    assert not (point.masked & prompt_positions).any()


@pytest.mark.parametrize(
    "rule,mc_samples",
    [
        ("gauss-1", 1),
        ("gauss-2", 1),
        ("gauss-3", 1),
        ("gauss-3", 3),
        ("gauss-5", 2),
        ("simpson", 1),
        ("mc", 1),
        ("mc", 4),
    ],
)
def test_every_rule_recovers_a_known_closed_form_elbo(rule, mc_samples):
    """Pins the absolute scale of every integration rule.

    For a model with constant per-token log probability ``c``, the integrand is
    ``(1/t) * sum_masked c = (1/t) * (tL) * c = Lc`` -- flat in ``t`` -- so the
    ELBO is exactly ``L*c`` under any correct quadrature. Any error in a weight,
    in the 1/t factor, or in how Monte Carlo draws are averaged shows up here as
    a scale error, whatever the rule.
    """
    const = -0.25
    estimator = SdmcElboEstimator(
        make_cfg(quadrature=rule, mc_samples=mc_samples), MASK_ID
    )
    input_ids, completion_mask = make_batch(batch=2, completion_len=128)
    scorable = completion_mask.sum(-1).float()

    total = torch.zeros(input_ids.shape[0])
    for point in estimator.mask_points(input_ids, completion_mask, seed=42):
        logprobs = torch.full(input_ids.shape, const)
        total += estimator.accumulate(point, logprobs).sum(-1)

    torch.testing.assert_close(total, scorable * const, rtol=0.02, atol=0.0)


def _elbo_of_reference_model(estimator, input_ids, completion_mask, base, seed):
    """Scores a toy 'model' that degrades as more of the canvas is masked.

    A model with constant per-token log probabilities would make the ELBO
    integrand constant in ``t``, which every integration rule nails exactly --
    and the comparison below would be vacuous. Real denoisers predict worse the
    more context is hidden, so the stand-in scales its log probabilities by
    ``sqrt(mask density)``. That makes the integrand vary with ``t`` (and be
    non-polynomial, so quadrature gets no free exactness).
    """
    total = torch.zeros(input_ids.shape[0])
    scorable = completion_mask.sum(-1, keepdim=True).float()
    for point in estimator.mask_points(input_ids, completion_mask, seed=seed):
        density = point.masked.sum(-1, keepdim=True).float() / scorable
        logprobs = base * (1.0 + 4.0 * density.sqrt())
        total += estimator.accumulate(point, logprobs).sum(-1)
    return total


def test_sdmc_has_lower_variance_than_double_monte_carlo():
    """The paper's core claim: deterministic time beats sampled time (Fig. 3).

    Both estimators get the same forward-pass budget, so this isolates *where*
    that budget is spent -- fixed quadrature nodes versus randomly drawn times.
    """
    torch.manual_seed(0)
    input_ids, completion_mask = make_batch(batch=1, completion_len=64)
    base = -torch.rand(input_ids.shape)

    budget = 3
    sdmc = SdmcElboEstimator(make_cfg(quadrature="gauss-3"), MASK_ID)
    double_mc = SdmcElboEstimator(make_cfg(quadrature="mc", mc_samples=budget), MASK_ID)
    assert sdmc.num_forwards == double_mc.num_forwards == budget

    sdmc_estimates = torch.tensor(
        [
            _elbo_of_reference_model(
                sdmc, input_ids, completion_mask, base, seed
            ).item()
            for seed in range(60)
        ]
    )
    mc_estimates = torch.tensor(
        [
            _elbo_of_reference_model(
                double_mc, input_ids, completion_mask, base, seed
            ).item()
            for seed in range(60)
        ]
    )

    assert sdmc_estimates.std() < mc_estimates.std()
    # And it should not have bought that stability with a biased mean.
    assert sdmc_estimates.mean().item() == pytest.approx(
        mc_estimates.mean().item(), rel=0.15
    )


@pytest.mark.parametrize(
    "rule,mc_samples", [("gauss-2", 1), ("gauss-3", 2), ("mc", 3), ("mc", 1)]
)
def test_yielded_points_match_the_advertised_forward_budget(rule, mc_samples):
    """num_forwards is what callers size their compute against, so it must be real."""
    estimator = SdmcElboEstimator(
        make_cfg(quadrature=rule, mc_samples=mc_samples), MASK_ID
    )
    input_ids, completion_mask = make_batch()

    points = list(estimator.mask_points(input_ids, completion_mask, seed=0))
    assert len(points) == estimator.num_forwards


class _FakeModelConfig:
    """Stands in for a Hugging Face config exposing a mask token id."""

    def __init__(self, **attrs):
        for key, value in attrs.items():
            setattr(self, key, value)


def test_mask_id_is_read_from_the_model_when_unset():
    """LLaDA publishes mask_token_id=126336, so users should not retype it."""
    cfg = MaskedDiffusionConfig(enabled=True)
    assert cfg.mask_id is None
    assert resolve_mask_id(cfg, _FakeModelConfig(mask_token_id=MASK_ID)) == MASK_ID


def test_explicit_mask_id_overrides_the_model():
    """An escape hatch for a model whose config is wrong or absent."""
    cfg = MaskedDiffusionConfig(enabled=True, mask_id=999)
    assert resolve_mask_id(cfg, _FakeModelConfig(mask_token_id=MASK_ID)) == 999


def test_missing_mask_id_everywhere_raises_with_guidance():
    """Silently guessing a mask id would corrupt the likelihood, not crash."""
    cfg = MaskedDiffusionConfig(enabled=True)
    with pytest.raises(ValueError, match="no mask token id is available"):
        resolve_mask_id(cfg, _FakeModelConfig(vocab_size=126464))


def test_defaults_match_the_papers_recommended_setting():
    cfg = SdmcLikelihoodConfig()
    assert cfg.quadrature == "gauss-2"
    assert cfg.mc_samples == 1
    assert SdmcElboEstimator(cfg, MASK_ID).num_forwards == 2


def test_estimator_is_dtype_preserving():
    cfg = make_cfg(quadrature="gauss-2")
    estimator = SdmcElboEstimator(cfg, MASK_ID)
    input_ids, completion_mask = make_batch()

    for point in estimator.mask_points(input_ids, completion_mask, seed=2):
        contribution = estimator.accumulate(
            point, torch.rand(input_ids.shape, dtype=torch.bfloat16).neg()
        )
        assert contribution.dtype == torch.bfloat16
        assert math.isfinite(contribution.float().sum().item())


def test_accumulate_runs_one_forward_per_quadrature_point():
    estimator = SdmcElboEstimator(make_cfg(quadrature="gauss-3", mc_samples=2), MASK_ID)
    input_ids, completion_mask = make_batch()
    seen = []

    def score_fn(masked_ids, clean_ids):
        seen.append((masked_ids, clean_ids))
        return -torch.rand(input_ids.shape)

    out = accumulate_elbo_logprobs(
        estimator,
        input_ids=input_ids,
        completion_mask=completion_mask,
        seed=3,
        score_fn=score_fn,
    )
    assert len(seen) == estimator.num_forwards == 6
    assert out.shape == input_ids.shape


def test_accumulate_scores_clean_targets_against_masked_inputs():
    """The model must see the corrupted sequence but be scored on the clean one."""
    estimator = SdmcElboEstimator(make_cfg(quadrature="gauss-2"), MASK_ID)
    input_ids, completion_mask = make_batch()

    def score_fn(masked_ids, clean_ids):
        # Targets are always the untouched sequence ...
        assert torch.equal(clean_ids, input_ids)
        # ... while the model input carries mask tokens the clean one lacks.
        assert (masked_ids == MASK_ID).any()
        assert not torch.equal(masked_ids, clean_ids)
        return -torch.rand(input_ids.shape)

    accumulate_elbo_logprobs(
        estimator,
        input_ids=input_ids,
        completion_mask=completion_mask,
        seed=1,
        score_fn=score_fn,
    )


def test_accumulate_matches_the_manual_loop():
    """The helper is exactly the documented accumulate-in-a-loop, nothing more."""
    cfg = make_cfg(quadrature="gauss-3")
    input_ids, completion_mask = make_batch()

    def score_fn(masked_ids, clean_ids):
        # Deterministic in the masked input, so both paths see identical scores.
        return -(masked_ids == MASK_ID).float()

    helper = accumulate_elbo_logprobs(
        SdmcElboEstimator(cfg, MASK_ID),
        input_ids=input_ids,
        completion_mask=completion_mask,
        seed=17,
        score_fn=score_fn,
    )

    manual_est = SdmcElboEstimator(cfg, MASK_ID)
    manual = torch.zeros(input_ids.shape)
    for point in manual_est.mask_points(input_ids, completion_mask, seed=17):
        manual += manual_est.accumulate(point, score_fn(point.input_ids, input_ids))

    torch.testing.assert_close(helper, manual)


def test_accumulate_is_differentiable_through_every_forward():
    """The training step needs gradients from all N forwards, not just the last."""
    estimator = SdmcElboEstimator(make_cfg(quadrature="gauss-3"), MASK_ID)
    input_ids, completion_mask = make_batch()
    scale = torch.ones(1, requires_grad=True)

    out = accumulate_elbo_logprobs(
        estimator,
        input_ids=input_ids,
        completion_mask=completion_mask,
        seed=2,
        score_fn=lambda masked, clean: -torch.rand(input_ids.shape) * scale,
    )
    out.sum().backward()
    assert scale.grad is not None and scale.grad.abs().item() > 0
