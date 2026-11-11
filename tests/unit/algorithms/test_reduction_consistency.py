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

"""Every ClippedPGLossFn term must follow the configured objective reduction.

Each case builds the per-token values of the active terms independently, sums
them, and reduces that sum with a hand-written oracle for the normalization.
A term that picks its own denominator (for example from ``loss_type``) breaks
the equality as soon as response lengths differ, which the batch below
guarantees. Add new loss terms here: give them an entry in ``TERMS``, a
constructor override in ``_loss_config`` and their per-token value in
``_per_token_terms``.
"""

from __future__ import annotations

import pytest
import torch

from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.loss.interfaces import group_token_mean_weights
from nemo_rl.algorithms.loss.score_centering import compute_score_centering_term
from nemo_rl.algorithms.opd import OnPolicyDistillationFullConfig
from nemo_rl.algorithms.utils import calculate_kl, masked_mean
from nemo_rl.data_plane.schema import GROUP_TOKEN_MEAN_WEIGHT
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

# Two prompt groups with unequal response lengths and unequal group masses:
# group 0 holds 5 + 1 valid tokens, group 1 holds 2 + 3.
VALID_LENGTHS = (5, 1, 2, 3)
GROUP_INDICES = (0, 0, 1, 1)
SEQ_LEN = 6
VOCAB = 7
TOP_K = 2
KL_COEF = 0.1

NORMALIZATIONS = ("global_token_mean", "sequence_mean", "group_token_mean")
# Formulas that existed before the reducer refactor.
LEGACY_NORMALIZATIONS = ("global_token_mean", "sequence_mean")
TERMS = ("none", "kl", "score_centering", "opd_full", "opd_full_kl")
# Dropped (0) and fractionally weighted rows must leave both the sums and the
# denominators exactly as the oracle weights them.
SAMPLE_MASKS = {
    "all_kept": (1.0, 1.0, 1.0, 1.0),
    "longest_dropped": (0.0, 1.0, 1.0, 1.0),
    "fractional": (1.0, 0.0, 0.5, 0.25),
}
# Combinations the loss rejects at construction.
UNSUPPORTED = {
    ("group_token_mean", "opd_full"),
    ("group_token_mean", "opd_full_kl"),
}
CASES = [
    (normalization, term)
    for normalization in NORMALIZATIONS
    for term in TERMS
    if (normalization, term) not in UNSUPPORTED
]


def _batch(
    sample_mask: tuple[float, ...] = SAMPLE_MASKS["all_kept"],
) -> tuple[BatchedDataDict, dict[str, torch.Tensor]]:
    """Return the loss data and the differentiable model outputs."""
    generator = torch.Generator().manual_seed(1234)
    batch_size = len(VALID_LENGTHS)
    token_mask = torch.zeros(batch_size, SEQ_LEN)
    for row, length in enumerate(VALID_LENGTHS):
        token_mask[row, 1 : 1 + length] = 1.0
    logits = torch.randn(batch_size, SEQ_LEN, VOCAB, generator=generator)
    sampler_logprobs = logits.log_softmax(-1)
    sampler_topk_logprobs, sampler_topk_ids = sampler_logprobs.topk(TOP_K, dim=-1)
    policy_logprobs = (
        logits + 0.3 * torch.randn(batch_size, SEQ_LEN, VOCAB, generator=generator)
    ).log_softmax(-1)
    data = BatchedDataDict(
        input_ids=sampler_topk_ids[..., 0],
        token_mask=token_mask,
        sample_mask=torch.tensor(sample_mask),
        advantages=torch.randn(batch_size, SEQ_LEN, generator=generator),
        prev_logprobs=sampler_topk_logprobs[..., 0],
        generation_logprobs=sampler_topk_logprobs[..., 0],
        reference_policy_logprobs=(
            sampler_topk_logprobs[..., 0]
            + 0.2 * torch.randn(batch_size, SEQ_LEN, generator=generator)
        ),
        sampler_topk_token_ids=sampler_topk_ids,
        sampler_topk_logprobs=sampler_topk_logprobs,
    )
    data[GROUP_TOKEN_MEAN_WEIGHT] = group_token_mean_weights(
        [f"s{row}" for row in range(batch_size)],
        token_mask,
        data["sample_mask"],
        group_indices=torch.tensor(GROUP_INDICES),
        expected_group_size=2,
    )
    topk_policy = policy_logprobs.gather(-1, sampler_topk_ids)[:, 1:]
    outputs = {
        "next_token_logprobs": topk_policy[..., 0].clone().requires_grad_(True),
        "policy_logprobs_at_sampler_topk": topk_policy.clone().requires_grad_(True),
        "opd_full_divergence": torch.rand(
            batch_size, SEQ_LEN - 1, generator=generator
        ).requires_grad_(True),
    }
    return data, outputs


def _loss_config(normalization: str, term: str) -> tuple[ClippedPGLossConfig, dict]:
    overrides: dict = {
        "normalization": normalization,
        "disable_ppo_ratio": True,
        "reference_policy_kl_penalty": KL_COEF if term.endswith("kl") else 0.0,
        "score_centering": term == "score_centering",
    }
    extra = {}
    if term.startswith("opd_full"):
        extra["opd_full"] = OnPolicyDistillationFullConfig(enabled=True)
    return ClippedPGLossConfig(**overrides), extra


def _per_token_terms(
    term: str, data: BatchedDataDict, outputs: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Sum of every active term's per-token value, before any reduction."""
    mask = data["token_mask"][:, 1:] * data["sample_mask"].unsqueeze(-1)
    current = outputs["next_token_logprobs"]
    advantages = data["advantages"][:, 1:]
    if term.startswith("opd_full"):
        values = outputs["opd_full_divergence"]
    else:
        # REINFORCE actor term with disable_ppo_ratio and IS off.
        values = -advantages * current
    if term.endswith("kl"):
        values = values + KL_COEF * calculate_kl(
            logprobs=current,
            logprobs_reference=data["reference_policy_logprobs"][:, 1:],
            kl_type="k3",
            importance_sampling_weights=torch.exp(current - current.detach()),
        )
    if term == "score_centering":
        centering, _ = compute_score_centering_term(
            outputs["policy_logprobs_at_sampler_topk"],
            sampler_topk_logprobs=data["sampler_topk_logprobs"][:, 1:],
            importance_sampling_mode="off",
            is_bounds=None,
            token_mask=mask,
            sampler_replay=False,
        )
        values = values + advantages * centering
    return values


def _oracle(normalization: str, values: torch.Tensor, data: BatchedDataDict):
    """Reduce per-token values by the normalization's defining formula."""
    token_mask = data["token_mask"][:, 1:]
    sample_mask = data["sample_mask"]
    rows = [(values[i] * token_mask[i]).sum() for i in range(len(VALID_LENGTHS))]
    lengths = [token_mask[i].sum() for i in range(len(VALID_LENGTHS))]
    if normalization == "global_token_mean":
        total_tokens = sum(n * s for n, s in zip(lengths, sample_mask))
        return sum(r * s for r, s in zip(rows, sample_mask)) / total_tokens
    if normalization == "sequence_mean":
        per_response = [r / n * s for r, n, s in zip(rows, lengths, sample_mask)]
        return sum(per_response) / sample_mask.sum()
    if normalization == "group_token_mean":
        # Each prompt group's token mean, then the mean over groups with tokens.
        group_tokens: dict[int, torch.Tensor] = {}
        for group, n, s in zip(GROUP_INDICES, lengths, sample_mask):
            group_tokens[group] = group_tokens.get(group, 0.0) + n * s
        nonempty_groups = sum(1 for total in group_tokens.values() if total > 0)
        weighted = [
            r * s / group_tokens[group]
            for r, s, group in zip(rows, sample_mask, GROUP_INDICES)
        ]
        return sum(weighted) / nonempty_groups
    raise AssertionError(f"no oracle for {normalization}")


def _pre_refactor(normalization: str, values: torch.Tensor, data: BatchedDataDict):
    """The loss_type-branched reduction ClippedPGLossFn used before reducers."""
    token_mask = data["token_mask"][:, 1:]
    sample_mask = data["sample_mask"]
    mask = token_mask * sample_mask.unsqueeze(-1)
    if normalization == "global_token_mean":
        return masked_mean(values, mask, global_normalization_factor=mask.sum())
    return masked_mean(
        masked_mean(values, token_mask, dim=-1),
        sample_mask,
        global_normalization_factor=sample_mask.sum(),
    )


def _run_loss(normalization: str, term: str, data, outputs, **call_kwargs):
    cfg, extra = _loss_config(normalization, term)
    loss_fn = ClippedPGLossFn(cfg, **extra)
    mask = data["token_mask"][:, 1:] * data["sample_mask"].unsqueeze(-1)
    if loss_fn.objective_reduction.requires_explicit_denominator:
        # What the split worker would all-reduce for this (single-chunk) batch.
        _, _, denominator = loss_fn.objective_reduction.local_counts(data)
        call_kwargs.setdefault("global_objective_denominator", denominator)
    kwargs = {
        "data": data,
        "next_token_logprobs": outputs["next_token_logprobs"],
        "global_valid_toks": mask.sum(),
        "global_valid_seqs": data["sample_mask"].sum(),
        **call_kwargs,
    }
    if term == "score_centering":
        kwargs["policy_logprobs_at_sampler_topk"] = outputs[
            "policy_logprobs_at_sampler_topk"
        ]
    if term.startswith("opd_full"):
        kwargs["opd_full_divergence"] = outputs["opd_full_divergence"]
    loss, _ = loss_fn(**kwargs)
    return loss


def _assert_same_value_and_grads(actual, expected, outputs) -> None:
    torch.testing.assert_close(actual, expected)
    inputs = list(outputs.values())
    actual_grads = torch.autograd.grad(
        actual, inputs, retain_graph=True, allow_unused=True
    )
    expected_grads = torch.autograd.grad(
        expected, inputs, retain_graph=True, allow_unused=True
    )
    for name, got, want in zip(outputs, actual_grads, expected_grads):
        if want is None:
            assert got is None or torch.count_nonzero(got) == 0, name
        else:
            assert got is not None, name
            torch.testing.assert_close(got, want, msg=name)


@pytest.mark.parametrize("sample_mask", SAMPLE_MASKS)
@pytest.mark.parametrize("normalization,term", CASES)
def test_every_loss_term_follows_the_configured_reduction(
    normalization: str, term: str, sample_mask: str
) -> None:
    data, outputs = _batch(SAMPLE_MASKS[sample_mask])
    loss = _run_loss(normalization, term, data, outputs)
    values = _per_token_terms(term, data, outputs)
    _assert_same_value_and_grads(loss, _oracle(normalization, values, data), outputs)


@pytest.mark.parametrize("sample_mask", SAMPLE_MASKS)
@pytest.mark.parametrize("term", TERMS)
@pytest.mark.parametrize("normalization", LEGACY_NORMALIZATIONS)
def test_reducers_preserve_the_pre_refactor_formula(
    normalization: str, term: str, sample_mask: str
) -> None:
    data, outputs = _batch(SAMPLE_MASKS[sample_mask])
    values = _per_token_terms(term, data, outputs)
    expected = _pre_refactor(normalization, values, data)
    _assert_same_value_and_grads(
        _oracle(normalization, values, data), expected, outputs
    )
    _assert_same_value_and_grads(
        _run_loss(normalization, term, data, outputs), expected, outputs
    )


def test_batch_distinguishes_the_reductions() -> None:
    # Guard the fixture: equal lengths would make every reduction coincide.
    data, outputs = _batch()
    values = _per_token_terms("none", data, outputs)
    results = [_oracle(mode, values, data) for mode in NORMALIZATIONS]
    for i, first in enumerate(results):
        for second in results[i + 1 :]:
            assert not torch.allclose(first, second)


@pytest.mark.parametrize("normalization,term", sorted(UNSUPPORTED))
def test_unsupported_combinations_are_rejected(normalization: str, term: str) -> None:
    cfg, extra = _loss_config(normalization, term)
    with pytest.raises(ValueError, match="does not support"):
        ClippedPGLossFn(cfg, **extra)
