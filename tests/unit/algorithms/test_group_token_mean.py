# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
import torch

from nemo_rl.algorithms.loss import (
    ClippedPGLossConfig,
    ClippedPGLossFn,
    SequencePackingFusionLossWrapper,
    SequencePackingLossWrapper,
)
from nemo_rl.algorithms.loss.interfaces import (
    GroupTokenMeanReduction,
    MetricNormalizer,
    group_token_mean_weights,
    make_objective_reduction,
)
from nemo_rl.algorithms.loss.wrapper import StreamingObjectiveLoss
from nemo_rl.data_plane.schema import GROUP_TOKEN_MEAN_WEIGHT
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

GROUP_INDICES = torch.tensor([0, 0, 1, 1, 1, 2, 2])


def _batch(*, empty: bool = False) -> tuple[BatchedDataDict[Any], torch.Tensor]:
    lengths = torch.tensor([2, 8, 3, 6, 1, 7, 4])
    mask = (torch.arange(10)[None, :] <= lengths[:, None]).float()
    mask[3, 3] = 0.3
    sample = torch.tensor([1, 0.7, 1, 1, 0, 0, 0])
    if empty:
        sample.zero_()
    previous = torch.full_like(mask, -2.0)
    # Keep rejection decisions away from the 0.1/5.0 bounds of ICEPOP.
    ratio = torch.tensor([1, 0.05, 0.2, 0.5, 1, 2, 4, 6, 1.1, 0.9])
    advantages = (
        torch.tensor([1, -0.5, 2, -1, 0, 3, -2])[:, None].expand_as(mask).clone()
    )
    advantages[1, 3] = 0  # Zero advantage must not remove denominator mass.
    data = BatchedDataDict(
        input_ids=torch.zeros_like(mask, dtype=torch.long),
        input_lengths=lengths + 1,
        token_mask=mask,
        sample_mask=sample,
        advantages=advantages,
        prev_logprobs=previous,
        generation_logprobs=previous - ratio.log(),
        reference_policy_logprobs=previous - 0.2,
    )
    data.update(
        GroupTokenMeanReduction().prepare_batch(
            sample_ids=[str(i) for i in range(7)],
            token_mask=mask,
            sample_mask=sample,
            group_indices=GROUP_INDICES,
            expected_group_size=None,
        )
    )
    return data, (previous[:, 1:] + torch.linspace(-0.3, 0.3, 9)).requires_grad_()


def _loss_fn() -> ClippedPGLossFn:
    return ClippedPGLossFn(
        ClippedPGLossConfig(
            normalization="group_token_mean",
            use_cispo=True,
            use_importance_sampling_correction=True,
            truncated_importance_sampling_type="icepop",
            truncated_importance_sampling_ratio_min=0.1,
            truncated_importance_sampling_ratio=5.0,
            reference_policy_kl_penalty=0.03,
        )
    )


def _call(
    loss_fn: ClippedPGLossFn,
    data: BatchedDataDict[Any],
    current: torch.Tensor,
    *,
    raw: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    sequences, tokens, groups = loss_fn.objective_reduction.local_counts(data)
    if raw:
        return StreamingObjectiveLoss(loss_fn)(
            data=data,
            next_token_logprobs=current,
            global_valid_toks=tokens.new_tensor(1),
            global_valid_seqs=sequences.new_tensor(1),
        )
    return loss_fn(
        data=data,
        next_token_logprobs=current,
        global_valid_toks=tokens,
        global_valid_seqs=sequences,
        global_objective_denominator=groups,
    )


@pytest.mark.parametrize("empty", [False, True])
def test_group_loss_gradient_and_partition_invariance(empty: bool) -> None:
    data, current = _batch(empty=empty)
    loss_fn = _loss_fn()
    actual, metrics = _call(loss_fn, data, current)
    # Independent FP64 oracle: no production row weights or reduction helpers.
    reference = {key: value.double() for key, value in data.items()}
    logprobs = current.double()
    mask = reference["token_mask"][:, 1:] * reference["sample_mask"][:, None]
    previous = reference["prev_logprobs"][:, 1:]
    cispo = (logprobs - previous).exp().clamp(0.8, 1.2).detach()
    importance = (previous - reference["generation_logprobs"][:, 1:]).exp()
    importance = importance * ((importance >= 0.1) & (importance <= 5.0))
    values = -reference["advantages"][:, 1:] * cispo * logprobs * importance
    delta = reference["reference_policy_logprobs"][:, 1:] - logprobs
    values = (
        values + 0.03 * (delta.exp() - 1 - delta) * (logprobs - logprobs.detach()).exp()
    )
    group_means = []
    for indices in ([0, 1], [2, 3, 4], [5, 6]):
        denominator = mask[indices].sum()
        if denominator > 0:
            group_means.append((values[indices] * mask[indices]).sum() / denominator)
    expected = torch.stack(group_means).mean() if group_means else logprobs.sum() * 0
    expected_grad = torch.autograd.grad(expected, current, retain_graph=True)[0]
    torch.testing.assert_close(actual, expected.float(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        torch.autograd.grad(actual, current, retain_graph=True)[0], expected_grad
    )

    # Split siblings across chunks; the complete-group weights must stay fixed.
    losses, token_errors, chunk_groups = [], [], []
    for indices in ([6, 2], [0], [5, 4, 1], [3]):
        chunk = BatchedDataDict({key: value[indices] for key, value in data.items()})
        loss, chunk_metrics = _call(loss_fn, chunk, current[indices], raw=True)
        losses.append(loss)
        token_errors.append(chunk_metrics["token_mult_prob_error"])
        # Mirror the worker, which accumulates the denominator per chunk.
        chunk_groups.append(loss_fn.objective_reduction.local_counts(chunk)[2])
    _, tokens, groups = loss_fn.objective_reduction.local_counts(data)
    assert groups.item() == pytest.approx(0 if empty else 2)
    torch.testing.assert_close(torch.stack(chunk_groups).sum(), groups)
    streamed = torch.stack(losses).sum() / groups.clamp_min(1)
    torch.testing.assert_close(streamed, actual)
    torch.testing.assert_close(torch.autograd.grad(streamed, current)[0], expected_grad)
    assert sum(token_errors) / tokens.clamp_min(1).item() == pytest.approx(
        metrics["token_mult_prob_error"]
    )


@pytest.mark.parametrize("fused", [False, True])
def test_sequence_packing_preserves_group_weights_and_gradients(fused: bool) -> None:
    data, current = _batch()
    loss_fn = _loss_fn()
    full, _ = _call(loss_fn, data, current)
    bounds = (0, *data["input_lengths"].cumsum(0).tolist())
    packed = torch.cat(
        [
            torch.cat((current[row, : length - 1], current.new_zeros(1)))
            for row, length in enumerate(data["input_lengths"].tolist())
        ]
    )[None, :]

    def prepare(
        *, logits: torch.Tensor, data: BatchedDataDict[Any], **kwargs: Any
    ) -> tuple[dict[str, torch.Tensor], BatchedDataDict[Any]]:
        if fused:
            # Fused prepare returns unpacked response logprobs.
            rows = [
                torch.nn.functional.pad(
                    logits[:, start : end - 1],
                    (0, current.shape[1] - (end - start - 1)),
                )
                for start, end in zip(bounds[:-1], bounds[1:])
            ]
            return {"next_token_logprobs": torch.cat(rows)}, data
        return {"next_token_logprobs": logits[:, :-1]}, data

    wrapper_type = (
        SequencePackingFusionLossWrapper if fused else SequencePackingLossWrapper
    )
    packed_sum, _ = wrapper_type(StreamingObjectiveLoss(loss_fn), prepare, bounds)(
        packed,
        data,
        global_valid_seqs=current.new_tensor(1),
        global_valid_toks=current.new_tensor(1),
    )
    packed_loss = packed_sum / loss_fn.objective_reduction.local_counts(data)[2]
    torch.testing.assert_close(packed_loss, full)
    torch.testing.assert_close(
        torch.autograd.grad(packed_loss, current, retain_graph=True)[0],
        torch.autograd.grad(full, current)[0],
    )


def _weights_inputs() -> dict[str, Any]:
    """Valid inputs for group_token_mean_weights: two groups of two rows."""
    token_mask = torch.ones(4, 4)
    token_mask[:, 0] = 0
    return {
        "sample_ids": ["a", "b", "c", "d"],
        "token_mask": token_mask,
        "sample_mask": torch.ones(4),
        "group_indices": torch.tensor([0, 1, 0, 1]),
        "expected_group_size": 2,
    }


def _loss_data() -> tuple[BatchedDataDict[Any], torch.Tensor]:
    """One group of two responses with three valid tokens each (T_g = 6)."""
    token_mask = torch.tensor([[0.0, 1, 1, 1], [0, 1, 1, 1]])
    sample_mask = torch.ones(2)
    data = BatchedDataDict(
        token_mask=token_mask,
        sample_mask=sample_mask,
        advantages=torch.tensor([[0.0, 1, -1, 2], [0, -2, 1, 1]]),
        prev_logprobs=torch.full((2, 4), -2.0),
        generation_logprobs=torch.full((2, 4), -2.0),
        reference_policy_logprobs=torch.full((2, 4), -2.1),
    )
    data[GROUP_TOKEN_MEAN_WEIGHT] = group_token_mean_weights(
        ["a", "b"], token_mask, sample_mask, group_indices=torch.tensor([0, 0])
    )
    return data, torch.full((2, 3), -2.5, requires_grad=True)


def test_group_token_mean_requires_an_explicit_denominator() -> None:
    data, current = _loss_data()
    loss_fn = ClippedPGLossFn(ClippedPGLossConfig(normalization="group_token_mean"))
    with pytest.raises(ValueError, match="split training path"):
        loss_fn(
            data=data,
            next_token_logprobs=current,
            global_valid_toks=torch.tensor(6.0),
            global_valid_seqs=torch.tensor(2.0),
        )


def test_group_token_mean_rejects_malformed_row_weights() -> None:
    data, current = _loss_data()
    data[GROUP_TOKEN_MEAN_WEIGHT] = torch.full((2, 1), 1 / 6)
    loss_fn = ClippedPGLossFn(ClippedPGLossConfig(normalization="group_token_mean"))
    with pytest.raises(ValueError, match="must match sample_mask"):
        loss_fn(
            data=data,
            next_token_logprobs=current,
            global_valid_toks=torch.tensor(6.0),
            global_valid_seqs=torch.tensor(2.0),
            global_objective_denominator=torch.tensor(1.0),
        )


def test_group_token_mean_defers_loss_metrics_to_the_group_denominator() -> None:
    loss_fn = ClippedPGLossFn(ClippedPGLossConfig(normalization="group_token_mean"))
    assert "loss" not in loss_fn.metric_normalizations
    assert "kl_penalty" not in loss_fn.metric_normalizations
    assert (
        loss_fn.metric_normalizations["token_mult_prob_error"]
        is MetricNormalizer.TOKENS
    )


@pytest.mark.parametrize(
    "override,match",
    [
        ({"sample_ids": ["a", "a", "c", "d"]}, "unique sample ID"),
        ({"sample_mask": torch.tensor([1.0, -0.5, 1.0, 1.0])}, "nonnegative"),
        ({"token_mask": torch.full((4, 4), float("nan"))}, "finite"),
        (
            {"group_indices": torch.tensor([0, 1, 0, 1], dtype=torch.int32)},
            "integer group index",
        ),
        ({"group_indices": torch.tensor([0, 1, 0, 2])}, "incomplete group"),
    ],
)
def test_group_token_mean_weights_rejects_bad_inputs(override, match) -> None:
    with pytest.raises(ValueError, match=match):
        group_token_mean_weights(**{**_weights_inputs(), **override})


@pytest.mark.parametrize(
    "overrides",
    [
        {"trainer": "grpo"},
        {"trainer": "ppo", "algorithm": "ppo"},
        {"algorithm": "ppo"},
        {"on_policy_distillation": True},
    ],
)
def test_group_token_mean_validate_setup_rejects_unsupported_paths(overrides) -> None:
    supported = dict(
        trainer="single_controller",
        algorithm="grpo",
        megatron_enabled=True,
        on_policy_distillation=False,
    )
    reduction = make_objective_reduction("group_token_mean")
    reduction.validate_setup(**supported)
    with pytest.raises(ValueError, match="group_token_mean requires"):
        reduction.validate_setup(**{**supported, **overrides})
    make_objective_reduction("global_token_mean").validate_setup(
        **{**supported, **overrides}
    )


@pytest.mark.parametrize(
    "override,match",
    [
        (
            {"sequence_level_importance_ratios": True},
            "sequence_level_importance_ratios",
        ),
        ({"opd_full": True}, "OPD-full or NLL"),
        ({"positive_example_nll_weight": 0.1}, "OPD-full or NLL"),
        ({"seq_logprob_error_in_loss": True}, "masks finalized before training"),
    ],
)
def test_group_token_mean_validate_loss_options_rejects_each_option(
    override: dict[str, Any], match: str
) -> None:
    supported = dict(
        sequence_level_importance_ratios=False,
        opd_full=False,
        positive_example_nll_weight=0.0,
        seq_logprob_error_in_loss=False,
    )
    make_objective_reduction("group_token_mean").validate_loss_options(**supported)
    with pytest.raises(ValueError, match=match):
        make_objective_reduction("group_token_mean").validate_loss_options(
            **{**supported, **override}
        )
    # The base reductions place no such restriction.
    make_objective_reduction("global_token_mean").validate_loss_options(
        **{**supported, **override}
    )
