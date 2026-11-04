# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import pickle
from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.loss.interfaces import (
    GlobalTokenMeanReduction,
    LossType,
    MetricNormalizer,
    SequenceMeanReduction,
    get_objective_reduction,
    make_objective_reduction,
)
from nemo_rl.algorithms.loss.utils import (
    metric_normalizer_scales,
    rescale_loss_metrics,
)
from nemo_rl.algorithms.loss.wrapper import StreamingObjectiveLoss
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@pytest.mark.parametrize("mode", ["global_token_mean", "sequence_mean"])
def test_named_reduction_fractional_masks(mode: str) -> None:
    # An empty but sample-valid response still counts in sequence normalization.
    data = {
        "token_mask": torch.tensor([[0, 1, 0.5], [0, 1, 1], [0, 0, 0]]),
        "sample_mask": torch.tensor([1.0, 0.25, 1.0]),
    }
    reduction = make_objective_reduction(mode)
    sequences, tokens, denominator = reduction.local_counts(data)
    assert sequences.item() == 2.25
    assert tokens.item() == 2.0
    values = torch.tensor([[1.0, 3.0], [2.0, 4.0], [7.0, 9.0]], requires_grad=True)
    token_mask = data["token_mask"][:, 1:]
    sample_mask = data["sample_mask"]
    actual = reduction.reduce(
        values,
        token_mask=token_mask,
        sample_mask=sample_mask,
        mask=token_mask * sample_mask[:, None],
        denominator=denominator,
        per_response_weights=None,
    )
    rows = (values * token_mask).sum(-1)
    if reduction.loss_type == LossType.SEQUENCE_LEVEL:
        rows = rows / (token_mask.sum(-1) + 1e-8)
        expected_denominator = sample_mask.sum()
    else:
        expected_denominator = (token_mask * sample_mask[:, None]).sum()
    expected = (rows * sample_mask).sum() / expected_denominator
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, values, retain_graph=True)[0],
        torch.autograd.grad(expected, values)[0],
    )


@pytest.mark.parametrize("mode", ["global_token_mean", "sequence_mean", "explicit"])
def test_streaming_preserves_legacy_objective(mode: str) -> None:
    # "explicit" needs the wrapper's denominator: a bypass trips its assert.
    data = BatchedDataDict(
        token_mask=torch.tensor([[0, 1, 0.5], [0, 1, 1], [0, 0, 0]]),
        sample_mask=torch.tensor([1.0, 0.25, 1.0]),
        advantages=torch.tensor([[0.0, 2, -1], [0, 3, 1], [0, 4, 5]]),
        prev_logprobs=torch.full((3, 3), -2.0),
        generation_logprobs=torch.full((3, 3), -2.0),
    )
    current = torch.full((3, 2), -3.0, requires_grad=True)
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            normalization="global_token_mean" if mode == "explicit" else mode,
            disable_ppo_ratio=True,
            reference_policy_kl_penalty=0,
        )
    )
    if mode == "explicit":
        loss_fn.objective_reduction = _ExplicitDenominatorReduction()
    sequences, tokens, denominator = loss_fn.objective_reduction.local_counts(data)
    full, _ = loss_fn(
        data=data,
        next_token_logprobs=current,
        global_valid_toks=tokens,
        global_valid_seqs=sequences,
        global_objective_denominator=denominator,
    )
    chunks = []
    for indices in ([2, 0], [1]):
        chunk = BatchedDataDict({key: value[indices] for key, value in data.items()})
        loss, _ = StreamingObjectiveLoss(loss_fn)(
            data=chunk,
            next_token_logprobs=current[indices],
            global_valid_toks=tokens.new_tensor(1),
            global_valid_seqs=sequences.new_tensor(1),
        )
        chunks.append(loss)
    streamed = torch.stack(chunks).sum() / denominator
    torch.testing.assert_close(streamed, full)
    torch.testing.assert_close(
        torch.autograd.grad(streamed, current, retain_graph=True)[0],
        torch.autograd.grad(full, current)[0],
    )


@dataclass(frozen=True)
class _ExplicitDenominatorReduction(GlobalTokenMeanReduction):
    requires_explicit_denominator: ClassVar[bool] = True

    def loss_divisor(
        self, *, global_valid_toks, global_valid_seqs, global_objective_denominator
    ):
        assert global_objective_denominator is not None
        return global_objective_denominator


def test_streaming_objective_loss_forwards_denominator_and_proxies_attributes() -> None:
    data = BatchedDataDict(
        token_mask=torch.tensor([[0, 1, 1], [0, 1, 0]], dtype=torch.float32),
        sample_mask=torch.tensor([1.0, 1.0]),
        advantages=torch.tensor([[0.0, 2.0, -1.0], [0.0, 3.0, 1.0]]),
        prev_logprobs=torch.full((2, 3), -2.0),
        generation_logprobs=torch.full((2, 3), -2.0),
    )
    current = torch.full((2, 2), -3.0, requires_grad=True)
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            normalization="global_token_mean",
            disable_ppo_ratio=True,
            reference_policy_kl_penalty=0,
        )
    )
    loss_fn.objective_reduction = _ExplicitDenominatorReduction()
    streaming = StreamingObjectiveLoss(loss_fn)

    assert streaming.loss_type is loss_fn.loss_type
    assert streaming.metric_normalizations is loss_fn.metric_normalizations
    assert not hasattr(streaming, "definitely_missing_attribute")

    placeholder = torch.tensor(7.0)
    loss, _ = streaming(
        data=data,
        next_token_logprobs=current,
        global_valid_toks=placeholder,
        global_valid_seqs=torch.tensor(99.0),
    )
    token_sum = -(data["advantages"][:, 1:] * current * data["token_mask"][:, 1:]).sum()
    torch.testing.assert_close(loss, token_sum / placeholder)
    with pytest.raises(TypeError, match="global_valid_toks"):
        streaming(current, data, torch.tensor(99.0), placeholder)


def test_streaming_objective_loss_survives_deepcopy_and_pickle() -> None:
    # __getattr__ must not recurse when copy/pickle probe before __init__ ran.
    streaming = StreamingObjectiveLoss(
        ClippedPGLossFn(ClippedPGLossConfig(normalization="sequence_mean"))
    )
    for clone in (copy.deepcopy(streaming), pickle.loads(pickle.dumps(streaming))):
        assert isinstance(clone.loss_fn, ClippedPGLossFn)
        assert clone.normalization == "sequence_mean"


def test_get_objective_reduction_prefers_reducer_then_legacy_loss_type() -> None:
    explicit = ClippedPGLossFn(ClippedPGLossConfig(normalization="sequence_mean"))
    assert get_objective_reduction(explicit) is explicit.objective_reduction
    assert isinstance(
        get_objective_reduction(SimpleNamespace(loss_type=LossType.SEQUENCE_LEVEL)),
        SequenceMeanReduction,
    )
    assert isinstance(
        get_objective_reduction(SimpleNamespace(loss_type=LossType.TOKEN_LEVEL)),
        GlobalTokenMeanReduction,
    )
    assert isinstance(get_objective_reduction(object()), GlobalTokenMeanReduction)
    assert isinstance(
        get_objective_reduction(MagicMock(loss_type=LossType.SEQUENCE_LEVEL)),
        SequenceMeanReduction,
    )


@pytest.mark.parametrize("mode", ["global_token_mean", "sequence_mean"])
def test_local_counts_without_token_mask_counts_every_position(mode: str) -> None:
    data = {
        "sample_mask": torch.tensor([1.0, 0.0, 1.0]),
        "input_ids": torch.zeros(3, 7, dtype=torch.long),
    }
    sequences, tokens, denominator = make_objective_reduction(mode).local_counts(data)
    assert sequences.item() == 2.0
    assert tokens.item() == 14.0
    assert denominator.item() == (14.0 if mode == "global_token_mean" else 2.0)


@pytest.mark.parametrize(
    "mode,legacy_kind",
    [
        ("global_token_mean", MetricNormalizer.TOKENS),
        ("sequence_mean", MetricNormalizer.SEQUENCES),
    ],
)
def test_loss_divisor_metrics_rescale_by_the_reduction_divisor(
    mode: str, legacy_kind: MetricNormalizer
) -> None:
    """Split-API rescaling of LOSS_DIVISOR metrics divides by loss_divisor()."""
    data = BatchedDataDict(
        token_mask=torch.tensor([[0, 1, 0.5], [0, 1, 1], [0, 0, 0]]),
        sample_mask=torch.tensor([1.0, 0.25, 1.0]),
        advantages=torch.tensor([[0.0, 2, -1], [0, 3, 1], [0, 4, 5]]),
        prev_logprobs=torch.full((3, 3), -2.0),
        generation_logprobs=torch.full((3, 3), -2.0),
        reference_policy_logprobs=torch.tensor(
            [[0.0, -2.5, -1.5], [0, -2.0, -3.5], [0, -1.0, -2.0]]
        ),
    )
    current = torch.tensor([[-3.0, -1.0], [-2.0, -2.5], [-1.5, -0.5]])
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(normalization=mode, reference_policy_kl_penalty=0.1)
    )
    norms = loss_fn.metric_normalizations
    assert norms["loss"] is MetricNormalizer.LOSS_DIVISOR
    assert norms["kl_penalty"] is MetricNormalizer.LOSS_DIVISOR
    reduction = loss_fn.objective_reduction
    sequences, tokens, denominator = reduction.local_counts(data)
    _, synchronous = loss_fn(
        data=data,
        next_token_logprobs=current,
        global_valid_toks=tokens,
        global_valid_seqs=sequences,
    )
    raw: dict[str, float] = {}
    one = tokens.new_tensor(1.0)
    for indices in ([2, 0], [1]):
        chunk = BatchedDataDict({key: value[indices] for key, value in data.items()})
        _, metrics = loss_fn(
            data=chunk,
            next_token_logprobs=current[indices],
            global_valid_toks=one,
            global_valid_seqs=one,
        )
        for key, value in metrics.items():
            raw[key] = raw.get(key, 0.0) + value

    scales = metric_normalizer_scales(
        reduction,
        global_valid_toks=tokens,
        global_valid_seqs=sequences,
        global_objective_denominator=denominator,
    )
    divisor = reduction.loss_divisor(
        global_valid_toks=tokens,
        global_valid_seqs=sequences,
        global_objective_denominator=denominator,
    )
    assert scales[MetricNormalizer.LOSS_DIVISOR] == pytest.approx(1 / divisor.item())
    # Same factor as the TOKENS/SEQUENCES entry these metrics used to advertise.
    assert scales[MetricNormalizer.LOSS_DIVISOR] == scales[legacy_kind]
    assert synchronous["kl_penalty"] != 0
    for key in ("loss", "kl_penalty"):
        assert raw[key] * scales[norms[key]] == pytest.approx(synchronous[key])


def test_rescale_loss_metrics_applies_the_gradient_factor_to_loss_divisor() -> None:
    normalizers = {
        "loss": MetricNormalizer.LOSS_DIVISOR,
        "tok": MetricNormalizer.TOKENS,
        "seq": MetricNormalizer.SEQUENCES,
        "count": MetricNormalizer.NONE,
    }
    metrics = {"loss": 1.0, "tok": 1.0, "seq": 1.0, "count": 1.0}
    assert rescale_loss_metrics(
        metrics,
        normalizers,
        token_factor=2.0,
        sequence_factor=3.0,
        loss_divisor_factor=5.0,
    ) == {"loss": 5.0, "tok": 2.0, "seq": 3.0, "count": 1.0}
