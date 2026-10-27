# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.loss.interfaces import LossType, make_objective_reduction
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


@pytest.mark.parametrize("mode", ["global_token_mean", "sequence_mean"])
def test_streaming_preserves_legacy_objective(mode: str) -> None:
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
            normalization=mode,
            disable_ppo_ratio=True,
            reference_policy_kl_penalty=0,
        )
    )
    sequences, tokens, denominator = loss_fn.objective_reduction.local_counts(data)
    full, _ = loss_fn(
        data=data,
        next_token_logprobs=current,
        global_valid_toks=tokens,
        global_valid_seqs=sequences,
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
