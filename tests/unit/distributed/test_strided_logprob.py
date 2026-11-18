# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Filtered log-probs accept the transposed output of a sequence-major LM head."""

import functools

import pytest
import torch

from nemo_rl.algorithms.logits_sampling_utils import apply_top_k_top_p
from nemo_rl.distributed.model_utils import (
    ChunkedDistributedLogprobWithSampling,
    DistributedLogprobWithSampling,
)


def _run_strided_logprob(
    rank: int, world_size: int, *, chunk_size: int | None, top_p: float
) -> None:
    torch.manual_seed(31)
    full_logits = torch.randn(5, 2, 32, device="cuda", requires_grad=True)
    target = full_logits.detach().transpose(0, 1).argmax(dim=-1)
    start, end = rank * 16, (rank + 1) * 16
    local_logits = full_logits.detach()[..., start:end].contiguous().requires_grad_()
    strided_logits = local_logits.transpose(0, 1)
    contiguous_logits = strided_logits.detach().contiguous().requires_grad_()
    assert not strided_logits.is_contiguous()
    top_k = 4 if top_p == 1.0 else None
    args = (target, torch.distributed.group.WORLD, top_k, top_p)

    if chunk_size is None:
        actual = DistributedLogprobWithSampling.apply(strided_logits, *args)
        contiguous = DistributedLogprobWithSampling.apply(contiguous_logits, *args)
    else:
        actual = ChunkedDistributedLogprobWithSampling.apply(
            strided_logits, *args, chunk_size
        )
        contiguous = ChunkedDistributedLogprobWithSampling.apply(
            contiguous_logits, *args, chunk_size
        )

    filtered, _ = apply_top_k_top_p(
        full_logits.transpose(0, 1).reshape(-1, 32), top_k=top_k, top_p=top_p
    )
    expected = (
        filtered.log_softmax(dim=-1)
        .gather(-1, target.reshape(-1, 1))
        .reshape_as(target)
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual, contiguous, rtol=0, atol=0)

    actual_grad = torch.autograd.grad(actual.sum(), local_logits)[0].transpose(0, 1)
    contiguous_grad = torch.autograd.grad(contiguous.sum(), contiguous_logits)[0]
    expected_grad = torch.autograd.grad(expected.sum(), full_logits)[0].transpose(0, 1)[
        ..., start:end
    ]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_grad, contiguous_grad, rtol=0, atol=0)


@pytest.mark.parametrize("chunk_size", [None, 2])
@pytest.mark.parametrize("top_p", [1.0, 0.8])
def test_filtered_logprob_accepts_strided_logits(
    distributed_test_runner, chunk_size: int | None, top_p: float
) -> None:
    test_fn = functools.partial(
        _run_strided_logprob, chunk_size=chunk_size, top_p=top_p
    )
    distributed_test_runner(test_fn, world_size=2)
