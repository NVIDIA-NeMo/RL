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

import pytest
import torch
import torch.distributed.nn.functional

from nemo_rl.distributed.model_utils import ChunkedDistributedLogprob


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk_size", [1, 4, 32])
@pytest.mark.parametrize("logit_scale", [1.0, 40.0])
def test_chunked_logprob_gradients(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    chunk_size: int,
    logit_scale: float,
) -> None:
    """Check weighted gradients, partial chunks, and saved-input reuse at TP=1."""
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        torch.distributed.nn.functional,
        "all_reduce",
        lambda tensor, *args, **kwargs: tensor,
    )
    torch.manual_seed(42)
    source = (torch.randn(2, 11, 74) * logit_scale).to(dtype)
    logits = source[..., ::2].requires_grad_(True)
    before = logits.detach().clone()
    reference = logits.detach().float().requires_grad_(True)
    target = torch.randint(logits.shape[-1], logits.shape[:-1])
    grad_output = torch.randn(*logits.shape[:-1])

    actual = ChunkedDistributedLogprob.apply(
        logits, target, 0, logits.shape[-1], chunk_size, None, False
    )
    expected = reference.log_softmax(-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    expected_grad = torch.autograd.grad(expected, reference, grad_output)[0].to(dtype)
    actual_grad = torch.autograd.grad(actual, logits, grad_output, retain_graph=True)[0]
    repeated_grad = torch.autograd.grad(actual, logits, grad_output)[0]

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)
    torch.testing.assert_close(actual_grad, repeated_grad, rtol=0, atol=0)
    torch.testing.assert_close(logits, before, rtol=0, atol=0)
