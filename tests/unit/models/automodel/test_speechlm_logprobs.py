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

import pytest
import torch
import torch.nn.functional as F

from nemo_rl.models.automodel.speechlm_logprobs import response_logprobs


def fixture():
    torch.manual_seed(42)
    hidden = torch.randn(2, 8, 5, requires_grad=True)
    native = torch.tensor(
        [
            [-100, -100, -100, -100, 2, 3, 1, -100],
            [-100, -100, 4, 1, -100, -100, -100, -100],
        ]
    )
    ids = torch.tensor([[0, 7, 2, 3, 1, 0], [0, 7, 4, 1, 0, 0]])
    mask = torch.tensor([[0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0]])
    weight = torch.randn(6, 5)
    return hidden, native, ids, mask, weight


@pytest.mark.parametrize("chunk_size", [1, 2, 256])
def test_logprobs_and_gradient_match_dense_projection(chunk_size):
    hidden, native, ids, mask, weight = fixture()
    expected_hidden = hidden.detach().clone().requires_grad_()
    dense = -F.cross_entropy(
        F.linear(expected_hidden, weight).float().flatten(0, 1) / 0.7,
        native.flatten(),
        reduction="none",
    ).reshape_as(native)
    expected = torch.zeros((2, 5)).masked_scatter(
        mask[:, 1:].bool(), dense[native != -100]
    )
    actual = response_logprobs(
        hidden, native, ids, mask, weight, chunk_size=chunk_size, temperature=0.7
    )
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(hidden.grad, expected_hidden.grad)
    assert torch.count_nonzero(hidden.grad[native == -100]) == 0


def test_wrong_target_order_rejected():
    hidden, native, ids, mask, weight = fixture()
    native[0, 4] = 3
    with pytest.raises(ValueError, match="token IDs"):
        response_logprobs(hidden, native, ids, mask, weight)


def test_empty_responses_have_zero_logprobs_and_zero_gradient():
    hidden, native, ids, mask, weight = fixture()
    native.fill_(-100)
    mask.zero_()
    actual = response_logprobs(hidden, native, ids, mask, weight)
    actual.sum().backward()
    assert torch.count_nonzero(actual) == 0
    assert torch.count_nonzero(hidden.grad) == 0
