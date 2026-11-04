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

"""Numerical and retained-activation contracts for non-distributed top-k KD."""

import pytest
import torch

from nemo_rl.distributed.model_utils import (
    get_distillation_topk_logprobs_from_logits,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request: pytest.FixtureRequest) -> torch.device:
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch.device(request.param)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("zero_outside_topk", [False, True])
@pytest.mark.parametrize("defer_fp32_upcast", [False, True])
def test_topk_matches_full_upcast_with_masked_padding(
    device: torch.device,
    dtype: torch.dtype,
    zero_outside_topk: bool,
    defer_fp32_upcast: bool,
) -> None:
    """Compare actual outputs and gradients, including duplicate padding indices."""
    torch.manual_seed(42)
    logits = torch.randn(2, 8, 128, dtype=dtype, device=device)
    teacher_logits, indices = torch.randn(2, 8, 128, device=device).topk(4, dim=-1)
    indices[:, 6:] = 0
    upstream = torch.randn(2, 7, 4, device=device)
    upstream[:, 6:] = 0
    student = logits.clone().requires_grad_(True)
    actual, teacher, entropy = get_distillation_topk_logprobs_from_logits(
        student,
        teacher_logits,
        indices,
        zero_outside_topk=zero_outside_topk,
        calculate_entropy=zero_outside_topk,
        defer_fp32_upcast=defer_fp32_upcast,
    )
    (actual * upstream).sum().backward()

    reference = logits.clone().requires_grad_(True)
    full_logits = reference.float()
    if zero_outside_topk:
        full_logprobs = full_logits.log_softmax(-1)
        expected = full_logprobs.gather(-1, indices)
        expected_entropy = (full_logprobs.exp() * full_logprobs).sum(-1)[:, :-1]
        torch.testing.assert_close(entropy, expected_entropy, rtol=0, atol=0)
    else:
        expected = full_logits.gather(-1, indices).log_softmax(-1)
        assert entropy is None
    expected = expected[:, :-1]
    (expected * upstream).sum().backward()

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(student.grad, reference.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        teacher, teacher_logits.log_softmax(-1)[:, :-1], rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("defer_fp32_upcast", [False, True])
def test_topk_retained_activation_follows_opt_in(
    device: torch.device, dtype: torch.dtype, defer_fp32_upcast: bool
) -> None:
    """Only the opted-in local subset omits the full-vocabulary fp32 copy."""
    student = torch.randn(1, 8, 128, dtype=dtype, device=device, requires_grad=True)
    teacher_logits, indices = torch.randn(1, 8, 128, device=device).topk(4, dim=-1)
    saved: list[tuple[torch.Size, torch.dtype]] = []

    def record(tensor: torch.Tensor) -> torch.Tensor:
        saved.append((tensor.shape, tensor.dtype))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
        result, _, _ = get_distillation_topk_logprobs_from_logits(
            student,
            teacher_logits,
            indices,
            zero_outside_topk=False,
            calculate_entropy=False,
            defer_fp32_upcast=defer_fp32_upcast,
        )

    assert ((student.shape, torch.float32) not in saved) == defer_fp32_upcast
    result.sum().backward()
    assert student.grad is not None


class _PermutationTokenLayout:
    """Non-identity layout with Automodel's shard/gather token interface."""

    def __init__(self, order: torch.Tensor) -> None:
        self.order = order
        self.inverse_order = torch.argsort(order)

    def shard_token_tensor(
        self, tensor: torch.Tensor, *, seq_dim: int, fill: int
    ) -> torch.Tensor:
        del fill
        return tensor.index_select(seq_dim, self.order)

    def gather_token_tensor(
        self, tensor: torch.Tensor, *, seq_dim: int, trim: bool
    ) -> torch.Tensor:
        assert trim
        return tensor.index_select(seq_dim, self.inverse_order)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_automodel_layout_preserves_full_upcast(dtype: torch.dtype) -> None:
    """Automodel CP must retain its old upcast even without a legacy CP group."""
    torch.manual_seed(42)
    order = torch.tensor([0, 3, 6, 1, 4, 7, 2, 5])
    layout = _PermutationTokenLayout(order)
    logits = torch.randn(2, 8, 128, dtype=dtype)
    teacher_logits, indices = torch.randn(2, 8, 128).topk(4, dim=-1)
    indices[:, 6:] = 0
    upstream = torch.randn(2, 7, 4)
    upstream[:, 6:] = 0
    student = logits.index_select(1, order).clone().requires_grad_(True)
    saved: list[tuple[torch.Size, torch.dtype]] = []

    def record(tensor: torch.Tensor) -> torch.Tensor:
        saved.append((tensor.shape, tensor.dtype))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
        actual, teacher, entropy = get_distillation_topk_logprobs_from_logits(
            student,
            teacher_logits,
            indices,
            zero_outside_topk=False,
            calculate_entropy=False,
            cp_sharder=layout,
            defer_fp32_upcast=True,
        )
    (actual * upstream).sum().backward()

    reference = logits.clone().requires_grad_(True)
    expected = reference.float().gather(-1, indices).log_softmax(-1)[:, :-1]
    (expected * upstream).sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        student.grad, reference.grad.index_select(1, order), rtol=0, atol=0
    )
    torch.testing.assert_close(
        teacher, teacher_logits.log_softmax(-1)[:, :-1], rtol=0, atol=0
    )
    assert entropy is None
    assert (student.shape, torch.float32) in saved
