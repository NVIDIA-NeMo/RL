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

from collections.abc import Callable
from typing import Any

import pytest
import torch

from nemo_rl.distributed.fused_linear_logprobs import linear_selected_logprobs
from nemo_rl.distributed.model_utils import ChunkedDistributedLogprob

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

BATCH, SEQ, HIDDEN, VOCAB = 2, 37, 48, 257

LogprobFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@pytest.fixture
def deterministic_bf16_matmul(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep cuBLAS from choosing a bf16 split-K reduction that depends on the chunk's row count."""
    monkeypatch.setattr(
        torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", False
    )


def _unfused_logprobs(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    *,
    chunk_size: int,
    temperature: float,
) -> torch.Tensor:
    """The AutoModel TP=1 training path: full logits, then ChunkedDistributedLogprob."""
    logits = torch.nn.functional.linear(hidden_states, weight).to(torch.float32)
    if temperature != 1.0:
        logits = logits.div_(temperature)
    return ChunkedDistributedLogprob.apply(
        logits, target, 0, weight.shape[0], chunk_size, None, False
    )


def _forward_backward(
    fn: LogprobFn,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_states = hidden_states.detach().clone().requires_grad_()
    weight = weight.detach().clone().requires_grad_()
    logprobs = fn(hidden_states, weight, target)
    logprobs.backward(grad_output)
    assert hidden_states.grad is not None and weight.grad is not None
    return logprobs.detach(), hidden_states.grad, weight.grad


def _inputs(
    num_tokens: int, hidden: int, vocab: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    hidden_states = torch.randn(1, num_tokens, hidden, device="cuda").to(dtype)
    weight = (torch.randn(vocab, hidden, device="cuda") * 0.3).to(dtype)
    target = torch.randint(vocab, (1, num_tokens), device="cuda")
    grad_output = torch.randn(1, num_tokens, device="cuda")
    return hidden_states, weight, target, grad_output


@pytest.fixture
def no_tp_all_reduce(monkeypatch: pytest.MonkeyPatch) -> None:
    """ChunkedDistributedLogprob all-reduces over TP; at TP=1 that is the identity."""
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)


@pytest.mark.usefixtures("deterministic_bf16_matmul", "no_tp_all_reduce")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
@pytest.mark.parametrize("chunk_size", [1, 7, BATCH * SEQ, 1000])
def test_matches_unfused_logits_path(
    dtype: torch.dtype, temperature: float, chunk_size: int
) -> None:
    """Values and both gradients match the unfused path, over ragged and oversized chunks."""
    torch.manual_seed(0)
    hidden_states = torch.randn(BATCH, SEQ, HIDDEN, device="cuda").to(dtype)
    weight = (torch.randn(VOCAB, HIDDEN, device="cuda") * 0.3).to(dtype)
    target = torch.randint(VOCAB, (BATCH, SEQ), device="cuda")
    grad_output = torch.randn(BATCH, SEQ, device="cuda")

    fused = _forward_backward(
        lambda h, w, t: linear_selected_logprobs(
            h, w, t, chunk_size=chunk_size, temperature=temperature
        ),
        hidden_states,
        weight,
        target,
        grad_output,
    )
    unfused = _forward_backward(
        lambda h, w, t: _unfused_logprobs(
            h, w, t, chunk_size=chunk_size, temperature=temperature
        ),
        hidden_states,
        weight,
        target,
        grad_output,
    )

    assert fused[0].dtype == torch.float32 and fused[0].shape == (BATCH, SEQ)
    assert fused[1].dtype == dtype and fused[2].dtype == dtype
    torch.testing.assert_close(fused[0], unfused[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fused[1], unfused[1])
    torch.testing.assert_close(fused[2], unfused[2])


@pytest.mark.usefixtures("deterministic_bf16_matmul", "no_tp_all_reduce")
def test_weight_grad_is_as_accurate_as_one_unfused_matmul() -> None:
    """Summing 256 bf16 chunk partials in float32 loses no accuracy against the fp64 gradient."""
    hidden_states, weight, target, grad_output = _inputs(4096, 64, 512, torch.bfloat16)

    def exact_logprobs(
        h: torch.Tensor, w: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        logits = torch.nn.functional.linear(h, w)
        return logits.log_softmax(-1).gather(-1, t.unsqueeze(-1)).squeeze(-1)

    exact = _forward_backward(
        exact_logprobs,
        hidden_states.double(),
        weight.double(),
        target,
        grad_output.double(),
    )[2]
    fused = _forward_backward(
        lambda h, w, t: linear_selected_logprobs(h, w, t, chunk_size=16),
        hidden_states,
        weight,
        target,
        grad_output,
    )[2]
    unfused = _forward_backward(
        lambda h, w, t: _unfused_logprobs(h, w, t, chunk_size=256, temperature=1.0),
        hidden_states,
        weight,
        target,
        grad_output,
    )[2]

    def relative_error(grad: torch.Tensor) -> float:
        return ((grad.double() - exact).norm() / exact.norm()).item()

    assert relative_error(fused) <= 1.1 * relative_error(unfused)


def test_frozen_weight_and_no_grad() -> None:
    """Skip the weight gradient when frozen, and keep no graph under no_grad."""
    hidden_states, weight, target, grad_output = _inputs(
        BATCH * SEQ, HIDDEN, VOCAB, torch.bfloat16
    )
    hidden_before = hidden_states.clone()
    weight_before = weight.clone()

    with torch.no_grad():
        no_grad_logprobs = linear_selected_logprobs(
            hidden_states, weight, target, chunk_size=7
        )
    assert not no_grad_logprobs.requires_grad

    trainable_hidden = hidden_states.clone().requires_grad_()
    logprobs = linear_selected_logprobs(trainable_hidden, weight, target, chunk_size=7)
    logprobs.backward(grad_output)

    assert trainable_hidden.grad is not None and weight.grad is None
    torch.testing.assert_close(logprobs.detach(), no_grad_logprobs, rtol=0, atol=0)
    torch.testing.assert_close(hidden_states, hidden_before, rtol=0, atol=0)
    torch.testing.assert_close(weight, weight_before, rtol=0, atol=0)


def test_peak_memory_stays_far_below_one_logits_table() -> None:
    """Forward plus backward never allocates anything near ``tokens * vocab`` float32 values."""
    num_tokens, vocab, chunk_size = 8192, 32768, 512
    hidden_states, weight, target, grad_output = _inputs(
        num_tokens, 256, vocab, torch.bfloat16
    )
    hidden_states.requires_grad_()
    weight.requires_grad_()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()

    linear_selected_logprobs(
        hidden_states, weight, target, chunk_size=chunk_size
    ).backward(grad_output)
    torch.cuda.synchronize()

    full_fp32_logits_bytes = num_tokens * vocab * 4
    assert torch.cuda.max_memory_allocated() - baseline < full_fp32_logits_bytes / 4


@pytest.mark.parametrize(
    ("hidden_shape", "weight_shape", "weight_dtype", "kwargs", "error"),
    [
        ((4, 8), (16, 9), torch.float32, {"chunk_size": 2}, ValueError),
        ((4, 8), (16, 8), torch.bfloat16, {"chunk_size": 2}, TypeError),
        ((4, 8), (16, 8), torch.float32, {"chunk_size": 0}, ValueError),
        ((4, 8), (16, 8), torch.float32, {"chunk_size": 2.0}, TypeError),
        (
            (4, 8),
            (16, 8),
            torch.float32,
            {"chunk_size": 2, "temperature": 0.0},
            ValueError,
        ),
    ],
)
def test_rejects_invalid_arguments(
    hidden_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    weight_dtype: torch.dtype,
    kwargs: dict[str, Any],
    error: type[Exception],
) -> None:
    hidden_states = torch.zeros(hidden_shape, device="cuda")
    weight = torch.zeros(weight_shape, device="cuda", dtype=weight_dtype)
    target = torch.zeros(hidden_shape[:-1], device="cuda", dtype=torch.int64)
    with pytest.raises(error):
        linear_selected_logprobs(hidden_states, weight, target, **kwargs)


def test_rejects_misaligned_target() -> None:
    hidden_states = torch.zeros(2, 5, 8, device="cuda")
    weight = torch.zeros(16, 8, device="cuda")
    with pytest.raises(ValueError, match="target must have shape"):
        linear_selected_logprobs(
            hidden_states,
            weight,
            torch.zeros(2, 4, device="cuda", dtype=torch.int64),
            chunk_size=2,
        )
