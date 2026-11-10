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

"""Chunked logprob backward preserves gradients without dense vocabulary masks."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import TypeVar

import pytest
import torch

from nemo_rl.distributed.model_utils import (
    ChunkedDistributedLogprob,
    _compute_distributed_log_softmax,
)


@pytest.fixture
def local_collectives(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed.nn.functional, "all_reduce", lambda tensor, **kwargs: tensor)


def _old_backward(ctx: SimpleNamespace, grad_output: torch.Tensor) -> torch.Tensor:
    logits, target_mask, masked_target = ctx.saved_tensors
    grad = torch.zeros_like(logits)
    for start in range(0, logits.shape[1], ctx.chunk_size):
        end = min(start + ctx.chunk_size, logits.shape[1])
        chunk = logits[:, start:end].float()
        softmax = _compute_distributed_log_softmax(chunk, group=None).exp()
        chosen = (~target_mask[:, start:end]).unsqueeze(-1) * torch.nn.functional.one_hot(
            masked_target[:, start:end], num_classes=logits.shape[-1]
        )
        chunk_grad = chosen.float().sub_(softmax)
        chunk_grad.mul_(grad_output[:, start:end].unsqueeze(-1))
        grad[:, start:end].copy_(chunk_grad)
        del chunk, softmax, chosen, chunk_grad
    return grad


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("offset", [0, 20])
@pytest.mark.parametrize("underflow", [False, True])
def test_chunked_backward_is_bitwise_equal_to_dense_mask(
    local_collectives: None, monkeypatch: pytest.MonkeyPatch,
    device: str, dtype: torch.dtype, reuse: bool, offset: int, underflow: bool,
) -> None:
    torch.manual_seed(44)
    leaf = torch.randn(2, 5, 17, device=device, dtype=dtype)
    if underflow:
        leaf[:, ::2, 0] = -1000
        leaf[:, ::2, 1] = 1000
    leaf.requires_grad_()
    logits = leaf * 1 if reuse else leaf
    target = torch.randint(offset, offset + 17, (2, 5), device=device)
    target[0, 0], target[1, -1] = offset - 1, offset + 17
    mask = (target < offset) | (target >= offset + 17)
    masked = (target - offset).masked_fill(mask, 0)
    upstream = torch.tensor([[0.0, -0.0, 1.0, -0.75, 2.3], [-0.0, 0.0, -1.0, 0.125, -3.0]], device=device)
    ctx = SimpleNamespace(saved_tensors=(logits.detach().clone(), mask, masked), chunk_size=2)
    expected = _old_backward(ctx, upstream)

    def forbid_one_hot(*args: object, **kwargs: object) -> None:
        raise AssertionError("backward must not allocate a dense vocabulary mask")

    monkeypatch.setattr(torch.nn.functional, "one_hot", forbid_one_hot)
    output = ChunkedDistributedLogprob.apply(logits, target, offset, offset + 17, 2, None, False, reuse)
    actual, = torch.autograd.grad(output, leaf, grad_outputs=upstream)
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    assert torch.equal(actual.view(bits), expected.view(bits))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_chunked_backward_reduces_real_peak_memory(local_collectives: None) -> None:
    logits = torch.randn(1, 512, 131072, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(0, 131072, (1, 512), device="cuda")
    mask = torch.zeros_like(target, dtype=torch.bool)
    ctx = SimpleNamespace(saved_tensors=(logits, mask, target), chunk_size=512, tp_group=None, reuse_logits_storage=False)
    upstream = torch.randn(1, 512, device="cuda")

    T = TypeVar("T")

    def measure(fn: Callable[[SimpleNamespace, torch.Tensor], T]) -> tuple[T, int]:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = fn(ctx, upstream)
        torch.cuda.synchronize()
        return result, torch.cuda.max_memory_allocated() - baseline

    old, old_peak = measure(_old_backward)
    new_result, new_peak = measure(ChunkedDistributedLogprob.backward)
    new = new_result[0]
    assert torch.equal(old.view(torch.int16), new.view(torch.int16))
    assert new_peak < old_peak * 0.8, (new_peak, old_peak)
    print(f"chunked backward peak bytes: old={old_peak} new={new_peak}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_chunked_backward_graph_observes_changed_logits(local_collectives: None) -> None:
    logits = torch.randn(1, 4, 17, device="cuda")
    target = torch.tensor([[0, 2, 4, 6]], device="cuda")
    ctx = SimpleNamespace(saved_tensors=(logits, torch.zeros_like(target, dtype=torch.bool), target), chunk_size=3, tp_group=None, reuse_logits_storage=False)
    upstream = torch.tensor([[0.0, -0.0, -0.75, 2.3]], device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            ChunkedDistributedLogprob.backward(ctx, upstream)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = ChunkedDistributedLogprob.backward(ctx, upstream)[0]
    logits.add_(torch.randn_like(logits))
    graph.replay()
    torch.cuda.synchronize()
    expected = _old_backward(ctx, upstream)
    assert torch.equal(result.view(torch.int32), expected.view(torch.int32))
