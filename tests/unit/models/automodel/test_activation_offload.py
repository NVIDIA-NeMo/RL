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

import copy
from contextlib import nullcontext

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)

from nemo_rl.models.automodel import activation_offload
from nemo_rl.models.automodel.activation_offload import (
    MIN_OFFLOAD_BYTES,
    ActivationCpuOffload,
    _pack,
    _unpack,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]

HIDDEN = 512
TOKENS = 4096
NUM_BLOCKS = 4


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(HIDDEN)
        self.up = nn.Linear(HIDDEN, 4 * HIDDEN)
        self.down = nn.Linear(4 * HIDDEN, HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.down(torch.nn.functional.gelu(self.up(self.norm(x))))


def _checkpointed_model() -> nn.Module:
    torch.manual_seed(0)
    blocks = [checkpoint_wrapper(_Block()) for _ in range(NUM_BLOCKS)]
    return nn.Sequential(nn.Linear(HIDDEN, HIDDEN), *blocks).cuda()


def _train(model: nn.Module, inputs: list[torch.Tensor], offload):
    """Run one AdamW step per input; return losses, step-1 grads and retained bytes."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_weights = torch.randn(
        TOKENS,
        HIDDEN,
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(2),
    )
    losses, first_grads, retained = [], None, []
    for x in inputs:
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        with offload if offload is not None else nullcontext():
            out = model(x)
        # Saves only loss_weights, so what stays allocated is what the model saved.
        loss = (out * loss_weights).sum()
        del out
        torch.cuda.synchronize()
        retained.append(torch.cuda.memory_allocated() - before)
        loss.backward()
        if first_grads is None:
            first_grads = [p.grad.clone() for p in model.parameters()]
        losses.append(loss.item())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return losses, first_grads, retained


def test_offload_matches_gpu_training_and_frees_saved_block_inputs():
    reference = _checkpointed_model()
    offloaded = copy.deepcopy(reference)
    generator = torch.Generator(device="cuda").manual_seed(1)
    inputs = [
        torch.randn(TOKENS, HIDDEN, device="cuda", generator=generator)
        for _ in range(3)
    ]

    ref_losses, ref_grads, ref_retained = _train(reference, inputs, None)
    off_losses, off_grads, off_retained = _train(
        offloaded, inputs, ActivationCpuOffload()
    )

    # The host round trip is an exact copy, so training is bitwise identical.
    assert off_losses == ref_losses
    for ref_grad, off_grad in zip(ref_grads, off_grads):
        torch.testing.assert_close(off_grad, ref_grad, rtol=0, atol=0)
    for ref_param, off_param in zip(reference.parameters(), offloaded.parameters()):
        torch.testing.assert_close(off_param, ref_param, rtol=0, atol=0)

    block_input_bytes = TOKENS * HIDDEN * 4
    # Without offload every checkpointed block keeps its input on the GPU; with
    # offload none of them does.
    assert min(ref_retained) >= NUM_BLOCKS * block_input_bytes
    assert max(off_retained) < block_input_bytes


def test_pack_keeps_parameters_their_views_and_small_tensors_on_gpu():
    weight = nn.Parameter(torch.randn(1024, 1024, device="cuda"))
    small = torch.randn(MIN_OFFLOAD_BYTES // 8, device="cuda")
    assert weight.numel() * weight.element_size() >= MIN_OFFLOAD_BYTES

    assert _pack(weight) is weight
    transposed = weight.t()
    assert _pack(transposed) is transposed
    assert _pack(small) is small


def test_pack_moves_activations_to_pinned_host_memory_and_back():
    activation = torch.randn(1024, 1024, device="cuda")[:, :512]

    packed = _pack(activation)

    assert isinstance(packed, tuple)
    device, host = packed
    assert device == activation.device
    assert host.device.type == "cpu" and host.is_pinned()
    restored = _unpack(packed)
    assert restored.device == activation.device
    torch.testing.assert_close(restored, activation, rtol=0, atol=0)


def test_a_tensor_saved_twice_is_copied_to_the_host_once(monkeypatch):
    copies = []
    pack = activation_offload._pack

    def counting_pack(tensor):
        packed = pack(tensor)
        copies.append(packed)
        return packed

    monkeypatch.setattr(activation_offload, "_pack", counting_pack)
    x = torch.randn(1024, 1024, device="cuda", requires_grad=True)

    with ActivationCpuOffload():
        # mul saves both operands, which are the same tensor here.
        loss = (x * x).sum()
    loss.backward()

    assert len(copies) == 1
    torch.testing.assert_close(x.grad, 2 * x.detach(), rtol=0, atol=0)


def test_a_tensor_modified_in_place_after_a_save_gets_a_new_host_copy():
    offload = ActivationCpuOffload()
    activation = torch.randn(1024, 1024, device="cuda")
    before_write = activation.clone()

    with offload:
        first = offload.pack_hook(activation)
        again = offload.pack_hook(activation)
        activation.add_(1)
        after_write = offload.pack_hook(activation)

    assert again is first
    assert after_write is not first
    torch.testing.assert_close(_unpack(first), before_write, rtol=0, atol=0)
    torch.testing.assert_close(_unpack(after_write), activation, rtol=0, atol=0)
