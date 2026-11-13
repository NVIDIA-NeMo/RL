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

"""Bitwise contract of the zero-KL log-prob replacement, on GPU with real MCore.

Forward: equal to Megatron Inference's raw log-probs over the full row.
Backward: equal to the replaced NeMo-RL function's gradient with the preset off.
"""

import functools
import types

import pytest
import torch

pytestmark = pytest.mark.mcore

# Nemotron Nano v3's padded vocabulary, so the large-row log-softmax kernel runs.
_VOCAB = 131072
_BATCH = 2
_SEQ = 37


def _run_logprob_contract(rank, world_size, *, dtype, chunk_size):
    from megatron.core.inference.contexts.dynamic_context import (
        DynamicInferenceContext,
    )
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        disable_batch_invariant_mode,
        enable_batch_invariant_mode,
    )

    from nemo_rl.distributed import model_utils
    from nemo_rl.models.megatron import zero_train_gen_mismatch as zgm

    group = torch.distributed.new_group(ranks=list(range(world_size)))
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(1234)

    full = torch.randn(_BATCH, _SEQ, _VOCAB, device=device, generator=generator) * 4
    # A large common offset: the gradient must not depend on it.
    full[0, 0] += 2.0**24
    full = full.to(dtype)
    target = torch.randint(
        0, _VOCAB, (_BATCH, _SEQ), device=device, generator=generator
    )
    # Targets in every rank's shard, including both shard edges.
    shard = _VOCAB // world_size
    for r in range(world_size):
        target[0, 1 + 2 * r] = r * shard
        target[0, 2 + 2 * r] = (r + 1) * shard - 1
    grad_output = torch.randn(_BATCH, _SEQ, device=device, generator=generator)
    grad_output[:, -3:] = 0.0  # padding tokens

    start, end = rank * shard, (rank + 1) * shard
    local = full[..., start:end].contiguous()
    if chunk_size is None:
        original = model_utils.DistributedLogprob
        original_args = (start, end, group)
    else:
        original = model_utils.ChunkedDistributedLogprob
        original_args = (start, end, chunk_size, group)

    # Preset off: the replaced function with stock kernels.
    off_logits = local.clone().requires_grad_()
    off_out = original.apply(off_logits, target, *original_args, False)
    off_out.backward(grad_output)

    enable_batch_invariant_mode(backend="te_native", collective="ordered")
    try:
        stub = types.SimpleNamespace(
            config=types.SimpleNamespace(logprobs_mode="raw_logprobs")
        )
        rows = full.reshape(-1, _VOCAB).float()
        generation = (
            DynamicInferenceContext._processed_log_probs(
                stub, rows, rows.shape[0], None, None
            )
            .gather(-1, target.reshape(-1, 1))
            .reshape(_BATCH, _SEQ)
        )

        shim = (
            zgm._DistributedFullVocabLogprob(original)
            if chunk_size is None
            else zgm._ChunkedFullVocabLogprob(original)
        )
        on_logits = local.clone().requires_grad_()
        on_out = shim.apply(on_logits, target, *original_args, False)
        on_out.backward(grad_output)
        inference_out = shim.apply(local, target, *original_args, True)
    finally:
        disable_batch_invariant_mode()

    assert torch.equal(on_out.detach(), generation)
    assert torch.equal(inference_out, generation)
    assert on_logits.grad.dtype == off_logits.grad.dtype == dtype
    assert torch.equal(on_logits.grad, off_logits.grad)


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("chunk_size", [None, 16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=str)
def test_logprob_forward_matches_generation_and_backward_matches_preset_off(
    distributed_test_runner, world_size, chunk_size, dtype
):
    distributed_test_runner(
        functools.partial(_run_logprob_contract, dtype=dtype, chunk_size=chunk_size),
        world_size=world_size,
    )
