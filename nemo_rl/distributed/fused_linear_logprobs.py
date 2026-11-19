# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
# Adapted from https://github.com/volcengine/verl/blob/e42d6af6e85dd37c907af5ea99326355c376bd97/verl/utils/experimental/torch_functional.py

"""Selected-token log-probabilities computed directly from LM-head inputs.

The output projection, the full-vocabulary normalization and the backward pass
all run over chunks of tokens, so no ``[tokens, vocab]`` tensor larger than one
chunk is allocated. Backward recomputes each chunk's logits from the saved
LM-head input and weight.

Each chunk follows the arithmetic of the unfused logits path: the LM-head
matmul runs in the input dtype, its output is cast to float32 and divided by
the sampling temperature, and the log-softmax matches
:class:`~nemo_rl.distributed.model_utils.ChunkedDistributedLogprob`. The
gradient of those logits is cast back to the input dtype before the matmul
backward. The LM-head weight gradient is accumulated across chunks in float32
and cast to the weight dtype once.
"""

import math
from typing import Any, Optional

import torch


class ChunkedLinearSelectedLogprobs(torch.autograd.Function):
    """``log_softmax(hidden @ weight.T / temperature)`` gathered at ``target``.

    Shapes: ``hidden_states`` is ``[N, H]``, ``weight`` is ``[V, H]`` and
    ``target`` is ``[N]`` (int64, values in ``[0, V)``). Returns ``[N]`` float32.
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]  torch.autograd.Function.forward is always more specific than the base class
        ctx: Any,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        chunk_size: int,
        temperature: float,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        log_probs = torch.empty(
            num_tokens, device=hidden_states.device, dtype=torch.float32
        )
        logits_max = torch.empty_like(log_probs)
        sum_exp_logits = torch.empty_like(log_probs)

        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)
            logits = _chunk_logits(hidden_states[start:end], weight, temperature)
            chunk_max = logits.amax(dim=-1, keepdim=True)
            logits.sub_(chunk_max)
            selected = logits.gather(-1, target[start:end].unsqueeze(-1)).squeeze(-1)
            chunk_sum_exp = logits.exp_().sum(dim=-1)
            log_probs[start:end] = selected - chunk_sum_exp.log()
            logits_max[start:end] = chunk_max.squeeze(-1)
            sum_exp_logits[start:end] = chunk_sum_exp
            del logits, chunk_max, selected, chunk_sum_exp

        ctx.save_for_backward(hidden_states, weight, target, logits_max, sum_exp_logits)
        # pyrefly: ignore[implicitly-defined-attribute]
        ctx.chunk_size = chunk_size
        # pyrefly: ignore[implicitly-defined-attribute]
        ctx.temperature = temperature
        return log_probs

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], None, None, None]:
        grad_output = grad_outputs[0]
        hidden_states, weight, target, logits_max, sum_exp_logits = ctx.saved_tensors
        chunk_size: int = ctx.chunk_size
        temperature: float = ctx.temperature
        needs_hidden_grad, needs_weight_grad = ctx.needs_input_grad[:2]

        grad_hidden = torch.empty_like(hidden_states) if needs_hidden_grad else None
        grad_weight_fp32 = (
            torch.zeros(weight.shape, device=weight.device, dtype=torch.float32)
            if needs_weight_grad
            else None
        )
        num_tokens = hidden_states.shape[0]

        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)
            hidden_chunk = hidden_states[start:end]
            # d(log p_target)/d(logits) = one_hot(target) - softmax(logits),
            # computed in place in the recomputed float32 logits buffer.
            grad_logits = _chunk_logits(hidden_chunk, weight, temperature)
            grad_logits.sub_(logits_max[start:end].unsqueeze(-1)).exp_()
            grad_logits.div_(sum_exp_logits[start:end].unsqueeze(-1)).neg_()
            grad_logits.scatter_add_(
                -1,
                target[start:end].unsqueeze(-1),
                torch.ones(
                    (end - start, 1), device=grad_logits.device, dtype=torch.float32
                ),
            )
            grad_logits.mul_(grad_output[start:end].unsqueeze(-1))
            if temperature != 1.0:
                grad_logits.div_(temperature)
            # The unfused path casts this gradient back to the LM-head output
            # dtype before the matmul backward; do the same here.
            grad_logits = grad_logits.to(hidden_states.dtype)

            if grad_hidden is not None:
                torch.matmul(grad_logits, weight, out=grad_hidden[start:end])
            if grad_weight_fp32 is not None:
                _accumulate_fp32(grad_weight_fp32, grad_logits.t(), hidden_chunk)
            del grad_logits

        grad_weight = None
        if grad_weight_fp32 is not None:
            grad_weight = grad_weight_fp32.to(weight.dtype)
            del grad_weight_fp32
        return grad_hidden, grad_weight, None, None, None


def _chunk_logits(
    hidden_chunk: torch.Tensor, weight: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Float32, temperature-scaled logits for one chunk, in a buffer the caller owns."""
    logits = torch.matmul(hidden_chunk, weight.t())
    # The matmul result is a fresh tensor, so a no-op cast for float32 inputs
    # still leaves the caller free to modify it in place.
    logits = logits.to(torch.float32)
    if temperature != 1.0:
        logits.div_(temperature)
    return logits


def _accumulate_fp32(
    accumulator: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor
) -> None:
    """``accumulator += mat1 @ mat2`` with a float32 accumulator and output.

    Half-precision operands use ``addmm(out_dtype=float32)``, which PyTorch
    supports only on CUDA.
    """
    if mat1.dtype == torch.float32:
        accumulator.addmm_(mat1, mat2)
    else:
        # pyrefly: ignore[no-matching-overload]  The torch stubs omit out_dtype.
        torch.addmm(accumulator, mat1, mat2, out_dtype=torch.float32, out=accumulator)


def linear_selected_logprobs(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    *,
    chunk_size: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Log-probabilities of ``target`` under ``softmax(hidden_states @ weight.T / temperature)``.

    The full-vocabulary normalization is exact; chunking only bounds the
    working set to ``chunk_size`` tokens at a time, in both forward and
    backward. Gradients flow to ``hidden_states`` and ``weight``.

    Args:
        hidden_states: LM-head input with shape ``[..., H]``.
        weight: LM-head weight with shape ``[V, H]`` and the same dtype as
            ``hidden_states``. A bias is not supported.
        target: Token ids with shape ``hidden_states.shape[:-1]``, aligned with
            ``hidden_states`` (row ``i`` scores ``target[i]``). Values must lie
            in ``[0, V)``; they are not range-checked here because doing so
            would synchronize with the device.
        chunk_size: Number of tokens per chunk. The largest temporary buffers
            are about ``chunk_size * V`` float32 values.
        temperature: Sampling temperature dividing the float32 logits.

    Returns:
        Float32 log-probabilities with shape ``target.shape``.
    """
    if hidden_states.ndim < 2:
        raise ValueError(
            f"hidden_states must have shape [..., H]; got {tuple(hidden_states.shape)}."
        )
    if weight.ndim != 2 or weight.shape[1] != hidden_states.shape[-1]:
        raise ValueError(
            f"weight must have shape [V, {hidden_states.shape[-1]}]; got "
            f"{tuple(weight.shape)}."
        )
    if hidden_states.dtype != weight.dtype:
        raise TypeError(
            "hidden_states and weight must share a dtype; got "
            f"{hidden_states.dtype} and {weight.dtype}."
        )
    if target.shape != hidden_states.shape[:-1]:
        raise ValueError(
            f"target must have shape {tuple(hidden_states.shape[:-1])}; got "
            f"{tuple(target.shape)}."
        )
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError(f"chunk_size must be an int; got {type(chunk_size)}.")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive; got {chunk_size}.")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be finite and > 0; got {temperature}.")

    output_shape = target.shape
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_target = target.reshape(-1).to(torch.int64)
    log_probs: torch.Tensor = ChunkedLinearSelectedLogprobs.apply(  # type: ignore[assignment]
        flat_hidden, weight, flat_target, chunk_size, temperature
    )
    return log_probs.reshape(output_shape)
