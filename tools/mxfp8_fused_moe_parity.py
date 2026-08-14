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

"""Compare fused and unfused Torch grouped-MoE MXFP8 inference paths."""

import torch

from megatron.core.inference.moe.fused_moe import ActivationType, mcore_fused_moe
from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor


def _stack_mxfp8(weights: torch.Tensor) -> MXFP8Tensor:
    quantized = [MXFP8Tensor.from_bf16(weight, backend="torch") for weight in weights]
    return MXFP8Tensor(
        data=torch.stack([weight.data for weight in quantized]),
        scale=torch.stack([weight.scale for weight in quantized]),
        backend="torch",
    )


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> str:
    diff = (actual.float() - expected.float()).abs()
    relative_l2 = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(
        expected.float()
    )
    return (
        f"mean_abs={diff.mean().item():.8f} "
        f"max_abs={diff.max().item():.8f} "
        f"relative_l2={relative_l2.item():.8f} "
        f"p99_abs={torch.quantile(diff, 0.99).item():.8f}"
    )


@torch.no_grad()
def main() -> None:
    """Run fused, unfused, and BF16 MoE paths on identical inputs."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    torch.manual_seed(1234)
    num_tokens = 64
    hidden_size = 2688
    ffn_hidden_size = 1856
    num_experts = 4
    topk = 2
    hidden_states = torch.randn(
        num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    logits = torch.randn(num_tokens, num_experts, device="cuda")
    all_probs = torch.softmax(logits, dim=-1)
    probs, routing_map = torch.topk(all_probs, topk, dim=-1)
    fc1_bf16 = torch.randn(
        num_experts,
        ffn_hidden_size,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    fc2_bf16 = torch.randn(
        num_experts,
        hidden_size,
        ffn_hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    fc1_mxfp8 = _stack_mxfp8(fc1_bf16)
    fc2_mxfp8 = _stack_mxfp8(fc2_bf16)
    valid_tokens = torch.tensor(num_tokens, device="cuda", dtype=torch.int32)
    common_args = {
        "hidden_states": hidden_states,
        "probs": probs,
        "activation_type": ActivationType.SQUARED_RELU,
        "num_local_experts": num_experts,
        "local_expert_start": 0,
        "valid_tokens": valid_tokens,
        "routing_map": routing_map,
    }

    fused_output = mcore_fused_moe(
        fc1_weight=fc1_mxfp8,
        fc2_weight=fc2_mxfp8,
        disable_fused_quant_kernels=False,
        **common_args,
    )
    unfused_output = mcore_fused_moe(
        fc1_weight=fc1_mxfp8,
        fc2_weight=fc2_mxfp8,
        disable_fused_quant_kernels=True,
        **common_args,
    )
    bf16_output = mcore_fused_moe(
        fc1_weight=fc1_bf16,
        fc2_weight=fc2_bf16,
        disable_fused_quant_kernels=True,
        **common_args,
    )
    print(f"fused_vs_unfused {_metrics(fused_output, unfused_output)}")
    print(f"fused_vs_bf16 {_metrics(fused_output, bf16_output)}")
    print(f"unfused_vs_bf16 {_metrics(unfused_output, bf16_output)}")

    # Keep the first token, its routing decisions, and all weights identical,
    # but execute it once alone and once alongside the rest of the batch. This
    # isolates batch-shape variance in grouped expert execution/accumulation.
    single_token_output = mcore_fused_moe(
        hidden_states=hidden_states[:1],
        probs=probs[:1],
        fc1_weight=fc1_mxfp8,
        fc2_weight=fc2_mxfp8,
        activation_type=ActivationType.SQUARED_RELU,
        num_local_experts=num_experts,
        local_expert_start=0,
        valid_tokens=torch.tensor(1, device="cuda", dtype=torch.int32),
        routing_map=routing_map[:1],
        disable_fused_quant_kernels=False,
    )
    print(
        "same_token_batch1_vs_batch64 "
        f"{_metrics(single_token_output[0], fused_output[0])}"
    )


if __name__ == "__main__":
    main()
