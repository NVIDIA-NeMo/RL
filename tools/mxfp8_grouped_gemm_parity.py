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

"""Compare grouped MXFP8 GEMM with equivalent per-expert dense GEMMs."""

import torch

from megatron.core.inference.moe.fused_moe import _mxfp8_grouped_mm
from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
from megatron.core.inference.quantization.utils import mm_mxfp8


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
    """Run Nano-shaped grouped and dense MXFP8 GEMMs on identical operands."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    torch.manual_seed(1234)
    num_experts = 4
    rows_per_expert = 128
    in_features = 2688
    out_features = 1856
    activations = torch.randn(
        num_experts * rows_per_expert,
        in_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weights = torch.randn(
        num_experts,
        out_features,
        in_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    quantized_activations = MXFP8Tensor.from_bf16(activations, backend="torch")
    quantized_weights = [
        MXFP8Tensor.from_bf16(weight, backend="torch") for weight in weights
    ]
    stacked_weight = MXFP8Tensor(
        data=torch.stack([weight.data for weight in quantized_weights]),
        scale=torch.stack([weight.scale for weight in quantized_weights]),
        backend="torch",
    )
    offsets = torch.arange(
        rows_per_expert,
        (num_experts + 1) * rows_per_expert,
        rows_per_expert,
        device="cuda",
        dtype=torch.int32,
    )

    grouped_output = _mxfp8_grouped_mm(
        quantized_activations, stacked_weight, offsets
    )
    dense_outputs = []
    for expert_index, weight in enumerate(quantized_weights):
        start = expert_index * rows_per_expert
        end = start + rows_per_expert
        dense_outputs.append(
            mm_mxfp8(activations[start:end].unsqueeze(1), weight).squeeze(1)
        )
    dense_output = torch.cat(dense_outputs)
    print(f"grouped_vs_dense {_metrics(grouped_output, dense_output)}")


if __name__ == "__main__":
    main()
