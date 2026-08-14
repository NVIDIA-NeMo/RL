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

"""Compare FlashInfer and Torch dense MXFP8 GEMMs on identical operands."""

import torch
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch import Linear, fp8_autocast

from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
from megatron.core.inference.quantization.utils import mm_mxfp8


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> str:
    diff = (actual.float() - expected.float()).abs()
    expected_abs = expected.float().abs()
    relative_l2 = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(
        expected.float()
    )
    return (
        f"mean_abs={diff.mean().item():.8f} "
        f"max_abs={diff.max().item():.8f} "
        f"relative_l2={relative_l2.item():.8f} "
        f"p99_abs={torch.quantile(diff, 0.99).item():.8f} "
        f"expected_mean_abs={expected_abs.mean().item():.8f}"
    )


@torch.no_grad()
def main() -> None:
    """Run representative Nano dense GEMM shapes."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    torch.manual_seed(1234)
    for rows, in_features, out_features in (
        (1, 2688, 512),
        (16, 2688, 512),
        (32, 2688, 512),
        (32, 1856, 512),
    ):
        inputs = torch.randn(
            rows, 1, in_features, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(
            out_features, in_features, device="cuda", dtype=torch.bfloat16
        )
        flashinfer_weight = MXFP8Tensor.from_bf16(weight, backend="flashinfer")
        torch_weight = MXFP8Tensor.from_bf16(weight, backend="torch")
        if not torch.equal(flashinfer_weight.data, torch_weight.data):
            raise RuntimeError("FlashInfer and Torch weight data differ.")
        if not torch.equal(
            flashinfer_weight.scale.view(torch.uint8),
            torch_weight.scale.view(torch.uint8),
        ):
            raise RuntimeError("FlashInfer and Torch weight scales differ.")

        flashinfer_output = mm_mxfp8(inputs, flashinfer_weight)
        torch_output = mm_mxfp8(inputs, torch_weight)
        bf16_output = torch.matmul(inputs.squeeze(1), weight.t()).unsqueeze(1)
        print(
            f"shape=({rows},{in_features},{out_features}) "
            f"flashinfer_vs_torch {_metrics(flashinfer_output, torch_output)}"
        )
        print(
            f"shape=({rows},{in_features},{out_features}) "
            f"flashinfer_vs_bf16 {_metrics(flashinfer_output, bf16_output)}"
        )
        print(
            f"shape=({rows},{in_features},{out_features}) "
            f"torch_vs_bf16 {_metrics(torch_output, bf16_output)}"
        )
        if rows % 32 == 0:
            te_linear = Linear(
                in_features,
                out_features,
                bias=False,
                params_dtype=torch.bfloat16,
                device="cuda",
            )
            te_linear.weight.copy_(weight)
            with fp8_autocast(enabled=True, fp8_recipe=MXFP8BlockScaling()):
                te_output = te_linear(inputs.squeeze(1)).unsqueeze(1)
            print(
                f"shape=({rows},{in_features},{out_features}) "
                f"flashinfer_vs_te {_metrics(flashinfer_output, te_output)}"
            )
            print(
                f"shape=({rows},{in_features},{out_features}) "
                f"te_vs_bf16 {_metrics(te_output, bf16_output)}"
            )


if __name__ == "__main__":
    main()
