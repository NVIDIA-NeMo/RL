# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_rl.utils.flops_tracker import get_theoretical_tflops, is_using_tf32


def test_worker_total_flops_aggregation_megatron_path():
    """Verify the aggregation logic for total_flops and theoretical_tflops from Megatron workers.

    The Megatron worker returns gpu_name, model_dtype, total_flops, and num_ranks.
    The elif branch in lm_policy.py computes theoretical_tflops from gpu_name/model_dtype,
    mirroring the FSDP/dtensor pattern.
    """
    import warnings

    world_size = 8
    results = [
        {
            "total_flops": 1.0e15,
            "num_ranks": world_size,
            "gpu_name": "NVIDIA H100 80GB HBM3",
            "model_dtype": torch.bfloat16,
        }
    ]

    aggregated_results: dict = {}
    # Mirrors the elif branch in lm_policy.py
    if results and "total_flops" in results[0]:
        aggregated_results["total_flops"] = results[0]["total_flops"]
        aggregated_results["num_ranks"] = results[0].get("num_ranks", world_size)
        try:
            aggregated_results["theoretical_tflops"] = aggregated_results[
                "num_ranks"
            ] * get_theoretical_tflops(
                results[0]["gpu_name"], results[0]["model_dtype"]
            )
        except Exception as e:
            warnings.warn(f"Error getting theoretical flops: {e}")

    assert aggregated_results["total_flops"] == pytest.approx(1.0e15)
    assert aggregated_results["num_ranks"] == 8
    # 8 GPUs × (1979/2 TFLOPS) for H100 bfloat16
    assert aggregated_results["theoretical_tflops"] == pytest.approx(8 * 1979 / 2)


@pytest.mark.parametrize(
    "device_name, model_dtype, tflops",
    [
        ("NVIDIA A100 80GB PCIe", torch.bfloat16, 624 / 2),
        ("NVIDIA A100 80GB PCIe", torch.float32, 312 / 2 if is_using_tf32() else 19.5),
        ("NVIDIA H100 80GB HBM3", torch.bfloat16, 1979 / 2),
        ("NVIDIA H100 80GB HBM3", torch.float32, 989 / 2 if is_using_tf32() else 67.0),
        ("NVIDIA H200", torch.bfloat16, 1979 / 2),
        ("NVIDIA H200", torch.float32, 989 / 2 if is_using_tf32() else 67.0),
        ("NVIDIA B200", torch.bfloat16, 4500 / 2),
        ("NVIDIA B200", torch.float32, 2200 / 2 if is_using_tf32() else 80.0),
        ("NVIDIA B300", torch.bfloat16, 4500 / 2),
        ("NVIDIA B300", torch.float32, 2200 / 2 if is_using_tf32() else 80.0),
        ("NVIDIA GB200", torch.bfloat16, 4900 / 2),
        ("NVIDIA GB200", torch.float32, 2500 / 2 if is_using_tf32() else 80.0),
        ("NVIDIA GB300", torch.bfloat16, 4900 / 2),
        ("NVIDIA GB300", torch.float32, 2500 / 2 if is_using_tf32() else 80.0),
    ],
)
def test_theoretical_tflops(device_name, model_dtype, tflops):
    assert get_theoretical_tflops(device_name, model_dtype) == pytest.approx(tflops)
