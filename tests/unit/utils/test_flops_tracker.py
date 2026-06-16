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

from nemo_rl.models.policy.lm_policy import _aggregate_megatron_flops_metrics
from nemo_rl.utils.flops_formulas import FLOPSConfig, qwen3
from nemo_rl.utils.flops_tracker import get_theoretical_tflops, is_using_tf32


def _qwen3_flops_config(head_dim):
    # Qwen3-235B-A22B-like shape (smaller layer count for a cheap test).
    return FLOPSConfig(
        gbs=1,
        enc_seq_len=4096,
        hs=4096,
        layers=2,
        attention_heads=64,
        query_groups=8,
        head_dim=head_dim,
        moe_ffn_hidden_size=1536,
        moe_router_topk=8,
        vocab_size=151936,
    )


def test_qwen3_flops_head_dim_backward_compat():
    """head_dim=None falls back to hidden_size // num_heads, matching the old formula."""
    assert qwen3(_qwen3_flops_config(None)) == qwen3(_qwen3_flops_config(4096 // 64))


def test_qwen3_flops_wide_attention():
    """Wide attention (num_heads*head_dim > hidden) must count MORE attention FLOPs.

    Qwen3-235B-A22B has head_dim=128, num_heads=64, hidden=4096, so num_heads*head_dim=8192=2*hidden.
    The QKV/output projections and the O(seq^2) scores scale with num_heads*head_dim, not hidden_size,
    so the formula must not collapse head_dim to hidden_size/num_heads.
    """
    standard = qwen3(_qwen3_flops_config(4096 // 64))  # head_dim=64 == hidden/num_heads
    wide = qwen3(_qwen3_flops_config(128))  # head_dim=128 (Qwen3-235B)
    assert wide > standard


def test_worker_total_flops_aggregation_megatron_path():
    """Verify _aggregate_megatron_flops_metrics for the basic case (no train_elapsed_seconds)."""
    world_size = 8
    results = [
        {
            "total_flops": 1.0e15,
            "num_ranks": world_size,
            "gpu_name": "NVIDIA H100 80GB HBM3",
            "model_dtype": torch.bfloat16,
        }
    ]

    aggregated_results = _aggregate_megatron_flops_metrics(results, world_size)

    assert aggregated_results["total_flops"] == pytest.approx(1.0e15)
    assert aggregated_results["num_ranks"] == 8
    assert "train_elapsed_seconds" not in aggregated_results
    # 8 GPUs × (1979/2 TFLOPS) for H100 bfloat16
    assert aggregated_results["theoretical_tflops"] == pytest.approx(8 * 1979 / 2)


def test_worker_total_flops_aggregation_megatron_path_with_elapsed():
    """Verify train_elapsed_seconds is forwarded when present in worker results."""
    world_size = 4
    results = [
        {
            "total_flops": 2.0e15,
            "num_ranks": world_size,
            "gpu_name": "NVIDIA H100 80GB HBM3",
            "model_dtype": torch.bfloat16,
            "train_elapsed_seconds": 3.5,
        }
    ]

    aggregated_results = _aggregate_megatron_flops_metrics(results, world_size)

    assert aggregated_results["total_flops"] == pytest.approx(2.0e15)
    assert aggregated_results["num_ranks"] == 4
    assert aggregated_results["train_elapsed_seconds"] == pytest.approx(3.5)
    assert aggregated_results["theoretical_tflops"] == pytest.approx(4 * 1979 / 2)


def test_worker_total_flops_aggregation_unknown_gpu_warns():
    """Verify a warning is emitted and theoretical_tflops is absent for unknown GPUs."""
    world_size = 2
    results = [
        {
            "total_flops": 1.0e14,
            "num_ranks": world_size,
            "gpu_name": "NVIDIA UNKNOWN GPU XYZ",
            "model_dtype": torch.bfloat16,
        }
    ]

    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        aggregated_results = _aggregate_megatron_flops_metrics(results, world_size)

    assert aggregated_results["total_flops"] == pytest.approx(1.0e14)
    assert aggregated_results["num_ranks"] == 2
    assert "theoretical_tflops" not in aggregated_results
    assert len(w) == 1
    assert "theoretical flops" in str(w[0].message).lower()


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
