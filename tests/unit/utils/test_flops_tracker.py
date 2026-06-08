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
