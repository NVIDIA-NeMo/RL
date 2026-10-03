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

"""GPU probe adapted from NVIDIA-NeMo/RL#4167.

Requires CUDA, Transformer Engine, and Automodel NemotronV3Attention.
Skipped on CPU-only boxes; unit coverage lives in test_automodel_packing.py.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch


def _has_te_and_nemotron() -> bool:
    if not torch.cuda.is_available():
        return False
    if importlib.util.find_spec("transformer_engine") is None:
        return False
    try:
        from nemo_automodel.components.models.nemotron_v3.layers import (  # noqa: F401
            NemotronV3Attention,
        )
        from nemo_automodel.components.models.common.utils import BackendConfig  # noqa: F401
        from nemo_rl.models.huggingface.common import get_flash_attention_kwargs  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.automodel
@pytest.mark.skipif(
    not _has_te_and_nemotron(),
    reason="CUDA + Transformer Engine + Automodel NemotronV3Attention required",
)
def test_packing_boundaries_isolate_second_trajectory():
    """Changing trajectory A must not change trajectory B under native packing."""
    from types import SimpleNamespace

    from nemo_automodel.components.models.common.utils import BackendConfig
    from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Attention
    from nemo_rl.models.huggingface.common import get_flash_attention_kwargs

    torch.manual_seed(42)
    cfg = SimpleNamespace(
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        torch_dtype="bfloat16",
        attention_dropout=0.0,
    )
    layer = (
        NemotronV3Attention(cfg, BackendConfig(attn="te", linear="torch")).cuda().eval()
    )
    length = 64
    first = torch.randn(1, length, 512, device="cuda", dtype=torch.bfloat16)
    second = torch.randn_like(first)
    changed_first = torch.randn_like(first)
    flash_kwargs = get_flash_attention_kwargs(
        torch.tensor([length, length], device="cuda")
    )
    bounds = torch.tensor([0, length, 2 * length], device="cuda", dtype=torch.int32)

    with torch.no_grad():
        packed = torch.cat((first, second), dim=1)
        changed = torch.cat((changed_first, second), dim=1)
        # Nested HF FA2 metadata (what NeMo RL used to pass only): expected leak.
        nested_actual = layer(packed, flash_attn_kwargs=flash_kwargs)[:, length:]
        nested_perturbed = layer(changed, flash_attn_kwargs=flash_kwargs)[:, length:]
        # Native top-level cu_seqlens + THD layout (positive control / fixed contract).
        native = layer(packed.squeeze(0), cu_seqlens=bounds, max_seqlen=length)[length:]
        native_changed = layer(
            changed.squeeze(0), cu_seqlens=bounds, max_seqlen=length
        )[length:]

    nested_delta = (nested_actual - nested_perturbed).abs().max().item()
    native_delta = (native - native_changed).abs().max().item()
    assert native_delta == 0.0, "native packing control failed"
    # Nested HF metadata is ignored by TE attention, so packed tokens attend as one
    # sequence (nonzero delta). Native top-level cu_seqlens isolates trajectories.
    print(
        {
            "nested_second_sequence_max_delta": nested_delta,
            "native_second_sequence_max_delta": native_delta,
        }
    )
