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

"""Token-alignment requirements shared by Megatron data paths."""

from collections.abc import Mapping
from typing import Any


def get_fp8_token_alignment(megatron_cfg: Mapping[str, Any]) -> int:
    """Return the token-dimension alignment required by FP8 or FP4."""
    fp8_cfg = megatron_cfg.get("fp8_cfg") or {}
    fp4_cfg = megatron_cfg.get("fp4_cfg") or {}
    use_fp8 = fp8_cfg.get("enabled", False)
    use_fp4 = (
        fp4_cfg.get("enabled", False) if isinstance(fp4_cfg, dict) else fp4_cfg.enabled
    )
    if use_fp8 and use_fp4:
        raise ValueError(
            "megatron_cfg.fp8_cfg and fp4_cfg cannot both have enabled: true "
            "(Megatron does not allow fp8 and fp4 together)."
        )
    if use_fp4:
        return 128
    if not use_fp8:
        return 1
    if fp8_cfg["fp8_recipe"] == "blockwise":
        return 128
    if fp8_cfg["fp8_recipe"] == "mxfp8":
        return 32
    return 16


def get_parallel_token_alignment(megatron_cfg: Mapping[str, Any]) -> int:
    """Return the token alignment required by context and sequence parallelism."""
    cp_size = megatron_cfg["context_parallel_size"]
    tp_size = megatron_cfg["tensor_model_parallel_size"]
    return (2 * cp_size if cp_size > 1 else 1) * (
        tp_size if tp_size > 1 and megatron_cfg["sequence_parallel"] else 1
    )
