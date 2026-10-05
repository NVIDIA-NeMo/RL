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

"""Shared parsing for the ``fp32_lm_head`` setting.

The same setting appears on the Megatron trainer (``policy.megatron_cfg``), the
vLLM engine (``policy.generation.vllm_cfg``), and every non-colocated MOPD
teacher (``on_policy_distillation.non_colocated_teachers``). Kept dependency-free
so each of those config layers can import it without import cycles.
"""

from typing import Literal, TypeAlias

# ``"tf32"`` is accepted as an alias of ``true`` so configs written for branches
# whose fp32 head upcasts to fp32 and runs on TF32 tensor cores keep working.
# NeMo-RL's Megatron head never needs TF32: it runs Transformer Engine's GEMM on
# the bf16 operands with fp32 output, which is at least as accurate as TF32 at
# bf16-GEMM speed.
Fp32LmHeadSetting: TypeAlias = bool | Literal["tf32"]

FP32_LM_HEAD_TF32_ALIAS = "tf32"


def fp32_lm_head_enabled(value: object, *, key: str = "fp32_lm_head") -> bool:
    """Return whether an ``fp32_lm_head`` setting enables the fp32 LM head.

    Args:
        value: The configured value. ``None`` (unset) and ``False`` disable it;
            ``True`` and ``"tf32"`` enable it.
        key: Dotted config path named in the error message.

    Returns:
        True when the fp32 LM head is enabled.

    Raises:
        ValueError: If ``value`` is anything else (for example ``"fp16"``).
    """
    if isinstance(value, str):
        if value == FP32_LM_HEAD_TF32_ALIAS:
            return True
    elif value is None or value is False or value is True:
        return bool(value)
    raise ValueError(
        f'{key} must be true, false, or "{FP32_LM_HEAD_TF32_ALIAS}" (an alias of '
        f"true); got {value!r}."
    )
