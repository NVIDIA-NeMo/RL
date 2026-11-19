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

"""Native MXFP4/MXFP8 refit for DeepSeek V4.1: trainer matrices to checkpoint-format tensors.

vLLM's own reload lifecycle (``start_reload`` / ``finish_reload``) restores the
checkpoint layout before the stream and rebuilds the kernel layout after it, so
this module only converts what the trainer sends into what the checkpoint holds.
"""

from collections.abc import Iterable, Iterator

import torch
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

from nemo_rl.models.generation.vllm.quantization.deepseek_v41_codec import (
    quantize_native_weight,
)


def is_native_model(model: torch.nn.Module) -> bool:
    """Detect native Flash weights without enabling legacy NRL FP8 patches."""
    config = getattr(model, "config", None)
    language_model = getattr(model, "language_model", model)
    quant = getattr(getattr(language_model, "model", None), "quant_config", None)
    return (
        getattr(config, "model_type", None) in ("deepseek_v41", "deepseek_v41_text")
        and quant is not None
        and quant.get_name() == "deepseek_v4_fp8"
        and quant.expert_dtype == "fp4"
        and quant.weight_block_size == [32, 32]
    )


def get_quantized_weight_iterator(
    weights: Iterable[tuple[str, torch.Tensor]], model: torch.nn.Module
) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream trainer matrices as native checkpoint values/scales.

    Args:
        weights: HF names and complete floating matrices [out, in], before TP/EP
            slicing. Non-quantized parameters retain their original layout.
        model: Constructed vLLM model, whose quant methods own backend dispatch.

    Yields:
        Weight then scale for each quantized matrix. Dense scales are [out/32,
        in/32]; expert scales [out, in/32]. No model-sized converted list is held.
    """
    from nemo_rl.models.generation.vllm.quantization.fp8 import (
        get_module_from_param_name,
    )

    for name, weight in weights:
        if not name.endswith(".weight") or ".engram.embed." in name:
            yield name, weight
            continue
        module = get_module_from_param_name(model, name)
        if isinstance(module, RoutedExperts):
            if not isinstance(module.quant_method, Mxfp4MoEMethod):
                raise ValueError(f"Native refit requires MXFP4 experts: {name}")
            fp4 = True
        elif (
            getattr(getattr(module, "weight", None), "dtype", None)
            == torch.float8_e4m3fn
        ):
            fp4 = False
        else:
            yield name, weight
            continue
        value, scale = quantize_native_weight(weight, fp4=fp4)
        yield name, value
        # vLLM's HF mapper dispatches .scale to the right linear/expert suffix.
        # E8M0 view is essential: uint8 -> float conversion corrupts exponents.
        yield name.removesuffix("weight") + "scale", scale.view(torch.float8_e8m0fnu)
