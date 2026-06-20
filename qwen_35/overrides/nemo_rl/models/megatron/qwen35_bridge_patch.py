# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Register Qwen 3.5 MoE with the existing Qwen 3 MoE Megatron bridge.

This overlay is intentionally narrow: Transformers 5.12 recognizes
Qwen/Qwen3.5-397B-A17B as Qwen3_5MoeForConditionalGeneration, while the
container's Megatron-Bridge only registers Qwen3MoeForCausalLM. The HF weight
layout is expected to match the text Qwen3 MoE bridge closely enough for a smoke
test; if it does not, the later mapping/import checks will fail explicitly.
"""

from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge
from transformers import Qwen3_5MoeForConditionalGeneration


@MegatronModelBridge.register_bridge(
    source=Qwen3_5MoeForConditionalGeneration,
    target=GPTModel,
    model_type="qwen3_5_moe",
)
class Qwen35MoEBridge(Qwen3MoEBridge):
    """Bridge alias for text-only Qwen 3.5 MoE models."""
