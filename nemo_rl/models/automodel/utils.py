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
from typing import Any, Dict

# Side-effect import: installs the resolver hook that routes FP8-native Mistral 3.5
# configs to Mistral3FP8VLM. Without it, HF's stock FP8Linear path runs and produces
# 0-d weight_scale_inv params that FSDP2 rejects.
import nemo_automodel.components.models.mistral3_vlm  # noqa: F401
from nemo_automodel._transformers.auto_model import (
    NeMoAutoModelForCausalLM,
    NeMoAutoModelForImageTextToText,
    NeMoAutoModelForTextToWaveform,
)
from nemo_automodel.components.moe.megatron.token_dispatcher import (
    MoEFlexTokenDispatcher,
)
from torch import nn

# Add an entry whenever a model's architecture isn't loadable via
# NeMoAutoModelForCausalLM (e.g. VLMs using ForConditionalGeneration /
# ForImageTextToText). Check MODEL_ARCH_MAPPING in the NeMo automodel registry to see
# which architectures have custom impls:
# https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/_transformers/registry.py#L32-L146
AUTOMODEL_FACTORY: Dict[str, Any] = {
    "qwen2_5_vl": NeMoAutoModelForImageTextToText,
    "qwen2_vl": NeMoAutoModelForImageTextToText,
    "qwen2_5_omni": NeMoAutoModelForTextToWaveform,
    "qwen3_5": NeMoAutoModelForImageTextToText,
    "llava": NeMoAutoModelForImageTextToText,
    "internvl": NeMoAutoModelForImageTextToText,
    "gemma3": NeMoAutoModelForImageTextToText,
    "gemma4": NeMoAutoModelForImageTextToText,
    "gemma4_unified": NeMoAutoModelForImageTextToText,
    "smolvlm": NeMoAutoModelForImageTextToText,
    "mistral3": NeMoAutoModelForImageTextToText,
    "llama4": NeMoAutoModelForImageTextToText,
}


def resolve_model_class(model_name: str) -> Any:
    """Resolve the model class for a model type.

    Args:
        model_name: Model type to resolve.
    """
    return AUTOMODEL_FACTORY.get(model_name.lower(), NeMoAutoModelForCausalLM)


# Per-dispatch tensors that Automodel's flex token-dispatch managers (DeepEP,
# HybridEP, UCCL-EP) keep as attributes. Each dispatch reassigns them, so none
# is needed across steps.
# TODO(https://github.com/NVIDIA-NeMo/Automodel/issues/4401): drop this workaround
# once Automodel releases this state itself.
_MOE_DISPATCH_STATE_ATTRS = (
    "token_probs",
    "token_indices",
    "routing_map",
    "handle",
    "tokens_per_expert",
    "num_permuted_tokens",
    "dispatched_indices",
    "dispatched_probs",
    "dispatched_routing_map",
    "reversed_mapping_for_combine",
)


def release_moe_dispatcher_state(model: nn.Module) -> int:
    """Drop the per-dispatch tensors cached on the model's MoE dispatch managers.

    The managers keep the last dispatch's routing tensors as attributes;
    ``combine`` clears only ``handle`` (and HybridEP's ``num_permuted_tokens``).
    After a backward with activation checkpointing, a manager's last dispatch
    is a recompute, so ``token_probs`` and ``dispatched_probs`` carry a
    ``grad_fn`` that keeps that recompute graph and its saved activations
    alive. With the default manager shared by all layers that is the first MoE
    layer (about 1 GB for Qwen3.5-35B-A3B at 9216 tokens per micro-batch); with
    per-layer managers every layer keeps one. The live blocks pin cached
    allocator segments, so ``torch.cuda.empty_cache()`` cannot return them
    before colocated generation wakes up.

    Args:
        model: Model whose ``token_dispatcher`` attributes are scanned. A
            manager shared across layers is cleared once.

    Returns:
        The number of attributes set to ``None``.
    """
    managers = {}
    for module in model.modules():
        # MoEFlexTokenDispatcher is a plain object held by the experts module,
        # not a submodule, so model.modules() does not yield it.
        dispatcher = getattr(module, "token_dispatcher", None)
        if isinstance(dispatcher, MoEFlexTokenDispatcher):
            manager = getattr(dispatcher, "_comm_manager", None)
            if manager is not None:
                managers[id(manager)] = manager

    cleared = 0
    for manager in managers.values():
        for attr in _MOE_DISPATCH_STATE_ATTRS:
            if getattr(manager, attr, None) is not None:
                setattr(manager, attr, None)
                cleared += 1
    return cleared
