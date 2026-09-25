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

# Add an entry whenever a model's architecture isn't loadable via
# NeMoAutoModelForCausalLM (e.g. VLMs using ForConditionalGeneration /
# ForImageTextToText). Check MODEL_ARCH_MAPPING in the NeMo automodel registry to see
# which architectures have custom impls:
# https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/_transformers/registry.py#L32-L146
AUTOMODEL_FACTORY: Dict[str, Any] = {
    # NeMo wrappers — keep in sync with the vanilla HF dict above.
    # See comment above for when to add entries.
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
