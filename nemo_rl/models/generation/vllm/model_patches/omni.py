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

from typing import Any

import torch

_NEMOTRON_OMNI_ARCHITECTURES = {
    "NemotronH_Nano_Omni_Reasoning_V3",
    "NemotronH_Super_Omni_Reasoning_V3",
}


def _is_nemotron_omni(model_runner: Any) -> bool:
    model_config = model_runner.vllm_config.model_config
    architectures = set(getattr(model_config, "architectures", None) or [])
    return not architectures.isdisjoint(_NEMOTRON_OMNI_ARCHITECTURES)


def validate_nemotron_omni_radio_layerscale_refit(
    model_runner: Any, state_dict_info: dict[str, Any]
) -> None:
    """Reject explicit LayerScale state that stock vLLM would ignore.

    This function can run while a colocated vLLM engine is asleep. It must not
    read or mutate model tensors because vLLM's level-1 sleep allocator has
    temporarily released their CUDA storage.
    """
    if not _is_nemotron_omni(model_runner):
        return

    explicit_layerscale = [
        name
        for name in state_dict_info
        if name.startswith("vision_model.radio_model.model.blocks.")
        and name.endswith((".ls1", ".ls2"))
    ]
    if explicit_layerscale:
        raise RuntimeError(
            "Nemotron Omni refit contains explicit RADIO LayerScale tensors, "
            "but the stock vLLM 0.20 loader ignores them. Refusing to replace "
            "checkpoint values with the folded-checkpoint identity behavior."
        )


def initialize_nemotron_omni_radio_layerscale(model_runner: Any) -> int:
    """Set folded RADIO LayerScale parameters to identity while vLLM is awake.

    Nano/Super Omni checkpoints fold RADIO LayerScale into the adjacent
    projection weights and therefore do not export ``ls1``/``ls2``. Stock
    vLLM 0.20 leaves those parameters at dummy-initialized values during
    direct load or refit, corrupting image inference. Initialize the
    vLLM-only parameters once, immediately after engine creation and before
    colocated sleep can release their CUDA storage.
    """
    if not _is_nemotron_omni(model_runner):
        return 0

    model = model_runner.model
    vision_model = getattr(model, "vision_model", None)
    if vision_model is None:
        raise RuntimeError(
            "Nemotron Omni vLLM model has no vision_model during initialization."
        )

    initializer_factor = getattr(vision_model.config, "initializer_factor", 1.0)
    initialized = 0
    with torch.no_grad():
        for name, parameter in vision_model.named_parameters():
            if name.rsplit(".", 1)[-1] in {"ls1", "ls2"}:
                parameter.fill_(initializer_factor)
                initialized += 1

    if initialized == 0:
        raise RuntimeError(
            "Nemotron Omni vLLM model exposes no RADIO ls1/ls2 parameters; "
            "the expected vLLM 0.20 model layout may have changed."
        )
    return initialized
