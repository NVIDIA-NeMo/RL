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

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "BaseSFTTaskEncoder": "nemo_rl.data.energon.multimodal.task_encoders.base",
    "GenericSFTTaskEncoder": (
        "nemo_rl.data.energon.multimodal.task_encoders.generic_sft"
    ),
    "HFMultimodalSFTProcessorAdapter": (
        "nemo_rl.data.energon.multimodal.task_encoders.generic_sft"
    ),
    "NemotronMultiModalProcessorAdapter": (
        "nemo_rl.data.energon.multimodal.task_encoders.nemotron_multimodal"
    ),
    "NemotronMultiModalTaskEncoder": (
        "nemo_rl.data.energon.multimodal.task_encoders.nemotron_multimodal"
    ),
    "SFTProcessorAdapter": (
        "nemo_rl.data.energon.multimodal.task_encoders.generic_sft"
    ),
    "build_processor_adapter": (
        "nemo_rl.data.energon.multimodal.task_encoders.generic_sft"
    ),
}


def __getattr__(name: str) -> Any:
    """Load optional task encoders only when a caller requests them."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = list(_EXPORT_MODULES)
