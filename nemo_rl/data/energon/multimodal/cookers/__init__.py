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
    "GRANARY_ENGLISH_PROMPT": ("nemo_rl.data.energon.multimodal.cookers.nemotron"),
    "cook_audio_conversation_jsonl": (
        "nemo_rl.data.energon.multimodal.cookers.nemotron"
    ),
    "cook_conversation": "nemo_rl.data.energon.multimodal.cookers.generic",
    "cook_general_conversations_jsonl": (
        "nemo_rl.data.energon.multimodal.cookers.nemotron"
    ),
    "cook_general_conversations_jsonl_explicit_loss_v1": (
        "nemo_rl.data.energon.multimodal.cookers.nemotron"
    ),
    "cook_general_conversations_webdataset": (
        "nemo_rl.data.energon.multimodal.cookers.nemotron"
    ),
    "cook_granary_english_jsonl": ("nemo_rl.data.energon.multimodal.cookers.nemotron"),
    "cook_granary_english_webdataset": (
        "nemo_rl.data.energon.multimodal.cookers.nemotron"
    ),
    "cook_nemotron_conversation": (
        "nemo_rl.data.energon.multimodal.cookers.nemotron"
    ),
    "cook_nano_openai_messages_jsonl": (
        "nemo_rl.data.energon.multimodal.cookers.nemotron"
    ),
    "cook_nano_openai_messages_offline_packed_jsonl": (
        "nemo_rl.data.energon.multimodal.cookers.nemotron"
    ),
    "cook_omcat_legacy_conversation_monolithic": (
        "nemo_rl.data.energon.multimodal.cookers.nemotron"
    ),
}


def __getattr__(name: str) -> Any:
    """Load optional cooker modules only when a caller requests them."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = list(_EXPORT_MODULES)
