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

from enum import Enum, auto

from transformers import AutoConfig


class ModelFlag(Enum):
    """Enum that defines special flags for model-specific behaviors.

    This enum provides a way to identify models that require special handling or
    configuration in different parts of the NeMo RL codebase.

    Flags:
        SKIP_DTENSOR_TIED_WEIGHTS_CHECK: Models that should skip the tied weights check
                                 for the DTensor Policy even without setting the
                                 NRL_SKIP_TIED_WEIGHT_CHECK flag.
        VLLM_LOAD_FORMAT_AUTO: Models that should use the "auto" load format when initializing
                               VLLM.

    Each flag has a `matches` method that determines if the flag applies to a given model_name.
    """

    SKIP_DTENSOR_TIED_WEIGHTS_CHECK = auto()
    VLLM_LOAD_FORMAT_AUTO = auto()

    def matches(self, model_name: str) -> bool:
        match self:
            case ModelFlag.SKIP_DTENSOR_TIED_WEIGHTS_CHECK:
                return is_gemma_model(model_name)
            case ModelFlag.VLLM_LOAD_FORMAT_AUTO:
                return is_gemma_model(model_name)
            case _:
                raise ValueError(f"Unknown ModelFlag: {self}")


def is_gemma_model(model_name: str) -> bool:
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    return hasattr(hf_config, "model_type") and hf_config.model_type in [
        "gemma2",
        "gemma3",
        "gemma3_text",
    ]
