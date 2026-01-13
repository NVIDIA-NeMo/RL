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


def is_lora_weight_name(name: str) -> bool:
    """Return True if a parameter name corresponds to a LoRA weight."""
    return (
        name.endswith(".lora_A.weight")
        or name.endswith(".lora_B.weight")
        or name.endswith(".lora_scaling.weight")
    )


def is_base_model_weight_name(name: str) -> bool:
    """Return True if a parameter name corresponds to a base (non-LoRA) weight."""
    return not is_lora_weight_name(name)
