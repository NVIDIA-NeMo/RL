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
import torch


def calculate_pass_rate_per_prompt(
    prompts: torch.Tensor, is_correct: torch.Tensor
) -> float:
    """Function to compute fraction of prompts that have at least one correct answer (reward > 0).

    prompts:    tensor (b, s)     Tensor of prompts the model used. May be on any device
    is_correct: tensor (b,)       bool-valued label. May be on any device

    Returns:
    pass rate: float
    """
    unique_prompts = torch.unique(prompts, dim=0)

    correct_prompt_ct = 0
    for i in range(len(unique_prompts)):
        is_matching_prompt = (prompts == unique_prompts[i]).all(1)
        if torch.any(is_correct[is_matching_prompt] > 0):
            correct_prompt_ct += 1

    return correct_prompt_ct / len(unique_prompts)
