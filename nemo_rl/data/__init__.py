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

from typing import NotRequired, TypedDict


class DataConfig(TypedDict):
    max_input_seq_length: int
    prompt_file: NotRequired[str]
    system_prompt_file: NotRequired[str]
    dataset_name: str
    val_dataset_name: NotRequired[str]
    add_bos: NotRequired[bool]
    add_eos: NotRequired[bool]
    input_key: NotRequired[str]
    output_key: NotRequired[str]
    add_generation_prompt: NotRequired[bool]
    add_system_prompt: NotRequired[bool]
    split: NotRequired[str]
    shuffle: NotRequired[bool]


class MathDataConfig(DataConfig):
    problem_key: str
    solution_key: str
