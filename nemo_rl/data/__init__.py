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


# TODO: split this typed dict up so it can be PreferenceDataConfig | ResponseDataConfig | etc
#       so that we can type check the configs more rigorously as opposed to saying everything
#       is not required.
class DataConfig(TypedDict):
    max_input_seq_length: int
    prompt_file: NotRequired[str | None]
    system_prompt_file: NotRequired[str | None]
    dataset_name: str
    val_dataset_name: NotRequired[str]
    add_bos: NotRequired[bool]
    add_eos: NotRequired[bool]
    input_key: NotRequired[str]
    output_key: NotRequired[str | None]
    add_generation_prompt: NotRequired[bool]
    add_system_prompt: NotRequired[bool]
    split: NotRequired[str | None]
    shuffle: NotRequired[bool]
    seed: NotRequired[int | None]
    download_dir: NotRequired[str]
    train_data_path: NotRequired[str]
    val_data_paths: NotRequired[dict[str, str]]
    # Number of data loader workers.
    # Set to 8 or 10 for large batches to improve loading speed.
    # This saturates CPU threads without consuming too much memory
    # However, setting it too high might cause memory issues for long seqlens.
    num_workers: NotRequired[int]


# TODO: split this typed dict up so it can be MMLUConfig | AIMEConfig | etc
#       so that we can type check the configs more rigorously as opposed to saying everything
#       is not required.
class MathDataConfig(DataConfig):
    problem_key: NotRequired[str]
    solution_key: NotRequired[str]
