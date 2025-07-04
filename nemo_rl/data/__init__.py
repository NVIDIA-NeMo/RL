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

from typing import Optional, TypedDict


class DatasetConfig(TypedDict):
    shuffle: bool
    seed: int
    jsonl_path: str
    filter_long_samples: bool
    drop_last: bool


class DataConfig(TypedDict):
    train: DatasetConfig
    val: Optional[DatasetConfig]
    max_input_seq_length: int


class MathDataConfig(DataConfig):
    problem_key: str
    solution_key: str
