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

import os
import sys

from datasets import Dataset

abspath = os.path.abspath(__file__)
sys.path.append("/".join(abspath.split("/")[:-4]))

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.processors import math_data_processor
from nemo_rl.models.policy import TokenizerConfig

basic_tokenizer_test_config: TokenizerConfig = {
    "name": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "chat_template": "default",
}


def test_math_data_processor():
    raw_dataset = Dataset.from_list(
        [
            {"problem": "problem1", "expected_answer": "answer1"},
            {"problem": "problem2", "expected_answer": "answer2"},
        ]
    )

    tokenizer = get_tokenizer(basic_tokenizer_test_config)

    math_task_spec = TaskDataSpec(
        task_name="math",
        prompt_file=None,
        system_prompt_file=None,
    )

    dataset = AllTaskProcessedDataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        default_task_data_spec=math_task_spec,
        task_data_processors=math_data_processor,
        max_seq_length=128,
    )

    assert dataset[0]["extra_env_info"]["ground_truth"] == "answer1"
    assert dataset[1]["extra_env_info"]["ground_truth"] == "answer2"
