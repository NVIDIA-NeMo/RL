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
import pytest
from transformers import AutoTokenizer

from nemo_rl.data.datasets import load_eval_dataset


@pytest.mark.skip(reason="dataset download is flaky")
def test_gpqa_dataset():
    # load the dataset
    data_config = {
        "dataset_name": "gpqa",
        "prompt_file": None,
        "system_prompt_file": None,
    }
    gpqa_dataset = load_eval_dataset(data_config)

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    # check that the dataset is formatted correctly
    for example in gpqa_dataset.rekeyed_ds.take(5):
        assert "question" in example
        assert "options" in example
        assert "answer" in example

        ## check that applying chat template works as expected
        default_templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["question"]}],
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )

        assert (
            default_templated
            == f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{example['question']}<|im_end|>\n"
        )


@pytest.mark.skip(reason="dataset download is flaky")
def test_math_dataset():
    # load the dataset
    data_config = {
        "dataset_name": "math",
        "prompt_file": None,
        "system_prompt_file": None,
    }
    math_dataset = load_eval_dataset(data_config)

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    # check that the dataset is formatted correctly
    for example in math_dataset.rekeyed_ds.take(5):
        assert "problem" in example
        assert "expected_answer" in example

        ## check that applying chat template works as expected
        default_templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["problem"]}],
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )

        assert (
            default_templated
            == f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{example['problem']}<|im_end|>\n"
        )


@pytest.mark.skip(reason="dataset download is flaky")
def test_mmlu_dataset():
    # load the dataset
    data_config = {
        "dataset_name": "mmlu",
        "prompt_file": None,
        "system_prompt_file": None,
    }
    mmlu_dataset = load_eval_dataset(data_config)

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    # check that the dataset is formatted correctly
    for example in mmlu_dataset.rekeyed_ds.take(5):
        assert "question" in example
        assert "options" in example
        assert "answer" in example
        assert "subject" in example

        ## check that applying chat template works as expected
        default_templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["question"]}],
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )

        assert (
            default_templated
            == f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{example['question']}<|im_end|>\n"
        )
