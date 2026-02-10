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
import tempfile
from collections import defaultdict

import pytest
import torch
from datasets import Dataset

abspath = os.path.abspath(__file__)
sys.path.append("/".join(abspath.split("/")[:-4]))

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.datasets.eval_datasets import (
    AIMEDataset,
    GPQADataset,
    MathDataset,
    MMLUDataset,
)
from nemo_rl.data.datasets.response_datasets import (
    DeepScalerDataset,
    OpenMathInstruct2Dataset,
)
from nemo_rl.data.interfaces import TaskDataProcessFnCallable, TaskDataSpec
from nemo_rl.data.processors import (
    helpsteer3_data_processor,
    math_data_processor,
    math_hf_data_processor,
)
from nemo_rl.models.policy import TokenizerConfig


class DummyTokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    ):
        content = "".join(
            f"{m.get('role', 'user')}: {m['content']}\n" for m in messages
        )
        if add_generation_prompt:
            content += "assistant:"
        return content

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        if isinstance(text, list):
            text = "".join(text)
        encoded = list(range(len(text)))
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([encoded], dtype=torch.long)}
        return {"input_ids": encoded}


def test_math_data_processor():
    raw_dataset = Dataset.from_list(
        [
            {"problem": "problem1", "expected_answer": "answer1"},
            {"problem": "problem2", "expected_answer": "answer2"},
        ]
    )

    tokenizer = get_tokenizer(
        TokenizerConfig(
            name="Qwen/Qwen2.5-Math-1.5B-Instruct",
            chat_template="default",
        )
    )

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


@pytest.mark.hf_gated
@pytest.mark.parametrize(
    "tokenizer_name",
    [
        "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",  # no bos token
        "google/gemma-3-1b-it",
        "Qwen/Qwen3-0.6B",  # no bos token
        "deepseek-ai/DeepSeek-V3",
        "moonshotai/Moonlight-16B-A3B-Instruct",
    ],
)
@pytest.mark.parametrize(
    "dataset_cls",
    [
        OpenMathInstruct2Dataset,
        DeepScalerDataset,
    ],
)
def test_math_hf_data_processor(tokenizer_name, dataset_cls):
    # Initialize dataset
    data = dataset_cls()
    task_name = data.task_name
    # Setup tokenizer
    tokenizer = get_tokenizer(
        TokenizerConfig(
            name=tokenizer_name,
            chat_template="default",
        )
    )

    # Configure task specification
    math_task_spec = TaskDataSpec(
        task_name=task_name,
        prompt_file=f"{os.path.dirname(abspath)}/../../../examples/prompts/cot.txt",
        system_prompt_file=None,
    )

    task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = (
        defaultdict(lambda: (math_task_spec, math_hf_data_processor))
    )
    task_data_processors[task_name] = (math_task_spec, math_hf_data_processor)

    dataset = AllTaskProcessedDataset(
        dataset=data.dataset,
        tokenizer=tokenizer,
        default_task_data_spec=math_task_spec,
        task_data_processors=task_data_processors,
        max_seq_length=128,
    )

    # Test that the first item can be retrieved when the BOS token assertion passes
    first_item = dataset[0]
    assert first_item is not None
    assert "message_log" in first_item
    assert len(first_item["message_log"]) > 0


def test_math_hf_data_processor_without_prompt():
    datum_dict = {
        "messages": [
            {"role": "user", "content": "Solve 1+1."},
            {"role": "assistant", "content": "2"},
        ],
        "task_name": "math",
    }
    tokenizer = DummyTokenizer()

    math_task_spec = TaskDataSpec(
        task_name="math",
        prompt_file=None,
        system_prompt_file=None,
    )

    result = math_hf_data_processor(
        datum_dict=datum_dict,
        task_data_spec=math_task_spec,
        tokenizer=tokenizer,
        max_seq_length=128,
        idx=0,
    )

    assert result["extra_env_info"]["ground_truth"] == "2"
    assert result["loss_multiplier"] == 1.0
    assert len(result["message_log"]) == 1
    assert result["message_log"][0]["role"] == "user"
    assert "Solve 1+1." in result["message_log"][0]["content"]


@pytest.fixture
def system_prompt_file(request):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as file:
        file.write("You are a helpful assistant.\n{}")

    return file.name


@pytest.mark.hf_gated
@pytest.mark.parametrize(
    "tokenizer_name",
    [
        "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",  # no bos token
        "google/gemma-3-1b-it",
        "Qwen/Qwen3-0.6B",  # no bos token
        "deepseek-ai/DeepSeek-V3",
        "moonshotai/Moonlight-16B-A3B-Instruct",
    ],
)
@pytest.mark.parametrize(
    "dataset_cls",
    [
        AIMEDataset,
        GPQADataset,
        MathDataset,
        MMLUDataset,
    ],
)
@pytest.mark.parametrize(
    "system_prompt_file", [system_prompt_file, None], indirect=True
)
def test_eval_math_hf_data_processor(tokenizer_name, dataset_cls, system_prompt_file):
    # Initialize dataset
    data = dataset_cls()

    # Setup tokenizer
    tokenizer = get_tokenizer(
        TokenizerConfig(
            name=tokenizer_name,
            chat_template="default",
        )
    )

    # Configure task specification
    math_task_spec = TaskDataSpec(
        task_name="math",
        prompt_file=f"{os.path.dirname(abspath)}/../../../examples/prompts/cot.txt",
        system_prompt_file=system_prompt_file,
    )

    dataset = AllTaskProcessedDataset(
        dataset=data.rekeyed_ds,
        tokenizer=tokenizer,
        default_task_data_spec=math_task_spec,
        task_data_processors=data.processor,
        max_seq_length=128,
    )

    # Test that the first item can be retrieved when the BOS token assertion passes
    first_item = dataset[0]
    assert first_item is not None
    assert "message_log" in first_item
    assert len(first_item["message_log"]) > 0


def test_helpsteer3_data_processor():
    tokenizer = DummyTokenizer()
    task_data_spec = TaskDataSpec(
        task_name="helpsteer3",
        prompt_file=None,
        system_prompt_file=None,
    )
    datum_dict = {
        "context": [
            {"role": "user", "content": "Hello"},
        ],
        "response": [
            {"role": "assistant", "content": "Hi"},
            {"role": "assistant", "content": "there"},
        ],
        "task_name": "helpsteer3",
    }

    out = helpsteer3_data_processor(
        datum_dict=datum_dict,
        task_data_spec=task_data_spec,
        tokenizer=tokenizer,
        max_seq_length=4096,
        idx=7,
    )

    # Basic structure
    assert isinstance(out, dict)
    assert out["idx"] == 7
    assert out.get("task_name") == "helpsteer3"
    assert "message_log" in out and isinstance(out["message_log"], list)
    assert "extra_env_info" in out and "ground_truth" in out["extra_env_info"]
    assert isinstance(out["length"], int)
    assert out["loss_multiplier"] == 1.0

    # Ground truth should be space-joined assistant responses
    assert out["extra_env_info"]["ground_truth"] == "Hi there"

    # Tokenization behavior: only first message has non-empty token_ids
    msg_log = out["message_log"]
    assert len(msg_log) >= 1
    assert "token_ids" in msg_log[0] and isinstance(
        msg_log[0]["token_ids"], torch.Tensor
    )

    for m in msg_log[1:]:
        assert "token_ids" in m and isinstance(m["token_ids"], torch.Tensor)
        assert int(m["token_ids"].numel()) == 0

    # Length equals sum of token lengths
    assert out["length"] == sum(int(m["token_ids"].numel()) for m in msg_log)
