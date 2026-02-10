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


from typing import Any

from datasets import Dataset, load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset

# Load and prep dataset
SYSTEM_PROMPT = system_prompt = """
You are a helpful AI assistant.

For every request, you should carefully think through the math problem step by step, then provide the fianl answer in integer format.

Steps for Each Request:
1. Think: Provide detailed, step-by-step reasoning, calculations, or derivations.
2. Produce Final Answer: After step-by-step reasoning, output the final answer in integer format.

Output Format:
<think>Your thoughts and reasoning</think>
<answer>Final answer in integer format</answer>

Important Notes:
1. You must include your reasoning steps inside <think>.
2. You must always output the Final Answer within <answer> after the reasoning steps is done.
3. You should consistently work through the solution step by step before giving the final answer.
4. The final answer can only be an integer.
"""

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def format_gsm8k(
    data: dict[str, str | float | int],
    task_name: str = "gsm8k",
) -> dict[str, list[Any] | str]:
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
            ,
            {
                "role": "user",
                "content": data["question"],
            },
            {
                "role": "assistant",
                "content": extract_hash_answer(data['answer']),
            },
        ],
        "task_name": task_name,
    }


def prepare_gsm8k_dataset(
    seed: int = 42, task_name: str = "gsm8k"
) -> dict[str, Dataset | None]:
    
    # Load the original dataset for training
    train_ds = load_dataset('openai/gsm8k', 'main')["train"]

    # Load hendrydong/aime24 dataset for validation
    val_ds = load_dataset('openai/gsm8k', 'main')["test"]

    # Shuffle the training dataset with the specified seed
    train_ds = train_ds.shuffle(seed=seed)

    # Format the examples, removing original columns
    train_formatted = train_ds.map(
        format_gsm8k,
        remove_columns=train_ds.column_names,
        fn_kwargs={"task_name": task_name},
    )
    val_formatted = val_ds.map(
        format_gsm8k,
        remove_columns=val_ds.column_names,
        fn_kwargs={"task_name": task_name},
    )

    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class gsm8kDataset(RawDataset):
    def __init__(self, seed: int = 42) -> None:
        """Initialize the DAPO Math 17K dataset with train split.

        Args:
            seed: Random seed for reproducible splitting
        """
        self.task_name = "gsm8k"
        self.formatted_ds = prepare_gsm8k_dataset(
            seed=seed, task_name=self.task_name
        )
