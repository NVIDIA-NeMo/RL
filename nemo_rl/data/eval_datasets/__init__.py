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

from nemo_rl.data.eval_datasets.aime2024 import AIME2024Dataset
from nemo_rl.data.eval_datasets.gpqa import GPQADataset
from nemo_rl.data.eval_datasets.local_math_dataset import LocalMathDataset
from nemo_rl.data.eval_datasets.math import MathDataset
from nemo_rl.data.eval_datasets.mmlu import MMLUDataset
from nemo_rl.data.eval_datasets.mmlu_pro import MMLUProDataset


def load_eval_dataset(data_config):
    """Loads evaluation dataset."""
    dataset_name = data_config["dataset_name"]
    if dataset_name.startswith("mmlu") and dataset_name != "mmlu_pro":
        if dataset_name == "mmlu":
            base_dataset = MMLUDataset(
                prompt_file=data_config["prompt_file"],
                system_prompt_file=data_config["system_prompt_file"],
            )
        else:
            language = dataset_name.split("_")[1]
            base_dataset = MMLUDataset(
                language=language,
                prompt_file=data_config["prompt_file"],
                system_prompt_file=data_config["system_prompt_file"],
            )
    elif dataset_name == "aime2024":
        base_dataset = AIME2024Dataset(
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "gpqa":
        base_dataset = GPQADataset(
            variant="main",
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "gpqa_diamond":
        base_dataset = GPQADataset(
            variant="diamond",
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "mmlu_pro":
        base_dataset = MMLUProDataset(
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "math":
        base_dataset = MathDataset(
            variant="math_test",
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "math500":
        base_dataset = MathDataset(
            variant="math_500_test",
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "local":
        base_dataset = LocalMathDataset(
            name=dataset_name,
            data_paths=data_config["data_paths"],
            problem_key=data_config["problem_key"],
            solution_key=data_config["solution_key"],
            file_format=data_config["file_format"],
            split=data_config["split"],
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}.")
    return base_dataset


__all__ = [
    "AIME2024Dataset",
    "GPQADataset",
    "LocalMathDataset",
    "MathDataset",
    "MMLUDataset",
    "MMLUProDataset",
]
