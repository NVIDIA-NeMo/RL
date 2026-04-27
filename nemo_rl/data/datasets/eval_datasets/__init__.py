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

from nemo_rl.data.datasets.eval_datasets.aime import AIMEDataset
from nemo_rl.data.datasets.eval_datasets.gpqa import GPQADataset
from nemo_rl.data.datasets.eval_datasets.humaneval_plus import HumanEvalPlusDataset
from nemo_rl.data.datasets.eval_datasets.local_math_dataset import LocalMathDataset
from nemo_rl.data.datasets.eval_datasets.math import MathDataset
from nemo_rl.data.datasets.eval_datasets.mbpp_plus import MBPPPlusDataset
from nemo_rl.data.datasets.eval_datasets.mmau import MMAUDataset
from nemo_rl.data.datasets.eval_datasets.mmlu import MMLUDataset
from nemo_rl.data.datasets.eval_datasets.mmlu_pro import MMLUProDataset

# Dataset names that require multimodal (VLM) processing
MULTIMODAL_DATASETS = {"mmau", "TwinkStart/MMAU"}


def _is_multimodal_dataset(dataset_name):
    """Check if the dataset requires multimodal processing."""
    return dataset_name in MULTIMODAL_DATASETS


def load_eval_dataset(data_config):
    """Loads evaluation dataset."""
    dataset_name = data_config["dataset_name"]

    # mmlu
    if dataset_name.startswith("mmlu") and dataset_name != "mmlu_pro":
        num_few_shot = data_config.get("num_few_shot", 0)
        if dataset_name == "mmlu":
            base_dataset = MMLUDataset(
                prompt_file=data_config["prompt_file"],
                system_prompt_file=data_config["system_prompt_file"],
                num_few_shot=num_few_shot,
            )
        else:
            language = dataset_name.split("_")[1]
            base_dataset = MMLUDataset(
                language=language,
                prompt_file=data_config["prompt_file"],
                system_prompt_file=data_config["system_prompt_file"],
                num_few_shot=num_few_shot,
            )
    elif dataset_name == "mmlu_pro":
        base_dataset = MMLUProDataset(
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    # aime
    elif dataset_name == "aime2024":
        base_dataset = AIMEDataset(
            variant="2024",
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "aime2025":
        base_dataset = AIMEDataset(
            variant="2025",
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    # gpqa
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
    # math
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
    # mmau
    elif dataset_name in ("mmau", "TwinkStart/MMAU"):
        split = data_config.get("split", "v05.15.25")
        base_dataset = MMAUDataset(
            dataset_name="TwinkStart/MMAU",
            split=split,
        )
    # code-execution benchmarks
    elif dataset_name == "mbpp_plus":
        base_dataset = MBPPPlusDataset(
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
            dataset_path=data_config.get("dataset_path") or "evalplus/mbppplus",
            split=data_config.get("split") or "test",
        )
    elif dataset_name in ("humaneval_plus", "human_eval_plus"):
        base_dataset = HumanEvalPlusDataset(
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
            dataset_path=data_config.get("dataset_path") or "evalplus/humanevalplus",
            split=data_config.get("split") or "test",
        )
    # fall back to local dataset
    else:
        print(f"Loading dataset from {dataset_name}...")
        base_dataset = LocalMathDataset(
            data_path=dataset_name,
            problem_key=data_config["problem_key"],
            solution_key=data_config["solution_key"],
            file_format=data_config["file_format"],
            split=data_config["split"],
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )

    return base_dataset


__all__ = [
    "AIMEDataset",
    "GPQADataset",
    "HumanEvalPlusDataset",
    "LocalMathDataset",
    "MathDataset",
    "MBPPPlusDataset",
    "MMAUDataset",
    "MMLUDataset",
    "MMLUProDataset",
    "MULTIMODAL_DATASETS",
    "_is_multimodal_dataset",
]
