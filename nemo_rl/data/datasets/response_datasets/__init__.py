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

from nemo_rl.data.datasets.response_datasets.clevr import CLEVRCoGenTDataset
from nemo_rl.data.datasets.response_datasets.dapo_math import DAPOMath17KDataset
from nemo_rl.data.datasets.response_datasets.deepscaler import DeepScalerDataset
from nemo_rl.data.datasets.response_datasets.geometry3k import Geometry3KDataset
from nemo_rl.data.datasets.response_datasets.oai_format_dataset import (
    OpenAIFormatDataset,
)
from nemo_rl.data.datasets.response_datasets.oasst import OasstDataset
from nemo_rl.data.datasets.response_datasets.openmathinstruct2 import (
    OpenMathInstruct2Dataset,
)
from nemo_rl.data.datasets.response_datasets.refcoco import RefCOCODataset
from nemo_rl.data.datasets.response_datasets.response_dataset import ResponseDataset
from nemo_rl.data.datasets.response_datasets.squad import SquadDataset
from nemo_rl.data.datasets.response_datasets.tulu3 import Tulu3Dataset
from nemo_rl.data.datasets.utils import get_extra_kwargs


def load_response_dataset(data_config, seed: int = 42):
    """Loads response dataset."""
    dataset_name = data_config["dataset_name"]

    # TODO @yukih: remove duplicated dataset_name (openmathinstruct2, clevr_cogent)
    # for sft training
    if dataset_name == "open_assistant":
        base_dataset = OasstDataset(
            output_dir="/tmp/open_assistant",
            seed=seed,
        )
    elif dataset_name == "squad":
        base_dataset = SquadDataset()
    elif dataset_name == "openmathinstruct2":
        base_dataset = OpenMathInstruct2Dataset(
            split=data_config["split"],
            output_key=data_config["output_key"],
            prompt_file=data_config["prompt_file"],
            seed=seed,
        )
    elif dataset_name == "tulu3":
        base_dataset = Tulu3Dataset(
            seed=seed,
        )
    elif dataset_name == "clevr_cogent":
        base_dataset = CLEVRCoGenTDataset(
            split=data_config["split"],
            prompt_file=data_config["prompt_file"],
        )
    elif dataset_name == "openai_format":
        base_dataset = OpenAIFormatDataset(
            data_config["train_data_path"],
            data_config["val_data_path"],
            data_config["chat_key"],
            data_config["system_key"],
            data_config["system_prompt"],
            data_config["tool_key"],
            data_config["use_preserving_dataset"],
        )
    # for rl training
    elif dataset_name == "OpenMathInstruct-2":
        print("Loading nvidia/OpenMathInstruct2Dataset for training and validation")
        base_dataset: Any = OpenMathInstruct2Dataset(seed=seed)
    elif dataset_name == "DeepScaler":
        print(
            "Loading agentica-org/DeepScaleR-Preview-Dataset for training and validation"
        )
        base_dataset: Any = DeepScalerDataset(seed=seed)
    elif dataset_name == "DAPOMath17K":
        print(
            "Loading BytedTsinghua-SIA/DAPO-Math-17k for training and AIME 2024 for validation"
        )
        base_dataset: Any = DAPOMath17KDataset(seed=seed)
    # for vlm rl training
    elif dataset_name == "clevr-cogent":
        base_dataset: Any = CLEVRCoGenTDataset(
            split=data_config["split"],
        )
    elif dataset_name == "refcoco":
        base_dataset: Any = RefCOCODataset(
            split=data_config["split"],
            download_dir=data_config["download_dir"],
        )
    elif dataset_name == "geometry3k":
        base_dataset: Any = Geometry3KDataset(
            split=data_config["split"],
        )
    # fall back to load from JSON file
    elif dataset_name == "ResponseDataset":
        if "train_data_path" not in data_config:
            raise ValueError(
                "train_data_path is required when dataset_name is not one of the built-ins."
            )
        extra_kwargs = get_extra_kwargs(
            data_config,
            [
                "val_data_path",
                "input_key",
                "output_key",
                "train_split",
                "val_split",
            ],
        )
        base_dataset = ResponseDataset(
            train_data_path=data_config["train_data_path"],
            **extra_kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported {dataset_name=}. "
            "Please either use a built-in dataset "
            "or set dataset_name=ResponseDataset to load from local JSONL file or HuggingFace."
        )

    return base_dataset


__all__ = [
    "CLEVRCoGenTDataset",
    "DeepScalerDataset",
    "DAPOMath17KDataset",
    "Geometry3KDataset",
    "OpenAIFormatDataset",
    "OasstDataset",
    "OpenMathInstruct2Dataset",
    "RefCOCODataset",
    "ResponseDataset",
    "SquadDataset",
]
