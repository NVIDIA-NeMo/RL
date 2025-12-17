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

from nemo_rl.data.datasets.response_datasets.aime24 import AIME2024Dataset
from nemo_rl.data.datasets.response_datasets.clevr import CLEVRCoGenTDataset
from nemo_rl.data.datasets.response_datasets.dapo_math import (
    DAPOMath17KDataset,
    DAPOMathAIME2024Dataset,
)
from nemo_rl.data.datasets.response_datasets.deepscaler import DeepScalerDataset
from nemo_rl.data.datasets.response_datasets.geometry3k import Geometry3KDataset
from nemo_rl.data.datasets.response_datasets.helpsteer3 import HelpSteer3Dataset
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
from nemo_rl.data.datasets.response_datasets.tulu3 import Tulu3SftMixtureDataset


# TODO: refactor this to use the new processor interface and RawDataset interface. https://github.com/NVIDIA-NeMo/RL/issues/1552
def load_response_dataset(data_config, seed: int = 42):
    """Loads response dataset."""
    dataset_name = data_config["dataset_name"]

    # for sft training
    if dataset_name == "open_assistant":
        base_dataset = OasstDataset(
            output_dir="/tmp/open_assistant",
            seed=seed,
        )
    elif dataset_name == "squad":
        base_dataset = SquadDataset(**data_config)
    elif dataset_name == "tulu3_sft_mixture":
        base_dataset: Any = Tulu3SftMixtureDataset(
            test_size=data_config.get("test_size", 0.05),
            prompt_file=data_config.get("prompt_file", None),
            max_samples=data_config.get("max_samples", None),
            seed=seed,
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
        # TODO: also test after SFT updated
        print("Loading nvidia/OpenMathInstruct2Dataset for training and validation")
        base_dataset: Any = OpenMathInstruct2Dataset(**data_config, seed=seed)
    elif dataset_name == "DeepScaler":
        base_dataset: Any = DeepScalerDataset(**data_config)
    elif dataset_name == "DAPOMath17K":
        base_dataset: Any = DAPOMath17KDataset(**data_config)
    elif dataset_name == "HelpSteer3":
        base_dataset: Any = HelpSteer3Dataset(**data_config)
    elif dataset_name == "AIME2024":
        base_dataset: Any = AIME2024Dataset(**data_config)
    elif dataset_name == "DAPOMathAIME2024":
        base_dataset: Any = DAPOMathAIME2024Dataset(**data_config)
    # for vlm training
    # TODO: test after GRPO-VLM updated
    elif dataset_name == "clevr-cogent":
        # TODO: also test after SFT updated
        base_dataset: Any = CLEVRCoGenTDataset(**data_config)
    elif dataset_name == "refcoco":
        base_dataset: Any = RefCOCODataset(**data_config)
    elif dataset_name == "geometry3k":
        base_dataset: Any = Geometry3KDataset(**data_config)
    # fall back to load from JSON file
    elif dataset_name == "ResponseDataset":
        base_dataset = ResponseDataset(**data_config, seed=seed)
    else:
        raise ValueError(
            f"Unsupported {dataset_name=}. "
            "Please either use a built-in dataset "
            "or set dataset_name=ResponseDataset to load from local JSONL file or HuggingFace."
        )

    base_dataset.set_task_spec(data_config)
    base_dataset.set_processor()

    return base_dataset


__all__ = [
    "AIME2024Dataset",
    "CLEVRCoGenTDataset",
    "DeepScalerDataset",
    "DAPOMath17KDataset",
    "DAPOMathAIME2024Dataset",
    "Geometry3KDataset",
    "OpenAIFormatDataset",
    "OasstDataset",
    "OpenMathInstruct2Dataset",
    "RefCOCODataset",
    "ResponseDataset",
    "SquadDataset",
    "Tulu3SftMixtureDataset",
    "HelpSteer3Dataset",
]
