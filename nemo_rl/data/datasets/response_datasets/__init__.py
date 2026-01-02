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

from nemo_rl.data import ResponseDatasetConfig
from nemo_rl.data.datasets.response_datasets.aime24 import AIME2024Dataset
from nemo_rl.data.datasets.response_datasets.clevr import CLEVRCoGenTDataset
from nemo_rl.data.datasets.response_datasets.dapo_math import (
    DAPOMath17KDataset,
    DAPOMathAIME2024Dataset,
)
from nemo_rl.data.datasets.response_datasets.deepscaler import DeepScalerDataset
from nemo_rl.data.datasets.response_datasets.geometry3k import Geometry3KDataset
from nemo_rl.data.datasets.response_datasets.helpsteer3 import HelpSteer3Dataset
from nemo_rl.data.datasets.response_datasets.nemogym_dataset import NemoGymDataset
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

DATASET_REGISTRY = {
    # built-in datasets
    "AIME2024": AIME2024Dataset,
    "clevr-cogent": CLEVRCoGenTDataset,
    "DAPOMath17K": DAPOMath17KDataset,
    "DAPOMathAIME2024": DAPOMathAIME2024Dataset,
    "DeepScaler": DeepScalerDataset,
    "geometry3k": Geometry3KDataset,
    "HelpSteer3": HelpSteer3Dataset,
    "open_assistant": OasstDataset,
    "OpenMathInstruct-2": OpenMathInstruct2Dataset,
    "refcoco": RefCOCODataset,
    "squad": SquadDataset,
    "tulu3_sft_mixture": Tulu3SftMixtureDataset,
    # load from local JSONL file or HuggingFace
    "openai_format": OpenAIFormatDataset,
    "ResponseDataset": ResponseDataset,
}


def load_response_dataset(data_config: ResponseDatasetConfig):
    """Loads response dataset."""
    dataset_name = data_config["dataset_name"]

    # load dataset
    if dataset_name in DATASET_REGISTRY:
        dataset_class = DATASET_REGISTRY[dataset_name]
        dataset = dataset_class(
            **data_config  # pyrefly: ignore[missing-argument]  `data_path` is required for some classes
        )
    elif dataset_name == "NemoGymDataset":
        dataset = NemoGymDataset(**data_config)
    else:
        raise ValueError(
            f"Unsupported {dataset_name=}. "
            "Please either use a built-in dataset "
            "or set dataset_name=ResponseDataset to load from local JSONL file or HuggingFace."
        )

    dataset.set_task_spec(data_config)
    # Remove this after the data processor is refactored. https://github.com/NVIDIA-NeMo/RL/issues/1658
    dataset.set_processor()

    return dataset


__all__ = [
    "AIME2024Dataset",
    "CLEVRCoGenTDataset",
    "DAPOMath17KDataset",
    "DAPOMathAIME2024Dataset",
    "DeepScalerDataset",
    "Geometry3KDataset",
    "HelpSteer3Dataset",
    "NemoGymDataset",
    "OasstDataset",
    "OpenAIFormatDataset",
    "OpenMathInstruct2Dataset",
    "RefCOCODataset",
    "ResponseDataset",
    "SquadDataset",
    "Tulu3SftMixtureDataset",
    "load_response_dataset",
]
