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
from nemo_rl.data.sft_datasets.clevr import CLEVRCoGenTDataset
from nemo_rl.data.sft_datasets.local_sft_dataset import LocalSFTDataset
from nemo_rl.data.sft_datasets.oai_format_dataset import OpenAIFormatDataset
from nemo_rl.data.sft_datasets.oasst import OasstDataset
from nemo_rl.data.sft_datasets.openmathinstruct2 import OpenMathInstruct2Dataset
from nemo_rl.data.sft_datasets.squad import SquadDataset


def load_sft_dataset(data_config, seed: int):
    """Loads SFT dataset."""
    dataset_name = data_config["dataset_name"]

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
        )
    # fall back to local dataset
    else:
        base_dataset = LocalSFTDataset(
            data_config["train_data_path"],
            data_config["val_data_path"],
            data_config["input_key"],
            data_config["output_key"],
        )

    return base_dataset


__all__ = [
    "OasstDataset",
    "SquadDataset",
    "OpenMathInstruct2Dataset",
    "CLEVRCoGenTDataset",
    "OpenAIFormatDataset",
    "LocalSFTDataset",
]
