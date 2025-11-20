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
from nemo_rl.data.datasets.preference_datasets.binary_preference_dataset import (
    BinaryPreferenceDataset,
)
from nemo_rl.data.datasets.preference_datasets.helpsteer3 import HelpSteer3Dataset
from nemo_rl.data.datasets.preference_datasets.preference_dataset import (
    PreferenceDataset,
)
from nemo_rl.data.datasets.preference_datasets.tulu3 import Tulu3PreferenceDataset
from nemo_rl.data.datasets.utils import get_extra_kwargs


# TODO: refactor this to use the new processor interface and RawDataset interface. https://github.com/NVIDIA-NeMo/RL/issues/1552
def load_preference_dataset(data_config):
    """Loads preference dataset."""
    dataset_name = data_config["dataset_name"]

    if dataset_name == "HelpSteer3":
        base_dataset = HelpSteer3Dataset()
    elif dataset_name == "Tulu3Preference":
        base_dataset = Tulu3PreferenceDataset()
    # fall back to load from JSON file
    elif dataset_name == "BinaryPreferenceDataset":
        if "train_data_path" not in data_config:
            raise ValueError(
                "train_data_path is required for dataset_name=BinaryPreferenceDataset."
            )
        extra_kwargs = get_extra_kwargs(
            data_config,
            [
                "val_data_path",
                "prompt_key",
                "chosen_key",
                "rejected_key",
                "train_split",
                "val_split",
            ],
        )
        base_dataset = BinaryPreferenceDataset(
            train_data_path=data_config["train_data_path"],
            **extra_kwargs,
        )
    elif dataset_name == "PreferenceDataset":
        if "train_data_path" not in data_config:
            raise ValueError(
                "train_data_path is required for dataset_name=PreferenceDataset."
            )
        extra_kwargs = get_extra_kwargs(
            data_config,
            [
                "val_data_path",
                "train_split",
                "val_split",
            ],
        )
        base_dataset = PreferenceDataset(
            train_data_path=data_config["train_data_path"],
            **extra_kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported {dataset_name=}. "
            "Please either set dataset_name in {'HelpSteer3', 'Tulu3Preference'} to use a built-in dataset "
            "or set dataset_name in {'PreferenceDataset', 'BinaryPreferenceDataset'} to load from local JSONL file or HuggingFace."
        )

    return base_dataset


__all__ = [
    "BinaryPreferenceDataset",
    "HelpSteer3Dataset",
    "PreferenceDataset",
    "Tulu3PreferenceDataset",
]
