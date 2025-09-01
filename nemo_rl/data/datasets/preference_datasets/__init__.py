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
from nemo_rl.data.datasets.preference_datasets.helpsteer3 import HelpSteer3Dataset
from nemo_rl.data.datasets.preference_datasets.preference_dataset import (
    PreferenceDataset,
)
from nemo_rl.data.datasets.preference_datasets.tulu3 import Tulu3PreferenceDataset


def load_preference_dataset(data_config):
    """Loads preference dataset."""
    dataset_name = (
        data_config["dataset_name"] if "dataset_name" in data_config else "from_json"
    )

    if dataset_name == "HelpSteer3":
        base_dataset = HelpSteer3Dataset()
    elif dataset_name == "Tulu3Preference":
        base_dataset = Tulu3PreferenceDataset()
    # fall back to load from JSON file
    else:
        prompt_key = data_config["prompt_key"] if "prompt_key" in data_config else None
        chosen_key = data_config["chosen_key"] if "chosen_key" in data_config else None
        rejected_key = (
            data_config["rejected_key"] if "rejected_key" in data_config else None
        )
        base_dataset = PreferenceDataset(
            train_ds_path=data_config["train_data_path"],
            val_ds_path=data_config["val_data_path"],
            prompt_key=prompt_key,
            chosen_key=chosen_key,
            rejected_key=rejected_key,
        )

    return base_dataset


__all__ = [
    "HelpSteer3Dataset",
    "PreferenceDataset",
    "Tulu3PreferenceDataset",
]
