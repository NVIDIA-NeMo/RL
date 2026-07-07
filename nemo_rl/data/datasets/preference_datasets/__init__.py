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
from nemo_rl.data import PreferenceDatasetConfig
from nemo_rl.data.datasets.preference_datasets.binary_preference_dataset import (
    BinaryPreferenceDataset,
)
from nemo_rl.data.datasets.preference_datasets.helpsteer3 import HelpSteer3Dataset
from nemo_rl.data.datasets.preference_datasets.preference_dataset import (
    PreferenceDataset,
)
from nemo_rl.data.datasets.preference_datasets.tulu3 import Tulu3PreferenceDataset
from nemo_rl.data.datasets.utils import resolve_external_dataset_class

DATASET_REGISTRY = {
    # built-in datasets
    "HelpSteer3": HelpSteer3Dataset,
    "Tulu3Preference": Tulu3PreferenceDataset,
    # load from local JSONL file or HuggingFace
    "BinaryPreferenceDataset": BinaryPreferenceDataset,
    "PreferenceDataset": PreferenceDataset,
}


def load_preference_dataset(data_config: PreferenceDatasetConfig):
    """Loads preference dataset.

    Resolution order for ``data_config["dataset_name"]``:

    1. If the name matches a key in ``DATASET_REGISTRY``, use the built-in
       class.
    2. Otherwise, if the name contains a ``.``, treat it as a fully qualified
       dotted import path (e.g. ``my_pkg.my_module.MyDataset``) and import
       the class dynamically. This lets users register custom datasets
       without editing ``nemo_rl``.
    3. Otherwise, raise ``ValueError`` with a helpful message.
    """
    dataset_name = data_config["dataset_name"]

    # load dataset
    if dataset_name in DATASET_REGISTRY:
        dataset_class = DATASET_REGISTRY[dataset_name]
    elif "." in dataset_name:
        dataset_class = resolve_external_dataset_class(dataset_name)
    else:
        raise ValueError(
            f"Unsupported {dataset_name=}. Please set dataset_name to one of: "
            "(1) a built-in dataset name, "
            "(2) 'BinaryPreferenceDataset' or 'PreferenceDataset' to load from a local JSONL file or HuggingFace, or "
            "(3) an importable dotted path to a dataset class "
            "(ensure it is installed and importable from PYTHONPATH)."
        )

    dataset = dataset_class(
        **data_config  # pyrefly: ignore[missing-argument]  `data_path` is required for some classes
    )

    # bind prompt and system prompt
    dataset.set_task_spec(data_config)

    return dataset


__all__ = [
    "BinaryPreferenceDataset",
    "HelpSteer3Dataset",
    "PreferenceDataset",
    "Tulu3PreferenceDataset",
    "load_preference_dataset",
]
