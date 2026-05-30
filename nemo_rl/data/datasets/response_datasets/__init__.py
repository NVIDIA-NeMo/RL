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

import importlib

from nemo_rl.data import ResponseDatasetConfig
from nemo_rl.data.datasets.response_datasets.aime24 import AIME2024Dataset
from nemo_rl.data.datasets.response_datasets.avqa import AVQADataset
from nemo_rl.data.datasets.response_datasets.clevr import CLEVRCoGenTDataset
from nemo_rl.data.datasets.response_datasets.daily_omni import DailyOmniDataset
from nemo_rl.data.datasets.response_datasets.dapo_math import (
    DAPOMath17KDataset,
    DAPOMathAIME2024Dataset,
)
from nemo_rl.data.datasets.response_datasets.deepscaler import DeepScalerDataset
from nemo_rl.data.datasets.response_datasets.general_conversations_dataset import (
    GeneralConversationsJsonlDataset,
)
from nemo_rl.data.datasets.response_datasets.geometry3k import Geometry3KDataset
from nemo_rl.data.datasets.response_datasets.gsm8k import GSM8KDataset
from nemo_rl.data.datasets.response_datasets.helpsteer3 import HelpSteer3Dataset
from nemo_rl.data.datasets.response_datasets.nemogym_dataset import NemoGymDataset
from nemo_rl.data.datasets.response_datasets.nemotron_cascade2_sft import (
    NemotronCascade2SFTMathDataset,
)
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
    "avqa": AVQADataset,
    "AIME2024": AIME2024Dataset,
    "clevr-cogent": CLEVRCoGenTDataset,
    "daily-omni": DailyOmniDataset,
    "general-conversation-jsonl": GeneralConversationsJsonlDataset,
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
    "gsm8k": GSM8KDataset,
    "Nemotron-Cascade-2-SFT-Math": NemotronCascade2SFTMathDataset,
    # load from local JSONL file or HuggingFace
    "openai_format": OpenAIFormatDataset,
    "NemoGymDataset": NemoGymDataset,
    "ResponseDataset": ResponseDataset,
}


def _resolve_external_dataset_class(dataset_name: str) -> type:
    """Resolve a fully-qualified dotted dataset path to a class.

    Supports user-defined datasets that live outside ``nemo_rl`` so users do
    not have to edit the built-in ``DATASET_REGISTRY`` to plug in their own
    dataset class. The class must be importable from ``PYTHONPATH`` (or the
    active virtual environment).
    """
    if "." not in dataset_name:
        raise ValueError(
            f"Unsupported {dataset_name=}. "
            "Please either use a built-in dataset "
            "(see nemo_rl.data.datasets.response_datasets.DATASET_REGISTRY "
            "for the full list), set dataset_name=ResponseDataset to load "
            "from a local JSONL file or HuggingFace, or pass a fully "
            "qualified import path like 'my_pkg.my_module.MyDataset' to a "
            "class importable from PYTHONPATH."
        )

    module_path, _, class_name = dataset_name.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ValueError(
            f"Could not import module {module_path!r} for "
            f"dataset_name={dataset_name!r}. Ensure the module is "
            "installed and importable from PYTHONPATH."
        ) from e
    if not hasattr(module, class_name):
        raise ValueError(
            f"Module {module_path!r} has no attribute {class_name!r} "
            f"(referenced by dataset_name={dataset_name!r})."
        )
    dataset_class = getattr(module, class_name)
    if not isinstance(dataset_class, type):
        raise ValueError(
            f"dataset_name={dataset_name!r} resolved to {dataset_class!r}, "
            "which is not a class. Expected a dataset class."
        )
    return dataset_class


def load_response_dataset(data_config: ResponseDatasetConfig):
    """Loads response dataset.

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

    if dataset_name in DATASET_REGISTRY:
        dataset_class = DATASET_REGISTRY[dataset_name]
    else:
        dataset_class = _resolve_external_dataset_class(dataset_name)

    dataset = dataset_class(
        **data_config  # pyrefly: ignore[missing-argument]  `data_path` is required for some classes
    )

    # bind prompt, system prompt and data processor
    dataset.set_task_spec(data_config)
    # Remove this after the data processor is refactored. https://github.com/NVIDIA-NeMo/RL/issues/1658
    dataset.set_processor()

    return dataset


__all__ = [
    "AVQADataset",
    "AIME2024Dataset",
    "CLEVRCoGenTDataset",
    "DailyOmniDataset",
    "GeneralConversationsJsonlDataset",
    "DAPOMath17KDataset",
    "DAPOMathAIME2024Dataset",
    "GSM8KDataset",
    "DeepScalerDataset",
    "Geometry3KDataset",
    "HelpSteer3Dataset",
    "NemoGymDataset",
    "NemotronCascade2SFTMathDataset",
    "OasstDataset",
    "OpenAIFormatDataset",
    "OpenMathInstruct2Dataset",
    "RefCOCODataset",
    "ResponseDataset",
    "SquadDataset",
    "Tulu3SftMixtureDataset",
    "load_response_dataset",
]
