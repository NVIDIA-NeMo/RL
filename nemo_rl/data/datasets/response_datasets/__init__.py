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

from functools import partial

from nemo_rl.data import ResponseDatasetConfig
from nemo_rl.data.datasets.response_datasets.aime import AIMEDataset
from nemo_rl.data.datasets.response_datasets.arrow_text_dataset import ArrowTextDataset
from nemo_rl.data.datasets.response_datasets.audiomcq import AudioMCQDataset
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
from nemo_rl.data.datasets.response_datasets.gpqa import GPQADataset
from nemo_rl.data.datasets.response_datasets.gsm8k import GSM8KDataset
from nemo_rl.data.datasets.response_datasets.helpsteer3 import HelpSteer3Dataset
from nemo_rl.data.datasets.response_datasets.intent import (
    IntentBenchDataset,
    IntentTrainDataset,
)
from nemo_rl.data.datasets.response_datasets.math import MathDataset
from nemo_rl.data.datasets.response_datasets.mmau import MMAUDataset
from nemo_rl.data.datasets.response_datasets.mmlu import MMLUDataset
from nemo_rl.data.datasets.response_datasets.mmlu_pro import MMLUProDataset
from nemo_rl.data.datasets.response_datasets.mmpr_tiny import MMPRTinyDataset
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
from nemo_rl.data.datasets.response_datasets.openr1_math import OpenR1Math220KDataset
from nemo_rl.data.datasets.response_datasets.refcoco import RefCOCODataset
from nemo_rl.data.datasets.response_datasets.response_dataset import ResponseDataset
from nemo_rl.data.datasets.response_datasets.squad import SquadDataset
from nemo_rl.data.datasets.response_datasets.tulu3 import Tulu3SftMixtureDataset
from nemo_rl.data.datasets.utils import resolve_external_dataset_class

DATASET_REGISTRY = {
    # built-in datasets
    "audiomcq": AudioMCQDataset,
    "arrow_text": ArrowTextDataset,
    "avqa": AVQADataset,
    "AIME2024": partial(AIMEDataset, variant="2024"),
    "AIME2025": partial(AIMEDataset, variant="2025"),
    "AIME2026": partial(AIMEDataset, variant="2026"),
    "clevr-cogent": CLEVRCoGenTDataset,
    "daily-omni": DailyOmniDataset,
    "general-conversation-jsonl": GeneralConversationsJsonlDataset,
    "DAPOMath17K": DAPOMath17KDataset,
    "DAPOMathAIME2024": DAPOMathAIME2024Dataset,
    "DeepScaler": DeepScalerDataset,
    "GSM8K": GSM8KDataset,
    "geometry3k": Geometry3KDataset,
    "gpqa": partial(GPQADataset, variant="main"),
    "gpqa_diamond": partial(GPQADataset, variant="diamond"),
    "mmpr-tiny": MMPRTinyDataset,
    "math": partial(MathDataset, variant="math_test"),
    "math500": partial(MathDataset, variant="math_500_test"),
    "mmau": MMAUDataset,
    "TwinkStart/MMAU": MMAUDataset,
    "mmlu": partial(MMLUDataset, language="EN-US"),
    "mmlu_AR-XY": partial(MMLUDataset, language="AR-XY"),
    "mmlu_BN-BD": partial(MMLUDataset, language="BN-BD"),
    "mmlu_DE-DE": partial(MMLUDataset, language="DE-DE"),
    "mmlu_EN-US": partial(MMLUDataset, language="EN-US"),
    "mmlu_ES-LA": partial(MMLUDataset, language="ES-LA"),
    "mmlu_FR-FR": partial(MMLUDataset, language="FR-FR"),
    "mmlu_HI-IN": partial(MMLUDataset, language="HI-IN"),
    "mmlu_ID-ID": partial(MMLUDataset, language="ID-ID"),
    "mmlu_IT-IT": partial(MMLUDataset, language="IT-IT"),
    "mmlu_JA-JP": partial(MMLUDataset, language="JA-JP"),
    "mmlu_KO-KR": partial(MMLUDataset, language="KO-KR"),
    "mmlu_PT-BR": partial(MMLUDataset, language="PT-BR"),
    "mmlu_ZH-CN": partial(MMLUDataset, language="ZH-CN"),
    "mmlu_SW-KE": partial(MMLUDataset, language="SW-KE"),
    "mmlu_YO-NG": partial(MMLUDataset, language="YO-NG"),
    "mmlu_pro": MMLUProDataset,
    "HelpSteer3": HelpSteer3Dataset,
    "intent-train": IntentTrainDataset,
    "intent-bench": IntentBenchDataset,
    "open_assistant": OasstDataset,
    "OpenMathInstruct-2": OpenMathInstruct2Dataset,
    "OpenR1-Math-220k": OpenR1Math220KDataset,
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


def _resolve_response_dataset_factory(dataset_name: str):
    """Resolve a built-in or external response dataset factory."""
    if dataset_name in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset_name]
    if "." in dataset_name:
        return resolve_external_dataset_class(dataset_name)
    raise ValueError(
        f"Unsupported {dataset_name=}. Please set dataset_name to one of: "
        "(1) a built-in dataset name, "
        "(2) 'ResponseDataset' to load from a local JSONL file or HuggingFace, or "
        "(3) an importable dotted path to a dataset class "
        "(ensure it is installed and importable from PYTHONPATH)."
    )


def is_multimodal_response_dataset(dataset_name: str) -> bool:
    """Return whether a response dataset requires a multimodal processor.

    Built-in and external dataset classes declare this capability through the
    ``RawDataset.is_multimodal`` class attribute. For a ``partial`` registry
    entry, the capability is read from the underlying dataset class.
    """
    dataset_factory = _resolve_response_dataset_factory(dataset_name)
    dataset_type = (
        dataset_factory.func
        if isinstance(dataset_factory, partial)
        else dataset_factory
    )
    return bool(getattr(dataset_type, "is_multimodal", False))


def load_response_dataset(data_config: ResponseDatasetConfig):
    """Load a response dataset.

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

    dataset_class = _resolve_response_dataset_factory(dataset_name)

    dataset = dataset_class(
        **data_config  # pyrefly: ignore[missing-argument]  `data_path` is required for some classes
    )

    # bind prompt, system prompt and data processor
    dataset.set_task_spec(data_config)
    # Remove this after the data processor is refactored. https://github.com/NVIDIA-NeMo/RL/issues/1658
    dataset.set_processor()

    return dataset


__all__ = [
    "AudioMCQDataset",
    "ArrowTextDataset",
    "AVQADataset",
    "AIMEDataset",
    "CLEVRCoGenTDataset",
    "DailyOmniDataset",
    "GeneralConversationsJsonlDataset",
    "DAPOMath17KDataset",
    "DAPOMathAIME2024Dataset",
    "GSM8KDataset",
    "DeepScalerDataset",
    "Geometry3KDataset",
    "GPQADataset",
    "HelpSteer3Dataset",
    "IntentBenchDataset",
    "IntentTrainDataset",
    "MathDataset",
    "MMAUDataset",
    "MMPRTinyDataset",
    "MMLUDataset",
    "MMLUProDataset",
    "NemoGymDataset",
    "NemotronCascade2SFTMathDataset",
    "OasstDataset",
    "OpenAIFormatDataset",
    "OpenMathInstruct2Dataset",
    "OpenR1Math220KDataset",
    "RefCOCODataset",
    "ResponseDataset",
    "SquadDataset",
    "Tulu3SftMixtureDataset",
    "is_multimodal_response_dataset",
    "load_response_dataset",
]
