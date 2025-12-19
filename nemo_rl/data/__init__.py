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

from typing import Literal, NotRequired, TypedDict


class ResponseDatasetConfig(TypedDict):
    dataset_name: str
    data_path: NotRequired[str]
    input_key: NotRequired[str]
    output_key: NotRequired[str]
    split: NotRequired[str]
    prompt_file: NotRequired[str | None]
    system_prompt_file: NotRequired[str | None]
    env_name: NotRequired[str]
    processor: NotRequired[str]  # remove once processor is refactored
    download_dir: NotRequired[str]
    split_validation_size: NotRequired[float]


# TODO: split this typed dict up so it can be PreferenceDatasetConfig | ResponseDatasetConfig | etc
#       so that we can type check the configs more rigorously as opposed to saying everything
#       is not required.
class DataConfig(TypedDict):
    max_input_seq_length: int
    add_bos: NotRequired[bool]
    add_eos: NotRequired[bool]
    add_generation_prompt: NotRequired[bool]
    add_system_prompt: NotRequired[bool]
    shuffle: bool
    # Number of data loader workers.
    # Set to 8 or 10 for large batches to improve loading speed.
    # This saturates CPU threads without consuming too much memory
    # However, setting it too high might cause memory issues for long seqlens.
    num_workers: NotRequired[int]
    # dataset configs
    prompt_file: NotRequired[str | None]
    system_prompt_file: NotRequired[str | None]
    env_name: NotRequired[str]
    processor: NotRequired[str]  # remove once processor is refactored
    # TODO: remove NotRequired once preference dataset is refactored
    train: NotRequired[ResponseDatasetConfig]
    validation: NotRequired[ResponseDatasetConfig | None]
    # TODO: remove once preference dataset is refactored
    dataset_name: NotRequired[str]
    val_dataset_name: NotRequired[str]
    input_key: NotRequired[str]
    output_key: NotRequired[str | None]
    split: NotRequired[str]
    train_data_path: NotRequired[str]
    val_data_paths: NotRequired[dict[str, str]]


# ===============================================================================
# Eval Dataset Configs
# ===============================================================================
# These configs correspond to the eval datasets in data/datasets/eval_datasets/
# Note: TypedDict doesn't allow narrowing types in child classes, so each config
# is defined independently with common fields repeated.


class MMLUEvalDataConfig(TypedDict):
    """Config for MMLU and multilingual MMLU datasets.

    Supports dataset_name: "mmlu" or "mmlu_{language}" where language is one of:
    AR-XY, BN-BD, DE-DE, EN-US, ES-LA, FR-FR, HI-IN, ID-ID, IT-IT, JA-JP,
    KO-KR, PT-BR, ZH-CN, SW-KE, YO-NG
    """

    max_input_seq_length: int
    dataset_name: Literal[
        "mmlu",
        "mmlu_AR-XY",
        "mmlu_BN-BD",
        "mmlu_DE-DE",
        "mmlu_EN-US",
        "mmlu_ES-LA",
        "mmlu_FR-FR",
        "mmlu_HI-IN",
        "mmlu_ID-ID",
        "mmlu_IT-IT",
        "mmlu_JA-JP",
        "mmlu_KO-KR",
        "mmlu_PT-BR",
        "mmlu_ZH-CN",
        "mmlu_SW-KE",
        "mmlu_YO-NG",
    ]
    prompt_file: NotRequired[str | None]
    system_prompt_file: NotRequired[str | None]


class MMLUProEvalDataConfig(TypedDict):
    """Config for MMLU Pro dataset."""

    max_input_seq_length: int
    dataset_name: Literal["mmlu_pro"]
    prompt_file: NotRequired[str | None]
    system_prompt_file: NotRequired[str | None]


class AIMEEvalDataConfig(TypedDict):
    """Config for AIME datasets."""

    max_input_seq_length: int
    dataset_name: Literal["aime2024", "aime2025"]
    prompt_file: NotRequired[str | None]
    system_prompt_file: NotRequired[str | None]


class GPQAEvalDataConfig(TypedDict):
    """Config for GPQA datasets."""

    max_input_seq_length: int
    dataset_name: Literal["gpqa", "gpqa_diamond"]
    prompt_file: NotRequired[str | None]
    system_prompt_file: NotRequired[str | None]


class MathEvalDataConfig(TypedDict):
    """Config for Math datasets."""

    max_input_seq_length: int
    dataset_name: Literal["math", "math500"]
    prompt_file: NotRequired[str | None]
    system_prompt_file: NotRequired[str | None]


class LocalMathEvalDataConfig(TypedDict):
    """Config for local math datasets loaded from files.

    dataset_name can be a URL or local file path.
    Requires additional fields: problem_key, solution_key, file_format, split.
    """

    max_input_seq_length: int
    dataset_name: str  # URL or file path
    problem_key: str
    solution_key: str
    file_format: Literal["csv", "json"]
    split: NotRequired[str | None]
    prompt_file: NotRequired[str | None]
    system_prompt_file: NotRequired[str | None]


# Union type for all eval dataset configs
EvalDataConfigType = (
    MMLUEvalDataConfig
    | MMLUProEvalDataConfig
    | AIMEEvalDataConfig
    | GPQAEvalDataConfig
    | MathEvalDataConfig
    | LocalMathEvalDataConfig
)
