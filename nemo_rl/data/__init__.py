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

from typing import NotRequired, TypedDict


class ResponseDatasetConfig(TypedDict):
    dataset_name: NotRequired[str]
    data_path: NotRequired[str]
    input_key: NotRequired[str]
    output_key: NotRequired[str]
    subset: NotRequired[str | None]
    split: NotRequired[str]
    prompt_file: NotRequired[str | None]
    system_prompt_file: NotRequired[str | None]
    env_name: NotRequired[str]
    processor: NotRequired[str]  # remove once processor is refactored
    download_dir: NotRequired[str]
    # Size of the validation data
    split_validation_size: NotRequired[float]
    # Seed for train/validation split when split_validation_size > 0
    seed: NotRequired[int]


class PreferenceDatasetConfig(TypedDict):
    dataset_name: NotRequired[str]
    data_path: NotRequired[str]
    prompt_key: NotRequired[str]
    chosen_key: NotRequired[str]
    rejected_key: NotRequired[str]
    subset: NotRequired[str | None]
    split: NotRequired[str]
    prompt_file: NotRequired[str | None]
    system_prompt_file: NotRequired[str | None]


class DataConfig(TypedDict):
    max_input_seq_length: int | None
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
    # multiple dataloader configs
    # currently only supported for GRPO
    use_multiple_dataloader: NotRequired[bool]
    num_prompts_per_dataloader: NotRequired[int]
    custom_dataloader: NotRequired[str]
    # dataset configs
    train: ResponseDatasetConfig | PreferenceDatasetConfig | list[ResponseDatasetConfig]
    validation: NotRequired[
        ResponseDatasetConfig
        | PreferenceDatasetConfig
        | list[ResponseDatasetConfig]
        | None
    ]
    # default settings for all datasets, will be overridden by dataset-specific settings
    default: NotRequired[ResponseDatasetConfig | PreferenceDatasetConfig | None]


class EvalDataConfig(ResponseDatasetConfig):
    """Response-dataset configuration extended with eval-only settings.

    Kept as a ``TypedDict`` (v1) because it extends the still-v1
    ``ResponseDatasetConfig``; migrate both to ``BaseModel`` together.

    Fields:
        max_input_seq_length: Max prompt length passed to the generation backend.
        repeat: Number of dataset copies to evaluate. AIME defaults to one;
            training recipes that need repeated validation set a higher value.
        include_single_letter_instruction: Daily-Omni only. Set to false so the
            eval ``prompt_file`` dictates answer formatting instead of the
            training-only single-letter instruction.
    """

    max_input_seq_length: int
    repeat: NotRequired[int]
    include_single_letter_instruction: NotRequired[bool]


# Backward-compatible public type name. Eval now uses the response dataset schema.
EvalDataConfigType = EvalDataConfig
