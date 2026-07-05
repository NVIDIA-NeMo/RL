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

"""MMAU (Massive Multitask Audio Understanding) response dataset."""

import io
from typing import Any

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.response_datasets.avqa import _resample_audio

MMAU_DATASET_NAME = "TwinkStart/MMAU"
DEFAULT_TEMPLATE = (
    "{question} Please choose the answer from the following options: {choices}. "
    "Output the final answer in <answer> </answer>."
)


class MMAUDataset(RawDataset):
    task_name = "mmau"
    default_processor = "vlm_hf_data_processor"
    is_multimodal = True

    def __init__(self, split: str = "v05.15.25", **kwargs):
        dataset = load_dataset(MMAU_DATASET_NAME, split=split)
        dataset = dataset.cast_column("audio", Audio(decode=False))
        self.dataset = dataset.add_column("task_name", [self.task_name] * len(dataset))
        self.val_dataset = None
        self.preprocessor = self.format_data

    def format_data(self, datum_dict: dict[str, Any]) -> dict[str, Any]:
        audio_raw = datum_dict["audio"]
        audio_array, original_sample_rate = sf.read(io.BytesIO(audio_raw["bytes"]))

        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)
        if original_sample_rate != 16000:
            audio_array = _resample_audio(audio_array, original_sample_rate, 16000)

        choices = datum_dict["choices"]
        prompt_text = DEFAULT_TEMPLATE.format(
            question=datum_dict["question"], choices=choices
        )
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_array},
                        {"type": "text", "text": prompt_text},
                    ],
                },
                {"role": "assistant", "content": datum_dict["answer"]},
            ],
            "task_name": self.task_name,
            "choices": choices,
        }
