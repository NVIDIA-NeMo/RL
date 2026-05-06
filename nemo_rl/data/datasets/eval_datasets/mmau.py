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

"""MMAU (Massive Multitask Audio Understanding) evaluation dataset."""

import io
from typing import Any

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

from nemo_rl.data.datasets.response_datasets.avqa import _resample_audio
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.processors import vlm_hf_data_processor

DEFAULT_TEMPLATE = (
    "{question} Please choose the answer from the following options: {choices}. "
    "Output the final answer in <answer> </answer>."
)


class MMAUDataset:
    """MMAU evaluation dataset.

    Loads the TwinkStart/MMAU HF dataset and formats each item into the
    messages format expected by vlm_hf_data_processor.

    Args:
        dataset_name: HuggingFace dataset name.
        split: Dataset split to load.
    """

    def __init__(
        self,
        dataset_name: str = "TwinkStart/MMAU",
        split: str = "v05.15.25",
    ):
        ds = load_dataset(dataset_name, split=split)
        ds = ds.cast_column("audio", Audio(decode=False))

        self.rekeyed_ds = ds
        self.task_spec = TaskDataSpec(task_name="mmau")
        self.processor = vlm_hf_data_processor
        self.preprocessor = self.format_data

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert a raw MMAU item into messages format for vlm_hf_data_processor."""
        audio_raw = data["audio"]
        audio_array, orig_sr = sf.read(io.BytesIO(audio_raw["bytes"]))

        # Convert to mono if stereo
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)

        # Resample to 16kHz if needed
        if orig_sr != 16000:
            audio_array = _resample_audio(audio_array, orig_sr, 16000)

        question = data["question"]
        choices = data["choices"]
        answer = data["answer"]

        prompt_text = DEFAULT_TEMPLATE.format(question=question, choices=choices)

        user_content = [
            {"type": "audio", "audio": audio_array},
            {"type": "text", "text": prompt_text},
        ]
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer},
            ],
            "task_name": "mmau",
            "choices": choices,
        }
