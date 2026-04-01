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

import re
from typing import Any

import numpy as np
import torch
import torchaudio
from datasets import Dataset, load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset

DEFAULT_TEMPLATE = (
    "{question} Please choose the answer from the following options: {choices}. "
    "Output the final answer in <answer> </answer>."
)


def _resample_audio(audio_array, orig_sr, target_sr=16000):
    """Resample audio to target sample rate."""
    if isinstance(audio_array, np.ndarray):
        waveform = torch.from_numpy(audio_array).float()
    else:
        waveform = audio_array.float()

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    resampled = resampler(waveform)
    return resampled[0].numpy()


def _parse_question(question_text):
    r"""Parse the HF dataset question format.

    Input: "How many animals are there in the video?\nChoices:\nA. 3\nB. One\nC. 4\nD. 2"
    Returns: (question, choices_list)
    """
    parts = question_text.split("\nChoices:\n")
    if len(parts) == 2:
        question = parts[0]
        choices = []
        for line in parts[1].strip().split("\n"):
            line = line.strip()
            if line:
                match = re.match(r"^[A-Z]\.\s*(.+)$", line)
                choices.append(match.group(1) if match else line)
        return question, choices
    return question_text, []


class AVQADataset(RawDataset):
    """Wrapper around the AVQA (Audio-Visual Question Answering) dataset.

    Formats audio samples into OpenAI-style messages for audio QA
    fine-tuning with Qwen2.5-Omni.

    Args:
        split: Split name for the dataset. Supported: "train", "validation".
        max_samples: Maximum number of samples to load.
        seed: Random seed for splitting the dataset.
        split_validation_size: Size of the validation set.
    """

    task_name = "avqa"

    def __init__(
        self,
        split: str = "train",
        split_validation_size: float = 0,
        seed: int = 42,
        max_samples: int | None = None,
        **kwargs,
    ):
        VALID_SPLITS = ("train", "validation")
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split: {split}. Please use one of {VALID_SPLITS}."
            )

        if max_samples is not None:
            ds = load_dataset("gijs/avqa-processed", split=split, streaming=True)
            self.dataset = Dataset.from_list(list(ds.take(max_samples)))
        else:
            self.dataset = load_dataset("gijs/avqa-processed", split=split)

        self.dataset = self.dataset.add_column(
            "task_name", [self.task_name] * len(self.dataset)
        )

        self.preprocessor = self.format_data

        # `self.val_dataset` is used (not None) only when current dataset is used for both training and validation
        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        audio_data = data["audio"]
        audio_array = audio_data["array"]

        # Resample to 16kHz if needed
        if audio_data["sampling_rate"] != 16000:
            audio_array = _resample_audio(
                audio_array, audio_data["sampling_rate"], 16000
            )

        # Parse question and build prompt
        question, choices = _parse_question(data["question"])
        question = question.replace("video", "audio")

        prompt_text = DEFAULT_TEMPLATE.format(question=question, choices=choices)

        # Strip letter prefix from answer (e.g., "B. Yacht consignment" -> "Yacht consignment")
        answer = data["answer"]
        answer_match = re.match(r"^[A-Z]\.\s*(.+)$", answer)
        if answer_match:
            answer = answer_match.group(1)

        user_content = [
            {"type": "audio", "audio": audio_array},
            {"type": "text", "text": prompt_text},
        ]
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer},
            ],
            "task_name": self.task_name,
        }
