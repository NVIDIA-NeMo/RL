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

import os
from typing import Any

import numpy as np
from huggingface_hub import snapshot_download

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.utils import (
    get_huggingface_cache_path,
    load_dataset_from_path,
)


def _load_audio_16k_mono(path: str) -> np.ndarray:
    """Decode an audio file as a 1-D float32 array at 16 kHz mono.

    Daily-Omni ships each clip's audio track as a sibling ``*_audio.wav`` next
    to ``*_video.mp4``. We feed it as an independent ``{type: audio}`` content
    item (mirroring the IntentTrain training path) so the Qwen2.5-Omni chat
    template renders an ``<|AUDIO|>`` placeholder and vLLM populates
    ``multi_modal_data["audio"]``. The benchmark is audio-visual, so video
    frames alone leave audio-dependent questions unanswerable. Uses decord
    (already a project dependency for video decoding) for the same 16 kHz mono
    pipeline the training path uses.
    """
    import decord

    reader = decord.AudioReader(path, sample_rate=16000, mono=True)
    # Shape: (channels, T). With mono=True channels=1; squeeze to (T,).
    audio = reader[:].asnumpy()
    if audio.ndim > 1:
        audio = audio[0]
    return audio.astype(np.float32)


class DailyOmniDataset(RawDataset):
    """Simple wrapper around the Daily-Omni dataset.

    Args:
        split: Split name for the dataset, default is "train"
    """

    task_name = "daily-omni"

    def __init__(
        self,
        split: str = "train",
        split_validation_size: float = 0,
        seed: int = 42,
        **kwargs,
    ):
        # train, valA, and valB are supported splits.
        SPLIT_TO_HF_NAME = {
            "train": "liarliar/Daily-Omni",
        }
        if split not in SPLIT_TO_HF_NAME:
            raise ValueError(f"Invalid split: {split}. Please use 'train'.")

        self.hf_cache_dir = get_huggingface_cache_path(SPLIT_TO_HF_NAME[split])
        if not self.hf_cache_dir:
            # download the dataset
            self.hf_cache_dir = snapshot_download(
                repo_id=SPLIT_TO_HF_NAME[split], repo_type="dataset"
            )
        if not self.hf_cache_dir:
            raise ValueError("Cannot download DailyOmniDataset.")

        json_file = os.path.join(self.hf_cache_dir, "qa.json")

        if not os.path.isfile(json_file):
            raise ValueError(f"{json_file} cannot be found.")

        files_folder = os.path.join(self.hf_cache_dir, "Videos")
        if not os.path.isdir(files_folder):
            # prepare the dataset
            # TODO: move untar, unzip func to utils?
            import tarfile

            archive_filename = os.path.join(self.hf_cache_dir, "Videos.tar")
            if not os.path.isfile(archive_filename):
                raise ValueError(f"{archive_filename} cannot be found.")
            try:
                with tarfile.open(archive_filename, "r:*") as tar:
                    # Extract all contents to the specified path
                    tar.extractall(path=self.hf_cache_dir)
                if os.path.isdir(files_folder):
                    print(
                        f"Successfully extracted '{archive_filename}' to '{files_folder}'"
                    )
                else:
                    raise ValueError(
                        f"Cannot find the extracted folder {files_folder}. Extraction failed."
                    )
            except tarfile.ReadError:
                raise tarfile.ReadError(
                    "Error: Could not read the tar file. It might be corrupted or not a tar file."
                )
            except Exception as e:
                raise Exception(f"An unexpected error occurred: {e}")

        self.dataset = load_dataset_from_path(json_file)

        # format - disable features to avoid schema conflicts
        self.dataset = self.dataset.add_column(
            "task_name", [self.task_name] * len(self.dataset)
        )

        self.preprocessor = self.format_data

        # `self.val_dataset` is used (not None) only when current dataset is used for both training and validation
        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    @classmethod
    def get_prompt(cls, data: dict[str, Any]) -> str:
        # WARNING: model could have preference of a different prompt
        prompt = data["Question"] + "\n" + "\n".join(data["Choice"])
        candidate_answers = [chr(ord("A") + idx) for idx in range(len(data["Choice"]))]
        candidate_answers_all_but_last = ",".join(candidate_answers[:-1])
        prompt += (
            "\n"
            + "Your replies must contain only a single letter "
            + f"(either {candidate_answers_all_but_last} or {candidate_answers[-1]})."
        )
        return prompt

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        video_dir = os.path.join(self.hf_cache_dir, "Videos", data["video_id"])
        video_path = os.path.join(video_dir, data["video_id"] + "_video.mp4")
        audio_path = os.path.join(video_dir, data["video_id"] + "_audio.wav")
        # Audio + video flow as two independent content items so the
        # Qwen2.5-Omni chat template renders both <|VIDEO|> and <|AUDIO|>
        # placeholders (Daily-Omni is an audio-visual benchmark).
        user_content = [
            {"type": "video", "video": video_path},
            {"type": "audio", "audio": _load_audio_16k_mono(audio_path)},
            {"type": "text", "text": self.get_prompt(data)},
        ]
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": data["Answer"]},
            ],
            "task_name": self.task_name,
        }
