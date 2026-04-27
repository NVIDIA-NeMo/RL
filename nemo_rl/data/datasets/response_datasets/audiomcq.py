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
from math import gcd
from typing import Any

import numpy as np
import soundfile as sf
from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
from scipy.signal import resample_poly

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.utils import get_huggingface_cache_path

DEFAULT_TEMPLATE = (
    "{question} Please choose the answer from the following options: {choices}. "
    "Output the final answer in <answer> </answer>."
)

AUDIOMCQ_REPO_ID = "Harland/AudioMCQ-StrongAC-GeminiCoT"
AUDIOMCQ_MANIFEST = "data.jsonl"
TARGET_SAMPLE_RATE = 16000
STRONG_AC_VALUE = "strong"


def _resample_audio(
    audio_array: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    if not isinstance(audio_array, np.ndarray):
        audio_array = np.array(audio_array, dtype=np.float64)
    else:
        audio_array = audio_array.astype(np.float64)
    g = gcd(int(orig_sr), int(target_sr))
    return resample_poly(audio_array, target_sr // g, orig_sr // g)


def _resolve_snapshot_root() -> str:
    cached = get_huggingface_cache_path(AUDIOMCQ_REPO_ID)
    if cached:
        return cached
    return snapshot_download(repo_id=AUDIOMCQ_REPO_ID, repo_type="dataset")


class AudioMCQDataset(RawDataset):
    """Wrapper around the Harland/AudioMCQ-StrongAC-GeminiCoT dataset.

    The upstream dataset is already filtered to the StrongAC subset of AudioMCQ
    and additionally restricted to samples whose Gemini chain-of-thought
    annotations passed quality review. Each row contains a relative ``audio_path``
    pointing to a ``.wav`` or ``.mp3`` file shipped inline in the dataset
    snapshot, plus a four-item ``choices`` list and a free-text ``answer``.
    """

    task_name = "audiomcq"

    def __init__(
        self,
        split: str = "train",
        split_validation_size: float = 0,
        seed: int = 42,
        max_samples: int | None = None,
        populate_val_dataset: bool = True,
        **kwargs,
    ):
        """Construct the wrapper.

        Args:
            split: ``"train"`` or ``"validation"``. The upstream manifest only
                ships a native train split; both values share the same
                shuffled+filtered base dataset and partition deterministically
                from ``split_validation_size`` and ``seed``.
            split_validation_size: Fraction (or absolute count) of held-out
                rows. Required when ``split == "validation"``.
            seed: Shuffle and partition seed. Use the same value across the
                train and validation entries to share the same partition.
            max_samples: Optional cap, applied after the defensive
                ``audio-contribution`` filter and the deterministic shuffle.
            populate_val_dataset: For ``split == "train"``: when ``True``
                (the default), populate ``self.val_dataset`` with the
                held-out slice (the convention used by AVQA and the
                ``setup_response_data`` train-only path). Set to ``False``
                when the YAML configures an explicit ``data.validation``
                entry — the train slice still excludes the held-out rows
                (no leakage), but ``self.val_dataset`` stays ``None`` so
                ``setup_response_data`` does not double-count the same rows.
        """
        valid_splits = ("train", "validation")
        if split not in valid_splits:
            raise ValueError(
                f"Invalid split: {split}. Please use one of {valid_splits}."
            )

        self.snapshot_root = _resolve_snapshot_root()
        manifest_path = os.path.join(self.snapshot_root, AUDIOMCQ_MANIFEST)
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                f"AudioMCQ manifest not found at {manifest_path}. "
                f"Expected the snapshot of {AUDIOMCQ_REPO_ID} to contain "
                f"{AUDIOMCQ_MANIFEST} at its root."
            )

        # The upstream dataset has only a native 'train' split; we always
        # load the full manifest and synthesize a validation slice from it
        # when the caller requests one.
        ds = load_dataset("json", data_files=manifest_path, split="train")

        # Defensive filter: the dataset is already pre-filtered to the StrongAC
        # subset, but if the upstream schema ever ships an audio-contribution
        # column we keep only rows whose value is "strong".
        if "audio-contribution" in ds.column_names:
            ds = ds.filter(lambda ex: ex["audio-contribution"] == STRONG_AC_VALUE)

        ds = ds.shuffle(seed=seed)
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))

        self._eager_audio_probe(ds)

        ds = ds.add_column("task_name", [self.task_name] * len(ds))

        if split == "validation":
            if split_validation_size <= 0:
                raise ValueError(
                    "AudioMCQDataset(split='validation') requires "
                    "split_validation_size > 0; the upstream dataset has no "
                    "native validation split."
                )
            split_ds = ds.train_test_split(test_size=split_validation_size, seed=seed)
            self.dataset = split_ds["test"]
            self.preprocessor = self.format_data
            self.val_dataset = None
        else:
            self.preprocessor = self.format_data
            self.val_dataset = None
            if populate_val_dataset:
                # Train-and-validate-from-train convention: keep the train
                # slice on self.dataset and expose the held-out slice via
                # self.val_dataset. setup_response_data will pick it up.
                self.dataset = ds
                self.split_train_validation(split_validation_size, seed)
            else:
                # Caller is loading validation through a separate
                # data.validation entry; drop the held-out rows from train
                # so the two splits remain disjoint, but leave val_dataset
                # unset so setup_response_data does not duplicate the rows
                # the validation entry already contributes.
                if split_validation_size > 0:
                    split_ds = ds.train_test_split(
                        test_size=split_validation_size, seed=seed
                    )
                    self.dataset = split_ds["train"]
                else:
                    self.dataset = ds

    def _eager_audio_probe(self, ds: Dataset) -> None:
        """Verify the first row's audio file exists under the snapshot root.

        Catches missing audio archives at construction time so doomed runs do
        not boot Ray actors, vLLM, and Megatron before failing.
        """
        if len(ds) == 0:
            return
        head = ds[0]
        head_path = os.path.join(self.snapshot_root, head["audio_path"])
        if not os.path.isfile(head_path):
            source = head.get("source_dataset", "<unknown>")
            raise RuntimeError(
                f"AudioMCQ eager asset probe failed: audio file for the head "
                f"sample is missing. source_dataset={source!r} "
                f"audio_path={head['audio_path']!r} "
                f"snapshot_root={self.snapshot_root!r}. "
                f"Please re-run snapshot_download or verify the dataset "
                f"snapshot is complete."
            )

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        audio_path = data["audio_path"]
        absolute_path = os.path.join(self.snapshot_root, audio_path)
        if not os.path.isfile(absolute_path):
            source = data.get("source_dataset", "<unknown>")
            raise FileNotFoundError(
                f"AudioMCQ audio missing at {absolute_path} "
                f"(source_dataset={source!r}, audio_path={audio_path!r})."
            )

        audio_array, orig_sr = sf.read(absolute_path)

        # Mono downmix for multi-channel waveforms before resampling.
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)

        if orig_sr != TARGET_SAMPLE_RATE:
            audio_array = _resample_audio(audio_array, orig_sr, TARGET_SAMPLE_RATE)
        else:
            audio_array = audio_array.astype(np.float64)

        prompt_text = DEFAULT_TEMPLATE.format(
            question=data["question"], choices=data["choices"]
        )

        user_content = [
            {"type": "audio", "audio": audio_array},
            {"type": "text", "text": prompt_text},
        ]
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": data["answer"]},
            ],
            "task_name": self.task_name,
            "choices": data["choices"],
        }
