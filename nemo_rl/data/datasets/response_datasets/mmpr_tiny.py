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

import fcntl
import os
import re
import shutil
import zipfile
from typing import Any, Optional

import pandas as pd
from datasets import Dataset, Features, Sequence, Value
from huggingface_hub import hf_hub_download

from nemo_rl.data.datasets.raw_dataset import RawDataset

# Matches <image>, <image_1>, <image_2>, etc.
_IMAGE_PLACEHOLDER_RE = re.compile(r"<image(?:_\d+)?>")


def format_mmpr_tiny_dataset(example: dict[str, Any]) -> dict[str, Any]:
    """Format the MMPR-Tiny dataset into an OpenAI-API-like message log.

    Supports multi-image rows by splitting question text on <image> and
    <image_N> placeholders and interleaving image content items with text
    segments. Each placeholder consumes the next image path in order.
    """
    images = example["images"]
    question = str(example["question"])

    # Split question on image placeholders, interleaving text and images
    segments = _IMAGE_PLACEHOLDER_RE.split(question)
    user_content = []
    img_idx = 0
    for i, segment in enumerate(segments):
        text = segment.strip()
        if text:
            user_content.append({"type": "text", "text": text})
        # After each segment (except the last), insert the next image
        if i < len(segments) - 1 and img_idx < len(images):
            user_content.append({"type": "image", "image": images[img_idx]})
            img_idx += 1

    # If no placeholders were found but images exist, prepend the first image
    if img_idx == 0 and images:
        user_content.insert(0, {"type": "image", "image": images[0]})

    assistant_content = str(example["answer"])

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "task_name": example["task_name"],
    }


def _ensure_mmpr_cached(download_dir: str) -> None:
    """Download and extract MMPR-Tiny images if not already cached.

    Thread-safe: uses atomic marker file to prevent race conditions in distributed settings.
    """
    images_dir = os.path.join(download_dir, "MMPR-Tiny", "images")
    parquet_path = os.path.join(download_dir, "mmpr_tiny.parquet")
    ready_marker = os.path.join(download_dir, ".mmpr_ready")

    if os.path.exists(ready_marker):
        return

    if os.path.exists(images_dir) and os.path.exists(parquet_path):
        with open(ready_marker, "w") as f:
            f.write("ready\n")
        return

    lock_file = os.path.join(download_dir, ".mmpr_download.lock")
    os.makedirs(download_dir, exist_ok=True)

    try:
        with open(lock_file, "w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

            if os.path.exists(ready_marker):
                return

            print(f"Downloading MMPR-Tiny to {download_dir}...")

            temp = os.path.join(download_dir, "_temp")

            # Clean up any partial state from a previous interrupted attempt
            shutil.rmtree(temp, ignore_errors=True)
            if os.path.exists(images_dir):
                shutil.rmtree(images_dir)
            if os.path.exists(parquet_path):
                os.remove(parquet_path)

            zip_path = hf_hub_download(
                "OpenGVLab/MMPR-Tiny", "images.zip", repo_type="dataset"
            )
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(temp)
                extracted_images = os.path.join(temp, "images")
                os.makedirs(os.path.dirname(images_dir), exist_ok=True)
                shutil.move(extracted_images, images_dir)
                shutil.rmtree(temp, ignore_errors=True)

            pq = hf_hub_download(
                "OpenGVLab/MMPR-Tiny", "mmpr_tiny.parquet", repo_type="dataset"
            )
            shutil.copy(pq, parquet_path)

            with open(ready_marker, "w") as f:
                f.write("ready\n")
            print(f"MMPR-Tiny cached successfully at {download_dir}")
    finally:
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except OSError:
                pass


def _load_mmpr_tiny_from_cache(download_dir: str) -> Dataset:
    """Load MMPR-Tiny dataset from a preprocessed cache directory."""
    parquet_path = os.path.join(download_dir, "mmpr_tiny.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"MMPR-Tiny parquet file not found at {parquet_path}. "
            "Ensure the cache directory is correct or allow HuggingFace download."
        )

    df = pd.read_parquet(parquet_path)

    df["images"] = df["images"].apply(
        lambda imgs: [os.path.join(download_dir, img["path"]) for img in imgs]
    )
    df["question"] = df["extra_info"].apply(lambda ei: ei.get("question", ""))
    df["answer"] = df["reward_model"].apply(lambda rm: rm.get("ground_truth", ""))
    df = df[["images", "question", "answer"]]

    # Filter out multi-image rows — current model only supports single-image input
    multi_mask = df["images"].apply(lambda imgs: len(imgs) > 1)
    num_multi = multi_mask.sum()
    if num_multi > 0:
        df = df[~multi_mask].reset_index(drop=True)
        print(f"MMPR-Tiny: filtered out {num_multi} multi-image rows, {len(df)} rows remaining")

    features = Features(
        {
            "images": Sequence(Value("string")),
            "question": Value("string"),
            "answer": Value("string"),
        }
    )

    return Dataset.from_pandas(df, preserve_index=False, features=features)


class MMPRTinyDataset(RawDataset):
    """Wrapper around the MMPR-Tiny dataset (OpenGVLab/MMPR-Tiny).

    Supports loading from a local preprocessed cache or downloading from HuggingFace.

    Args:
        split: Dataset split to use. Only "train" is supported (validation is
            created via split_validation_size).
        download_dir: Directory containing the preprocessed MMPR-Tiny cache, or
            the target directory for downloading from HuggingFace.
        split_validation_size: Fraction of data to hold out for validation (default 0).
        seed: Random seed for train/validation split (default 42).
    """

    def __init__(
        self,
        split: str = "train",
        download_dir: str = "",
        split_validation_size: float = 0,
        seed: int = 42,
        **kwargs,
    ):
        if split != "train":
            raise ValueError(
                f"Invalid split: {split}. MMPR-Tiny only supports 'train'. "
                "Use split_validation_size to create a validation split."
            )

        if not download_dir:
            raise ValueError(
                "download_dir is required for MMPR-Tiny dataset. "
                "Set it in the YAML config under data.default.download_dir or data.train.download_dir."
            )

        self.task_name = "mmpr-tiny"
        self.download_dir = download_dir

        _ensure_mmpr_cached(download_dir)

        self.dataset = _load_mmpr_tiny_from_cache(download_dir)
        self.dataset = self.dataset.add_column(
            "task_name", [self.task_name] * len(self.dataset)
        )
        self.val_dataset = None

        self.split_train_validation(split_validation_size, seed)
