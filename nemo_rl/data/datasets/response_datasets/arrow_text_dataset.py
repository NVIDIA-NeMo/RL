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

import glob
import hashlib
import os
from typing import Any

from datasets import Dataset, concatenate_datasets

from nemo_rl.data.datasets.raw_dataset import RawDataset


class ArrowTextDataset(RawDataset):
    """Load ``.arrow`` files with a raw-text column for SFT training.

    Each row is wrapped as a single assistant turn
    (``{"messages": [{"role": "assistant", "content": text}]}``) so training
    runs language-modeling style on the whole sequence.

    Arrow files are memory-mapped via ``Dataset.from_file`` and concatenated
    without materializing/rewriting a HuggingFace cache, which is significantly
    faster than ``load_dataset("arrow", ...)`` for large glob patterns.

    Args:
        arrow_files: Glob pattern or explicit list of ``.arrow`` paths.
        text_key: Column in the arrow files that holds the text (default ``"text"``).
        split_validation_size: Fraction of rows reserved for validation
            (default 0.05; 0 means no split).
        seed: Seed for the train/validation split, default is 42.
        num_proc: Worker processes for the ``messages`` rewrite ``.map`` pass.
            Default is ``None`` (single process); pass e.g. ``os.cpu_count()``
            to parallelize across all cores.
        cache_dir: Directory to write the ``.map`` output cache. Required when
            the source arrow files live on a read-only filesystem, because HF
            otherwise derives ``.map``'s temp-file location from the source
            path. Cache filename is derived from a deterministic hash of
            ``(file_list, text_key, task_name)`` so subsequent runs with the
            same inputs hit the cache.
    """

    def __init__(
        self,
        arrow_files: str | list[str],
        text_key: str = "text",
        split_validation_size: float = 0.05,
        seed: int = 42,
        num_proc: int | None = None,
        cache_dir: str | None = None,
        **kwargs,
    ) -> None:
        self.task_name = "arrow_text"
        self.text_key = text_key

        # resolve glob pattern to a concrete file list
        if isinstance(arrow_files, str):
            file_list = sorted(glob.glob(arrow_files))
            if not file_list:
                raise ValueError(
                    f"No arrow files found matching pattern: {arrow_files}"
                )
        else:
            file_list = list(arrow_files)

        # memory-map each arrow file individually and concatenate (O(num_files),
        # no cache rewrite) instead of load_dataset("arrow", ...) which scans
        # and re-serializes every row into the HF cache.
        shards = [Dataset.from_file(p) for p in file_list]
        self.dataset = concatenate_datasets(shards) if len(shards) > 1 else shards[0]

        if self.text_key not in self.dataset.column_names:
            raise ValueError(
                f"Column '{self.text_key}' not found in arrow files. "
                f"Available columns: {self.dataset.column_names}"
            )
        print(f"Loaded {len(self.dataset)} samples from {len(file_list)} arrow files.")

        # format the dataset (parallel + batched for large corpora)
        map_kwargs: dict[str, Any] = {
            "batched": True,
            "remove_columns": self.dataset.column_names,
        }
        if num_proc is not None and num_proc > 1:
            map_kwargs["num_proc"] = num_proc
        if cache_dir is not None:
            # Route the .map() output to a writable directory. HF would
            # otherwise derive the cache path from the source arrow file's
            # directory, which may be read-only (e.g. another user's data).
            os.makedirs(cache_dir, exist_ok=True)
            fingerprint = hashlib.md5(
                ("|".join(file_list) + f"|{text_key}|{self.task_name}").encode()
            ).hexdigest()[:16]
            map_kwargs["cache_file_name"] = os.path.join(
                cache_dir, f"arrow_text_{fingerprint}.arrow"
            )
        self.dataset = self.dataset.map(self.format_batch, **map_kwargs)

        # `self.val_dataset` is used only when current dataset is used for both training and validation
        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    def format_batch(self, batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        texts = batch[self.text_key]
        return {
            "messages": [[{"role": "assistant", "content": t}] for t in texts],
            "task_name": [self.task_name] * len(texts),
        }
