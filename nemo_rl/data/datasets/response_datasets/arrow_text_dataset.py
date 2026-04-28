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

"""Arrow Text Dataset for loading arrow files with 'text' column."""

import glob
import hashlib
import json
import os
import time
from typing import Any, Optional

import numpy as np
from datasets import Dataset, load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


class _LazyPackedDataset:
    """Map-style dataset that packs text on demand via precomputed boundaries.

    Each ``__getitem__`` call loads only the raw rows needed for one packed
    sample, joins them with newlines, and returns the messages dict.  The
    underlying arrow dataset stays memory-mapped — no bulk text loading.
    """

    def __init__(
        self,
        arrow_dataset: Dataset,
        pack_ranges: list[tuple[int, int]],
        text_key: str = "text",
    ):
        self.arrow_dataset = arrow_dataset
        self.pack_ranges = pack_ranges
        self.text_key = text_key

    def __len__(self) -> int:
        return len(self.pack_ranges)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        start, end = self.pack_ranges[idx]
        texts = self.arrow_dataset[start:end][self.text_key]
        packed = "\n".join(t for t in texts if isinstance(t, str) and t)
        return {
            "messages": [{"role": "assistant", "content": packed}],
            "task_name": "arrow_text_dataset",
        }


class ArrowTextDataset(RawDataset):
    """Dataset class for loading arrow files containing raw text.

    This class loads arrow files with a 'text' column and converts them to
    the messages format expected by SFT training.

    The text is wrapped as an assistant message:
        {"messages": [{"role": "assistant", "content": <text>}]}

    This format allows training on all tokens (language modeling style).

    Optionally, multiple short samples can be concatenated (packed) into a
    single sample up to ``characters_per_sample`` characters, separated by
    newlines. This avoids padding waste when individual samples are short
    relative to the model's context window. Works correctly with
    cross-tokenizer distillation because packing happens at the raw-text
    level before any tokenizer sees the data.

    Packing uses **lazy loading**: only character lengths are scanned at
    init time (fast) to build pack boundaries; actual text is loaded
    on-demand in ``__getitem__``.

    Args:
        arrow_files: Path pattern (glob) or list of arrow file paths
        val_split: Fraction of data to use for validation (default: 0.05)
        seed: Random seed for train/val split
        text_key: Key for text column in arrow files (default: "text")
        characters_per_sample: If set, concatenate multiple texts (separated
            by "\\n") until the accumulated character count reaches this
            threshold before yielding one packed sample. To guarantee that
            every packed sample tokenizes to at least ``max_input_seq_length``
            tokens (so truncation always fires and every training sequence is
            exactly the context length), use ``max_input_seq_length * 8``
            (≈ 8 chars per token, matching tokenalign_upstream's default
            ``characters_multiplier``). Using a smaller multiplier (e.g. 4)
            may leave some samples shorter than the context window for dense
            text such as code or CJK. Set to None or 0 to disable packing.

    Example config:
        data:
          dataset_name: "arrow_text"
          arrow_files: "/path/to/data/*.arrow"
          val_split: 0.05
          max_input_seq_length: 4096
          characters_per_sample: 32768  # 4096 tokens * 8 chars/token (guarantees full context)
    """

    def __init__(
        self,
        arrow_files: str | list[str],
        val_split: float = 0.05,
        seed: int = 42,
        text_key: str = "text",
        characters_per_sample: Optional[int] = None,
        pack_cache_dir: Optional[str] = None,
        **kwargs,
    ):
        # Don't call super().__init__() since RawDataset raises NotImplementedError
        self.seed = seed
        self.text_key = text_key
        self.task_name = "arrow_text_dataset"

        # Resolve glob pattern if string
        if isinstance(arrow_files, str):
            file_list = glob.glob(arrow_files)
            if not file_list:
                raise ValueError(f"No arrow files found matching pattern: {arrow_files}")
        else:
            file_list = arrow_files

        print(f"Loading {len(file_list)} arrow files...")
        dataset = load_dataset("arrow", data_files=file_list, split="train")
        original_count = len(dataset)
        print(f"  ✓ Loaded {original_count} total samples")

        # Verify text column exists
        if self.text_key not in dataset.column_names:
            raise ValueError(
                f"Column '{self.text_key}' not found in arrow files. "
                f"Available columns: {dataset.column_names}"
            )

        if characters_per_sample is not None and characters_per_sample > 0:
            # Lazy packing: scan character lengths to build pack boundaries,
            # then load+concatenate text on demand in __getitem__.
            #
            # The scan is deterministic in (file set, text_key, chars/sample)
            # and dominates dataset init time on large arrow corpora
            # (~6 minutes for 70M rows). When ``pack_cache_dir`` is set we
            # fingerprint the inputs and store the resulting boundaries as
            # an ``int64[N, 2]`` ``.npy`` file so subsequent runs skip the
            # scan entirely.
            pack_ranges = _load_or_build_pack_ranges(
                dataset=dataset,
                file_list=file_list,
                text_key=text_key,
                characters_per_sample=characters_per_sample,
                pack_cache_dir=pack_cache_dir,
            )

            # Split pack_ranges into train/val
            if val_split > 0:
                import random
                rng = random.Random(seed)
                indices = list(range(len(pack_ranges)))
                rng.shuffle(indices)
                val_count = max(1, int(len(pack_ranges) * val_split))
                val_indices = sorted(indices[:val_count])
                train_indices = sorted(indices[val_count:])
                train_ranges = [pack_ranges[i] for i in train_indices]
                val_ranges = [pack_ranges[i] for i in val_indices]
            else:
                train_ranges = pack_ranges
                val_ranges = pack_ranges[:min(100, len(pack_ranges))]

            train_dataset = _LazyPackedDataset(dataset, train_ranges, text_key)
            val_dataset = _LazyPackedDataset(dataset, val_ranges, text_key)
        else:
            # No packing: convert text to messages format directly
            def text_to_messages(example: dict[str, Any]) -> dict[str, Any]:
                text = example[self.text_key]
                return {
                    "messages": [{"role": "assistant", "content": text}],
                    "task_name": "arrow_text_dataset",
                }

            formatted_dataset = dataset.map(text_to_messages, remove_columns=dataset.column_names)

            if val_split > 0:
                split = formatted_dataset.train_test_split(test_size=val_split, seed=seed)
                train_dataset = split["train"]
                val_dataset = split["test"]
            else:
                train_dataset = formatted_dataset
                val_dataset = formatted_dataset.select(range(min(100, len(formatted_dataset))))

        print(f"  ✓ Train: {len(train_dataset)}, Validation: {len(val_dataset)}")

        self.dataset = train_dataset
        self.val_dataset = val_dataset


def _pack_cache_key(
    file_list: list[str], text_key: str, characters_per_sample: int
) -> str:
    """Fingerprint the inputs that determine the pack boundary list.

    Uses (sorted resolved path, size, mtime-as-int) per arrow file so the
    cache invalidates automatically when any shard is rewritten or when the
    file set changes. ``text_key`` and ``characters_per_sample`` are part of
    the key because changing either yields different boundaries.
    """
    parts = []
    for path in sorted(file_list):
        st = os.stat(path)
        parts.append((path, st.st_size, int(st.st_mtime)))
    blob = json.dumps(
        {
            "files": parts,
            "text_key": text_key,
            "chars_per_sample": int(characters_per_sample),
        },
        sort_keys=True,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _load_or_build_pack_ranges(
    dataset: Dataset,
    file_list: list[str],
    text_key: str,
    characters_per_sample: int,
    pack_cache_dir: Optional[str],
) -> list[tuple[int, int]]:
    """Scan + cache pack boundaries, or load from disk if a cache hit exists."""
    cache_path: Optional[str] = None
    if pack_cache_dir:
        os.makedirs(pack_cache_dir, exist_ok=True)
        key = _pack_cache_key(file_list, text_key, characters_per_sample)
        cache_path = os.path.join(pack_cache_dir, f"{key}.npy")
        if os.path.exists(cache_path):
            print(f"  ↪ Loading cached pack boundaries from {cache_path}")
            arr = np.load(cache_path)
            pack_ranges = [(int(s), int(e)) for s, e in arr]
            print(f"  ✓ Loaded {len(pack_ranges)} pack boundaries (cache hit)")
            return pack_ranges

    print(
        f"  Scanning character lengths for lazy packing (target ~{characters_per_sample} chars)..."
    )
    t0 = time.time()
    pack_ranges = _build_pack_ranges(dataset, text_key, characters_per_sample)
    print(
        f"  ✓ Built {len(pack_ranges)} pack boundaries from {len(dataset)} samples in {time.time() - t0:.1f}s"
    )

    if cache_path is not None:
        # Atomic write so concurrent jobs don't corrupt the cache file.
        # Pass a file handle to np.save (not a path) — np.save auto-appends
        # ".npy" when given a path that doesn't already end in .npy, which
        # breaks the os.replace(tmp -> final) rename below.
        tmp_path = f"{cache_path}.tmp.{os.getpid()}"
        arr = np.asarray(pack_ranges, dtype=np.int64)
        with open(tmp_path, "wb") as f:
            np.save(f, arr)
        os.replace(tmp_path, cache_path)
        print(f"  ↳ Cached pack boundaries to {cache_path}")

    return pack_ranges


def _build_pack_ranges(
    dataset: Dataset,
    text_key: str,
    characters_per_sample: int,
    scan_batch_size: int = 10000,
) -> list[tuple[int, int]]:
    """Scan character lengths in batches and build pack boundary indices.

    Returns a list of (start_row, end_row) tuples. Each packed sample
    covers rows [start_row, end_row) whose combined character length
    (plus newline separators) meets or exceeds ``characters_per_sample``.
    """
    n = len(dataset)
    pack_ranges: list[tuple[int, int]] = []
    start = 0
    accum = 0

    for batch_start in range(0, n, scan_batch_size):
        batch_end = min(batch_start + scan_batch_size, n)
        batch_texts = dataset[batch_start:batch_end][text_key]
        for i, text in enumerate(batch_texts):
            row_idx = batch_start + i
            text_len = len(text) if isinstance(text, str) and text else 0
            if text_len == 0:
                continue
            # +1 for newline separator (except the first text in a pack)
            accum += text_len + (1 if accum > 0 else 0)
            if accum >= characters_per_sample:
                pack_ranges.append((start, row_idx + 1))
                start = row_idx + 1
                accum = 0

    # Flush remaining rows as a partial pack
    if start < n and accum > 0:
        pack_ranges.append((start, n))

    return pack_ranges
