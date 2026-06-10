# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Arrow-shard raw-text dataset for cross-tokenizer distillation.

A minimal dataset that loads an arrow file (or a glob of arrow files), takes
one column of raw text, and optionally packs consecutive rows together into
larger samples by character count. Tokenization is intentionally NOT done
here — the cross-tokenizer collator tokenizes both student and teacher
copies of each text on the fly.
"""

from __future__ import annotations

from typing import Any, Iterable

from datasets import Dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.utils import load_dataset_from_path


class ArrowTextDataset(RawDataset):
    """Load a stream of raw text strings for cross-tokenizer distillation.

    The source is resolved by ``load_dataset_from_path``, which infers the
    loader from ``data_files``: a ``.arrow`` / ``.parquet`` / ``.json`` /
    ``.txt`` path (local, glob, or HTTP/``hf://`` URL) uses the matching
    file-format builder, while a bare HuggingFace dataset id (no extension)
    is loaded by name. This lets the same recipe run on a packaged HF dataset
    by default and on user-supplied ``.arrow`` files via a single override.

    Args:
        data_files: Path, glob, or URL to a data file (``.arrow``/``.parquet``
            /``.json``/``.txt``), or a HuggingFace dataset id to load by name.
            A single string (globs allowed); not a list.
        subset: HuggingFace config/subset name. Only valid when ``data_files``
            is a dataset id (not a file path); selects the config for datasets
            that define multiple.
        split: Split to load (default ``"train"``).
        text_key: Column on the loaded dataset that contains the raw text.
        characters_per_sample: If set, pack consecutive rows together until
            the running character count reaches this threshold; emit a packed
            sample and start a fresh one. If ``None``, every input row is
            one sample.
        split_validation_size: Optional held-out fraction.
        seed: Seed for the train/validation split.
    """

    def __init__(
        self,
        data_files: str,
        subset: str | None = None,
        split: str = "train",
        text_key: str = "text",
        characters_per_sample: int | None = None,
        split_validation_size: float = 0.0,
        seed: int = 42,
        **kwargs: Any,
    ):
        self.text_key = text_key
        self.task_name = "x_token"

        raw = load_dataset_from_path(data_files, subset, split)
        # Filter at the source so the packed and non-packed branches see the
        # same corpus (the packed path also drops empty/non-string rows).
        raw = raw.filter(lambda d: isinstance(d[text_key], str) and bool(d[text_key]))

        if characters_per_sample is None or characters_per_sample <= 0:
            self.dataset = raw.map(
                self.format_data,
                remove_columns=raw.column_names,
            )
        else:
            self.dataset = Dataset.from_generator(
                _pack_generator,
                gen_kwargs={
                    "raw": raw,
                    "text_key": text_key,
                    "characters_per_sample": characters_per_sample,
                    "task_name": self.task_name,
                    # Part of the HF datasets cache fingerprint; bump it
                    # whenever the emitted row schema changes.
                    "schema_version": "messages-v1",
                },
            )

        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        # `text` feeds kd_data_processor; the assistant `messages` field feeds
        # sft_processor, so the same dataset serves both pipelines.
        text = data[self.text_key]
        return {
            "messages": [{"role": "assistant", "content": text}],
            "task_name": self.task_name,
        }


def _pack_generator(
    raw: Dataset,
    text_key: str,
    characters_per_sample: int,
    task_name: str,
    schema_version: str = "messages-v1",
) -> Iterable[dict[str, Any]]:
    """Pack consecutive rows until each pack hits ``characters_per_sample``.

    ``schema_version`` is accepted only so that HF datasets includes it in
    the ``from_generator`` cache fingerprint. Bump the value in the caller
    when the emitted row schema changes.
    """
    del schema_version
    buf: list[str] = []
    n = 0
    for row in raw:
        text = row[text_key]
        buf.append(text)
        n += len(text)
        if n >= characters_per_sample:
            packed = "\n".join(buf)
            yield {
                "messages": [{"role": "assistant", "content": packed}],
                "task_name": task_name,
            }
            buf = []
            n = 0
    if buf:
        packed = "\n".join(buf)
        yield {
            "messages": [{"role": "assistant", "content": packed}],
            "task_name": task_name,
        }
