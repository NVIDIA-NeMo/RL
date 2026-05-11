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

from datasets import Dataset, load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


class ArrowTextDataset(RawDataset):
    """Load arrow shards as a stream of raw text strings.

    Args:
        arrow_files: Path or glob to one or more ``.arrow`` files. Forwarded
            to ``datasets.load_dataset("arrow", data_files=...)``.
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
        arrow_files: str | list[str],
        text_key: str = "text",
        characters_per_sample: int | None = None,
        split_validation_size: float = 0.0,
        seed: int = 42,
        **kwargs: Any,
    ):
        self.text_key = text_key
        self.task_name = "x_token"

        raw = load_dataset("arrow", data_files=arrow_files, split="train")

        if characters_per_sample is None or characters_per_sample <= 0:
            self.dataset = raw.map(
                lambda d: {"text": d[text_key], "task_name": self.task_name},
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
                },
            )

        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)


def _pack_generator(
    raw: Dataset,
    text_key: str,
    characters_per_sample: int,
    task_name: str,
) -> Iterable[dict[str, Any]]:
    """Pack consecutive rows until each pack hits ``characters_per_sample``."""
    buf: list[str] = []
    n = 0
    for row in raw:
        text = row[text_key]
        if not isinstance(text, str) or not text:
            continue
        buf.append(text)
        n += len(text)
        if n >= characters_per_sample:
            yield {"text": "\n".join(buf), "task_name": task_name}
            buf = []
            n = 0
    if buf:
        yield {"text": "\n".join(buf), "task_name": task_name}
