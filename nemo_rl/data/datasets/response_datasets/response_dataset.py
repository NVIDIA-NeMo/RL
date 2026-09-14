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

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Optional

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.utils import load_dataset_from_path


_SOURCE_ID_KEYS = ("sample_id", "id", "uuid")


def resolve_source_sample_id(
    data: Mapping[str, Any],
    raw_ordinal: int,
    *,
    data_path: str,
    subset: str | None,
    split: str | None,
) -> str:
    """Resolve a durable identity before dataset transformations.

    Explicit source identities are retained verbatim (after conversion to a
    string). Otherwise, the identity combines a collision-resistant digest of
    the complete dataset namespace with the original row ordinal. JSON
    serialization keeps path/subset/split boundaries unambiguous.

    Args:
        data: Unmodified source row.
        raw_ordinal: Row position in the source before filtering or splitting.
        data_path: Local path, shard/glob, URL, or Hugging Face dataset name.
        subset: Optional Hugging Face subset/config name.
        split: Requested source split. ``None`` has the loader's effective
            default of ``"train"``.

    Returns:
        The upstream identity, when present, or a deterministic synthesized ID.
    """
    for key in _SOURCE_ID_KEYS:
        value = data.get(key)
        if value is not None:
            source_id = str(value)
            if source_id.strip():
                return source_id

    namespace = json.dumps(
        {
            "data_path": data_path,
            "split": split or "train",
            "subset": subset,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    namespace_digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
    return f"source-row:{namespace_digest}:{raw_ordinal}"


def attach_source_sample_id(
    data: Mapping[str, Any],
    raw_ordinal: int,
    *,
    data_path: str,
    subset: str | None,
    split: str | None,
) -> dict[str, str]:
    """Build the ``Dataset.map`` update that attaches source identity."""
    return {
        "sample_id": resolve_source_sample_id(
            data,
            raw_ordinal,
            data_path=data_path,
            subset=subset,
            split=split,
        )
    }


class ResponseDataset(RawDataset):
    """Dataset class for response data which can be loaded from a JSON file.

    This class handles loading of response data for SFT and RL training.
    The input JSONL files should contain valid JSON objects formatted like this:
    {
        input_key: str,     # The input prompt/context
        output_key: str,    # The output response/answer
    }
    Please refer to https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/sft.md#datasets for more details.

    Args:
        data_path: Path to the dataset JSON file
        input_key: Key for the input text, default is "input"
        output_key: Key for the output text, default is "output"
        subset: Optional subset name for the dataset, used for HuggingFace datasets
        split: Optional split name for the dataset, used for HuggingFace datasets
        split_validation_size: Size of the validation data, default is 0
        seed: Seed for train/validation split when split_validation_size > 0, default is 42
    """

    def __init__(
        self,
        data_path: str,
        input_key: str = "input",
        output_key: str = "output",
        subset: Optional[str] = None,
        split: Optional[str] = None,
        split_validation_size: float = 0,
        seed: int = 42,
        **kwargs,
    ):
        self.input_key = input_key
        self.output_key = output_key

        self.task_name = "-".join(data_path.split("/")[-2:]).split(".")[0]
        if self.task_name[0] == "-":
            self.task_name = self.task_name[1:]

        # load from local or huggingface
        self.dataset = load_dataset_from_path(data_path, subset, split)

        # Attach identity to untouched source rows. Formatting removes source
        # columns and validation splitting reorders/selects rows, so neither
        # operation may be allowed to invent identity afterward.
        self.dataset = self.dataset.map(
            attach_source_sample_id,
            with_indices=True,
            remove_columns=["sample_id"]
            if "sample_id" in self.dataset.column_names
            else None,
            fn_kwargs={
                "data_path": data_path,
                "subset": subset,
                "split": split,
            },
        )

        # format the dataset
        if "messages" not in self.dataset.column_names:
            self.dataset = self.dataset.map(
                self.format_data,
                remove_columns=self.dataset.column_names,
            )
        else:
            self.dataset = self.dataset.add_column(
                "task_name", [self.task_name] * len(self.dataset)
            )

        # `self.val_dataset` is used (not None) only when current dataset is used for both training and validation
        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": data[self.input_key]},
                {"role": "assistant", "content": data[self.output_key]},
            ],
            "task_name": self.task_name,
            "sample_id": data["sample_id"],
        }
