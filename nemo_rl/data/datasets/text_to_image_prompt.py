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
"""Plain text-to-image prompt dataset for flow-GRPO training.

Supports two formats:

- ``.txt``: one prompt per line, blank lines are ignored.
- ``.jsonl``: each line is a JSON object with the keys ``prompt`` (required),
  ``negative_prompt`` (optional), and ``metadata`` (optional dict).

The collate function produces a ``BatchedDataDict[DiffusionDatumSpec]`` that
the flow-GRPO trainer feeds into ``FlowGRPOPolicy.sample_trajectory``.
"""

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.diffusion.interfaces import DiffusionDatumSpec


class TextToImagePromptDataset(Dataset):
    """Loads prompts from ``.txt`` or ``.jsonl`` into ``DiffusionDatumSpec`` entries."""

    def __init__(
        self,
        path: str | Path,
        *,
        negative_prompt_default: str = " ",
        task_name: str = "text_to_image",
    ) -> None:
        self.path = Path(path)
        self.negative_prompt_default = negative_prompt_default
        self.task_name = task_name

        suffix = self.path.suffix.lower()
        if suffix == ".txt":
            self._records = self._load_txt(self.path)
        elif suffix == ".jsonl":
            self._records = self._load_jsonl(self.path)
        else:
            raise ValueError(
                f"Unsupported file extension {suffix!r} for {self.path}; "
                "use .txt or .jsonl"
            )

    @staticmethod
    def _load_txt(path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            records.append({"prompt": line})
        return records

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" not in obj:
                raise ValueError(f"jsonl entry without 'prompt' key in {path}: {obj!r}")
            records.append(obj)
        return records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> DiffusionDatumSpec:
        rec = self._records[idx]
        datum: DiffusionDatumSpec = {
            "prompt": rec["prompt"],
            "negative_prompt": rec.get("negative_prompt", self.negative_prompt_default),
            "metadata": rec.get("metadata", {}),
            "idx": idx,
            "task_name": self.task_name,
        }
        return datum


def text_to_image_collate_fn(
    batch: list[DiffusionDatumSpec],
) -> BatchedDataDict[DiffusionDatumSpec]:
    """Pack a list of ``DiffusionDatumSpec`` entries into a ``BatchedDataDict``."""
    return BatchedDataDict(
        {
            "prompts": [item["prompt"] for item in batch],
            "negative_prompts": [item.get("negative_prompt", " ") for item in batch],
            "metadata": [item.get("metadata", {}) for item in batch],
            "idx": [item["idx"] for item in batch],
            "task_names": [item.get("task_name", "text_to_image") for item in batch],
        }
    )
