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

"""Stream untruncated training tensors without retaining a second step batch."""

import json
import os
from pathlib import Path
from typing import Any

import torch


class TrainDataDump:
    """Write chunks to a partial file, publishing only a completed optimizer step.

    Sequence columns are trimmed to input_lengths (padding only). Scalar
    columns are written per row as given, so variable-length values such as
    prompt_ids must be passed as jagged tensors rather than padded. Masked rows
    are retained. Values use the legacy train_data JSONL singleton-batch shape.
    A failed step leaves a .partial file, never a completed-looking JSONL file.
    """

    def __init__(self, log_dir: str) -> None:
        self.log_dir = Path(log_dir)
        self.step: int | None = None
        self.rows = 0

    def add_chunk(
        self,
        *,
        step: int,
        sample_ids: list[str],
        tags: list[dict[str, Any]] | None,
        input_lengths: torch.Tensor,
        sequences: dict[str, torch.Tensor],
        scalars: dict[str, torch.Tensor],
    ) -> None:
        if self.step is not None and self.step != step:
            raise RuntimeError("Training dump has an unpublished previous step")
        lengths = input_lengths.detach().cpu().reshape(-1).tolist()
        # unbind() splits dense and jagged columns alike into per-row tensors.
        columns = {
            k: list(v.detach().cpu().unbind())
            for k, v in {**sequences, **scalars}.items()
        }
        if (
            len(lengths) != len(sample_ids)
            or any(len(v) != len(sample_ids) for v in columns.values())
            or (tags is not None and len(tags) != len(sample_ids))
        ):
            raise ValueError("Training dump column lengths do not match sample ids")
        for i, length in enumerate(lengths):
            if length < 0 or any(length > columns[k][i].shape[0] for k in sequences):
                raise ValueError("Training dump input length exceeds a sequence column")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        partial = self.log_dir / f"train_data_step{step + 1}.jsonl.partial"
        mode = "w" if self.step is None else "a"
        self.step = step
        with partial.open(mode) as stream:
            for i, length in enumerate(lengths):
                row = {
                    "idx": self.rows,
                    "step": step + 1,
                    "sample_id": [sample_ids[i]],
                    "input_lengths": [length],
                    "metadata": [tags[i] if tags is not None else {}],
                }
                for key in sequences:
                    row[key] = [columns[key][i][:length].tolist()]
                for key in scalars:
                    row[key] = [columns[key][i].tolist()]
                stream.write(json.dumps(row) + "\n")
                self.rows += 1

    def finish_step(self, step: int) -> None:
        if self.step != step or not self.rows:
            raise RuntimeError("Completed optimizer step has no training dump")
        partial = self.log_dir / f"train_data_step{step + 1}.jsonl.partial"
        os.replace(partial, self.log_dir / f"train_data_step{step + 1}.jsonl")
        self.step = None
        self.rows = 0
