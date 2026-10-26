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

"""Per-step sample table: one row of ids and summaries per trained sample.

The advantage stage summarizes each chunk from the tensors it already holds and
returns the columns in its result; the controller concatenates a step's chunks
and logs the table with the step's metrics. See docs/design-docs/sample-table.md.
"""

from typing import Any

import torch
from pydantic import BaseModel

from nemo_rl.utils.train_data_dump import json_safe_nonfinite

# One chunk's summary: column name -> one value per sample.
TableChunk = dict[str, list[Any]]

# Row tags copied into the table; rollout_id is the key that joins it to Gym logs.
_TAG_COLUMNS = ("group_id", "rollout_id", "weight_version")


class SampleTableConfig(BaseModel, extra="forbid"):
    # Log a per-step table with one row per trained sample (ids, scalars, and
    # response-token means of logprobs and advantages) to W&B and/or NeMo Lens.
    enabled: bool = False
    # Table name under the train/ prefix: the W&B panel and the Lens event name.
    name: str = "samples"


def _dense(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().cpu()
    return torch.nested.to_padded_tensor(t, 0.0) if t.is_nested else t


def summarize_chunk(
    *,
    sample_ids: list[str],
    tags: list[dict[str, Any]] | None,
    input_lengths: torch.Tensor,
    token_mask: torch.Tensor,
    sequences: dict[str, torch.Tensor],
    scalars: dict[str, torch.Tensor],
) -> TableChunk:
    """Summarize a chunk as columns of one value per sample.

    ``response_tokens`` counts ``token_mask == 1`` within each row's length, and
    ``mean_<name>`` averages each floating-point sequence over those positions,
    skipping NaN and inf; it is None when nothing is left. Integer sequences and
    columns with more than one value per sample are not summarized.
    """
    lengths = input_lengths.detach().cpu().reshape(-1)
    mask = _dense(token_mask)
    positions = torch.arange(mask.shape[-1]).unsqueeze(0)
    keep = (mask == 1) & (positions < lengths.unsqueeze(1))
    row_tags = tags if tags is not None else [{}] * len(sample_ids)

    columns: TableChunk = {"sample_id": list(sample_ids)}
    columns.update({key: [t.get(key) for t in row_tags] for key in _TAG_COLUMNS})
    columns["input_lengths"] = lengths.tolist()
    columns["response_tokens"] = keep.sum(-1).tolist()
    for name, sequence in sequences.items():
        values = _dense(sequence)
        if not values.is_floating_point():
            continue
        width = min(values.shape[-1], keep.shape[-1])
        valid = keep[:, :width] & torch.isfinite(values[:, :width])
        total = torch.where(valid, values[:, :width], 0.0).sum(-1).double()
        # 0/0 is NaN, which json_safe_nonfinite turns into None for empty rows.
        columns[f"mean_{name}"] = json_safe_nonfinite((total / valid.sum(-1)).tolist())
    for name, column in scalars.items():
        column = column.detach().cpu()
        if not column.is_nested and column.dim() == 1:
            columns[name] = json_safe_nonfinite(column.tolist())
    return columns


def merge_table_chunks(chunks: list[TableChunk]) -> tuple[list[str], list[list[Any]]]:
    """Concatenate a step's chunks, in arrival order, into table columns and rows."""
    if not chunks:
        return [], []
    columns = list(chunks[0])
    rows = [list(row) for chunk in chunks for row in zip(*(chunk[c] for c in columns))]
    return columns, rows
