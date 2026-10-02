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

"""Write training-data chunks to Parquet from the SingleController process."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, cast
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tensordict import TensorDict

from nemo_rl.data_plane.interfaces import KVBatchMeta
from nemo_rl.data_plane.schema import MASK_SAMPLE, TRUNCATED

TRAJECTORY_SCHEMA_VERSION = "nemorl-trajectories/1"

ROW_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("step", pa.int64()),
        ("attempt_start", pa.timestamp("us", tz="UTC")),
        ("chunk_index", pa.int32()),
        ("sample_id", pa.string()),
        ("prompt_idx", pa.int64()),
        ("weight_version", pa.int64()),
        ("reward", pa.float64()),
        ("final_sample_mask", pa.float64()),
        ("mask_sample", pa.bool_()),
        ("truncated", pa.bool_()),
        ("advantages", pa.list_(pa.float32())),
        ("values", pa.list_(pa.float32())),
        ("returns", pa.list_(pa.float32())),
        ("input_ids", pa.list_(pa.int32())),
        ("token_mask", pa.list_(pa.int8())),
        ("generation_logprobs", pa.list_(pa.float32())),
        ("prev_logprobs", pa.list_(pa.float32())),
    ]
)


def _scalar_column(value: Optional[torch.Tensor], n: int) -> np.ndarray | list[None]:
    """Return one scalar per sample, or nulls for a missing column."""
    return [None] * n if value is None else value.detach().cpu().reshape(n).numpy()


def _token_column(
    name: str,
    value: Optional[torch.Tensor],
    lengths: np.ndarray,
) -> pa.Array:
    """Turn batched tokens into one Arrow list value per rollout row."""
    n = len(lengths)
    list_type = cast(pa.ListType, ROW_SCHEMA.field(name).type)
    value_type = list_type.value_type
    if value is None:
        return pa.nulls(n, list_type)

    tensor = value.detach().cpu()
    if tensor.dtype in (torch.bfloat16, torch.float16):
        tensor = tensor.float()
    if tensor.is_nested:
        rows = tensor.unbind()
    elif tensor.dim() == 1:
        rows = (tensor[i : i + 1] for i in range(n))
    else:
        rows = (tensor[i, : int(lengths[i])] for i in range(n))

    parts = [row.reshape(-1).numpy() for row in rows]
    offsets = np.empty(n + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum([len(part) for part in parts], out=offsets[1:])
    dtype = np.dtype(str(value_type))
    values = (
        np.concatenate(parts).astype(dtype, copy=False)
        if parts
        else np.empty(0, dtype=dtype)
    )
    if name == "token_mask":
        values = (values > 0).astype(np.int8)
    return pa.ListArray.from_arrays(
        pa.array(offsets), pa.array(values, type=value_type)
    )


class TrajectoryLogWriter:
    """Write Arrow columns directly from batched training tensors."""

    FETCH_FIELDS = (
        "input_ids",
        "input_lengths",
        "token_mask",
        "generation_logprobs",
        "total_reward",
        MASK_SAMPLE,
        TRUNCATED,
        "prev_logprobs",
    )

    def __init__(self, *, root_dir: str) -> None:
        self._attempt_start = datetime.now(timezone.utc)
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._active_step: Optional[int] = None
        self._writer: Optional[pq.ParquetWriter] = None
        self._pending_path: Optional[Path] = None

    def record(
        self,
        meta: KVBatchMeta,
        td: TensorDict,
        *,
        advantages: torch.Tensor,
        final_sample_mask: torch.Tensor,
        step: int,
        chunk_index: int,
        values: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
    ) -> None:
        """Append one chunk to the step's Parquet file."""
        n = len(meta.sample_ids)
        if n == 0:
            return
        if "input_lengths" not in td:
            raise ValueError("trajectory log requires input_lengths in the batch")
        lengths = td["input_lengths"].cpu().reshape(-1).numpy()
        tags = meta.tags or [{}] * n

        columns: dict[str, Any] = {
            "schema_version": [TRAJECTORY_SCHEMA_VERSION] * n,
            "step": [step] * n,
            "attempt_start": [self._attempt_start] * n,
            "chunk_index": [chunk_index] * n,
            "sample_id": meta.sample_ids,
            "prompt_idx": [tag.get("prompt_idx") for tag in tags],
            "weight_version": [tag.get("weight_version") for tag in tags],
            "reward": _scalar_column(
                td["total_reward"] if "total_reward" in td else None, n
            ),
            "final_sample_mask": _scalar_column(final_sample_mask, n),
            "mask_sample": _scalar_column(
                td[MASK_SAMPLE] if MASK_SAMPLE in td else None, n
            ),
            "truncated": _scalar_column(td[TRUNCATED] if TRUNCATED in td else None, n),
        }
        for name in (
            "input_ids",
            "token_mask",
            "generation_logprobs",
            "prev_logprobs",
        ):
            columns[name] = _token_column(
                name, td[name] if name in td else None, lengths
            )
        columns["advantages"] = _token_column("advantages", advantages, lengths)
        columns["values"] = _token_column("values", values, lengths)
        columns["returns"] = _token_column("returns", returns, lengths)
        table = pa.table(columns, schema=ROW_SCHEMA)

        if self._active_step is None:
            directory = self._root / f"step={step:08d}"
            directory.mkdir(parents=True, exist_ok=True)
            self._pending_path = directory / f".part-{uuid4().hex}.parquet.tmp"
            self._writer = pq.ParquetWriter(
                str(self._pending_path), ROW_SCHEMA, compression="zstd"
            )
            self._active_step = step
        elif self._active_step != step:
            raise RuntimeError("trajectory log has an unpublished previous step")
        assert self._writer is not None
        self._writer.write_table(table)

    def commit_step(self, step: int) -> None:
        """Publish this attempt's completed file after the optimizer step."""
        if self._active_step is None:
            return
        if self._active_step != step:
            raise RuntimeError("trajectory log is publishing the wrong step")
        assert self._writer is not None and self._pending_path is not None
        self._writer.close()
        published_name = self._pending_path.name.removeprefix(".").removesuffix(".tmp")
        os.replace(self._pending_path, self._pending_path.with_name(published_name))
        self._writer = None
        self._pending_path = None
        self._active_step = None
