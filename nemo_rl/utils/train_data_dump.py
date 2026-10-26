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

"""Write training tensors as independent Parquet chunks."""

import json
import math
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

SCHEMA = pa.schema(
    [
        ("step", pa.int64()),
        ("sample_id", pa.string()),
        ("group_id", pa.string()),
        ("input_lengths", pa.int64()),
        ("metadata", pa.string()),
        ("token_ids", pa.list_(pa.int64())),
        ("prompt_ids", pa.list_(pa.int64())),
        ("token_loss_mask", pa.list_(pa.float32())),
        ("advantages", pa.list_(pa.float32())),
        ("generation_logprobs", pa.list_(pa.float32())),
        ("prev_logprobs", pa.list_(pa.float32())),
        ("teacher_logprobs", pa.list_(pa.float32())),
        ("reference_logprobs", pa.list_(pa.float32())),
        ("values", pa.list_(pa.float32())),
        ("returns", pa.list_(pa.float32())),
        ("sample_loss_mask", pa.float32()),
        ("pre_seq_error_sample_loss_mask", pa.float32()),
        ("rewards", pa.float32()),
    ]
)


def json_safe_nonfinite(value: Any) -> Any:
    """Represent nonfinite diagnostic values as JSON null without mutating data."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe_nonfinite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_nonfinite(item) for item in value]
    return value


def _list_array(
    rows: tuple[torch.Tensor, ...] | list[torch.Tensor], dtype: pa.ListType
) -> pa.ListArray:
    """Pack tensor rows."""
    parts = [row.numpy() for row in rows]
    offsets = np.zeros(len(parts) + 1, dtype=np.int32)
    np.cumsum([len(part) for part in parts], out=offsets[1:])
    flat = (
        np.concatenate(parts)
        if parts
        else np.empty(0, dtype=dtype.value_type.to_pandas_dtype())
    )
    return pa.ListArray.from_arrays(
        pa.array(offsets), pa.array(flat, type=dtype.value_type)
    )


class TrainDataDump:
    """Write Parquet chunks and mark completed steps with a manifest."""

    def __init__(self, log_dir: str, *, shard_id: str = "0") -> None:
        self.log_dir = Path(log_dir)
        self.shard_id = shard_id

    def add_chunk(
        self,
        *,
        step: int,
        sample_ids: list[str],
        tags: list[dict[str, Any]] | None,
        input_lengths: torch.Tensor,
        sequences: dict[str, torch.Tensor],
        scalars: dict[str, torch.Tensor],
    ) -> str:
        """Write one chunk from existing tensors and return its filename."""
        size = len(sample_ids)
        lengths = input_lengths.detach().cpu().reshape(-1).tolist()
        if len(lengths) != size or (tags is not None and len(tags) != size):
            raise ValueError("Training dump column lengths do not match sample ids")
        if any(length < 0 for length in lengths):
            raise ValueError("Training dump input length must be nonnegative")
        data: dict[str, Any] = {
            "step": [step + 1] * size,
            "sample_id": sample_ids,
            "input_lengths": lengths,
        }
        if tags is not None:
            data["group_id"] = [tag.get("group_id") for tag in tags]
            data["metadata"] = [
                json.dumps(json_safe_nonfinite(tag), allow_nan=False) for tag in tags
            ]
        for key, tensor in {**sequences, **scalars}.items():
            tensor = tensor.detach().cpu()
            if tensor.shape[0] != size:
                raise ValueError("Training dump column lengths do not match sample ids")
            if tensor.is_floating_point():
                tensor = tensor.float()
            dtype = SCHEMA.field(key).type
            if pa.types.is_list(dtype):
                # unbind() produces row views for both dense and jagged tensors.
                rows = tensor.unbind()
                if key in sequences:
                    if any(length > len(row) for row, length in zip(rows, lengths)):
                        raise ValueError(
                            "Training dump input length exceeds a sequence column"
                        )
                    rows = [row[:length] for row, length in zip(rows, lengths)]
                data[key] = _list_array(rows, dtype)
            else:
                data[key] = pa.array(tensor.reshape(size).numpy(), type=dtype)
        table = pa.Table.from_arrays(
            [
                pa.array(data[field.name], type=field.type)
                if field.name in data
                else pa.nulls(size, type=field.type)
                for field in SCHEMA
            ],
            schema=SCHEMA,
        )
        directory = self.log_dir / "train_data_parquet" / f"step-{step + 1:08d}"
        directory.mkdir(parents=True, exist_ok=True)
        name = f"step-{step + 1:08d}-shard-{self.shard_id}-{uuid4().hex}.parquet"
        partial = directory / ("." + name + ".partial")
        pq.write_table(table, partial, compression="zstd")
        os.replace(partial, directory / name)
        return name

    def finish_step(self, step: int, *, files: list[str]) -> None:
        """Publish the completed step's chunk list without reopening payloads."""
        directory = self.log_dir / "train_data_parquet" / f"step-{step + 1:08d}"
        if not files or len(set(files)) != len(files):
            raise ValueError("Training dump requires a nonempty, unique chunk list")
        prefix = f"step-{step + 1:08d}-shard-"
        if any(
            Path(name).name != name or not name.startswith(prefix) for name in files
        ):
            raise ValueError("Training dump chunk does not belong to this step")
        if any(
            not (directory / name).is_file() or (directory / name).stat().st_size == 0
            for name in files
        ):
            raise RuntimeError(
                "Training dump chunk missing or empty; workers must share this filesystem"
            )
        final = directory / "_manifest.json"
        if final.exists():
            raise FileExistsError(f"Training dump already published: {final}")
        partial = directory / f"._manifest-{uuid4().hex}.partial"
        partial.write_text(json.dumps({"step": step + 1, "files": files}) + "\n")
        os.replace(partial, final)
