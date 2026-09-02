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

"""Byte-offset indexed access to large trajectory-value JSONL files."""

import json
from pathlib import Path
from typing import Any, BinaryIO

from nemo_rl.data.datasets.raw_dataset import RawDataset


class IndexedJsonlDataset:
    """Map-style JSONL dataset that keeps only line byte offsets in memory."""

    def __init__(self, path: Path, task_name: str) -> None:
        self.path = path
        self.task_name = task_name
        self.offsets = self._load_offsets()
        self._input_file: BinaryIO | None = None
        self._reference_files: dict[Path, BinaryIO] = {}

    def _load_offsets(self) -> list[int]:
        index_path = self.path.with_name(f"{self.path.name}.idx")
        if index_path.is_file():
            with index_path.open("r", encoding="ascii") as index_file:
                offsets = [int(line) for line in index_file if line.strip()]
            if offsets and offsets[0] != 0:
                raise ValueError(f"First JSONL index offset must be zero: {index_path}")
            if offsets and offsets[-1] >= self.path.stat().st_size:
                raise ValueError(f"JSONL index points beyond end of file: {index_path}")
            return offsets

        offsets: list[int] = []
        with self.path.open("rb") as input_file:
            while True:
                offset = input_file.tell()
                line = input_file.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
        return offsets

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self.offsets)
        if not 0 <= index < len(self.offsets):
            raise IndexError(index)
        if self._input_file is None:
            self._input_file = self.path.open("rb")
        self._input_file.seek(self.offsets[index])
        raw_line = self._input_file.readline().decode("utf-8")
        row = json.loads(raw_line)
        if not isinstance(row, dict):
            raise ValueError(f"trajectory-value row {index} is not an object")
        if "trajectory_ref" in row:
            row = self._resolve_trajectory_ref(row, index)
            return {
                "trajectory_value_row": row,
                "task_name": self.task_name,
            }
        return {
            "trajectory_value_json": raw_line,
            "task_name": self.task_name,
        }

    def _resolve_trajectory_ref(
        self, experiment_row: dict[str, Any], row_index: int
    ) -> dict[str, Any]:
        reference = experiment_row.get("trajectory_ref")
        if not isinstance(reference, dict):
            raise ValueError(f"trajectory_ref in row {row_index} must be an object")
        raw_path = reference.get("path")
        byte_offset = reference.get("byte_offset")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"trajectory_ref.path in row {row_index} is invalid")
        if (
            not isinstance(byte_offset, int)
            or isinstance(byte_offset, bool)
            or byte_offset < 0
        ):
            raise ValueError(
                f"trajectory_ref.byte_offset in row {row_index} is invalid"
            )
        reference_path = Path(raw_path)
        if not reference_path.is_absolute():
            reference_path = (self.path.parent / reference_path).resolve()
        if not reference_path.is_file():
            raise FileNotFoundError(
                f"canonical trajectory store is missing: {reference_path}"
            )
        reference_file = self._reference_files.get(reference_path)
        if reference_file is None:
            reference_file = reference_path.open("rb")
            self._reference_files[reference_path] = reference_file
        reference_file.seek(byte_offset)
        raw_canonical = reference_file.readline()
        if not raw_canonical:
            raise ValueError(
                f"trajectory_ref in row {row_index} points beyond {reference_path}"
            )
        canonical = json.loads(raw_canonical)
        if not isinstance(canonical, dict):
            raise ValueError(
                f"canonical trajectory at offset {byte_offset} is not an object"
            )
        for identity_key in ("trajectory_id", "instance_id"):
            expected = reference.get(identity_key, experiment_row.get(identity_key))
            actual = canonical.get(identity_key)
            if expected is not None and actual != expected:
                raise ValueError(
                    f"trajectory_ref {identity_key} mismatch in row {row_index}: "
                    f"expected {expected!r}, found {actual!r}"
                )
        if "responses_create_params" in experiment_row:
            raise ValueError(
                "referenced experiment rows cannot override responses_create_params"
            )
        merged = dict(canonical)
        canonical_metadata = canonical.get("metadata")
        experiment_metadata = experiment_row.get("metadata")
        merged.update(
            {key: value for key, value in experiment_row.items() if key != "metadata"}
        )
        merged.pop("trajectory_ref", None)
        if isinstance(canonical_metadata, dict) or isinstance(
            experiment_metadata, dict
        ):
            merged["metadata"] = {
                **(canonical_metadata if isinstance(canonical_metadata, dict) else {}),
                **(
                    experiment_metadata if isinstance(experiment_metadata, dict) else {}
                ),
            }
        return merged

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_input_file"] = None
        state["_reference_files"] = {}
        return state

    def __del__(self) -> None:
        input_file = getattr(self, "_input_file", None)
        if input_file is not None:
            input_file.close()
        for reference_file in getattr(self, "_reference_files", {}).values():
            reference_file.close()


class TrajectoryValueDataset(RawDataset):
    """Expose scalar-labeled trajectory prefixes through the response API."""

    def __init__(self, data_path: str, **kwargs: Any) -> None:
        del kwargs
        path = Path(data_path)
        if not path.is_file():
            raise FileNotFoundError(f"Trajectory-value dataset not found: {path}")
        self.task_name = f"trajectory-value-{path.stem}"
        self.dataset = IndexedJsonlDataset(path, self.task_name)
        self.val_dataset = None
