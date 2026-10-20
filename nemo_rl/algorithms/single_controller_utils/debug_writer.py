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
from typing import Any, Iterable

import torch

from nemo_rl.utils.json_utils import json_safe_nonfinite


class DebugWriter:
    """Base class for writers that store a step's training rows chunk by chunk.

    Each shard calls add_chunk on its own instance, and the controller calls
    finish_step on another once the step completes.
    """

    def __init__(self) -> None:
        """Initialize the current step and the number of rows written for it."""
        self.step: int | None = None
        self.rows: int = 0

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
        """Validate one chunk of per-sample rows and write it to storage.

        The first chunk of a step replaces any rows already stored for it; later
        chunks append to them.

        Args:
            step: Optimizer step the rows belong to.
            sample_ids: One id per sample in the chunk.
            tags: Per-sample metadata dicts, or None for empty metadata.
            input_lengths: Valid token count per sample, used to trim sequence columns.
            sequences: Per-sample columns trimmed to input_lengths, such as token ids
                and advantages.
            scalars: Per-sample columns written as given; variable-length values must
                be jagged tensors rather than padded.
        """
        # unbind() splits dense and jagged columns alike into per-row tensors.
        columns = {
            k: list(v.detach().cpu().unbind())
            for k, v in {**sequences, **scalars}.items()
        }

        # Validate column lengths
        lengths = input_lengths.detach().cpu().reshape(-1).tolist()
        if (
            len(lengths) != len(sample_ids)
            or any(len(v) != len(sample_ids) for v in columns.values())
            or (tags is not None and len(tags) != len(sample_ids))
        ):
            raise ValueError("Training dump column lengths do not match sample ids")
        for i, length in enumerate(lengths):
            if length < 0 or any(length > columns[k][i].shape[0] for k in sequences):
                raise ValueError("Training dump input length exceeds a sequence column")

        # Truncate on the first chunk of a step so a retried step replaces its
        # rows instead of appending a second copy to the previous attempt's.
        is_first_chunk = self.step != step
        if is_first_chunk:
            self.rows = 0
        self.step = step

        # Write rows
        def _iter_rows() -> Iterable[dict[str, Any]]:
            for i, length in enumerate(lengths):
                row = {
                    "step": step + 1,
                    "sample_id": [sample_ids[i]],
                    "input_lengths": [length],
                    "metadata": [tags[i] if tags is not None else {}],
                }
                for key in sequences:
                    row[key] = [columns[key][i][:length].tolist()]
                for key in scalars:
                    row[key] = [columns[key][i].tolist()]
                yield row

        self.rows += self._write_rows(
            step=step,
            rows=_iter_rows(),
            replace=is_first_chunk,
        )

    def _write_rows(
        self, step: int, rows: Iterable[dict[str, Any]], replace: bool
    ) -> int:
        """Write one chunk's rows to storage and return how many were written."""
        raise NotImplementedError

    def finish_step(self, step: int, expected_rows: int | None = None) -> None:
        """Publish a completed optimizer step and reset the per-step state.

        Args:
            step: The completed optimizer step.
            expected_rows: Rows the shards reported writing, checked against the
                rows published.
        """
        self._publish_step(step, expected_rows)
        self.step = None
        self.rows = 0

    def _publish_step(self, step: int, expected_rows: int | None) -> None:
        """Publish a completed step's rows, raising on a row-count mismatch."""
        raise NotImplementedError


class JsonDebugWriter(DebugWriter):
    """Write each shard's rows to JSONL part files, merged into one file per step.

    Rows use the legacy train_data singleton-batch shape, and publishing numbers
    them with idx across the merged file. A failed step leaves part files, never a
    completed-looking JSONL file. Shards must share a filesystem with the controller.
    """

    def __init__(self, log_dir: str, *, shard_id: str = "0") -> None:
        """Set the directory to write under and the shard whose part files this owns."""
        super().__init__()
        self.log_dir = Path(log_dir)
        self.shard_id = shard_id

    def _write_rows(
        self, step: int, rows: Iterable[dict[str, Any]], replace: bool
    ) -> int:
        """Write the rows to this shard's part file, starting it over if replace is set."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        part = self.log_dir / f"train_data_step{step + 1}.jsonl.part-{self.shard_id}"
        mode = "w" if replace else "a"

        num_rows = 0
        with part.open(mode) as stream:
            for row in rows:
                stream.write(
                    json.dumps(json_safe_nonfinite(row), allow_nan=False) + "\n"
                )
                num_rows += 1
        return num_rows

    def _publish_step(self, step: int, expected_rows: int | None) -> None:
        """Merge the shards' part files into one JSONL file and publish it atomically."""
        parts = sorted(self.log_dir.glob(f"train_data_step{step + 1}.jsonl.part-*"))
        if not parts:
            raise RuntimeError("Completed optimizer step has no training dump")
        partial = self.log_dir / f"train_data_step{step + 1}.jsonl.partial"
        rows = 0
        with partial.open("w") as merged:
            for part in parts:
                with part.open() as stream:
                    for line in stream:
                        # Splice idx onto the front rather than reparsing: the
                        # writers emit it nowhere, and a dump large enough to
                        # want sharding is large enough to not re-serialize.
                        merged.write(f'{{"idx": {rows}, {line[1:]}')
                        rows += 1
        if not rows:
            partial.unlink()
            raise RuntimeError("Completed optimizer step has no training dump")
        if expected_rows is not None and rows != expected_rows:
            partial.unlink()
            raise RuntimeError(
                f"Training dump merged {rows} row(s) from {len(parts)} shard(s) "
                f"but the stage reported writing {expected_rows}; the advantage "
                "pool must share this filesystem with the controller"
            )
        os.replace(partial, self.log_dir / f"train_data_step{step + 1}.jsonl")
        for part in parts:
            part.unlink()
