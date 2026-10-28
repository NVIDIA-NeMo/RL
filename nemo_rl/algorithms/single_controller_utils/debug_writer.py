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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from hydra.utils import get_class
from pydantic import BaseModel

from nemo_rl.utils.json_utils import json_safe_nonfinite


class DebugWriterConfig(BaseModel, extra="allow"):
    # Key in DEBUG_WRITER_REGISTRY, or a dotted import path to a DebugWriter defined
    # outside this repo. Extra keys are forwarded to the constructor after log_dir
    # and shard_id.
    target: str = "json"


@dataclass(frozen=True)
class TrainChunk:
    """One chunk of per-sample training columns handed to a DebugWriter.

    Attributes:
        step: Optimizer step the rows belong to.
        sample_ids: One id per sample in the chunk.
        tags: Per-sample metadata dicts, or None for empty metadata.
        input_lengths: Valid token count per sample, used to trim sequence columns.
        sequences: Per-sample columns trimmed to input_lengths, such as token ids
            and advantages.
        scalars: Per-sample columns written as given; variable-length values must
            be jagged tensors rather than padded.
    """

    step: int
    sample_ids: list[str]
    tags: list[dict[str, Any]] | None
    input_lengths: torch.Tensor
    sequences: dict[str, torch.Tensor]
    scalars: dict[str, torch.Tensor]


@dataclass(frozen=True)
class ChunkReceipt:
    """What one add_chunk call stored, handed to the controller for finish_step.

    Attributes:
        shard_id: Id of the shard whose writer stored the chunk.
        rows: Number of rows stored.
        ref: Writer-specific location of the stored rows, such as a part file path
            or an object key.
    """

    shard_id: str
    rows: int
    ref: Any


class DebugWriter(ABC):
    """Base class for writers that store a step's training rows chunk by chunk.

    Each shard calls add_chunk on its own instance, and the controller calls
    finish_step on another once the step completes and close once at shutdown.
    """

    def __init__(self, log_dir: str, *, shard_id: str = "0") -> None:
        """Store the log directory and shard id and start with no rows written."""
        self.log_dir = Path(log_dir)
        self.shard_id = shard_id
        self.step: int | None = None
        self.rows: int = 0

    def add_chunk(self, chunk: TrainChunk) -> ChunkReceipt:
        """Validate one chunk of per-sample rows and write it to storage.

        The first chunk of a step replaces any rows already stored for it; later
        chunks append to them.

        Args:
            chunk: The chunk's step and per-sample columns.

        Returns:
            The receipt for the stored rows, to be passed on to finish_step.
        """
        self._validate(chunk)

        # Truncate on the first chunk of a step so a retried step replaces its
        # rows instead of appending a second copy to the previous attempt's.
        is_first_chunk = self.step != chunk.step
        if is_first_chunk:
            self.rows = 0
        self.step = chunk.step

        receipt = self._write_chunk(chunk, replace=is_first_chunk)
        self.rows += receipt.rows
        return receipt

    def _validate(self, chunk: TrainChunk) -> None:
        """Raise ValueError if the chunk's columns disagree in length or a sequence is too short."""
        num_samples = len(chunk.sample_ids)
        lengths = chunk.input_lengths.detach().cpu().reshape(-1).tolist()

        if len(lengths) != num_samples or (
            chunk.tags is not None and len(chunk.tags) != num_samples
        ):
            raise ValueError("Training dump column lengths do not match sample ids")
        for values in [*chunk.sequences.values(), *chunk.scalars.values()]:
            if values.size(0) != num_samples:
                raise ValueError("Training dump column lengths do not match sample ids")

        # unbind() returns per-row views of dense and jagged columns alike, so
        # each row is checked against its own length without copying data.
        sequence_rows = [values.unbind() for values in chunk.sequences.values()]
        for i, length in enumerate(lengths):
            if length < 0:
                raise ValueError("Training dump input length exceeds a sequence column")
            for rows in sequence_rows:
                if length > rows[i].shape[0]:
                    raise ValueError(
                        "Training dump input length exceeds a sequence column"
                    )

    def finish_step(self, step: int, receipts: list[ChunkReceipt]) -> None:
        """Publish a completed optimizer step and reset the per-step state.

        Args:
            step: The completed optimizer step.
            receipts: Receipts add_chunk returned for this step's chunks, naming
                what to publish and how many rows it must hold.
        """
        self._publish_step(step, receipts)
        self.step = None
        self.rows = 0

    def close(self) -> None:
        """Release anything the writer still holds; the base writer holds nothing."""

    @abstractmethod
    def _write_chunk(self, chunk: TrainChunk, *, replace: bool) -> ChunkReceipt:
        """Write one validated chunk to storage and return the receipt for its rows.

        The first chunk of a step comes with replace set, telling the writer to drop
        any rows already stored for that step.
        """

    @abstractmethod
    def _publish_step(self, step: int, receipts: list[ChunkReceipt]) -> None:
        """Publish a completed step's rows, raising on a row-count mismatch."""


class JsonDebugWriter(DebugWriter):
    """Write each shard's rows to JSONL part files, merged into one file per step.

    Rows use the legacy train_data singleton-batch shape, and publishing numbers
    them with idx across the merged file. A failed step leaves part files, never a
    completed-looking JSONL file. Shards must share a filesystem with the controller.
    """

    def _dump_path(self, step: int) -> Path:
        """Return the published JSONL path."""
        return self.log_dir / f"train_data_step{step + 1}.jsonl"

    def _shard_path(self, step: int) -> Path:
        """Return this shard's part file path."""
        return self.log_dir / f"train_data_step{step + 1}.jsonl.part-{self.shard_id}"

    def _write_chunk(self, chunk: TrainChunk, *, replace: bool) -> ChunkReceipt:
        """Write the chunk's rows to this shard's part file, starting it over if replace is set."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        part = self._shard_path(chunk.step)
        mode = "w" if replace else "a"

        # unbind() splits dense and jagged columns alike into per-row tensors.
        columns = {
            k: list(v.detach().cpu().unbind())
            for k, v in {**chunk.sequences, **chunk.scalars}.items()
        }
        lengths = chunk.input_lengths.detach().cpu().reshape(-1).tolist()

        num_rows = 0
        with part.open(mode) as stream:
            for i, length in enumerate(lengths):
                row = {
                    "step": chunk.step + 1,
                    "sample_id": [chunk.sample_ids[i]],
                    "input_lengths": [length],
                    "metadata": [chunk.tags[i] if chunk.tags is not None else {}],
                }
                for key in chunk.sequences:
                    row[key] = [columns[key][i][:length].tolist()]
                for key in chunk.scalars:
                    row[key] = [columns[key][i].tolist()]
                stream.write(
                    json.dumps(json_safe_nonfinite(row), allow_nan=False) + "\n"
                )
                num_rows += 1
        return ChunkReceipt(shard_id=self.shard_id, rows=num_rows, ref=str(part))

    def _publish_step(self, step: int, receipts: list[ChunkReceipt]) -> None:
        """Merge the part files the receipts name into one JSONL file and publish it atomically."""
        # A shard appends every chunk of a step to one part file, so its
        # receipts repeat the same ref.
        parts = [Path(ref) for ref in dict.fromkeys(r.ref for r in receipts)]
        if not parts:
            raise RuntimeError("Completed optimizer step has no training dump")
        missing = [str(part) for part in parts if not part.exists()]
        if missing:
            raise RuntimeError(
                f"Training dump part file(s) {missing} not found; the advantage "
                "pool must share this filesystem with the controller"
            )

        # Merge the part files into a partial JSONL file.
        partial = Path(f"{self._dump_path(step)}.partial")
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

        # Check that the merged file has the same number of rows as the receipts report.
        expected_rows = sum(receipt.rows for receipt in receipts)
        if rows != expected_rows:
            partial.unlink()
            raise RuntimeError(
                f"Training dump merged {rows} row(s) from {len(parts)} part file(s) "
                f"but the receipts report {expected_rows}"
            )

        # Replace the partial file with the completed JSONL file.
        os.replace(partial, self._dump_path(step))
        for part in parts:
            part.unlink()


DEBUG_WRITER_REGISTRY: dict[str, Callable[..., DebugWriter]] = {
    "json": JsonDebugWriter,
}


def make_debug_writer(
    cfg: DebugWriterConfig, log_dir: str, *, shard_id: str = "0"
) -> DebugWriter:
    """Build the debug writer that cfg selects for one shard.

    A target found in DEBUG_WRITER_REGISTRY picks that writer; any other target
    containing a dot is imported as a dotted path to a DebugWriter subclass.

    Args:
        cfg: Debug writer config naming the writer and any constructor keys.
        log_dir: Base log directory handed to the writer.
        shard_id: Identifier of the shard that owns the returned writer.

    Returns:
        The constructed writer.
    """
    if cfg.target in DEBUG_WRITER_REGISTRY:
        writer_cls = DEBUG_WRITER_REGISTRY[cfg.target]
    elif "." in cfg.target:
        writer_cls = get_class(cfg.target)
        if not issubclass(writer_cls, DebugWriter):
            raise ValueError(
                f"debug_writer target {cfg.target!r} resolved to {writer_cls!r}, "
                "which is not a DebugWriter subclass."
            )
    else:
        raise ValueError(
            f"Unsupported debug_writer target {cfg.target!r}. Set target to one of "
            f"{sorted(DEBUG_WRITER_REGISTRY)} or to an importable dotted path to a "
            "DebugWriter subclass."
        )
    return writer_cls(log_dir, shard_id=shard_id, **(cfg.model_extra or {}))
