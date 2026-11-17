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

import dataclasses
import functools
import json
import multiprocessing
import os
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch
from hydra.utils import get_class
from pydantic import BaseModel

from nemo_rl.utils.json_utils import json_safe_nonfinite

if TYPE_CHECKING:
    import pyarrow as pa


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
        sequences: Per-sample columns, dense or jagged, that the writer trims to
            input_lengths, such as token ids and advantages.
        scalars: Per-sample columns written as given; variable-length values must
            be jagged tensors rather than padded.

    With background writing, add_chunk moves the CPU tensors of the chunk to
    shared memory, so the caller must not modify them after it returns.
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

    A subclass that defines __init__ must accept log_dir, the keyword-only
    shard_id and background, and pass them to this base. Extra keys of the
    debug_writer config arrive as further keyword arguments.
    """

    def __init__(
        self, log_dir: str, *, shard_id: str = "0", background: bool = False
    ) -> None:
        """Store the log directory and shard id and start with no rows written.

        Args:
            log_dir: Directory the writer stores rows under.
            shard_id: Identifier of the shard that owns this writer.
            background: Whether _write_chunk hands its writing to a separate
                process; flush waits for the chunks handed over.
        """
        self.log_dir = Path(log_dir)
        self.shard_id = shard_id
        self.step: int | None = None
        self.rows: int = 0
        # Write in a separate process, with at most two pending chunks;
        # flush() waits until they finish.
        self._background = background
        self._executor: ProcessPoolExecutor | None = None
        self._pending: list[Future] = []

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

        # Share storages before any submission: the queue feeder pickles after
        # submit() returns and would race with the caller's later reads.
        if self._background:
            chunk = _share_chunk_storage(chunk)
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

    def _submit(self, fn: Callable[..., None], *args: Any) -> None:
        """Run fn(*args) in the background process, with at most two pending."""
        if self._executor is None:
            # spawn: never fork a process holding Ray and CUDA state.
            self._executor = ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn")
            )
        if len(self._pending) == 2:
            self._pending.pop(0).result()
        self._pending.append(self._executor.submit(fn, *args))

    def flush(self) -> None:
        """Wait until every chunk handed to the background is written."""
        pending, self._pending = self._pending, []
        for future in pending:
            future.result()

    def close(self) -> None:
        """Finish the pending chunks and stop the background process."""
        self.flush()
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    @abstractmethod
    def _write_chunk(self, chunk: TrainChunk, *, replace: bool) -> ChunkReceipt:
        """Write one validated chunk to storage and return the receipt for its rows.

        The first chunk of a step comes with replace set, telling the writer to drop
        any rows already stored for that step.
        """
        # With background set, add_chunk already shared the storages; hand the
        # write to self._submit(). See JsonDebugWriter._write_chunk() for an example.

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
        receipt = ChunkReceipt(
            shard_id=self.shard_id, rows=len(chunk.sample_ids), ref=str(part)
        )
        if not self._background:
            self._write_part(part, mode, chunk)
            return receipt
        self._submit(self._write_part, part, mode, chunk)
        return receipt

    @staticmethod
    def _write_part(part: Path, mode: str, chunk: TrainChunk) -> None:
        """Write one validated chunk's rows to its part file (maybe in the background)."""
        # unbind() splits dense and jagged columns alike into per-row tensors.
        columns = {
            k: list(v.detach().cpu().unbind())
            for k, v in {**chunk.sequences, **chunk.scalars}.items()
        }
        lengths = chunk.input_lengths.detach().cpu().reshape(-1).tolist()

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


@functools.cache
def _parquet_schema() -> "pa.Schema":
    """Return the Parquet schema of a training dump row."""
    import pyarrow as pa

    def floats() -> "pa.DataType":
        return pa.list_(pa.float32())

    return pa.schema(
        [
            ("step", pa.int64()),
            ("sample_id", pa.string()),
            ("group_id", pa.string()),
            ("input_lengths", pa.int64()),
            ("metadata", pa.string()),
            ("token_ids", pa.list_(pa.int64())),
            ("prompt_ids", pa.list_(pa.int64())),
            ("token_loss_mask", floats()),
            ("advantages", floats()),
            ("generation_logprobs", floats()),
            ("prev_logprobs", floats()),
            ("teacher_logprobs", floats()),
            ("reference_logprobs", floats()),
            ("values", floats()),
            ("returns", floats()),
            ("sample_loss_mask", pa.float32()),
            ("pre_seq_error_sample_loss_mask", pa.float32()),
            ("rewards", pa.float32()),
        ]
    )


class ParquetDebugWriter(DebugWriter):
    """Write each chunk as its own Parquet file, listed by a per-step manifest.

    Files go under train_data_parquet/step-<N>/ in the log directory. Columns a
    step does not produce are null. A step is complete only once _manifest.json
    exists in its directory; read only the files it lists. Shards must share a
    filesystem with the controller. pyarrow is imported when a chunk is written,
    so the json writer does not need it.
    """

    def _validate(self, chunk: TrainChunk) -> None:
        """Also raise ValueError for a column the Parquet schema lacks, before any background write."""
        super()._validate(chunk)
        names = _parquet_schema().names
        unknown = [k for k in (*chunk.sequences, *chunk.scalars) if k not in names]
        if unknown:
            raise ValueError(
                f"Parquet training dump has no column(s) {unknown}; add them to "
                "_parquet_schema."
            )

    def _step_dir(self, step: int) -> Path:
        """Return the directory holding the step's chunk files and manifest."""
        return self.log_dir / "train_data_parquet" / f"step-{step + 1:08d}"

    def _write_chunk(self, chunk: TrainChunk, *, replace: bool) -> ChunkReceipt:
        """Write the chunk to a new Parquet file; _publish_step drops the files of earlier attempts."""
        directory = self._step_dir(chunk.step)
        directory.mkdir(parents=True, exist_ok=True)
        name = (
            f"step-{chunk.step + 1:08d}-shard-{self.shard_id}-{uuid.uuid4().hex}"
            ".parquet"
        )
        receipt = ChunkReceipt(
            shard_id=self.shard_id, rows=len(chunk.sample_ids), ref=name
        )
        if not self._background:
            self._write_file(directory, name, chunk)
            return receipt
        self._submit(self._write_file, directory, name, chunk)
        return receipt

    @staticmethod
    def _write_file(directory: Path, name: str, chunk: TrainChunk) -> None:
        """Write one validated chunk to directory/name atomically (maybe in the background)."""
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = _parquet_schema()
        size = len(chunk.sample_ids)
        lengths = chunk.input_lengths.detach().cpu().reshape(-1).tolist()
        columns: dict[str, Any] = {
            "step": [chunk.step + 1] * size,
            "sample_id": chunk.sample_ids,
            "input_lengths": lengths,
        }
        if chunk.tags is not None:
            columns["group_id"] = [tag.get("group_id") for tag in chunk.tags]
            columns["metadata"] = [
                json.dumps(json_safe_nonfinite(tag), allow_nan=False)
                for tag in chunk.tags
            ]
        for key, tensor in {**chunk.sequences, **chunk.scalars}.items():
            dtype = schema.field(key).type
            tensor = tensor.detach().cpu()
            if tensor.is_floating_point():
                tensor = tensor.float()
            if not pa.types.is_list(dtype):
                columns[key] = pa.array(tensor.reshape(size).numpy(), type=dtype)
                continue
            # unbind() yields per-row views of dense and jagged columns alike.
            rows = tensor.unbind()
            if key in chunk.sequences:
                rows = [row[:length] for row, length in zip(rows, lengths)]
            parts = [row.numpy() for row in rows]
            offsets = np.zeros(len(parts) + 1, dtype=np.int32)
            np.cumsum([len(part) for part in parts], out=offsets[1:])
            flat = (
                np.concatenate(parts)
                if parts
                else np.empty(0, dtype=dtype.value_type.to_pandas_dtype())
            )
            columns[key] = pa.ListArray.from_arrays(
                pa.array(offsets), pa.array(flat, type=dtype.value_type)
            )
        table = pa.Table.from_arrays(
            [
                pa.array(columns[field.name], type=field.type)
                if field.name in columns
                else pa.nulls(size, type=field.type)
                for field in schema
            ],
            schema=schema,
        )
        partial = directory / f".{name}.partial"
        pq.write_table(table, partial, compression="zstd")
        os.replace(partial, directory / name)

    def _publish_step(self, step: int, receipts: list[ChunkReceipt]) -> None:
        """Write _manifest.json listing the step's chunk files, after checking them against the receipts."""
        import pyarrow.parquet as pq

        directory = self._step_dir(step)
        files = [receipt.ref for receipt in receipts]
        if not files:
            raise RuntimeError("Completed optimizer step has no training dump")
        missing = [name for name in files if not (directory / name).is_file()]
        if missing:
            raise RuntimeError(
                f"Training dump chunk(s) {missing} not found; the advantage pool "
                "must share this filesystem with the controller"
            )
        rows = sum(pq.ParquetFile(directory / name).metadata.num_rows for name in files)
        expected_rows = sum(receipt.rows for receipt in receipts)
        if rows != expected_rows:
            raise RuntimeError(
                f"Training dump holds {rows} row(s) in {len(files)} chunk(s) but "
                f"the receipts report {expected_rows}"
            )
        final = directory / "_manifest.json"
        partial = directory / f"._manifest-{uuid.uuid4().hex}.partial"
        partial.write_text(json.dumps({"step": step + 1, "files": files}) + "\n")
        os.replace(partial, final)
        # A retried step leaves the failed attempt's chunks and the previous
        # manifest's chunks behind; drop them only after the new manifest is in place.
        for stale in directory.glob("step-*.parquet"):
            if stale.name not in files:
                stale.unlink()


def _share_tensor_storage(tensor: torch.Tensor) -> torch.Tensor:
    """Move the tensor's CPU storages, jagged values and offsets included, to shared memory."""
    tensor = tensor.detach().cpu()
    parts = (
        [tensor.values(), tensor.offsets(), tensor.lengths()]
        if tensor.is_nested
        else [tensor]
    )
    for part in parts:
        if part is not None:
            part.share_memory_()
    return tensor


def _share_chunk_storage(chunk: TrainChunk) -> TrainChunk:
    """Return the chunk with every tensor column in shared memory, ready to pickle."""
    return dataclasses.replace(
        chunk,
        input_lengths=_share_tensor_storage(chunk.input_lengths),
        sequences={k: _share_tensor_storage(v) for k, v in chunk.sequences.items()},
        scalars={k: _share_tensor_storage(v) for k, v in chunk.scalars.items()},
    )


DEBUG_WRITER_REGISTRY: dict[str, Callable[..., DebugWriter]] = {
    "json": JsonDebugWriter,
    "parquet": ParquetDebugWriter,
}


def make_debug_writer(
    cfg: DebugWriterConfig,
    log_dir: str,
    *,
    shard_id: str = "0",
    background: bool = False,
) -> DebugWriter:
    """Build the debug writer that cfg selects for one shard.

    A target found in DEBUG_WRITER_REGISTRY picks that writer; any other target
    containing a dot is imported as a dotted path to a DebugWriter subclass.

    Args:
        cfg: Debug writer config naming the writer and any constructor keys.
        log_dir: Base log directory handed to the writer.
        shard_id: Identifier of the shard that owns the returned writer.
        background: Write chunks in a separate process; passed to the writer's
            constructor.

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
    kwargs = dict(cfg.model_extra or {})
    if "background" in kwargs:
        raise ValueError(
            "debug_writer sets 'background' itself; remove the 'background' key "
            "from the debug_writer config."
        )
    kwargs["background"] = background
    return writer_cls(log_dir, shard_id=shard_id, **kwargs)
