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
"""Process-local data-plane adapter for colocated SFT loaders and policies."""

from __future__ import annotations

import pickle
import shutil
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from tensordict import NonTensorData, NonTensorStack, TensorDict

from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.data_plane.codec import materialize
from nemo_rl.data_plane.interfaces import (
    DataPlaneClient,
    KVBatchMeta,
    LocalDataPlaneConfig,
)
from nemo_rl.data_plane.schema import Layout
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

_LOCAL_GENERATION_KEY = "local_partition_generation"
# One name for both save_checkpoint and load_checkpoint. They used to spell it
# differently, which made every local resume raise FileNotFoundError.
_CHECKPOINT_FILENAME = "local_data_plane_state.pkl"


@dataclass
class _LocalPartition:
    # Declared schema. put_samples rejects any field not listed here.
    fields: tuple[str, ...]
    # Expected row count. put_samples requires exactly this many sample_ids.
    num_samples: int
    # [NOT CURRENTLY READ] Kept for readability. The register_partition argument
    # seeds `consumed` below; this tuple itself is never read back.
    # SFT registers ["train"].
    consumer_tasks: tuple[str, ...]
    # [NOT CURRENTLY USED] Both come from the DataPlaneClient.register_partition
    # signature and are stored so this adapter matches its siblings. Nothing in
    # this file reads either. SFT passes neither, so they stay None and {}.
    grpo_group_size: int | None
    enums: dict[str, list[str]]
    # Bumped on every register_partition and echoed into KVBatchMeta.extra_info.
    # _validate_meta rejects a meta whose generation does not match, which is
    # what stops a stale meta from reading a re-registered partition's rows.
    # It is also the marker is_local_batch_meta() keys on.
    generation: int
    # Everything below is filled by put_samples, not by registration.
    sample_ids: list[str] = field(default_factory=list)
    batch: dict[str, Any] = field(default_factory=dict)
    tags: list[dict[str, Any]] | None = None
    # Per-task set of sample_ids already handed out by claim_meta. SFT does not
    # call claim_meta -- it holds the meta put_samples returned -- so on the SFT
    # path these sets stay empty and check_consumption_status is never asked.
    consumed: dict[str, set[str]] = field(default_factory=dict)


def is_local_batch_meta(meta: KVBatchMeta) -> bool:
    """Return whether metadata identifies a process-local partition version."""
    return _LOCAL_GENERATION_KEY in meta.extra_info


def _value_batch_size(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        if value.dim() == 0:
            raise ValueError("Local data-plane fields must have a batch dimension.")
        return int(value.shape[0])
    if isinstance(value, PackedTensor):
        return len(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    try:
        return len(value)
    except TypeError as error:
        raise TypeError(
            f"Unsupported process-local field type: {type(value).__name__}."
        ) from error


def _copy_batch_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return deepcopy(value)


def _select_batch_value(value: Any, indices: list[int]) -> Any:
    if isinstance(value, torch.Tensor):
        index = torch.tensor(indices, dtype=torch.long, device=value.device)
        return value.index_select(0, index).detach().clone()
    if isinstance(value, PackedTensor):
        if not indices:
            return PackedTensor.empty_rows_like(value, 0)
        return deepcopy(value.slice(indices))
    if isinstance(value, tuple):
        return tuple(deepcopy(value[index]) for index in indices)
    if isinstance(value, list):
        return [deepcopy(value[index]) for index in indices]
    try:
        return deepcopy(value[indices])
    except (IndexError, KeyError, TypeError) as error:
        raise TypeError(
            f"Cannot select rows from local field type {type(value).__name__}."
        ) from error


def _unwrap_local_value(value: Any) -> Any:
    if isinstance(value, NonTensorData):
        return value.data
    if isinstance(value, NonTensorStack):
        items = [
            item.data if isinstance(item, NonTensorData) else item
            for item in value.tolist()
        ]
        if items and all(isinstance(item, PackedTensor) for item in items):
            return PackedTensor.concat(
                [item for item in items if isinstance(item, PackedTensor)]
            )
        return items
    return value


def local_batch_to_tensordict(
    fields: Mapping[str, Any], *, batch_size: int
) -> TensorDict:
    """Wrap a prepared local batch without flattening multimodal values.

    Tensor leaves stay as tensors. Other leaves, including ``PackedTensor``,
    use ``NonTensorData`` because this adapter never serializes them.
    """
    if batch_size < 0:
        raise ValueError(f"batch_size must be non-negative, got {batch_size}.")
    result = TensorDict({}, batch_size=(batch_size,))
    for name, value in fields.items():
        actual_size = _value_batch_size(value)
        if actual_size != batch_size:
            raise ValueError(
                f"Local field {name!r} has batch size {actual_size}, "
                f"expected {batch_size}."
            )
        if isinstance(value, torch.Tensor):
            result.set(name, value)
        else:
            result.set(
                name,
                # pyrefly: ignore[bad-argument-type]  tensordict's stubs declare `TensorClass` without its `TensorCollection` base, so `NonTensorData` is not seen as a valid `set` item
                NonTensorData(value, batch_size=(batch_size,)),
            )
    return result


def materialize_local(
    td: TensorDict,
    layout: Layout = "padded",
    pad_value_dict: dict[str, int | float] | None = None,
    pad_to_seqlen: int = 0,
) -> BatchedDataDict[Any]:
    """Materialize local tensors and restore exact non-tensor field values.

    Not interchangeable with :func:`materialize`. ``local_batch_to_tensordict``
    stores each non-tensor column as one ``NonTensorData`` holding the whole
    column, which ``materialize`` reads as a single row. Batches written by
    this adapter must be read back through here.
    """
    tensor_fields: dict[str, torch.Tensor] = {}
    local_fields: dict[str, Any] = {}
    for name in td.keys(include_nested=False):
        value = td.get(name)
        if isinstance(value, torch.Tensor):
            tensor_fields[name] = value
        else:
            local_fields[name] = _unwrap_local_value(value)

    tensor_td = TensorDict(tensor_fields, batch_size=td.batch_size)
    result = materialize(
        tensor_td,
        layout=layout,
        pad_value_dict=pad_value_dict,
        pad_to_seqlen=pad_to_seqlen,
    )
    result.update(local_fields)
    return result


class LocalDataPlaneClient(DataPlaneClient):
    """Store a bounded set of complete SFT batches in the current process."""

    def __init__(self, cfg: LocalDataPlaneConfig) -> None:
        self._max_partitions = cfg.max_partitions
        self._partitions: dict[str, _LocalPartition] = {}
        self._next_generation = 0
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Local data-plane client is closed.")

    def _partition(self, partition_id: str) -> _LocalPartition:
        self._require_open()
        try:
            return self._partitions[partition_id]
        except KeyError as error:
            raise KeyError(f"Unknown local partition {partition_id!r}.") from error

    def _validate_meta(self, meta: KVBatchMeta) -> _LocalPartition:
        partition = self._partition(meta.partition_id)
        generation = meta.extra_info.get(_LOCAL_GENERATION_KEY)
        if generation != partition.generation:
            raise ValueError(
                f"Stale local metadata for partition {meta.partition_id!r}: "
                f"generation {generation!r}, expected {partition.generation}."
            )
        return partition

    def register_partition(
        self,
        partition_id: str,
        fields: list[str],
        num_samples: int,
        consumer_tasks: list[str],
        grpo_group_size: int | None = None,
        enums: dict[str, list[str]] | None = None,
    ) -> None:
        self._require_open()
        if partition_id in self._partitions:
            raise ValueError(f"Local partition {partition_id!r} is already registered.")
        if len(self._partitions) >= self._max_partitions:
            raise RuntimeError(
                "Local data-plane partition limit reached: "
                f"{self._max_partitions}. Clear an active or prefetched partition "
                "before registering another one."
            )
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        if len(fields) != len(set(fields)):
            raise ValueError("Local partition field names must be unique.")
        if len(consumer_tasks) != len(set(consumer_tasks)):
            raise ValueError("Local partition consumer task names must be unique.")

        self._next_generation += 1
        generation = self._next_generation
        self._partitions[partition_id] = _LocalPartition(
            fields=tuple(fields),
            num_samples=num_samples,
            consumer_tasks=tuple(consumer_tasks),
            grpo_group_size=grpo_group_size,
            enums=deepcopy(enums) if enums is not None else {},
            generation=generation,
            consumed={task: set() for task in consumer_tasks},
        )

    def claim_meta(
        self,
        partition_id: str,
        task_name: str,
        required_fields: list[str],
        batch_size: int,
        dp_rank: int | None = None,
        blocking: bool = True,
        timeout_s: float = 60.0,
    ) -> KVBatchMeta:
        del dp_rank, blocking, timeout_s
        partition = self._partition(partition_id)
        if task_name not in partition.consumed:
            raise KeyError(
                f"Task {task_name!r} is not registered for local partition "
                f"{partition_id!r}."
            )
        unknown_fields = set(required_fields) - set(partition.batch)
        if unknown_fields:
            raise KeyError(
                f"Fields are not ready in local partition {partition_id!r}: "
                f"{sorted(unknown_fields)}."
            )

        available = [
            sample_id
            for sample_id in partition.sample_ids
            if sample_id not in partition.consumed[task_name]
        ][:batch_size]
        partition.consumed[task_name].update(available)
        indices = [partition.sample_ids.index(sample_id) for sample_id in available]
        selected_tags = (
            [deepcopy(partition.tags[index]) for index in indices]
            if partition.tags is not None
            else None
        )
        sequence_lengths = self._sequence_lengths(partition, indices)
        return KVBatchMeta(
            partition_id=partition_id,
            task_name=task_name,
            sample_ids=available,
            fields=list(required_fields),
            sequence_lengths=sequence_lengths,
            extra_info={_LOCAL_GENERATION_KEY: partition.generation},
            tags=selected_tags,
        )

    def get_data(
        self,
        meta: KVBatchMeta,
        select_fields: list[str] | None = None,
    ) -> TensorDict:
        self._validate_meta(meta)
        fields = select_fields if select_fields is not None else meta.fields
        if fields is None:
            raise ValueError(
                "get_data requires either select_fields or meta.fields; "
                "fetching all fields silently is forbidden."
            )
        return self.get_samples(meta.sample_ids, meta.partition_id, list(fields))

    def check_consumption_status(
        self, partition_id: str, task_names: list[str]
    ) -> bool:
        partition = self._partition(partition_id)
        for task_name in task_names:
            if task_name not in partition.consumed:
                return False
            if len(partition.consumed[task_name]) < len(partition.sample_ids):
                return False
        return True

    def put_samples(
        self,
        sample_ids: list[str],
        partition_id: str,
        fields: TensorDict | None = None,
        tags: list[dict[str, Any]] | None = None,
    ) -> KVBatchMeta:
        partition = self._partition(partition_id)
        if partition.sample_ids:
            raise ValueError(
                f"Local partition {partition_id!r} already contains a batch; "
                "duplicate writes are not allowed."
            )
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Local sample IDs must be unique within a batch.")
        if len(sample_ids) != partition.num_samples:
            raise ValueError(
                f"Local partition {partition_id!r} expects {partition.num_samples} "
                f"samples, received {len(sample_ids)}."
            )
        if fields is None and tags is None:
            raise ValueError("Local put_samples requires fields or tags.")
        if tags is not None and len(tags) != len(sample_ids):
            raise ValueError(
                f"Local tags have {len(tags)} rows, expected {len(sample_ids)}."
            )

        batch: dict[str, Any] = {}
        if fields is not None:
            if tuple(fields.batch_size) != (len(sample_ids),):
                raise ValueError(
                    f"Local fields batch size {tuple(fields.batch_size)} does not "
                    f"match {len(sample_ids)} sample IDs."
                )
            field_names = list(fields.keys(include_nested=False))
            undeclared = set(field_names) - set(partition.fields)
            if undeclared:
                raise ValueError(
                    f"Local write contains undeclared fields: {sorted(undeclared)}."
                )
            for name in field_names:
                value = _unwrap_local_value(fields.get(name))
                actual_size = _value_batch_size(value)
                if actual_size != len(sample_ids):
                    raise ValueError(
                        f"Local field {name!r} has batch size {actual_size}, "
                        f"expected {len(sample_ids)}."
                    )
                batch[name] = _copy_batch_value(value)

        partition.sample_ids = list(sample_ids)
        partition.batch = batch
        partition.tags = deepcopy(tags) if tags is not None else None
        sequence_lengths = self._sequence_lengths(
            partition, list(range(len(sample_ids)))
        )
        return KVBatchMeta(
            partition_id=partition_id,
            task_name=None,
            sample_ids=list(sample_ids),
            fields=list(batch),
            sequence_lengths=sequence_lengths,
            extra_info={_LOCAL_GENERATION_KEY: partition.generation},
            tags=deepcopy(tags) if tags is not None else None,
        )

    def get_samples(
        self,
        sample_ids: list[str],
        partition_id: str,
        select_fields: list[str],
    ) -> TensorDict:
        partition = self._partition(partition_id)
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Local fetch sample IDs must be unique.")
        missing_samples = set(sample_ids) - set(partition.sample_ids)
        if missing_samples:
            raise KeyError(
                f"Unknown sample IDs in local partition {partition_id!r}: "
                f"{sorted(missing_samples)}."
            )
        missing_fields = set(select_fields) - set(partition.batch)
        if missing_fields:
            raise KeyError(
                f"Fields are not available in local partition {partition_id!r}: "
                f"{sorted(missing_fields)}."
            )

        positions = {
            sample_id: index for index, sample_id in enumerate(partition.sample_ids)
        }
        indices = [positions[sample_id] for sample_id in sample_ids]
        selected = {
            name: _select_batch_value(partition.batch[name], indices)
            for name in select_fields
        }
        return local_batch_to_tensordict(selected, batch_size=len(sample_ids))

    def list_sample_ids(self, partition_id: str) -> list[str]:
        """List stored sample IDs without reading their batch payloads."""
        self._require_open()
        partition = self._partitions.get(partition_id)
        return sorted(partition.sample_ids) if partition is not None else []

    def clear_samples(self, sample_ids: list[str] | None, partition_id: str) -> None:
        partition = self._partition(partition_id)
        if sample_ids is None:
            del self._partitions[partition_id]
            return
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Local clear sample IDs must be unique.")
        missing_samples = set(sample_ids) - set(partition.sample_ids)
        if missing_samples:
            raise KeyError(
                f"Cannot clear unknown local sample IDs: {sorted(missing_samples)}."
            )

        removed = set(sample_ids)
        keep_indices = [
            index
            for index, sample_id in enumerate(partition.sample_ids)
            if sample_id not in removed
        ]
        partition.sample_ids = [partition.sample_ids[index] for index in keep_indices]
        partition.batch = {
            name: _select_batch_value(value, keep_indices)
            for name, value in partition.batch.items()
        }
        if partition.tags is not None:
            partition.tags = [partition.tags[index] for index in keep_indices]
        for consumed in partition.consumed.values():
            consumed.difference_update(removed)
        if not partition.sample_ids:
            del self._partitions[partition_id]

    def save_checkpoint(
        self,
        checkpoint_dir: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist the in-process partitions to ``checkpoint_dir``.

        State never leaves this process, so pickle is sufficient; callers must
        not load checkpoints from untrusted paths. The write goes to a sibling
        ``.tmp`` directory and is renamed into place so a crash mid-save leaves
        the previous checkpoint intact.
        """
        self._require_open()
        checkpoint_dir = Path(checkpoint_dir)
        tmp_dir = checkpoint_dir.with_name(f"{checkpoint_dir.name}.tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        try:
            with (tmp_dir / _CHECKPOINT_FILENAME).open("wb") as checkpoint_file:
                pickle.dump(
                    {
                        "partitions": self._partitions,
                        "next_generation": self._next_generation,
                        "metadata": metadata or {},
                    },
                    checkpoint_file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            # Move the previous checkpoint aside rather than deleting it first,
            # so a failed rename can be rolled back instead of leaving neither.
            old_dir = checkpoint_dir.with_name(f"{checkpoint_dir.name}.old")
            if old_dir.exists():
                shutil.rmtree(old_dir)
            replaced = checkpoint_dir.exists()
            if replaced:
                checkpoint_dir.rename(old_dir)
            try:
                tmp_dir.rename(checkpoint_dir)
            except OSError:
                if replaced:
                    old_dir.rename(checkpoint_dir)
                raise
            if replaced:
                shutil.rmtree(old_dir)
        except Exception:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            raise

    def load_checkpoint(self, checkpoint_dir: str | Path) -> dict[str, Any]:
        """Restore partitions into a client that has not registered any yet."""
        self._require_open()
        if self._partitions:
            raise RuntimeError(
                "load_checkpoint requires a clean data-plane client with no "
                "registered partitions"
            )
        checkpoint_file = Path(checkpoint_dir) / _CHECKPOINT_FILENAME
        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"Local checkpoint not found: {checkpoint_file}")
        with checkpoint_file.open("rb") as state_file:
            state = pickle.load(state_file)
        if "partitions" not in state:
            raise ValueError(
                f"Local checkpoint at {checkpoint_file} has no 'partitions' key; "
                "it was written by an incompatible version of this adapter."
            )
        metadata = state.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Local checkpoint metadata must be a dictionary")
        self._partitions = state["partitions"]
        self._next_generation = state.get("next_generation", 0)
        return metadata

    def close(self) -> None:
        if self._closed:
            return
        self._partitions.clear()
        self._closed = True

    @staticmethod
    def _sequence_lengths(
        partition: _LocalPartition, indices: list[int]
    ) -> list[int] | None:
        value = partition.batch.get("input_lengths")
        if value is None:
            return None
        selected = _select_batch_value(value, indices)
        if isinstance(selected, torch.Tensor):
            return [int(item) for item in selected.detach().cpu().reshape(-1).tolist()]
        return [int(item) for item in selected]
