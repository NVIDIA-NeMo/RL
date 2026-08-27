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
from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

import torch


TRAIN_DATA_ARTIFACT_FORMAT = "nemo_rl.train_data.pt+safetensors.v1"
TRAIN_DATA_TENSOR_SIDECAR_FORMAT = "nemo_rl.train_data.tensors.safetensors.v1"
TRAIN_DATA_MAIN_FILENAME_TEMPLATE = "train_data_step{step}.pt"
TRAIN_DATA_TENSOR_FILENAME_TEMPLATE = "train_data_step{step}.tensors.safetensors"


@dataclass(frozen=True)
class TrainDataArtifactPaths:
    main: Path
    tensors: Path


class TrainDataSaveCallable(Protocol):
    def __call__(
        self,
        *,
        step: int,
        num_samples: int,
        non_tensor_data: Mapping[str, Any],
        tensors: Mapping[str, torch.Tensor],
    ) -> TrainDataArtifactPaths: ...


def train_data_artifact_paths(
    base_dir: str | os.PathLike[str], step: int
) -> TrainDataArtifactPaths:
    """Return the stable filenames for one 1-based training step."""
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError(f"Training-data artifact step must be an integer, got {step!r}")
    if step < 1:
        raise ValueError(f"Training-data artifact step must be >= 1, got {step!r}")
    base_path = Path(base_dir)
    return TrainDataArtifactPaths(
        main=base_path / TRAIN_DATA_MAIN_FILENAME_TEMPLATE.format(step=step),
        tensors=base_path / TRAIN_DATA_TENSOR_FILENAME_TEMPLATE.format(step=step),
    )


def _dtype_label(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _find_tensor_path(
    value: Any,
    path: str,
    seen_container_ids: set[int],
) -> Optional[str]:
    if torch.is_tensor(value):
        return path
    if isinstance(value, Mapping):
        container_id = id(value)
        if container_id in seen_container_ids:
            return None
        seen_container_ids.add(container_id)
        for key, child in value.items():
            tensor_path = _find_tensor_path(
                child,
                f"{path}.{key}",
                seen_container_ids,
            )
            if tensor_path is not None:
                return tensor_path
    elif isinstance(value, (list, tuple)):
        container_id = id(value)
        if container_id in seen_container_ids:
            return None
        seen_container_ids.add(container_id)
        for index, child in enumerate(value):
            tensor_path = _find_tensor_path(
                child,
                f"{path}[{index}]",
                seen_container_ids,
            )
            if tensor_path is not None:
                return tensor_path
    return None


def _validate_non_tensor_data(non_tensor_data: Mapping[str, Any]) -> None:
    for key, value in non_tensor_data.items():
        if not isinstance(key, str):
            raise TypeError(
                "Training-data non-tensor field names must be strings, "
                f"got {type(key).__name__}"
            )
        tensor_path = _find_tensor_path(value, key, set())
        if tensor_path is not None:
            raise TypeError(
                "Training-data tensor values must be stored in the safetensors "
                f"sidecar; found a tensor at {tensor_path}"
            )


def normalize_train_data_tensors(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Detach tensors into dense, contiguous CPU storage for safetensors."""
    normalized: dict[str, torch.Tensor] = {}
    seen_storage_ptrs: set[int] = set()
    for name, tensor in tensors.items():
        if not isinstance(name, str):
            raise TypeError(
                f"Training-data tensor names must be strings, got {type(name).__name__}"
            )
        if not torch.is_tensor(tensor):
            raise TypeError(
                f"Training-data tensor field {name!r} has type "
                f"{type(tensor).__name__}, expected torch.Tensor"
            )
        if tensor.layout != torch.strided:
            raise TypeError(
                f"Training-data tensor field {name!r} has unsupported layout "
                f"{tensor.layout}"
            )
        normalized_tensor = tensor.detach().cpu().contiguous()
        # safetensors rejects fields that share storage.  Clone only aliases;
        # independent multi-GiB tensors retain their existing CPU storage.
        if normalized_tensor.numel() > 0:
            storage_ptr = normalized_tensor.untyped_storage().data_ptr()
            if storage_ptr in seen_storage_ptrs:
                normalized_tensor = normalized_tensor.clone()
                storage_ptr = normalized_tensor.untyped_storage().data_ptr()
            seen_storage_ptrs.add(storage_ptr)
        normalized[name] = normalized_tensor
    if not normalized:
        raise ValueError(
            "Training-data tensor sidecar must contain at least one tensor"
        )
    return normalized


def _tensor_inventory(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "dtype": _dtype_label(tensor.dtype),
            "shape": list(tensor.shape),
        }
        for name, tensor in sorted(tensors.items())
    }


def _temporary_path(path: Path, transaction_id: str) -> Path:
    return path.with_name(f"{path.name}.tmp.{transaction_id}")


def save_train_data_artifacts(
    *,
    base_dir: str | os.PathLike[str],
    step: int,
    num_samples: int,
    non_tensor_data: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
) -> TrainDataArtifactPaths:
    """Save one train-data step as an authoritative ``.pt`` plus tensor sidecar.

    Both artifacts are first written under transaction-specific temporary names.
    The sidecar is published first and the ``.pt`` file last, so the latter acts
    as the completion marker.  A transaction id in both files lets readers
    detect the narrow crash window between the two atomic renames.
    """
    from safetensors.torch import save_file

    if isinstance(num_samples, bool) or not isinstance(num_samples, int):
        raise TypeError(
            f"Training-data num_samples must be an integer, got {num_samples!r}"
        )
    if num_samples < 0:
        raise ValueError(f"Training-data num_samples must be >= 0, got {num_samples!r}")

    paths = train_data_artifact_paths(base_dir, step)
    paths.main.parent.mkdir(parents=True, exist_ok=True)
    _validate_non_tensor_data(non_tensor_data)
    normalized_tensors = normalize_train_data_tensors(tensors)

    overlapping_fields = set(non_tensor_data) & set(normalized_tensors)
    if overlapping_fields:
        raise ValueError(
            "Training-data fields cannot appear in both artifacts: "
            f"{sorted(overlapping_fields)}"
        )

    transaction_id = uuid.uuid4().hex
    main_tmp_path = _temporary_path(paths.main, transaction_id)
    tensor_tmp_path = _temporary_path(paths.tensors, transaction_id)
    tensor_inventory = _tensor_inventory(normalized_tensors)
    main_artifact = {
        "format": TRAIN_DATA_ARTIFACT_FORMAT,
        "step": int(step),
        "num_samples": int(num_samples),
        "transaction_id": transaction_id,
        "data": dict(non_tensor_data),
        "tensor_sidecar": {
            "format": TRAIN_DATA_TENSOR_SIDECAR_FORMAT,
            "filename": paths.tensors.name,
            "transaction_id": transaction_id,
            "tensors": tensor_inventory,
        },
    }
    sidecar_metadata = {
        "format": TRAIN_DATA_TENSOR_SIDECAR_FORMAT,
        "step": str(step),
        "num_samples": str(num_samples),
        "transaction_id": transaction_id,
    }

    try:
        save_file(
            normalized_tensors,
            str(tensor_tmp_path),
            metadata=sidecar_metadata,
        )
        torch.save(main_artifact, main_tmp_path)

        # Publish the authoritative .pt last.  If publication is interrupted,
        # load_train_data_artifacts detects a stale/mismatched pair by tx id.
        os.replace(tensor_tmp_path, paths.tensors)
        os.replace(main_tmp_path, paths.main)
    finally:
        main_tmp_path.unlink(missing_ok=True)
        tensor_tmp_path.unlink(missing_ok=True)

    return paths


def load_train_data_artifacts(
    main_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Load and validate a trusted train-data artifact pair."""
    from safetensors import safe_open

    main_path = Path(main_path)
    main_artifact = torch.load(main_path, map_location="cpu", weights_only=False)
    if not isinstance(main_artifact, dict):
        raise TypeError(
            f"Expected a dict in train-data artifact {main_path}, "
            f"got {type(main_artifact).__name__}"
        )
    if main_artifact.get("format") != TRAIN_DATA_ARTIFACT_FORMAT:
        raise ValueError(
            f"Unsupported train-data artifact format: {main_artifact.get('format')!r}"
        )

    sidecar_info = main_artifact.get("tensor_sidecar")
    if not isinstance(sidecar_info, dict):
        raise TypeError("Train-data artifact has no valid tensor_sidecar metadata")
    if sidecar_info.get("format") != TRAIN_DATA_TENSOR_SIDECAR_FORMAT:
        raise ValueError(
            "Unsupported train-data tensor sidecar format: "
            f"{sidecar_info.get('format')!r}"
        )

    sidecar_filename = sidecar_info.get("filename")
    if (
        not isinstance(sidecar_filename, str)
        or Path(sidecar_filename).name != sidecar_filename
    ):
        raise ValueError(
            f"Invalid train-data tensor sidecar filename: {sidecar_filename!r}"
        )
    tensor_path = main_path.parent / sidecar_filename

    transaction_id = main_artifact.get("transaction_id")
    if not isinstance(transaction_id, str) or not transaction_id:
        raise ValueError("Train-data artifact has no valid transaction_id")
    if sidecar_info.get("transaction_id") != transaction_id:
        raise ValueError("Train-data .pt and sidecar metadata transaction ids differ")
    step = main_artifact.get("step")
    num_samples = main_artifact.get("num_samples")
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError(f"Train-data artifact has an invalid step: {step!r}")
    if (
        isinstance(num_samples, bool)
        or not isinstance(num_samples, int)
        or num_samples < 0
    ):
        raise ValueError(
            f"Train-data artifact has an invalid num_samples: {num_samples!r}"
        )

    expected_inventory = sidecar_info.get("tensors")
    if not isinstance(expected_inventory, dict):
        raise TypeError("Train-data artifact has no valid tensor inventory")

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(tensor_path), framework="pt", device="cpu") as reader:
        sidecar_metadata = reader.metadata() or {}
        if sidecar_metadata.get("transaction_id") != transaction_id:
            raise ValueError("Train-data .pt and safetensors transaction ids differ")
        if sidecar_metadata.get("format") != TRAIN_DATA_TENSOR_SIDECAR_FORMAT:
            raise ValueError("Safetensors file has an unsupported train-data format")
        if sidecar_metadata.get("step") != str(step):
            raise ValueError("Train-data .pt and safetensors step values differ")
        if sidecar_metadata.get("num_samples") != str(num_samples):
            raise ValueError("Train-data .pt and safetensors sample counts differ")

        tensor_names = list(reader.keys())
        if set(expected_inventory) != set(tensor_names):
            raise ValueError(
                "Train-data tensor inventory differs from safetensors keys: "
                f"expected={sorted(expected_inventory)}, actual={sorted(tensor_names)}"
            )
        for name in tensor_names:
            tensor = reader.get_tensor(name).contiguous()
            descriptor = expected_inventory[name]
            if not isinstance(descriptor, dict):
                raise TypeError(f"Invalid tensor inventory entry for {name!r}")
            if descriptor.get("dtype") != _dtype_label(tensor.dtype):
                raise ValueError(
                    f"Train-data tensor {name!r} dtype differs from inventory"
                )
            if descriptor.get("shape") != list(tensor.shape):
                raise ValueError(
                    f"Train-data tensor {name!r} shape differs from inventory"
                )
            tensors[name] = tensor

    non_tensor_data = main_artifact.get("data")
    if not isinstance(non_tensor_data, dict):
        raise TypeError("Train-data artifact has no valid non-tensor data mapping")
    overlapping_fields = set(non_tensor_data) & set(tensors)
    if overlapping_fields:
        raise ValueError(
            "Train-data fields appear in both loaded artifacts: "
            f"{sorted(overlapping_fields)}"
        )

    return {
        "format": TRAIN_DATA_ARTIFACT_FORMAT,
        "step": step,
        "num_samples": num_samples,
        "transaction_id": transaction_id,
        "data": {**non_tensor_data, **tensors},
        "paths": TrainDataArtifactPaths(main=main_path, tensors=tensor_path),
    }


class AsyncTrainDataArtifactWriter:
    """Bounded single-flight background writer for train-data artifact pairs."""

    def __init__(self, save: TrainDataSaveCallable) -> None:
        self._save = save
        self._lock = threading.Lock()
        self._active = False
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None

    def start(
        self,
        *,
        step: int,
        num_samples: int,
        non_tensor_data: Mapping[str, Any],
        tensors: Mapping[str, torch.Tensor],
    ) -> None:
        # Keep at most one batch retained by the logger.  Binary sidecar writes
        # should normally finish during the following optimizer step.
        self.finish(wait=True)

        non_tensor_snapshot = dict(non_tensor_data)
        # Retain tensor references here, but defer detach/CPU/contiguous copies
        # to the save callable running on the writer thread.  In particular,
        # GRPO advantages are commonly zero-stride expanded views, so making
        # them contiguous on this thread would keep the training loop blocked.
        tensor_snapshot = dict(tensors)
        main_filename = TRAIN_DATA_MAIN_FILENAME_TEMPLATE.format(step=step)
        tensor_filename = TRAIN_DATA_TENSOR_FILENAME_TEMPLATE.format(step=step)

        def run_save() -> None:
            started_at = time.perf_counter()
            try:
                self._save(
                    step=step,
                    num_samples=num_samples,
                    non_tensor_data=non_tensor_snapshot,
                    tensors=tensor_snapshot,
                )
            except BaseException as exc:
                with self._lock:
                    self._error = exc
            else:
                elapsed = time.perf_counter() - started_at
                print(
                    "Completed background train-data save: "
                    f"{main_filename} + {tensor_filename} ({elapsed:.2f}s)",
                    flush=True,
                )

        thread = threading.Thread(
            target=run_save,
            name="async-grpo-train-data-save",
            daemon=True,
        )
        with self._lock:
            self._active = True
            self._thread = thread
            self._error = None
        try:
            thread.start()
        except BaseException:
            with self._lock:
                self._active = False
                self._thread = None
                self._error = None
            raise
        print(
            f"Started background train-data save: {main_filename} + {tensor_filename}",
            flush=True,
        )

    def finish(self, wait: bool = True) -> bool:
        """Finish or poll the active save and surface any writer exception."""
        with self._lock:
            if not self._active:
                return True
            thread = self._thread
            if thread is None:
                raise RuntimeError("Train-data save has no writer thread")

        if wait:
            thread.join()
        elif thread.is_alive():
            return False
        else:
            thread.join(timeout=0)

        with self._lock:
            if thread.is_alive():
                return False
            error = self._error
            self._active = False
            self._thread = None
            self._error = None

        if error is not None:
            raise RuntimeError("Async train-data artifact save failed") from error
        return True
