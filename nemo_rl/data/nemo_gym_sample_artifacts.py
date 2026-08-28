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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Protocol

import torch


NEMO_GYM_TRAINING_SAMPLE_BATCH_KEY = "_nemo_gym_training_samples"
NEMO_GYM_TRAINING_SAMPLE_ARTIFACT_FORMAT = "nemo_rl.nemo_gym_training_samples.pt.v1"
# Retain the established filename from the earlier sample-batch logger. The
# contents now have an explicit format marker and deliberately omit tensors.
NEMO_GYM_TRAINING_SAMPLE_FILENAME_TEMPLATE = "sample_batch_data.step_{step}.pt"


class NemoGymTrainingSampleSaveCallable(Protocol):
    def __call__(
        self,
        *,
        step: int,
        samples: Sequence[Mapping[str, Any]],
    ) -> Path: ...


def nemo_gym_training_sample_artifact_path(
    base_dir: str | os.PathLike[str], step: int
) -> Path:
    """Return the stable filename for one 1-based training step."""
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError(
            f"NeMo Gym training-sample artifact step must be an integer, got {step!r}"
        )
    if step < 1:
        raise ValueError(
            f"NeMo Gym training-sample artifact step must be >= 1, got {step!r}"
        )
    return Path(base_dir) / NEMO_GYM_TRAINING_SAMPLE_FILENAME_TEMPLATE.format(step=step)


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


def _validate_samples(samples: Sequence[Mapping[str, Any]]) -> None:
    if not samples:
        raise ValueError("NeMo Gym training-sample artifact must not be empty")
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise TypeError(
                "NeMo Gym training samples must be mappings, "
                f"got {type(sample).__name__} at index {index}"
            )
        tensor_path = _find_tensor_path(sample, f"samples[{index}]", set())
        if tensor_path is not None:
            raise TypeError(
                "NeMo Gym training-sample artifacts intentionally omit tensor "
                f"payloads; found a tensor at {tensor_path}"
            )


def save_nemo_gym_training_samples(
    *,
    base_dir: str | os.PathLike[str],
    step: int,
    samples: Sequence[Mapping[str, Any]],
) -> Path:
    """Atomically save selected NeMo Gym responses without tensor sidecars."""
    path = nemo_gym_training_sample_artifact_path(base_dir, step)
    sample_list = list(samples)
    _validate_samples(sample_list)
    payload = {
        "format": NEMO_GYM_TRAINING_SAMPLE_ARTIFACT_FORMAT,
        "step": step,
        "num_samples": len(sample_list),
        "samples": sample_list,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


class AsyncNemoGymTrainingSampleWriter:
    """Bounded single-flight writer for selected NeMo Gym response metadata."""

    def __init__(self, save: NemoGymTrainingSampleSaveCallable) -> None:
        self._save = save
        self._lock = threading.Lock()
        self._active = False
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None

    def start(
        self,
        *,
        step: int,
        samples: Sequence[Mapping[str, Any]],
    ) -> None:
        # Bound retained response metadata to one step. The caller transfers
        # ownership of these rows and must not mutate their nested values after
        # this call; copying a complete Gym response on the training thread would
        # defeat the purpose of asynchronous persistence.
        self.finish(wait=True)
        sample_snapshot = list(samples)
        filename = NEMO_GYM_TRAINING_SAMPLE_FILENAME_TEMPLATE.format(step=step)

        def run_save() -> None:
            started_at = time.perf_counter()
            try:
                self._save(step=step, samples=sample_snapshot)
            except BaseException as exc:
                with self._lock:
                    self._error = exc
            else:
                elapsed = time.perf_counter() - started_at
                print(
                    "Completed background NeMo Gym training-sample save: "
                    f"{filename} ({elapsed:.2f}s)",
                    flush=True,
                )

        thread = threading.Thread(
            target=run_save,
            name="async-nemo-gym-training-sample-save",
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
            f"Started background NeMo Gym training-sample save: {filename}",
            flush=True,
        )

    def finish(self, wait: bool = True) -> bool:
        """Finish or poll the active save and surface any writer exception."""
        with self._lock:
            if not self._active:
                return True
            thread = self._thread
            if thread is None:
                raise RuntimeError("NeMo Gym training-sample save has no writer thread")

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
            raise RuntimeError(
                "Async NeMo Gym training-sample artifact save failed"
            ) from error
        return True
