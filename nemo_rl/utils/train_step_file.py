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
"""A tiny JSON file that mirrors the trainer's current step for side-car telemetry.

The single controller rewrites ``<logger.log_dir>/train_step.json`` when it starts
(with the resumed step) and after every optimizer step. Out-of-process writers that
share the filesystem -- NeMo Gym's live W&B sink (``NEMO_GYM_LIVE_WANDB_STEP_FILE``)
-- read it to stamp their rows with ``nemo_rl/step``, so their series continue from
the resumed step across chained Slurm jobs instead of restarting at 0.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from typing import Any, Optional

TRAIN_STEP_FILE_NAME = "train_step.json"
STEP_KEY = "step"


def train_step_file_path(log_dir: str) -> str:
    """Location of the step file for ``log_dir``."""
    return os.path.join(log_dir, TRAIN_STEP_FILE_NAME)


def write_train_step_file(
    log_dir: Optional[str],
    step: int,
    trainer_version: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Atomically write ``{"step": step, ...}`` under ``log_dir``.

    Never raises: telemetry must not take the trainer down. Returns the path
    written, or ``None`` when ``log_dir`` is unset or the write failed.
    """
    if not log_dir:
        return None
    path = train_step_file_path(log_dir)
    payload: dict[str, Any] = {STEP_KEY: int(step), "updated_at": time.time()}
    if trainer_version is not None:
        payload["trainer_version"] = int(trainer_version)
    if extra:
        payload.update(extra)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        os.makedirs(log_dir, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    except OSError as exc:
        warnings.warn(f"could not write {path}: {exc!r}", stacklevel=2)
        try:
            os.remove(tmp)
        except OSError:
            pass
        return None
    return path


def read_train_step_file(log_dir: str) -> Optional[int]:
    """The step recorded under ``log_dir``, or ``None`` when absent/unreadable."""
    try:
        with open(train_step_file_path(log_dir), encoding="utf-8") as fh:
            return int(json.load(fh)[STEP_KEY])
    except (OSError, ValueError, KeyError, TypeError):
        return None
