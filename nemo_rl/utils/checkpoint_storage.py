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
"""Opt-in MSC checkpoint storage; ordinary filesystem paths remain unchanged."""

import glob
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from nemo_rl.utils.path_utils import (
    as_path,
    is_msc_path,
    open_path,
    path_exists,
    path_isdir,
    rmtree,
)

_STEP_DIR = re.compile(r"step_(\d+)")


def _parse_step(dir_name: str) -> int | None:
    """Return N for a published ``step_N`` directory name, else ``None``."""
    match = _STEP_DIR.fullmatch(dir_name)
    return int(match.group(1)) if match else None


class CheckpointStorage(Protocol):
    """How a checkpoint root creates, publishes and lists ``step_N`` directories.

    Local filesystems publish atomically by renaming ``tmp_step_N`` to ``step_N``.
    Object stores cannot rename a prefix, so MSC writes ``step_N`` in place and
    publishes it by writing a ``_SUCCESS`` marker last.
    """

    def begin(self, step: int) -> str | Path:
        """Create and return the directory the checkpoint for ``step`` is written into."""
        ...

    def publish(self, path: str | os.PathLike[str]) -> None:
        """Make the directory returned by ``begin`` visible to ``published_steps``."""
        ...

    def published_steps(self) -> list[tuple[int, str]]:
        """Return ``(step, path)`` for every published checkpoint under the root."""
        ...


@dataclass
class LocalStorage:
    root: Path

    def begin(self, step: int) -> Path:
        save_dir = self.root / f"tmp_step_{step}"
        # Remove a stale tmp_step_{step} left behind by an interrupted prior
        # save so the new save starts clean.
        if save_dir.exists():
            t0 = time.monotonic()
            rmtree(save_dir)
            print(f"Removed stale {save_dir.name} in {time.monotonic() - t0:.2f}s")
        save_dir.mkdir(parents=True, exist_ok=True)
        return Path(os.path.abspath(save_dir))

    def publish(self, path: str | os.PathLike[str]) -> None:
        """Rename tmp_step_N to step_N.

        If step_N already exists (defensive guard for edge cases, e.g. resuming
        training), performs a pseudo-atomic swap via an intermediate old_step_N
        directory.
        """
        tmp_path = Path(path)
        step = tmp_path.name.split("_")[2]
        final_path = tmp_path.parent / f"step_{step}"
        if final_path.exists():
            old_path = tmp_path.parent / f"old_step_{step}"
            os.rename(final_path, old_path)
            os.rename(tmp_path, final_path)
            if old_path.exists():
                rmtree(old_path)
        else:
            os.rename(tmp_path, final_path)

    def published_steps(self) -> list[tuple[int, str]]:
        steps = [
            (_parse_step(Path(d).name), d) for d in glob.glob(str(self.root / "step_*"))
        ]
        return sorted((step, d) for step, d in steps if step is not None)


@dataclass
class MscStorage:
    root: Path

    def begin(self, step: int) -> str:
        save_dir = self.root / f"step_{step}"
        if path_exists(save_dir / "_SUCCESS"):
            raise FileExistsError(
                f"Refusing to overwrite MSC checkpoint {save_dir}; use a fresh run prefix."
            )
        # A step without _SUCCESS is an interrupted save (never discoverable
        # for resume); clear it so the resumed run can save the same step.
        if path_isdir(save_dir, strict=False):
            rmtree(save_dir)
            print(f"Removed unpublished {save_dir.name}")
        save_dir.mkdir(parents=True, exist_ok=True)
        return str(save_dir)

    def publish(self, path: str | os.PathLike[str]) -> None:
        # Publish only after all worker saves (including async completion) succeed.
        with open_path(str(path) + "/_SUCCESS", "w") as stream:
            json.dump({"version": 1}, stream)

    def published_steps(self) -> list[tuple[int, str]]:
        # List step prefixes only, then probe each marker, so the cost scales
        # with the number of steps rather than the shard objects inside them.
        candidates = [
            (step, step_dir)
            for step_dir in self.root.iterdir()
            if (step := _parse_step(step_dir.name)) is not None
        ]
        with ThreadPoolExecutor(max_workers=8) as pool:
            published = pool.map(lambda c: (c[1] / "_SUCCESS").exists(), candidates)
        return sorted(
            (step, str(step_dir))
            for (step, step_dir), ok in zip(candidates, published)
            if ok
        )


def make_storage(root: str | os.PathLike[str]) -> CheckpointStorage:
    """Pick the storage that matches ``root`` (``msc://`` URL or local directory)."""
    path = as_path(root)
    return MscStorage(path) if is_msc_path(root) else LocalStorage(path)
