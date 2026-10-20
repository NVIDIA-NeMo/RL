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
import shutil
import time
from pathlib import Path
from typing import Any, Protocol

import multistorageclient as msc
import torch


def is_msc_path(path: str | os.PathLike[str]) -> bool:
    """Return whether a path explicitly selects an MSC storage profile."""
    return str(path).startswith("msc://")


def maybe_enable_msc(*paths: str | os.PathLike[str] | None) -> None:
    """Enable MSC in Megatron(-Bridge) for this process if any path is ``msc://``.

    Megatron(-Bridge) only treats ``msc://`` as remote once its process-global
    MSC flag is on, so call this at every entry point that is handed a
    checkpoint path (load or save). Local-path runs never import Megatron.
    """
    if any(path and is_msc_path(path) for path in paths):
        # Megatron is optional for the other policy backends.
        from megatron.core.msc_utils import MultiStorageClientFeature

        MultiStorageClientFeature.enable()


def checkpoint_abspath(path: str | os.PathLike[str]) -> str:
    """Preserve MSC URLs when normalizing a checkpoint location."""
    return str(path) if is_msc_path(path) else os.path.abspath(path)


def save_checkpoint_object(value: Any, path: str | os.PathLike[str]) -> None:
    """Save controller checkpoint state beside local or MSC model shards."""
    if is_msc_path(path):
        msc.torch.save(value, str(path))
    else:
        torch.save(value, path)


def load_checkpoint_object(path: str | os.PathLike[str], **kwargs: Any) -> Any:
    """Load controller checkpoint state saved by ``save_checkpoint_object``."""
    if is_msc_path(path):
        return msc.torch.load(str(path), **kwargs)
    return torch.load(path, **kwargs)


def path_exists(path: str | os.PathLike[str]) -> bool:
    """``os.path.exists`` that also understands ``msc://`` URLs."""
    if is_msc_path(path):
        return msc.os.path.exists(str(path))
    return os.path.exists(path)


def open_path(path: str | os.PathLike[str], mode: str = "r") -> Any:
    """``open`` that also understands ``msc://`` URLs."""
    if is_msc_path(path):
        return msc.open(str(path), mode)
    return open(path, mode)


def as_path(path: str | os.PathLike[str]) -> Path:
    """``Path(path)``, or the MSC path class for ``msc://`` URLs."""
    return (msc.Path if is_msc_path(path) else Path)(path)


def path_isdir(path: str | os.PathLike[str], strict: bool = True) -> bool:
    """``os.path.isdir`` that also understands ``msc://`` URLs.

    ``strict=False`` treats a missing MSC prefix as "not a directory" instead of raising.
    """
    if is_msc_path(path):
        return msc.os.path.isdir(str(path), strict=strict)
    return os.path.isdir(path)


def listdir(path: str | os.PathLike[str]) -> list[str]:
    """``os.listdir`` that also understands ``msc://`` URLs."""
    if is_msc_path(path):
        return [entry.name for entry in msc.Path(str(path)).iterdir()]
    return os.listdir(path)


def rmtree(path: str | os.PathLike[str]) -> None:
    """``shutil.rmtree`` that also understands ``msc://`` URLs."""
    if is_msc_path(path):
        msc.delete(str(path), recursive=True)
    else:
        shutil.rmtree(path)


_STEP_DIR = re.compile(r"step_\d+")


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


class LocalStorage:
    def __init__(self, root: Path):
        self.root = root

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
        dirs = glob.glob(str(self.root / "step_*"))
        return sorted(
            (int(Path(d).name.split("_")[1]), d)
            for d in dirs
            if _STEP_DIR.fullmatch(Path(d).name)
        )


class MscStorage:
    def __init__(self, root: Path):
        self.root = root

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
        steps = []
        for marker in msc.glob(str(self.root) + "/step_*/_SUCCESS"):
            step_dir = msc.Path(marker).parent
            if _STEP_DIR.fullmatch(step_dir.name):
                steps.append((int(step_dir.name.split("_")[1]), str(step_dir)))
        return sorted(steps)


def make_storage(root: str | os.PathLike[str]) -> CheckpointStorage:
    """Pick the storage that matches ``root`` (``msc://`` URL or local directory)."""
    path = as_path(root)
    return MscStorage(path) if is_msc_path(root) else LocalStorage(path)
