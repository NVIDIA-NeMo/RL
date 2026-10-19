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

import os
import shutil
from pathlib import Path
from typing import Any

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


def path_isdir(path: str | os.PathLike[str]) -> bool:
    """``os.path.isdir`` that also understands ``msc://`` URLs."""
    if is_msc_path(path):
        return msc.os.path.isdir(str(path))
    return os.path.isdir(path)


def listdir(path: str | os.PathLike[str]) -> list[str]:
    """``os.listdir`` that also understands ``msc://`` URLs."""
    if is_msc_path(path):
        return msc.os.listdir(str(path))
    return os.listdir(path)


def rmtree(path: str | os.PathLike[str]) -> None:
    """``shutil.rmtree`` that also understands ``msc://`` URLs."""
    if is_msc_path(path):
        msc.delete(str(path), recursive=True)
    else:
        shutil.rmtree(path)
