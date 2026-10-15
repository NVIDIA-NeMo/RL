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
from types import ModuleType
from typing import Any


def is_msc_path(path: str | os.PathLike[str]) -> bool:
    """Return whether a path explicitly selects an MSC storage profile."""
    return str(path).startswith("msc://")


def import_msc() -> ModuleType:
    """Import ``multistorageclient`` without touching Megatron.

    Plain file helpers (``open_path``, ``save_checkpoint_object``, ...) use this so
    actors whose venv has no Megatron, e.g. the async replay buffer, can read
    and write ``msc://`` URLs.
    """
    # MSC is optional: only runs that use msc:// need it installed.
    import multistorageclient as msc

    return msc


def enable_msc() -> ModuleType:
    """Enable MSC in Megatron(-Bridge) for the calling process and return MSC."""
    # Megatron is optional for the other policy backends.
    from megatron.core.msc_utils import MultiStorageClientFeature

    msc = import_msc()
    MultiStorageClientFeature.enable()
    return msc


def enable_msc_for(*paths: str | os.PathLike[str] | None) -> ModuleType | None:
    """Enable MSC in this process if any of ``paths`` is an ``msc://`` URL.

    Megatron(-Bridge) only treats ``msc://`` as remote once its process-global
    MSC flag is on, so call this at every entry point that is handed a
    checkpoint path (load or save). Returns the ``multistorageclient`` module
    when enabled, else ``None``; local-path runs never import it.
    """
    if any(path and is_msc_path(path) for path in paths):
        return enable_msc()
    return None


def checkpoint_abspath(path: str | os.PathLike[str]) -> str:
    """Preserve MSC URLs when normalizing a checkpoint location."""
    return str(path) if is_msc_path(path) else os.path.abspath(path)


def save_checkpoint_object(value: Any, path: str | os.PathLike[str]) -> None:
    """Save controller checkpoint state beside local or MSC model shards."""
    # Keep torch optional for callers that only normalize paths.
    import torch

    if is_msc_path(path):
        import_msc().torch.save(value, str(path))
    else:
        torch.save(value, path)


def load_checkpoint_object(path: str | os.PathLike[str], **kwargs: Any) -> Any:
    """Load controller checkpoint state saved by ``save_checkpoint_object``."""
    import torch

    if is_msc_path(path):
        return import_msc().torch.load(str(path), **kwargs)
    return torch.load(path, **kwargs)


def path_exists(path: str | os.PathLike[str]) -> bool:
    """``os.path.exists`` that also understands ``msc://`` URLs."""
    if is_msc_path(path):
        return import_msc().os.path.exists(str(path))
    return os.path.exists(path)


def open_path(path: str | os.PathLike[str], mode: str = "r") -> Any:
    """``open`` that also understands ``msc://`` URLs."""
    if is_msc_path(path):
        return import_msc().open(str(path), mode)
    return open(path, mode)
