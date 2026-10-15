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


def enable_msc() -> ModuleType:
    """Enable the optional Megatron storage backend in the calling process."""
    # MSC and Megatron are optional for the other policy backends.
    import multistorageclient as msc
    from megatron.core.msc_utils import MultiStorageClientFeature

    MultiStorageClientFeature.enable()
    return msc


def checkpoint_abspath(path: str | os.PathLike[str]) -> str:
    """Preserve MSC URLs when normalizing a checkpoint location."""
    return str(path) if is_msc_path(path) else os.path.abspath(path)


def save_checkpoint_object(value: Any, path: str | os.PathLike[str]) -> None:
    """Save controller checkpoint state beside local or MSC model shards."""
    # Keep torch optional for callers that only normalize paths.
    import torch

    if is_msc_path(path):
        enable_msc().torch.save(value, str(path))
    else:
        torch.save(value, path)
