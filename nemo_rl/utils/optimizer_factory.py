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
"""Construct optimizers from config with dispatch for model-aware builders.

The default path resolves the configured ``optimizer.name`` to a class
(e.g. ``torch.optim.AdamW``) and instantiates it with ``model.parameters()``.
Builders that need to walk the model (e.g. to split parameters across more
than one optimizer, like Muon for linear weights and AdamW for the rest)
mark themselves with ``_builds_optimizer_from_model = True`` and receive
the module instance directly.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch
import torch.nn as nn


def _resolve_dotted(path: str) -> Any:
    """Resolve ``module.qualified.Name`` to either a class or a function.

    ``hydra.utils.get_class`` rejects callables that are not classes, which
    breaks dispatching to factory functions like
    ``nemo_rl.algorithms.muon.build_dtensor_muon``. This minimal
    ``importlib`` lookup accepts both.
    """
    module_path, _, attr = path.rpartition(".")
    if not module_path:
        raise ValueError(f"Optimizer name must be a dotted path, got: {path!r}")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def build_optimizer_from_cfg(
    model: nn.Module,
    optimizer_cfg: dict[str, Any],
) -> torch.optim.Optimizer:
    """Resolve ``optimizer_cfg.name`` and build the optimizer.

    Args:
        model: The model whose trainable parameters the optimizer will own.
        optimizer_cfg: A mapping with ``name`` (dotted import path) and
            ``kwargs`` (forwarded to the resolved callable).

    Returns:
        An optimizer instance. Either the resolved class instantiated with
        ``model.parameters()``, or the resolved builder called with the
        whole module if it advertises model-aware construction via the
        ``_builds_optimizer_from_model`` attribute.
    """
    cls_or_builder = _resolve_dotted(optimizer_cfg["name"])
    kwargs = optimizer_cfg["kwargs"]
    if getattr(cls_or_builder, "_builds_optimizer_from_model", False):
        return cls_or_builder(model, **kwargs)
    return cls_or_builder(model.parameters(), **kwargs)
