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

import importlib
from dataclasses import dataclass
from typing import Any, Literal

from nemo_rl.data.energon.multimodal.model_families import (
    ALL_MODEL_FAMILIES,
    ModelFamily,
    get_supported_model_families,
    supports_model_family,
)

RegistryKind = Literal["cooker", "task_encoder"]


@dataclass(frozen=True)
class LazyRegistryEntry:
    """Import path and stable implementation version for one registry key."""

    import_path: str
    version: str


class LazyRegistry:
    """Resolve configured components inside the process that owns the loader."""

    def __init__(self, kind: RegistryKind) -> None:
        self.kind = kind
        self._entries: dict[str, LazyRegistryEntry] = {}

    def register(self, key: str, *, import_path: str, version: str) -> None:
        """Register one unused key without importing its implementation."""
        if not key:
            raise ValueError("Registry keys must be non-empty strings.")
        if key in self._entries:
            raise ValueError(f"Duplicate {self.kind} registry key {key!r}.")
        module_name, separator, attribute_name = import_path.partition(":")
        if not separator or not module_name or not attribute_name:
            raise ValueError(
                f"Invalid {self.kind} import path {import_path!r}; expected module:name."
            )
        if not version:
            raise ValueError(f"Registry version for {key!r} must be non-empty.")
        self._entries[key] = LazyRegistryEntry(
            import_path=import_path,
            version=version,
        )

    def resolve(self, key: str) -> Any:
        """Import and type-check one configured implementation."""
        entry = self._entries.get(key)
        if entry is None:
            raise ValueError(f"Unknown {self.kind} registry key {key!r}.")
        module_name, _, attribute_name = entry.import_path.partition(":")
        module = importlib.import_module(module_name)
        try:
            resolved = getattr(module, attribute_name)
        except AttributeError as error:
            raise TypeError(
                f"{self.kind} registry key {key!r} does not resolve to "
                f"{entry.import_path!r}."
            ) from error
        self._validate(key, resolved)
        return resolved

    def identity(self, key: str) -> dict[str, str]:
        """Return stable fingerprint data without importing the implementation."""
        entry = self._entries.get(key)
        if entry is None:
            raise ValueError(f"Unknown {self.kind} registry key {key!r}.")
        return {"key": key, "version": entry.version}

    def resolve_for_model_family(self, key: str, *, model_family: ModelFamily) -> Any:
        """Resolve a cooker or task encoder and validate its model family."""
        resolved = self.resolve(key)
        try:
            supported = get_supported_model_families(resolved)
        except TypeError as error:
            raise TypeError(
                f"{self.kind.replace('_', ' ').capitalize()} registry key {key!r} "
                "must declare its supported model families."
            ) from error
        if supports_model_family(resolved, model_family):
            return resolved
        supported_names = ", ".join(
            sorted(name for name in supported if name != ALL_MODEL_FAMILIES)
        )
        raise ValueError(
            f"{self.kind.replace('_', ' ').capitalize()} registry key {key!r} "
            f"does not support model family {model_family!r}; supported model "
            f"families: {supported_names}."
        )

    def _validate(self, key: str, resolved: Any) -> None:
        if self.kind == "task_encoder":
            # Deferred with the component so registry import stays dependency-light.
            from nemo_rl.data.energon.multimodal.task_encoders.base import (
                BaseSFTTaskEncoder,
            )

            if not isinstance(resolved, type) or not issubclass(
                resolved, BaseSFTTaskEncoder
            ):
                raise TypeError(
                    f"Task encoder registry key {key!r} must resolve to a "
                    "BaseSFTTaskEncoder subclass."
                )
            return
        if not callable(resolved):
            raise TypeError(
                f"{self.kind.capitalize()} registry key {key!r} must resolve "
                "to a callable."
            )


COOKER_REGISTRY = LazyRegistry("cooker")
COOKER_REGISTRY.register(
    "generic_conversation",
    import_path=("nemo_rl.data.energon.multimodal.cookers.generic:cook_conversation"),
    version="1",
)

TASK_ENCODER_REGISTRY = LazyRegistry("task_encoder")
TASK_ENCODER_REGISTRY.register(
    "generic_sft",
    import_path=(
        "nemo_rl.data.energon.multimodal.task_encoders.generic_sft:"
        "GenericSFTTaskEncoder"
    ),
    version="1",
)


def selected_registry_identity(
    *, task_encoder: str, cookers: list[str]
) -> dict[str, Any]:
    """Return stable identity data for all configured multimodal components."""
    return {
        "task_encoder": TASK_ENCODER_REGISTRY.identity(task_encoder),
        "cookers": [COOKER_REGISTRY.identity(cooker) for cooker in cookers],
    }


__all__ = [
    "COOKER_REGISTRY",
    "LazyRegistry",
    "LazyRegistryEntry",
    "TASK_ENCODER_REGISTRY",
    "selected_registry_identity",
]
