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

import hashlib
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Protocol

from nemo_rl.data.energon.multimodal.model_families import (
    ALL_MODEL_FAMILIES,
    ModelFamily,
    get_supported_model_families,
    supports_model_family,
)

RegistryKind = Literal["cooker", "task_encoder"]


class ComponentConfig(Protocol):
    """Configuration fields shared by task encoders and cookers."""

    name: str | None
    python_file: str | None
    object_name: str | None


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

    def resolve_configured(
        self,
        *,
        name: str | None,
        python_file: str | None,
        object_name: str | None,
    ) -> Any:
        """Resolve either a built-in key or an object from a Python file."""
        if name is not None:
            return self.resolve(name)
        path, digest = self._file_details(python_file)
        if not object_name:
            raise ValueError(f"File-backed {self.kind} requires an object name.")
        module = self._load_file_module(path, digest)
        try:
            resolved = getattr(module, object_name)
        except AttributeError as error:
            raise TypeError(
                f"File-backed {self.kind} {path}:{object_name} does not resolve "
                "to an object."
            ) from error
        self._validate(f"file {path}:{object_name}", resolved)
        return resolved

    def identity(self, key: str) -> dict[str, str]:
        """Return stable fingerprint data without importing the implementation."""
        entry = self._entries.get(key)
        if entry is None:
            raise ValueError(f"Unknown {self.kind} registry key {key!r}.")
        return {"key": key, "version": entry.version}

    def configured_identity(
        self,
        *,
        name: str | None,
        python_file: str | None,
        object_name: str | None,
    ) -> dict[str, str]:
        """Return resume identity for a built-in or file-backed component."""
        if name is not None:
            return self.identity(name)
        path, digest = self._file_details(python_file)
        if not object_name:
            raise ValueError(f"File-backed {self.kind} requires an object name.")
        return {
            "python_file": str(path),
            "object": object_name,
            "sha256": digest,
        }

    def resolve_for_model_family(self, key: str, *, model_family: ModelFamily) -> Any:
        """Resolve a cooker or task encoder and validate its model family."""
        return self._validate_model_family(key, self.resolve(key), model_family)

    def resolve_configured_for_model_family(
        self,
        *,
        name: str | None,
        python_file: str | None,
        object_name: str | None,
        model_family: ModelFamily,
    ) -> Any:
        """Resolve a configured component and validate its model family."""
        resolved = self.resolve_configured(
            name=name,
            python_file=python_file,
            object_name=object_name,
        )
        label = name if name is not None else f"{python_file}:{object_name}"
        return self._validate_model_family(label, resolved, model_family)

    def _validate_model_family(
        self, label: str, resolved: Any, model_family: ModelFamily
    ) -> Any:
        try:
            supported = get_supported_model_families(resolved)
        except TypeError as error:
            raise TypeError(
                f"{self.kind.replace('_', ' ').capitalize()} {label!r} "
                "must declare its supported model families."
            ) from error
        if supports_model_family(resolved, model_family):
            return resolved
        supported_names = ", ".join(
            sorted(name for name in supported if name != ALL_MODEL_FAMILIES)
        )
        raise ValueError(
            f"{self.kind.replace('_', ' ').capitalize()} {label!r} "
            f"does not support model family {model_family!r}; supported model "
            f"families: {supported_names}."
        )

    def _file_details(self, python_file: str | None) -> tuple[Path, str]:
        if not python_file:
            raise ValueError(f"File-backed {self.kind} requires python_file.")
        requested_path = Path(python_file)
        if not requested_path.is_absolute():
            raise ValueError(f"File-backed {self.kind} python_file must be absolute.")
        path = requested_path.resolve()
        if path.suffix != ".py":
            raise ValueError(f"File-backed {self.kind} python_file must end in .py.")
        try:
            source = self._source_bytes(path)
        except OSError as error:
            raise ValueError(
                f"Cannot read file-backed {self.kind} python_file {path}."
            ) from error
        return path, hashlib.sha256(source).hexdigest()

    @staticmethod
    def _source_bytes(path: Path) -> bytes:
        """Return one digest input for a standalone module or plugin package."""
        if path.name != "__init__.py":
            return path.read_bytes()
        source_files = sorted(path.parent.rglob("*.py"))
        return b"".join(
            source_file.relative_to(path.parent).as_posix().encode()
            + b"\0"
            + source_file.read_bytes()
            + b"\0"
            for source_file in source_files
        )

    def _load_file_module(self, path: Path, digest: str) -> ModuleType:
        module_name = (
            "_nemo_rl_energon_component_"
            f"{hashlib.sha256(f'{path}:{digest}'.encode()).hexdigest()}"
        )
        existing = sys.modules.get(module_name)
        if isinstance(existing, ModuleType):
            return existing
        package_paths = [str(path.parent)] if path.name == "__init__.py" else None
        spec = importlib.util.spec_from_file_location(
            module_name,
            path,
            submodule_search_locations=package_paths,
        )
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load file-backed {self.kind} from {path}.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module

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
    *, task_encoder: ComponentConfig, cookers: list[ComponentConfig]
) -> dict[str, Any]:
    """Return stable identity data for all configured multimodal components."""
    return {
        "task_encoder": TASK_ENCODER_REGISTRY.configured_identity(
            name=task_encoder.name,
            python_file=task_encoder.python_file,
            object_name=task_encoder.object_name,
        ),
        "cookers": [
            COOKER_REGISTRY.configured_identity(
                name=cooker.name,
                python_file=cooker.python_file,
                object_name=cooker.object_name,
            )
            for cooker in cookers
        ],
    }


__all__ = [
    "COOKER_REGISTRY",
    "ComponentConfig",
    "LazyRegistry",
    "LazyRegistryEntry",
    "TASK_ENCODER_REGISTRY",
    "selected_registry_identity",
]
