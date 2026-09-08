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

from collections.abc import Callable
from typing import Final, Literal, TypeVar, cast

ModelFamily = Literal["nemotron", "qwen"]
AllModelFamilies = Literal["*"]
SupportedModelFamily = ModelFamily | AllModelFamilies

ALL_MODEL_FAMILIES: Final[AllModelFamilies] = "*"
KNOWN_MODEL_FAMILIES: Final = cast(
    frozenset[ModelFamily], frozenset({"nemotron", "qwen"})
)
_MODEL_FAMILIES_ATTRIBUTE = "__nemo_rl_supported_model_families__"

ComponentT = TypeVar("ComponentT")


def supports_model_families(
    *model_families: SupportedModelFamily,
) -> Callable[[ComponentT], ComponentT]:
    """Declare the model families supported by a cooker or task encoder."""
    if not model_families:
        raise ValueError("At least one supported model family must be declared.")
    unknown = set(model_families) - KNOWN_MODEL_FAMILIES - {ALL_MODEL_FAMILIES}
    if unknown:
        raise ValueError(f"Unknown model families: {sorted(unknown)!r}.")
    if ALL_MODEL_FAMILIES in model_families and len(set(model_families)) != 1:
        raise ValueError(
            "The all-model-families marker cannot be combined with named families."
        )
    metadata = frozenset(model_families)

    def decorate(component: ComponentT) -> ComponentT:
        setattr(component, _MODEL_FAMILIES_ATTRIBUTE, metadata)
        return component

    return decorate


def get_supported_model_families(
    component: object,
) -> frozenset[SupportedModelFamily]:
    """Read one component's immutable model-family declaration."""
    namespace = getattr(component, "__dict__", {})
    metadata = namespace.get(_MODEL_FAMILIES_ATTRIBUTE)
    valid_values = KNOWN_MODEL_FAMILIES | {ALL_MODEL_FAMILIES}
    if (
        not isinstance(metadata, frozenset)
        or not metadata
        or not metadata.issubset(valid_values)
    ):
        raise TypeError(
            f"Component {component!r} has no supported model-family declaration."
        )
    return cast(frozenset[SupportedModelFamily], metadata)


def supports_model_family(component: object, model_family: ModelFamily) -> bool:
    """Return whether a declared component supports one model family."""
    supported = get_supported_model_families(component)
    return ALL_MODEL_FAMILIES in supported or model_family in supported


__all__ = [
    "ALL_MODEL_FAMILIES",
    "KNOWN_MODEL_FAMILIES",
    "ModelFamily",
    "SupportedModelFamily",
    "get_supported_model_families",
    "supports_model_families",
    "supports_model_family",
]
