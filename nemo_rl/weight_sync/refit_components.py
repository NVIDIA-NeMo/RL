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

"""Version-neutral metadata for ordered NCCL refit tensor components."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from torch.distributed._tensor import Shard
from torch.distributed.tensor.placement_types import Replicate

RefitComponentRole = Literal["weight", "weight_scale"]


@dataclass(frozen=True)
class RefitComponentMeta:
    """Immutable metadata for one transfer component of a logical parameter."""

    logical_name: str
    checkpoint_name: str
    role: RefitComponentRole
    dtype: str
    global_shape: tuple[int, ...]
    src_placements: tuple[Any, ...] = ()
    dst_placements: tuple[Any, ...] = ()

    def to_wire(self) -> dict[str, Any]:
        """Return the component fields in the transport metadata representation."""
        return {
            "logical_name": self.logical_name,
            "checkpoint_name": self.checkpoint_name,
            "role": self.role,
            "dtype": self.dtype,
            "global_shape": list(self.global_shape),
            "src_placements": list(self.src_placements),
            "dst_placements": list(self.dst_placements),
        }


def normalize_refit_components(
    logical_name: str,
    metadata: Mapping[str, Any],
) -> tuple[RefitComponentMeta, ...]:
    """Normalize legacy and native metadata into ordered component metadata.

    Args:
        logical_name: Canonical HF name for the logical parameter.
        metadata: Logical parameter metadata, optionally including ``components``.

    Returns:
        The ordered, validated components for the parameter.

    Raises:
        ValueError: If component roles, shapes, or MXFP8 scale metadata are invalid.
    """
    logical_shape = _positive_shape(
        _required(metadata, "shape", logical_name), logical_name
    )
    serialized = metadata.get("components")
    if serialized is None:
        serialized = [
            {
                "role": "weight",
                "shape": logical_shape,
                "dtype": _required(metadata, "dtype", logical_name),
            }
        ]
    if (
        not isinstance(serialized, Sequence)
        or isinstance(serialized, (str, bytes))
        or not serialized
    ):
        raise ValueError(f"{logical_name} components must not be empty")

    result: list[RefitComponentMeta] = []
    roles: set[str] = set()
    for item in serialized:
        if not isinstance(item, Mapping):
            raise ValueError(f"{logical_name} component metadata must be mappings")
        role = _required(item, "role", f"{logical_name} component")
        if role not in ("weight", "weight_scale"):
            raise ValueError(f"{logical_name} has unsupported component role {role!r}")
        if role in roles:
            raise ValueError(f"{logical_name} has duplicate component role {role!r}")
        roles.add(role)
        shape = _positive_shape(
            _required(item, "shape", f"{logical_name} {role}"),
            f"{logical_name} {role}",
        )
        dtype = str(_required(item, "dtype", f"{logical_name} {role}"))
        result.append(
            RefitComponentMeta(
                logical_name=logical_name,
                checkpoint_name=(
                    logical_name if role == "weight" else f"{logical_name}_scale"
                ),
                role=role,
                dtype=dtype,
                global_shape=shape,
            )
        )

    if "weight" not in roles:
        raise ValueError(f"{logical_name} components must include 'weight'")
    _validate_weight_scale_pair(logical_name, logical_shape, result)
    role_sequence = tuple(component.role for component in result)
    if role_sequence not in (("weight",), ("weight", "weight_scale")):
        raise ValueError(
            f"{logical_name} components must be ordered as ('weight', 'weight_scale')"
        )
    return tuple(result)


def component_plan_digest(refit_info: Mapping[str, Any]) -> str:
    """Return the deterministic SHA-256 digest for a refit component plan."""
    canonical_plan: list[dict[str, Any]] = []
    per_layer_params = refit_info.get("per_layer_params", {})
    for layer_name in refit_info.get("layer_names", []):
        for param_info in per_layer_params.get(layer_name, []):
            components = param_info.get("components")
            if components is None:
                components = [
                    {
                        "role": "weight",
                        "dtype": param_info["dtype"],
                        "global_shape": param_info["global_shape"],
                        "src_placements": param_info.get("src_placements", []),
                        "dst_placements": param_info.get("dst_placements", []),
                    }
                ]
            canonical_plan.append(
                {
                    "logical_name": param_info["name"],
                    "components": [
                        {
                            "role": component["role"],
                            "checkpoint_name": component.get("checkpoint_name"),
                            "dtype": str(component["dtype"]),
                            "global_shape": list(component["global_shape"]),
                            "src_placements": _canonical_placements(
                                component.get(
                                    "src_placements",
                                    param_info.get("src_placements", []),
                                )
                            ),
                            "dst_placements": _canonical_placements(
                                component.get(
                                    "dst_placements",
                                    param_info.get("dst_placements", []),
                                )
                            ),
                        }
                        for component in components
                    ],
                    "pp_stage": param_info.get("pp_stage"),
                    "grouped_expert_proj": param_info.get("grouped_expert_proj"),
                    "src_mesh": _canonical_mesh(param_info.get("src_mesh_info")),
                    "dst_mesh": _canonical_mesh(param_info.get("dst_mesh_info")),
                }
            )
    misc_plan = [
        {
            "name": name,
            "shape": list(metadata["shape"]),
            "dtype": str(metadata["dtype"]),
        }
        for name, metadata in refit_info.get("misc_meta", {}).items()
    ]
    payload = json.dumps(
        {"bulk": canonical_plan, "misc": misc_plan},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def native_mxfp8_param_names(refit_info: Mapping[str, Any]) -> set[str]:
    """Return parameters represented by a canonical MXFP8 value/scale pair."""
    result: set[str] = set()
    per_layer_params = refit_info.get("per_layer_params", {})
    for layer_name in refit_info.get("layer_names", []):
        for param_info in per_layer_params.get(layer_name, []):
            components = param_info.get("components", [])
            if [component.get("role") for component in components] != [
                "weight",
                "weight_scale",
            ]:
                continue
            if (
                str(components[0].get("dtype")) == "torch.float8_e4m3fn"
                and str(components[1].get("dtype")) == "torch.uint8"
            ):
                result.add(param_info["name"])
    return result


def _required(metadata: Mapping[str, Any], key: str, context: str) -> Any:
    """Read one required metadata field with a public validation error."""
    if key not in metadata:
        raise ValueError(f"{context} metadata must include {key!r}")
    return metadata[key]


def _positive_shape(value: Any, context: str) -> tuple[int, ...]:
    """Convert a serialized shape to a non-empty tuple of positive integers."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"{context} shape must contain positive integers")
    shape = tuple(value)
    if any(
        not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape
    ):
        raise ValueError(f"{context} shape must contain positive integers")
    return shape


def _validate_weight_scale_pair(
    logical_name: str,
    logical_shape: tuple[int, ...],
    components: Sequence[RefitComponentMeta],
) -> None:
    """Validate the logical weight and optional compact MXFP8 scale component."""
    by_role = {component.role: component for component in components}
    if by_role["weight"].global_shape != logical_shape:
        raise ValueError(f"{logical_name} weight shape must equal the logical shape")

    scale = by_role.get("weight_scale")
    if scale is None:
        return
    if scale.dtype != "torch.uint8":
        raise ValueError(f"{logical_name} weight_scale dtype must be torch.uint8")
    k = logical_shape[-1]
    if k % 32:
        raise ValueError(
            f"{logical_name} weight last dimension must be divisible by 32"
        )
    expected_shape = (*logical_shape[:-1], k // 32)
    if scale.global_shape != expected_shape:
        raise ValueError(
            f"{logical_name} weight_scale shape must equal the scale shape {expected_shape}"
        )


def _canonical_placements(placements: object) -> list[dict[str, int | str]]:
    """Represent Shard and Replicate placements without runtime object identity."""
    if not isinstance(placements, Sequence) or isinstance(placements, (str, bytes)):
        raise ValueError("refit placements must be a sequence")
    result: list[dict[str, int | str]] = []
    for placement in placements:
        if isinstance(placement, Shard):
            result.append({"kind": "shard", "dim": placement.dim})
        elif isinstance(placement, Replicate):
            result.append({"kind": "replicate"})
        elif isinstance(placement, Mapping):
            if "dim" in placement:
                result.append({"kind": "shard", "dim": int(placement["dim"])})
            else:
                result.append({"kind": "replicate"})
        else:
            raise ValueError(f"unsupported refit placement {placement!r}")
    return result


def _canonical_mesh(mesh_info: Any) -> Any:
    """Return nested mesh ranks without tensor or wrapper identity."""
    if mesh_info is None:
        return None
    mesh = (
        mesh_info.get("mesh")
        if isinstance(mesh_info, Mapping)
        else getattr(mesh_info, "mesh", None)
    )
    if mesh is None:
        raise ValueError(f"unsupported refit mesh {mesh_info!r}")
    tolist = getattr(mesh, "tolist", None)
    return tolist() if callable(tolist) else mesh
