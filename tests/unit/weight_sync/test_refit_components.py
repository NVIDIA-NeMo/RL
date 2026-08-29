# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Tests for version-neutral NCCL refit component metadata."""

import copy
from typing import Any

import pytest

from nemo_rl.weight_sync.nccl_reshard_utils import build_nccl_reshard_refit_info
from nemo_rl.weight_sync.refit_components import (
    component_plan_digest,
    native_mxfp8_param_names,
    normalize_refit_components,
)


def test_legacy_weight_becomes_one_component() -> None:
    components = normalize_refit_components(
        "model.layers.0.mlp.down_proj.weight",
        {"shape": [64, 256], "dtype": "torch.bfloat16"},
    )

    assert [(c.role, c.global_shape, c.dtype) for c in components] == [
        ("weight", (64, 256), "torch.bfloat16")
    ]


def test_native_mxfp8_requires_ordered_value_and_scale() -> None:
    components = normalize_refit_components(
        "model.layers.0.mlp.down_proj.weight",
        {
            "shape": [64, 256],
            "dtype": "torch.float8_e4m3fn",
            "components": [
                {
                    "role": "weight",
                    "shape": [64, 256],
                    "dtype": "torch.float8_e4m3fn",
                },
                {
                    "role": "weight_scale",
                    "shape": [64, 8],
                    "dtype": "torch.uint8",
                },
            ],
        },
    )

    assert [c.role for c in components] == ["weight", "weight_scale"]
    assert components[1].checkpoint_name == "model.layers.0.mlp.down_proj.weight_scale"


def test_native_mxfp8_rejects_reversed_value_and_scale() -> None:
    with pytest.raises(
        ValueError, match=r"components must be ordered as \('weight', 'weight_scale'\)"
    ):
        normalize_refit_components(
            "model.layers.0.mlp.down_proj.weight",
            {
                "shape": [64, 256],
                "dtype": "torch.float8_e4m3fn",
                "components": [
                    {
                        "role": "weight_scale",
                        "shape": [64, 8],
                        "dtype": "torch.uint8",
                    },
                    {
                        "role": "weight",
                        "shape": [64, 256],
                        "dtype": "torch.float8_e4m3fn",
                    },
                ],
            },
        )


@pytest.mark.parametrize(
    ("logical_shape", "components", "match"),
    [
        (
            [64, 256],
            [
                {"role": "weight", "shape": [64, 256], "dtype": "torch.float8_e4m3fn"},
                {"role": "weight", "shape": [64, 256], "dtype": "torch.float8_e4m3fn"},
            ],
            "duplicate component role",
        ),
        (
            [64, 256],
            [{"role": "weight_scale", "shape": [64, 8], "dtype": "torch.uint8"}],
            "must include 'weight'",
        ),
        (
            [64, 256],
            [
                {"role": "weight", "shape": [64, 256], "dtype": "torch.float8_e4m3fn"},
                {"role": "weight_scale", "shape": [64, 8], "dtype": "torch.float16"},
            ],
            "torch.uint8",
        ),
        (
            [64, 255],
            [
                {"role": "weight", "shape": [64, 255], "dtype": "torch.float8_e4m3fn"},
                {"role": "weight_scale", "shape": [64, 8], "dtype": "torch.uint8"},
            ],
            "divisible by 32",
        ),
        (
            [64, 256],
            [
                {"role": "weight", "shape": [64, 256], "dtype": "torch.float8_e4m3fn"},
                {"role": "weight_scale", "shape": [64, 9], "dtype": "torch.uint8"},
            ],
            "scale shape",
        ),
    ],
)
def test_normalize_refit_components_rejects_invalid_native_pairs(
    logical_shape: list[int], components: list[dict[str, object]], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        normalize_refit_components(
            "model.layers.0.mlp.down_proj.weight",
            {
                "shape": logical_shape,
                "dtype": "torch.float8_e4m3fn",
                "components": components,
            },
        )


@pytest.mark.parametrize("shape", [[64, 0], [64, True], [64, 1.5]])
def test_normalize_refit_components_rejects_invalid_shape(shape: list[object]) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        normalize_refit_components(
            "model.layers.0.mlp.down_proj.weight",
            {"shape": shape, "dtype": "torch.bfloat16"},
        )


def _native_refit_info() -> dict[str, Any]:
    return build_nccl_reshard_refit_info(
        {
            "model.layers.0.mlp.down_proj.weight": {
                "shape": [64, 256],
                "dtype": "torch.float8_e4m3fn",
                "components": [
                    {
                        "role": "weight",
                        "shape": [64, 256],
                        "dtype": "torch.float8_e4m3fn",
                    },
                    {
                        "role": "weight_scale",
                        "shape": [64, 8],
                        "dtype": "torch.uint8",
                    },
                ],
            }
        },
        train_parallelism={"tp_size": 2, "ep_size": 1, "pp_size": 1},
        gen_parallelism={"tp_size": 4, "ep_size": 1, "pp_size": 1},
        train_world_size=2,
        gen_world_size=4,
    )


def test_component_plan_digest_is_stable_and_order_sensitive() -> None:
    first = _native_refit_info()
    second = copy.deepcopy(first)

    assert component_plan_digest(first) == component_plan_digest(second)

    second["per_layer_params"]["model.layers.0"][0]["components"].reverse()
    assert component_plan_digest(first) != component_plan_digest(second)


def test_component_plan_digest_includes_mesh_shape_and_misc_metadata() -> None:
    first = _native_refit_info()
    second = copy.deepcopy(first)

    first_param = first["per_layer_params"]["model.layers.0"][0]
    second_param = second["per_layer_params"]["model.layers.0"][0]
    first_param["src_mesh_info"] = {"mesh": [0, 1]}
    second_param["src_mesh_info"] = {"mesh": [0, 1, 2, 3]}
    assert component_plan_digest(first) != component_plan_digest(second)

    second = copy.deepcopy(first)
    first["misc_meta"] = {
        "model.embed_tokens.weight": {"shape": [2, 4], "dtype": "torch.bfloat16"}
    }
    second["misc_meta"] = {
        "model.embed_tokens.weight": {"shape": [3, 4], "dtype": "torch.bfloat16"}
    }
    assert component_plan_digest(first) != component_plan_digest(second)


@pytest.mark.parametrize("placements", [None, "replicate"])
def test_component_plan_digest_rejects_invalid_placements(placements: object) -> None:
    refit_info = _native_refit_info()
    component = refit_info["per_layer_params"]["model.layers.0"][0]["components"][0]
    component["src_placements"] = placements

    with pytest.raises(ValueError, match="refit placements must be a sequence"):
        component_plan_digest(refit_info)


def test_native_mxfp8_param_names_requires_canonical_dtype_pair() -> None:
    refit_info = _native_refit_info()

    assert native_mxfp8_param_names(refit_info) == {
        "model.layers.0.mlp.down_proj.weight"
    }

    refit_info["per_layer_params"]["model.layers.0"][0]["components"][1]["dtype"] = (
        "torch.float32"
    )
    assert native_mxfp8_param_names(refit_info) == set()


@pytest.mark.parametrize("missing", ["shape", "dtype"])
def test_normalize_refit_components_reports_missing_required_metadata(
    missing: str,
) -> None:
    metadata = {"shape": [32, 64], "dtype": "torch.bfloat16"}
    del metadata[missing]

    with pytest.raises(ValueError, match=missing):
        normalize_refit_components("model.layers.0.weight", metadata)
