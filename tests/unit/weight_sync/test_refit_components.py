"""Tests for version-neutral NCCL refit component metadata."""

import copy
from typing import Any

import pytest

from nemo_rl.weight_sync.nccl_reshard_utils import build_nccl_reshard_refit_info
from nemo_rl.weight_sync.refit_components import (
    component_plan_digest,
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


@pytest.mark.parametrize(
    ("components", "match"),
    [
        (
            [
                {"role": "weight", "shape": [64, 256], "dtype": "torch.float8_e4m3fn"},
                {"role": "weight", "shape": [64, 256], "dtype": "torch.float8_e4m3fn"},
            ],
            "duplicate component role",
        ),
        (
            [{"role": "weight_scale", "shape": [64, 8], "dtype": "torch.uint8"}],
            "must include 'weight'",
        ),
        (
            [
                {"role": "weight", "shape": [64, 256], "dtype": "torch.float8_e4m3fn"},
                {"role": "weight_scale", "shape": [64, 8], "dtype": "torch.float16"},
            ],
            "torch.uint8",
        ),
        (
            [
                {"role": "weight", "shape": [64, 255], "dtype": "torch.float8_e4m3fn"},
                {"role": "weight_scale", "shape": [64, 8], "dtype": "torch.uint8"},
            ],
            "divisible by 32",
        ),
        (
            [
                {"role": "weight", "shape": [64, 256], "dtype": "torch.float8_e4m3fn"},
                {"role": "weight_scale", "shape": [64, 9], "dtype": "torch.uint8"},
            ],
            "scale shape",
        ),
    ],
)
def test_normalize_refit_components_rejects_invalid_native_pairs(
    components: list[dict[str, object]], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        normalize_refit_components(
            "model.layers.0.mlp.down_proj.weight",
            {
                "shape": [64, 256],
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
