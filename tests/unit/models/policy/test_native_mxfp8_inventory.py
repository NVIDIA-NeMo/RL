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

from collections import OrderedDict

import pytest

from nemo_rl.models.policy.workers.native_mxfp8_inventory import (
    assert_native_mxfp8_storage_inventory,
)


def _native_entry() -> dict[str, object]:
    return {
        "components": [
            {"role": "weight", "dtype": "torch.float8_e4m3fn"},
            {"role": "weight_scale", "dtype": "torch.uint8"},
        ]
    }


def _routed_name(layer: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.experts.0.{projection}_proj.weight"


def _bf16_non_routed_entries(*, include_shared_experts: bool) -> dict[str, dict[str, str]]:
    entries = {
        "model.layers.0.mixer.gate.weight": {"dtype": "torch.bfloat16"},
        "model.layers.0.mixer.qkv_proj.weight": {"dtype": "torch.bfloat16"},
        "model.layers.0.mixer.o_proj.weight": {"dtype": "torch.bfloat16"},
        "lm_head.weight": {"dtype": "torch.bfloat16"},
    }
    if include_shared_experts:
        entries["model.layers.0.mixer.shared_experts.linear_fc1.weight"] = {
            "dtype": "torch.bfloat16"
        }
    return entries


def _nano_inventory_inputs() -> tuple[OrderedDict[str, dict[str, object]], dict[str, dict[str, str]]]:
    native = OrderedDict(
        (
            (
                _routed_name(layer, projection),
                _native_entry(),
            )
            for layer in range(44)
            for projection in ("up", "down")
        )
    )
    misc = {
        **{
            _routed_name(layer, projection): {
                "dtype": "torch.bfloat16"
            }
            for layer in range(44, 52)
            for projection in ("up", "down")
        },
        **_bf16_non_routed_entries(include_shared_experts=True),
    }
    return native, misc


def _hybrid_nano_inventory_inputs(
    routed_layers: frozenset[int],
) -> tuple[OrderedDict[str, dict[str, object]], dict[str, dict[str, str]]]:
    native = OrderedDict(
        (
            (_routed_name(layer, projection), _native_entry())
            for layer in sorted(routed_layers & frozenset(range(44)))
            for projection in ("up", "down")
        )
    )
    misc = {
        **{
            _routed_name(layer, projection): {"dtype": "torch.bfloat16"}
            for layer in sorted(routed_layers & frozenset(range(44, 52)))
            for projection in ("up", "down")
        },
        **_bf16_non_routed_entries(include_shared_experts=True),
    }
    return native, misc


def test_nano_inventory_requires_routed_native_and_last_eight_bf16() -> None:
    native, misc = _nano_inventory_inputs()

    inventory = assert_native_mxfp8_storage_inventory(
        native_metadata=native,
        misc_metadata=misc,
        model_scope="nano",
        num_layers_at_end_in_bf16=8,
    )

    assert inventory["routed_experts"]["native"] == 88
    assert inventory["routed_experts"]["bf16"] == 16
    assert inventory["last_layers_bf16"]["layers"] == list(range(44, 52))
    assert inventory["shared_experts"]["bf16"] == 1
    assert inventory["router"]["bf16"] == 1
    assert inventory["qkvo"]["bf16"] == 2
    assert inventory["lm_head"]["bf16"] == 1


def test_nano_inventory_only_requires_hybrid_expert_layers() -> None:
    hybrid_pattern = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
    routed_layers = frozenset(
        layer for layer, layer_type in enumerate(hybrid_pattern) if layer_type == "E"
    )
    native, misc = _hybrid_nano_inventory_inputs(routed_layers)

    inventory = assert_native_mxfp8_storage_inventory(
        native_metadata=native,
        misc_metadata=misc,
        model_scope="nano",
        num_layers_at_end_in_bf16=8,
        routed_layer_indices=routed_layers,
    )

    assert inventory["routed_experts"]["native"] == 38
    assert inventory["routed_experts"]["bf16"] == 8
    assert inventory["last_layers_bf16"]["layers"] == list(range(44, 52))


def test_inventory_rejects_native_bf16_only_scope() -> None:
    native, misc = _nano_inventory_inputs()
    native["model.layers.0.mixer.qkv_proj.weight"] = _native_entry()
    misc.pop("model.layers.0.mixer.qkv_proj.weight")

    with pytest.raises(ValueError, match="qkvo.*BF16"):
        assert_native_mxfp8_storage_inventory(
            native_metadata=native,
            misc_metadata=misc,
            model_scope="nano",
            num_layers_at_end_in_bf16=8,
        )


def test_nano_inventory_rejects_missing_shared_experts() -> None:
    native, misc = _nano_inventory_inputs()
    misc.pop("model.layers.0.mixer.shared_experts.linear_fc1.weight")

    with pytest.raises(ValueError, match="shared_experts.*missing"):
        assert_native_mxfp8_storage_inventory(
            native_metadata=native,
            misc_metadata=misc,
            model_scope="nano",
            num_layers_at_end_in_bf16=8,
        )


def test_qwen_inventory_rejects_native_shared_expert() -> None:
    native = OrderedDict(
        (
            (_routed_name(layer, projection), _native_entry())
            for layer in range(48)
            for projection in ("gate", "up", "down")
        )
    )
    native["model.layers.0.mlp.shared_experts.linear_fc1.weight"] = _native_entry()

    with pytest.raises(ValueError, match="shared_experts.*BF16"):
        assert_native_mxfp8_storage_inventory(
            native_metadata=native,
            misc_metadata=_bf16_non_routed_entries(include_shared_experts=False),
            model_scope="qwen30",
            num_layers_at_end_in_bf16=0,
        )


def test_qwen_inventory_rejects_missing_all_gate_proj_entries() -> None:
    native = OrderedDict(
        (
            (_routed_name(layer, projection), _native_entry())
            for layer in range(48)
            for projection in ("up", "down")
        )
    )

    with pytest.raises(ValueError, match="layer 0 gate_proj.*native"):
        assert_native_mxfp8_storage_inventory(
            native_metadata=native,
            misc_metadata=_bf16_non_routed_entries(include_shared_experts=False),
            model_scope="qwen30",
            num_layers_at_end_in_bf16=0,
        )


def test_qwen_inventory_rejects_linear_fc1_alias_for_missing_gate_proj() -> None:
    native = OrderedDict(
        (
            (_routed_name(layer, projection), _native_entry())
            for layer in range(48)
            for projection in ("up", "down")
        )
    )
    native.update(
        (
            (f"model.layers.{layer}.mlp.experts.0.linear_fc1.weight", _native_entry())
            for layer in range(48)
        )
    )

    with pytest.raises(ValueError, match="not a supported projection"):
        assert_native_mxfp8_storage_inventory(
            native_metadata=native,
            misc_metadata=_bf16_non_routed_entries(include_shared_experts=False),
            model_scope="qwen30",
            num_layers_at_end_in_bf16=0,
        )


def test_qwen_inventory_allows_absent_shared_expert_when_no_native_entry() -> None:
    native = OrderedDict(
        (
            (_routed_name(layer, projection), _native_entry())
            for layer in range(48)
            for projection in ("gate", "up", "down")
        )
    )

    assert_native_mxfp8_storage_inventory(
        native_metadata=native,
        misc_metadata=_bf16_non_routed_entries(include_shared_experts=False),
        model_scope="qwen30",
        num_layers_at_end_in_bf16=0,
    )


def test_qwen235_inventory_requires_all_94_routed_expert_layers() -> None:
    native = OrderedDict(
        (
            (_routed_name(layer, projection), _native_entry())
            for layer in range(94)
            for projection in ("gate", "up", "down")
        )
    )

    inventory = assert_native_mxfp8_storage_inventory(
        native_metadata=native,
        misc_metadata=_bf16_non_routed_entries(include_shared_experts=False),
        model_scope="qwen235",
        num_layers_at_end_in_bf16=0,
    )

    assert inventory["routed_experts"]["native"] == 282
    assert inventory["routed_experts"]["bf16"] == 0


def test_nano_inventory_rejects_early_bf16_routed_expert() -> None:
    native, misc = _nano_inventory_inputs()
    early_fc1 = _routed_name(3, "up")
    misc[early_fc1] = {"dtype": "torch.bfloat16"}
    native.pop(early_fc1)

    with pytest.raises(ValueError, match="layer 3 up_proj.*native"):
        assert_native_mxfp8_storage_inventory(
            native_metadata=native,
            misc_metadata=misc,
            model_scope="nano",
            num_layers_at_end_in_bf16=8,
        )


def test_nano_inventory_rejects_omitted_final_bf16_routed_expert() -> None:
    native, misc = _nano_inventory_inputs()
    misc.pop(_routed_name(51, "down"))

    with pytest.raises(ValueError, match="layer 51 down_proj.*BF16"):
        assert_native_mxfp8_storage_inventory(
            native_metadata=native,
            misc_metadata=misc,
            model_scope="nano",
            num_layers_at_end_in_bf16=8,
        )
