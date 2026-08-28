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


def _nano_inventory_inputs() -> tuple[OrderedDict[str, dict[str, object]], dict[str, dict[str, str]]]:
    native = OrderedDict(
        (
            (
                f"model.layers.{layer}.mlp.experts.0.{projection}_proj.weight",
                _native_entry(),
            )
            for layer in range(2)
            for projection in ("gate", "up", "down")
        )
    )
    misc = {
        **{
            f"model.layers.{layer}.mlp.experts.0.{projection}_proj.weight": {
                "dtype": "torch.bfloat16"
            }
            for layer in range(2, 10)
            for projection in ("gate", "up", "down")
        },
        "model.layers.0.mixer.shared_experts.linear_fc1.weight": {"dtype": "torch.bfloat16"},
        "model.layers.0.mixer.gate.weight": {"dtype": "torch.bfloat16"},
        "model.layers.0.mixer.qkv_proj.weight": {"dtype": "torch.bfloat16"},
        "model.layers.0.mixer.o_proj.weight": {"dtype": "torch.bfloat16"},
        "lm_head.weight": {"dtype": "torch.bfloat16"},
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

    assert inventory["routed_experts"]["native"] == 6
    assert inventory["routed_experts"]["bf16"] == 24
    assert inventory["last_layers_bf16"]["layers"] == list(range(2, 10))
    assert inventory["shared_experts"]["bf16"] == 1
    assert inventory["router"]["bf16"] == 1
    assert inventory["qkvo"]["bf16"] == 2
    assert inventory["lm_head"]["bf16"] == 1


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
