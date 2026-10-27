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

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.utils.config import load_config


def test_normalization_selectors_and_roundtrip() -> None:
    cases = [
        ({}, "global_token_mean"),
        ({"token_level_loss": True}, "global_token_mean"),
        ({"token_level_loss": False}, "sequence_mean"),
        ({"normalization": "global_token_mean"}, "global_token_mean"),
        ({"normalization": "sequence_mean"}, "sequence_mean"),
    ]
    for fields, expected in cases:
        config = ClippedPGLossConfig(**fields)
        assert config.resolved_normalization == expected
        saved = config.model_dump()
        assert "resolved_normalization" not in saved
        reloaded = ClippedPGLossConfig.model_validate_json(config.model_dump_json())
        assert reloaded.resolved_normalization == expected
        assert reloaded.model_dump() == saved
        if config.token_level_loss is None:
            loss = ClippedPGLossFn(config)
        else:
            with pytest.warns(DeprecationWarning, match="normalization"):
                loss = ClippedPGLossFn(config)
        assert loss.normalization == expected

    for legacy in (True, False):
        with pytest.raises(ValidationError, match="Set only one"):
            ClippedPGLossConfig(normalization="sequence_mean", token_level_loss=legacy)
    with pytest.raises(ValidationError):
        ClippedPGLossConfig(normalization="typo")


def test_normalization_yaml_inheritance(tmp_path: Path) -> None:
    config_dir = Path(__file__).resolve().parents[3] / "examples" / "configs"
    parent = config_dir / "grpo_math_1B_megatron.yaml"
    old_child = tmp_path / "old.yaml"
    old_child.write_text(f"defaults: {parent}\nloss_fn:\n  token_level_loss: false\n")
    new_child = tmp_path / "new.yaml"
    new_child.write_text(
        f"defaults: {parent}\nloss_fn:\n  normalization: global_token_mean\n"
    )
    for child, expected in (
        (old_child, "sequence_mean"),
        (new_child, "global_token_mean"),
    ):
        loss_config = OmegaConf.to_container(load_config(child).loss_fn, resolve=True)
        assert ClippedPGLossConfig(**loss_config).resolved_normalization == expected

    # An explicitly configured legacy parent must be cleared, not silently overridden.
    new_child.write_text(
        f"defaults: {old_child}\nloss_fn:\n  normalization: global_token_mean\n"
    )
    fields = OmegaConf.to_container(load_config(new_child).loss_fn, resolve=True)
    with pytest.raises(ValidationError, match="token_level_loss=null"):
        ClippedPGLossConfig(**fields)
    new_child.write_text(
        f"defaults: {old_child}\nloss_fn:\n  token_level_loss: null\n  normalization: global_token_mean\n"
    )
    fields = OmegaConf.to_container(load_config(new_child).loss_fn, resolve=True)
    assert ClippedPGLossConfig(**fields).resolved_normalization == "global_token_mean"
