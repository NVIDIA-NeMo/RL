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


@pytest.mark.parametrize(
    "fields, expected",
    [
        ({}, "global_token_mean"),
        ({"normalization": None}, "global_token_mean"),
        ({"normalization": "global_token_mean"}, "global_token_mean"),
        ({"normalization": "sequence_mean"}, "sequence_mean"),
        ({"normalization": "group_token_mean"}, "group_token_mean"),
    ],
)
def test_normalization_roundtrip(fields: dict, expected: str) -> None:
    config = ClippedPGLossConfig(**fields)
    saved = config.model_dump()
    assert "resolved_normalization" not in saved
    reloaded = ClippedPGLossConfig.model_validate_json(config.model_dump_json())
    assert reloaded.model_dump() == saved
    assert ClippedPGLossFn(reloaded).normalization == expected


def test_unknown_normalization_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ClippedPGLossConfig(normalization="typo")


@pytest.mark.parametrize("legacy", [True, False, None])
def test_removed_normalization_selector_is_rejected(legacy: bool | None) -> None:
    with pytest.raises(ValidationError, match=r"loss_fn\.token_level_loss was removed"):
        ClippedPGLossConfig(token_level_loss=legacy)


def test_normalization_yaml_inheritance(tmp_path: Path) -> None:
    config_dir = Path(__file__).resolve().parents[3] / "examples" / "configs"
    parent = config_dir / "grpo_math_1B_megatron.yaml"
    sequence_mean = tmp_path / "sequence.yaml"
    sequence_mean.write_text(
        f"defaults: {parent}\nloss_fn:\n  normalization: sequence_mean\n"
    )
    override = tmp_path / "override.yaml"
    override.write_text(f"defaults: {sequence_mean}\nloss_fn:\n  normalization: null\n")
    for config_file, expected in (
        (sequence_mean, "sequence_mean"),
        (override, "global_token_mean"),
    ):
        fields = OmegaConf.to_container(load_config(config_file).loss_fn, resolve=True)
        assert ClippedPGLossConfig(**fields).resolved_normalization == expected
