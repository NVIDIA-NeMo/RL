# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The math exemplar must not select the dataset for the SWE training recipe."""

import copy
from pathlib import Path

import pytest
from nano35.check_training_data import validate_data_config
from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE = (
    REPO_ROOT
    / "examples/configs/recipes/llm/grpo-nano3.5-swe-32n4g-tp4cp16-async-trtllm.v1.yaml"
)


@pytest.fixture
def data_config(monkeypatch, tmp_path):
    monkeypatch.setenv("NANO35_DATA", str(tmp_path / "swe.jsonl"))
    register_omegaconf_resolvers()
    return OmegaConf.to_container(load_config(RECIPE).data, resolve=True)


def test_inherited_recipe_selects_only_swe_data(data_config):
    validate_data_config(
        data_config, expected_path=Path(data_config["train"]["data_path"])
    )
    assert data_config["train"]["dataset_name"] == "NemoGymDataset"
    assert "split_validation_size" not in data_config["train"]
    assert "seed" not in data_config["train"]


def test_failed_job_configuration_is_rejected_before_loading(data_config):
    failed = copy.deepcopy(data_config)
    failed["train"].update(
        dataset_name="OpenMathInstruct-2", split_validation_size=0.05
    )
    with pytest.raises(
        ValueError, match=r"data.train.dataset_name must be NemoGymDataset"
    ):
        validate_data_config(
            failed, expected_path=Path(data_config["train"]["data_path"])
        )


@pytest.mark.parametrize("split", ["train", "validation"])
def test_wrong_input_path_is_rejected(data_config, split, tmp_path):
    expected_path = Path(data_config["train"]["data_path"])
    data_config[split]["data_path"] = str(tmp_path / "other.jsonl")
    with pytest.raises(ValueError, match="differs from the recorded input"):
        validate_data_config(data_config, expected_path=expected_path)


def test_validation_split_cannot_silently_remove_training_rows(data_config):
    data_config["train"]["split_validation_size"] = 0.05
    with pytest.raises(ValueError, match="must not split"):
        validate_data_config(
            data_config, expected_path=Path(data_config["train"]["data_path"])
        )
