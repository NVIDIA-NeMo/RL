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
"""Validate CSD artifact choices before training allocates workers."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_rl.algorithms.xtoken_off_policy_distillation import CSDMasterConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


@pytest.mark.parametrize(
    ("common_from_tables", "projection", "cross_tokenizer", "valid"),
    [
        (True, None, True, True),
        (False, "projection.pt", True, True),
        (False, None, True, False),
        (False, None, False, True),
    ],
)
def test_csd_common_event_mapping_requires_the_selected_artifact(
    common_from_tables: bool,
    projection: str | None,
    cross_tokenizer: bool,
    valid: bool,
) -> None:
    register_omegaconf_resolvers()
    config_path = Path(__file__).resolve().parents[4] / "examples/configs/csd.yaml"
    config = OmegaConf.to_container(load_config(str(config_path)), resolve=True)
    config["loss_fn"]["common_indices_from_subtoks"] = common_from_tables
    aligner = config["teachers"][0]["aligner"]
    aligner["projection_matrix_path"] = projection
    if not cross_tokenizer:
        aligner["student_to_teacher_table_path"] = None
        aligner["teacher_to_student_table_path"] = None

    if valid:
        validated = CSDMasterConfig.model_validate(config)
        assert validated.loss_fn.common_indices_from_subtoks == common_from_tables
    else:
        with pytest.raises(ValidationError, match="projection_matrix_path is required"):
            CSDMasterConfig.model_validate(config)
