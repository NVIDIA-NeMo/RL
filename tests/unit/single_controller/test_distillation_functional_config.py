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
"""The distillation functional test's overrides must survive SC validation.

`tests/functional/distillation_single_controller.sh` needs two GPUs, so nobody
finds a bad override until a two-GPU runner picks it up. Everything up to the
first model load is pure config resolution, though, and that part runs here.

This caught a real one: `max_buffered_rollouts=6` sits below the `in_order`
sampler's required capacity of `num_prompts_per_step * (max_lookahead_versions
+ 1)` = 8, which `validate_sampler_buffer_capacity` rejects. The script would
have died at setup on the first paid run.

Reads the overrides out of the shell script rather than restating them, so the
two cannot drift.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    is_distillation_run,
    validate_single_controller_config,
)
from nemo_rl.utils.config import (
    load_config_with_inheritance,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "tests/functional/distillation_single_controller.sh"
_CONFIG = (
    _REPO / "examples/configs/distillation_math_1B_megatron_single_controller.yaml"
)


def _overrides_from_script() -> list[str]:
    """The `key=value` lines inside the script's TRAIN_CMD array."""
    src = _SCRIPT.read_text()
    start = src.index("TRAIN_CMD=(")
    block = src[start : src.index("\n)", start)]
    out = []
    for line in block.split("\n"):
        stripped = line.strip()
        if not re.match(r"^\+?\+?[a-z][a-z0-9_.]*=", stripped):
            continue
        # The script interpolates a temp dir here; any path resolves the same.
        out.append(stripped.replace('"${CKPT_DIR}"', "/tmp/ckpt"))
    return out


def test_the_functional_test_overrides_pass_sc_validation():
    register_omegaconf_resolvers()
    overrides = _overrides_from_script()
    assert len(overrides) > 20, (
        f"only parsed {len(overrides)} overrides -- parser drifted"
    )

    config = parse_hydra_overrides(
        load_config_with_inheritance(str(_CONFIG)),
        # max_num_steps and log_dir are appended per-invocation by the script.
        overrides + ["distillation.max_num_steps=2", "logger.log_dir=/tmp/logs"],
    )
    master_config = MasterConfig(**OmegaConf.to_container(config, resolve=True))

    assert is_distillation_run(master_config)
    validate_single_controller_config(master_config)


def test_the_script_still_points_at_the_shipped_recipe():
    """A renamed recipe would leave the script resolving nothing, and this file
    would keep passing against a config the script no longer uses."""
    assert _CONFIG.name in _SCRIPT.read_text()


@pytest.mark.parametrize(
    ("key", "why"),
    [
        (
            "policy.generation.colocated.enabled=false",
            "SC requires disaggregated generation",
        ),
        ("checkpointing.metric_name=null", "SC collects no val: metrics"),
    ],
)
def test_the_constraints_sc_enforces_are_actually_set(key, why):
    assert key in _SCRIPT.read_text(), why
