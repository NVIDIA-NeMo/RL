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
import re
from pathlib import Path

import pytest

from nemo_rl.utils.outdated_config_checks import check_outdated_config

EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples"
entrypoints = sorted(EXAMPLES_DIR.rglob("run_*.py"))
assert entrypoints, "No entrypoints found"

# Matches both construction styles in use: MasterConfig(**resolved) and
# MasterConfig.model_validate(resolved).
_BUILDS_MASTER_CONFIG = re.compile(r"MasterConfig(\(\*\*|\.model_validate)")


@pytest.mark.parametrize("entrypoint", entrypoints, ids=lambda p: p.name)
def test_every_entrypoint_checks_outdated_config(entrypoint):
    """An entrypoint that builds a MasterConfig must also reject outdated config.

    Without this, a new run_*.py silently skips the check and a stale config gets as far
    as worker startup before failing.
    """
    source = entrypoint.read_text()
    if not _BUILDS_MASTER_CONFIG.search(source):
        pytest.skip(f"{entrypoint.name} does not build a MasterConfig")

    assert "check_outdated_config(" in source, (
        f"{entrypoint.name} builds a MasterConfig but never calls "
        "check_outdated_config(). Add the call right after the config is built."
    )


def test_absent_key_passes():
    check_outdated_config({"policy": {"dtensor_cfg": {"enabled": True}}})


@pytest.mark.parametrize("value", [True, False])
def test_outdated_dtensor_v2_key_is_rejected(value):
    with pytest.raises(ValueError, match=r"policy\.dtensor_cfg\._v2"):
        check_outdated_config({"policy": {"dtensor_cfg": {"_v2": value}}})


@pytest.mark.parametrize("section", ["policy", "value"])
def test_both_top_level_sections_are_checked(section):
    with pytest.raises(ValueError, match=rf"{section}\.dtensor_cfg\._v2"):
        check_outdated_config({section: {"dtensor_cfg": {"_v2": True}}})


def test_each_teacher_is_checked():
    with pytest.raises(ValueError, match=r"teachers\.1\.dtensor_cfg\._v2"):
        check_outdated_config(
            {"teachers": [{"dtensor_cfg": {}}, {"dtensor_cfg": {"_v2": False}}]}
        )
