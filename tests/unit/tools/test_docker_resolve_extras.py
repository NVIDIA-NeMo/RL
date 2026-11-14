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

import subprocess
from pathlib import Path

import pytest

from nemo_rl.distributed.actor_environments import ACTOR_ENVIRONMENTS

REPO_ROOT = Path(__file__).parents[3]
SCRIPT = REPO_ROOT / "docker" / "resolve_extras.sh"
ALL_EXTRAS = "vllm sglang mcore automodel modelopt nemo_gym trtllm"


def _run(function: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-euc", f'source "{SCRIPT}"; {function}'],
        env={"PATH": "/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
    )


def test_all_extras_matches_actor_environments() -> None:
    actor_extras = {e for extras in ACTOR_ENVIRONMENTS.values() for e in extras or ()}
    assert set(ALL_EXTRAS.split()) == actor_extras
    assert _run('echo "$NRL_ALL_EXTRAS"').stdout.strip() == ALL_EXTRAS


@pytest.mark.parametrize(
    ("env", "wanted", "skipped"),
    [
        ({}, ALL_EXTRAS, ""),
        (
            {"NRL_EXTRAS": "nemo_gym mcore vllm"},
            "vllm mcore nemo_gym",
            "sglang automodel modelopt trtllm",
        ),
        (
            {"NRL_EXTRAS": "mcore"},
            "mcore",
            "vllm sglang automodel modelopt nemo_gym trtllm",
        ),
    ],
)
def test_wanted_and_skipped_extras(
    env: dict[str, str], wanted: str, skipped: str
) -> None:
    assert _run("nrl_wanted_extras", **env).stdout.strip() == wanted
    assert _run("nrl_skipped_extras", **env).stdout.strip() == skipped


def test_unknown_extra_fails() -> None:
    result = _run("nrl_wanted_extras", NRL_EXTRAS="mcore megatron")
    assert result.returncode != 0
    assert "unknown extra: 'megatron'" in result.stderr
    assert _run("nrl_skipped_extras", NRL_EXTRAS="mcore megatron").returncode != 0


@pytest.mark.parametrize(
    "removed", ["SKIP_VLLM_BUILD", "SKIP_SGLANG_BUILD", "SKIP_TRTLLM_BUILD"]
)
def test_removed_skip_arg_fails(removed: str) -> None:
    result = _run("nrl_wanted_extras", **{removed: "1"})
    assert result.returncode != 0
    assert f"{removed} was removed" in result.stderr
    assert "NRL_EXTRAS" in result.stderr
    assert _run("nrl_skipped_extras", **{removed: "1"}).returncode != 0
