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

"""``RAY_process_group_cleanup_enabled`` must reach fleet-health runs and nothing else.

It was originally set at ``import nemo_rl``, which handed a raylet-wide behaviour change
to every SFT, DPO and distillation run in the repo -- none of which own an EngineCore to
reap. See ``maybe_configure_engine_reaping_env`` for what the flag is for.
"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_rl.models.generation import maybe_configure_engine_reaping_env

REPO_ROOT = Path(__file__).resolve().parents[4]
FLAG = "RAY_process_group_cleanup_enabled"


@pytest.fixture(autouse=True)
def _clean_flag():
    """The function mutates process state, so no test may leak it into the next."""
    before = os.environ.pop(FLAG, None)
    yield
    os.environ.pop(FLAG, None)
    if before is not None:
        os.environ[FLAG] = before


class TestTheGate:
    def test_a_fleet_health_run_gets_the_flag(self):
        maybe_configure_engine_reaping_env(SimpleNamespace(enabled=True))
        assert os.environ[FLAG] == "1"

    @pytest.mark.parametrize(
        "cfg",
        [None, SimpleNamespace(enabled=False)],
        ids=["no-fleet-health-at-all", "fleet-health-off"],
    )
    def test_everything_else_is_left_alone(self, cfg):
        maybe_configure_engine_reaping_env(cfg)
        assert FLAG not in os.environ

    def test_an_explicit_operator_setting_wins(self):
        """setdefault, not assignment: someone debugging a wedged engine wants the corpse."""
        os.environ[FLAG] = "0"
        maybe_configure_engine_reaping_env(SimpleNamespace(enabled=True))
        assert os.environ[FLAG] == "0"


class TestItIsNotGlobal:
    """The regression that matters: a plain training run must not inherit the flag.

    Asserted against the source rather than by importing, because ``import nemo_rl``
    has already happened by the time this test runs and any assertion about
    ``os.environ`` would be measuring the test session, not the import.
    """

    def test_importing_nemo_rl_does_not_set_it(self):
        init_py = (REPO_ROOT / "nemo_rl" / "__init__.py").read_text()
        offenders = [
            line
            for line in init_py.splitlines()
            if FLAG in line and not line.lstrip().startswith("#")
        ]
        assert offenders == [], (
            f"nemo_rl/__init__.py sets {FLAG} at import time: {offenders}. "
            "That applies a raylet-wide change to every run in the repo, including "
            "pure-training lanes with no EngineCore to reap. Gate it on fleet health "
            "in maybe_configure_engine_reaping_env instead."
        )


class TestTheCallSite:
    """The flag is read by the raylet that ``ray.init`` spawns, so ordering is load-bearing.

    Same shape as the membership-ordering check in
    ``tests/unit/models/policy/test_worker_refit_signatures.py``: setting the variable
    after ``init_ray()`` would be silently too late rather than an error.
    """

    ENTRYPOINT = REPO_ROOT / "examples" / "run_grpo_single_controller.py"

    def _call_lines(self, name):
        tree = ast.parse(self.ENTRYPOINT.read_text())
        return [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        ]

    def test_the_sc_entrypoint_configures_reaping_before_it_starts_ray(self):
        reaping = self._call_lines("maybe_configure_engine_reaping_env")
        init_ray = self._call_lines("init_ray")
        assert reaping, (
            "run_grpo_single_controller.py no longer calls "
            "maybe_configure_engine_reaping_env; every shard killed by fleet health "
            "will leak its EngineCore's GPU for the rest of the run."
        )
        assert init_ray, "expected run_grpo_single_controller.py to call init_ray()"
        assert max(reaping) < min(init_ray), (
            f"maybe_configure_engine_reaping_env at line {max(reaping)} runs after "
            f"init_ray() at line {min(init_ray)}; the raylet has already been spawned "
            "with the old environment by then."
        )
