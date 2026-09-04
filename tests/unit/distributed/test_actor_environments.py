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
"""Guards on ACTOR_ENVIRONMENTS, the actor -> uv extras table.

docker/Dockerfile runs nemo_rl/distributed/actor_environments.py as a script from
the dependency layer to decide which venvs to pre-build, and the runtime registry
imports the same dict. These tests keep the two readers honest.
"""

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from nemo_rl.distributed.actor_environments import ACTOR_ENVIRONMENTS
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
    USE_SYSTEM_EXECUTABLE,
)
from nemo_rl.distributed.virtual_cluster import (
    PY_EXECUTABLES,
    git_root,
    uv_py_executable,
)

MODULE_PATH = Path(git_root) / "nemo_rl" / "distributed" / "actor_environments.py"

with open(Path(git_root) / "pyproject.toml", "rb") as _f:
    DECLARED_EXTRAS = set(tomllib.load(_f)["project"]["optional-dependencies"])


@pytest.mark.parametrize("actor_fqn", sorted(ACTOR_ENVIRONMENTS))
def test_actor_extras_are_declared(actor_fqn):
    """Every extra names a real [project.optional-dependencies] entry."""
    extras = ACTOR_ENVIRONMENTS[actor_fqn]
    if extras is None:
        return
    assert isinstance(extras, list) and all(isinstance(e, str) for e in extras), (
        f"{actor_fqn}: value must be None or a list of extras, got {extras!r}"
    )
    undeclared = set(extras) - DECLARED_EXTRAS
    assert not undeclared, (
        f"{actor_fqn} names extras {sorted(undeclared)} that are not in "
        "[project.optional-dependencies] of pyproject.toml"
    )


@pytest.mark.parametrize("actor_fqn", sorted(ACTOR_ENVIRONMENTS))
def test_actor_module_exists(actor_fqn):
    """The FQN still points at a module that exists.

    Parsed, not imported: most of these pull in vllm or megatron.
    """
    module_name, _, class_name = actor_fqn.rpartition(".")
    path = Path(git_root) / (module_name.replace(".", "/") + ".py")
    if not path.exists():
        path = Path(git_root) / module_name.replace(".", "/") / "__init__.py"
    assert path.exists(), f"{actor_fqn}: no module file for {module_name}"

    defined = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            defined.update(a.asname or a.name.split(".")[-1] for a in node.names)
    assert class_name in defined, f"{actor_fqn}: {class_name} not defined in {path}"


@pytest.mark.skipif(
    USE_SYSTEM_EXECUTABLE,
    reason="NEMO_RL_PY_EXECUTABLES_SYSTEM=1 puts every actor on the driver interpreter",
)
def test_registry_matches_py_executables():
    """The generated py_executable is the string the worker actually needs."""
    expected = {
        ("vllm",): PY_EXECUTABLES.VLLM,
        ("sglang",): PY_EXECUTABLES.SGLANG,
        ("fsdp",): PY_EXECUTABLES.FSDP,
        ("automodel",): PY_EXECUTABLES.AUTOMODEL,
        ("mcore",): PY_EXECUTABLES.MCORE,
        ("trtllm",): PY_EXECUTABLES.TRTLLM,
        ("nemo_gym",): PY_EXECUTABLES.NEMO_GYM,
    }
    for actor_fqn, extras in ACTOR_ENVIRONMENTS.items():
        got = ACTOR_ENVIRONMENT_REGISTRY[actor_fqn]
        if extras is None:
            assert got == PY_EXECUTABLES.SYSTEM, actor_fqn
        else:
            assert got == expected.get(tuple(extras), uv_py_executable(extras)), (
                actor_fqn
            )


def test_actor_environments_module_is_stdlib_only():
    """docker/Dockerfile runs this module from the dependency layer.

    Only pyproject.toml, uv.lock and a couple of nemo_rl files exist there, so an
    import of anything else -- especially anything from nemo_rl -- breaks the image
    build in its most expensive layer.
    """
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    tree = ast.parse(MODULE_PATH.read_text())
    for node in ast.walk(tree):
        roots = []
        if isinstance(node, ast.Import):
            roots = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = [(node.module or "").split(".")[0]]
        for root in roots:
            assert root in stdlib, (
                f"{MODULE_PATH.name} imports {root!r}, which is not in the standard "
                "library. This module must stay dependency-free -- docker/Dockerfile "
                "runs it from a layer where only pyproject.toml and uv.lock exist."
            )


def test_script_output_matches_the_registry():
    """Running the module as a script lists exactly the venvs the runtime expects."""
    proc = subprocess.run(
        [sys.executable, str(MODULE_PATH), "all"],
        capture_output=True,
        text=True,
        check=True,
        cwd=git_root,
    )
    listed = {line.split("\t")[0] for line in proc.stdout.splitlines() if line.strip()}
    expected = {fqn for fqn, extras in ACTOR_ENVIRONMENTS.items() if extras is not None}
    assert listed == expected


def test_script_skips_by_extra_not_by_name():
    """SKIP_VLLM_BUILD must drop actors that need vllm even without 'vllm' in the name."""
    proc = subprocess.run(
        [sys.executable, str(MODULE_PATH), "all", "vllm"],
        capture_output=True,
        text=True,
        check=True,
        cwd=git_root,
    )
    listed = {line.split("\t")[0] for line in proc.stdout.splitlines() if line.strip()}
    assert "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector" not in listed
    assert "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor" not in listed
    assert (
        "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
        in listed
    )
