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
import hashlib
import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from nemo_rl.distributed.actor_environments import ACTOR_ENVIRONMENTS, _build_stage
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


def test_script_rejects_a_typo_instead_of_printing_nothing():
    """A mistyped stage or skip extra must fail, not emit an empty list.

    The Dockerfile pipes this into a loop, so silently printing zero rows would
    build zero venvs. `test -s` catches that in the deps layer, but the guard is
    what makes the failure say which word was wrong.
    """
    for argv in (["badstage"], ["all", "notanextra"]):
        proc = subprocess.run(
            [sys.executable, str(MODULE_PATH), *argv],
            capture_output=True,
            text=True,
            cwd=git_root,
        )
        assert proc.returncode == 2, f"{argv} should be rejected, got {proc.returncode}"
        assert not proc.stdout.strip(), f"{argv} printed rows despite being invalid"


def test_registry_import_rejects_an_undeclared_extra(monkeypatch):
    """A typo'd extra must fail while IMPORTING the registry, not at venv creation.

    On the image-build path `prefetch_venvs.py` catches the per-actor error and
    exits 0, so without this guard a typo ships a green image missing one venv.
    Importing the module fresh is the point -- calling the check directly would
    still pass if nothing invoked it at import.
    """
    import nemo_rl.distributed.actor_environments as table

    registry_path = (
        Path(git_root) / "nemo_rl" / "distributed" / "ray_actor_environment_registry.py"
    )

    def _import_registry_fresh(name):
        spec = importlib.util.spec_from_file_location(name, registry_path)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))

    _import_registry_fresh("_registry_clean")  # real table: must not raise

    monkeypatch.setattr(
        table,
        "ACTOR_ENVIRONMENTS",
        {
            **table.ACTOR_ENVIRONMENTS,
            "nemo_rl.fake.Worker": ["definitely_not_an_extra"],
        },
    )
    with pytest.raises(ValueError, match="definitely_not_an_extra"):
        _import_registry_fresh("_registry_typo")


def test_script_emits_the_stage_and_extra_flags_each_actor_needs():
    """The Dockerfile builds each venv from columns 2 and 3, not just the name.

    Column 3 is what `uv sync $extras` consumes, and column 2 is what the deps
    layer branches on to leave the TRT-LLM venv base-only until its wheel exists.
    Checking only column 1 lets both go wrong silently: emitting one extra per
    actor, or staging everything as "deps", passes a name-only assertion.
    """
    proc = subprocess.run(
        [sys.executable, str(MODULE_PATH), "all"],
        capture_output=True,
        text=True,
        check=True,
        cwd=git_root,
    )
    rows = {
        line.split("\t")[0]: line.split("\t")[1:]
        for line in proc.stdout.splitlines()
        if line.strip()
    }
    expected = {
        fqn: [_build_stage(extras), " ".join(f"--extra {e}" for e in extras)]
        for fqn, extras in ACTOR_ENVIRONMENTS.items()
        if extras is not None
    }
    assert rows == expected
    assert (
        rows[
            "nemo_rl.models.generation.trtllm.trtllm_worker_async.TrtllmAsyncGenerationWorker"
        ][0]
        == "trtllm"
    )


def test_fingerprint_covers_the_actor_table():
    """Editing an actor's extras must invalidate the container fingerprint.

    Venvs at NEMO_RL_VENV_DIR are reused rather than rebuilt, and nothing prunes
    them (the base sync runs --inexact and `uv run` is inexact by default). So a
    changed extras list has to trip _check_container_fingerprint(), which is what
    tells the user to set NRL_FORCE_REBUILD_VENVS=true.
    """
    spec = importlib.util.spec_from_file_location(
        "_gen_fingerprint", Path(git_root) / "tools" / "generate_fingerprint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fingerprint = module.generate_fingerprint()
    assert "nemo_rl/distributed/actor_environments.py" in fingerprint, (
        "tools/generate_fingerprint.py must hash the actor -> extras table, or a "
        "changed actor environment leaves a stale venv with no warning"
    )
    # Check the value is this file's hash, not merely present and non-empty --
    # pointing the entry at some other file passes the weaker check.
    assert (
        fingerprint["nemo_rl/distributed/actor_environments.py"]
        == hashlib.md5(MODULE_PATH.read_bytes()).hexdigest()
    )


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
