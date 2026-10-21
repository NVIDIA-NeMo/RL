#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Retry native Flint GRPO with CUDA launch blocking scoped to SGLang actors.

This is a diagnostic launcher, not a correction to the failing CUDA kernel.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import runpy
import sys
from pathlib import Path

BASE_LAUNCHER = Path("/flint-work/tools/launch_grpo.py")
BASE_LAUNCHER_SHA256 = (
    "7fa8b1359e55d4c08882f730f39ae1535e44551073a3dab0ec0ed646244ed39a"
)
SGLANG_DIAGNOSTIC_EXECUTABLE = "/flint-run/tools/python-sglang-diagnostic.sh"


def actor_executable_mapping(*, synthetic_smoke=False, validate=True):
    """Return the native actor mapping without importing Ray or GPU packages."""
    mapping = {
        "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor": "/opt/flint/bin/python-driver",
        "nemo_rl.models.policy.workers.automodel_policy_worker.AutomodelPolicyWorker": "/opt/flint/bin/python-training",
        "nemo_rl.models.value.workers.automodel_value_worker.AutomodelValueWorker": "/opt/flint/bin/python-training",
        "nemo_rl.models.generation.sglang.sglang_worker.SGLangGenerationWorker": SGLANG_DIAGNOSTIC_EXECUTABLE,
        "nemo_rl.models.generation.sglang.sglang_router.RouterActor": SGLANG_DIAGNOSTIC_EXECUTABLE,
    }
    if synthetic_smoke:
        mapping["flint_smoke_environment.FlintSmokeEnvironment"] = (
            "/opt/flint/bin/python-driver"
        )
    if validate:
        for value in sorted(set(mapping.values())):
            executable = Path(value)
            if not executable.is_file():
                raise FileNotFoundError(executable)
            if not os.access(executable, os.X_OK):
                raise PermissionError(
                    f"Actor executable is not executable: {executable}"
                )
        if not Path("/opt/flint/bin/python-sglang").is_file():
            raise FileNotFoundError("/opt/flint/bin/python-sglang")
        if not os.access("/opt/flint/bin/python-sglang", os.X_OK):
            raise PermissionError(
                "SGLang profile is not executable: /opt/flint/bin/python-sglang"
            )
    return mapping


def load_reviewed_capture_metrics():
    """Import only the pinned launcher's helpers, using its verified bytes."""
    if BASE_LAUNCHER.is_symlink() or BASE_LAUNCHER.resolve() != BASE_LAUNCHER:
        raise RuntimeError("Refusing a redirected base launcher")
    source = BASE_LAUNCHER.read_bytes()
    observed = hashlib.sha256(source).hexdigest()
    if observed != BASE_LAUNCHER_SHA256:
        raise RuntimeError(f"Unknown base launcher hash: {observed}")
    specification = importlib.util.spec_from_file_location(
        "flint_reviewed_launcher_helpers", BASE_LAUNCHER
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Cannot import the reviewed base launcher")
    module = importlib.util.module_from_spec(specification)
    # Execute the bytes already checked above; no second source read, bytecode
    # cache, or write to the read-only earlier workspace is needed.
    exec(compile(source, str(BASE_LAUNCHER), "exec"), module.__dict__)
    if not callable(getattr(module, "capture_metrics", None)):
        raise RuntimeError("Reviewed launcher has no metrics-capture helper")
    return module.capture_metrics


def main():
    capture_metrics = load_reviewed_capture_metrics()
    # A driver-wide setting would also force synchronous Automodel execution.
    # The SGLang executable wrapper supplies the diagnostic setting per actor.
    if os.environ.get("CUDA_LAUNCH_BLOCKING") not in (None, "", "0"):
        raise RuntimeError(
            "CUDA_LAUNCH_BLOCKING must be scoped to the SGLang diagnostic executable"
        )
    os.environ["NEMO_RL_PY_EXECUTABLES_SYSTEM"] = "1"
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "flint_sglang"
    assert os.environ["NEMO_RL_PY_EXECUTABLES_SYSTEM"] == "1"

    # Preserve native Flint registration before AutoConfig is used.
    from nemo_automodel._transformers import registry

    from nemo_rl.distributed.ray_actor_environment_registry import (
        ACTOR_ENVIRONMENT_REGISTRY,
    )

    assert "FlintForCausalLM" in registry.MODEL_ARCH_MAPPING
    synthetic_smoke = os.environ.get("FLINT_SYNTHETIC_SMOKE") == "1"
    ACTOR_ENVIRONMENT_REGISTRY.update(
        actor_executable_mapping(synthetic_smoke=synthetic_smoke)
    )
    if synthetic_smoke:
        from nemo_rl.environments.utils import register_env

        register_env("flint_smoke", "flint_smoke_environment.FlintSmokeEnvironment")
    if os.environ.get("FLINT_METRICS_PATH"):
        capture_metrics(os.environ["FLINT_METRICS_PATH"])
    sys.argv = ["examples/run_grpo.py", *sys.argv[1:]]
    try:
        runpy.run_path("examples/run_grpo.py", run_name="__main__")
    finally:
        import ray

        ray.shutdown()


if __name__ == "__main__":
    main()
