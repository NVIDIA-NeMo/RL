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
"""Launch GRPO with the three isolated, pinned Flint Python profiles."""

import os
import runpy
import sys
from pathlib import Path


def actor_executable_mapping() -> dict[str, str]:
    serving = str(Path(__file__).with_name("python-sglang.sh"))
    return {
        "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor": "/opt/flint/bin/python-driver",
        "nemo_rl.models.policy.workers.automodel_policy_worker.AutomodelPolicyWorker": "/opt/flint/bin/python-training",
        "nemo_rl.models.value.workers.automodel_value_worker.AutomodelValueWorker": "/opt/flint/bin/python-training",
        "nemo_rl.models.generation.sglang.sglang_worker.SGLangGenerationWorker": serving,
        "nemo_rl.models.generation.sglang.sglang_router.RouterActor": serving,
    }


def main() -> None:
    if os.environ.get("CUDA_LAUNCH_BLOCKING") not in (None, "", "0"):
        raise RuntimeError("CUDA_LAUNCH_BLOCKING must be scoped to the serving wrapper")
    for name in [
        "WANDB_RUN_ID",
        "FLINT_FLASHINFER_MANIFEST_SHA256",
        "FLINT_FLASHINFER_RECEIPT_DIR",
    ]:
        if not os.environ.get(name):
            raise RuntimeError(f"Required environment variable is missing: {name}")
    os.environ["NEMO_RL_PY_EXECUTABLES_SYSTEM"] = "1"
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "flint_sglang"
    from nemo_automodel._transformers import registry

    from nemo_rl.distributed.ray_actor_environment_registry import (
        ACTOR_ENVIRONMENT_REGISTRY,
    )

    if "FlintForCausalLM" not in registry.MODEL_ARCH_MAPPING:
        raise RuntimeError("The native Flint Automodel implementation is missing")
    mapping = actor_executable_mapping()
    for executable in set(mapping.values()):
        if not os.access(executable, os.X_OK):
            raise RuntimeError(
                f"Actor executable is missing or not executable: {executable}"
            )
    ACTOR_ENVIRONMENT_REGISTRY.update(mapping)
    from nemo_rl.utils.logger import WandbLogger

    WandbLogger._git_artifacts_skip_reason = staticmethod(
        lambda: "Flint reference launcher records metrics without source artifacts"
    )
    repository = Path(__file__).resolve().parents[4]
    os.chdir(repository)
    sys.argv = [str(repository / "examples/run_grpo.py"), *sys.argv[1:]]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        import ray

        ray.shutdown()


if __name__ == "__main__":
    main()
