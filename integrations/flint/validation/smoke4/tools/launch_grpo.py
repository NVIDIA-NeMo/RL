"""Launch the private Flint GRPO check with the container's actor environments."""

import os
import runpy
import sys
from pathlib import Path


def capture_metrics(path):
    """Record the existing logger payload without changing training metrics."""
    import json
    import math
    from functools import wraps

    from nemo_rl.utils.logger import Logger

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    original = Logger.log_metrics

    def normalize(value):
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [normalize(item) for item in value]
        if hasattr(value, "tolist"):
            return normalize(value.tolist())
        if isinstance(value, float) and not math.isfinite(value):
            return {"nonfinite": str(value)}
        return value

    @wraps(original)
    def record(self, metrics, step, prefix="", step_metric=None, step_finished=False):
        original(self, metrics, step, prefix, step_metric, step_finished)
        with destination.open("a") as stream:
            stream.write(json.dumps(normalize({"step": step, "prefix": prefix, "metrics": metrics}), allow_nan=False) + "\n")

    Logger.log_metrics = record


def main():
    os.environ["NEMO_RL_PY_EXECUTABLES_SYSTEM"] = "1"
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "flint_sglang"
    # Register the local, native architecture before AutoConfig is used.
    from nemo_automodel._transformers import registry
    from nemo_rl.distributed.ray_actor_environment_registry import (
        ACTOR_ENVIRONMENT_REGISTRY,
    )

    assert "FlintForCausalLM" in registry.MODEL_ARCH_MAPPING
    mapping = {
        "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor": "driver",
        "nemo_rl.models.policy.workers.automodel_policy_worker.AutomodelPolicyWorker": "training",
        "nemo_rl.models.value.workers.automodel_value_worker.AutomodelValueWorker": "training",
        "nemo_rl.models.generation.sglang.sglang_worker.SGLangGenerationWorker": "sglang",
        "nemo_rl.models.generation.sglang.sglang_router.RouterActor": "sglang",
    }
    for actor, profile in mapping.items():
        executable = Path(f"/opt/flint/bin/python-{profile}")
        if not executable.is_file():
            raise FileNotFoundError(executable)
        ACTOR_ENVIRONMENT_REGISTRY[actor] = str(executable)
    if os.environ.get("FLINT_SYNTHETIC_SMOKE") == "1":
        from nemo_rl.environments.utils import register_env

        register_env("flint_smoke", "flint_smoke_environment.FlintSmokeEnvironment")
        ACTOR_ENVIRONMENT_REGISTRY["flint_smoke_environment.FlintSmokeEnvironment"] = str(
            Path("/opt/flint/bin/python-driver")
        )
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
