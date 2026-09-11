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

"""MLPerf deferred (offline) evaluation for GRPO.

With ``grpo.deferred_evaluation.enabled`` the training run disables inline
validation and instead saves a native checkpoint after every step from
``grpo.val_start_at`` through the final step. Each checkpoint's
``training_info.json`` records the end timestamp of its weight update
(``training_step_end_time_ms``).

Once training stops, this module evaluates the saved checkpoints in step
order against the validation set and stops at the first checkpoint that
reaches the target accuracy. The single ``run_stop`` event is emitted with
the passing checkpoint's weight-update timestamp, so checkpoint writing and
evaluation time are excluded from the score. If no checkpoint crosses the
target, ``run_stop`` carries ``status=aborted`` at the final checkpoint's
weight-update timestamp; operational failures fail the job.

The evaluation runs as a second driver process in the same allocation/Ray
cluster: the training driver releases its actors and placement groups before
exiting (see ``teardown_run_resources``), and the launcher re-enters the
container with ``run_and_time.sh --deferred-eval``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "configure_deferred_evaluation",
    "teardown_run_resources",
    "wait_for_ray_resources_released",
]


def configure_deferred_evaluation(config: dict) -> dict | None:
    """Switch an MLPerf GRPO config to checkpoint-based offline evaluation.

    Disables inline validation and enables native checkpointing (weights
    only) at every step from ``grpo.val_start_at`` on, plus the unconditional
    final-step save. Returns the deferred config when enabled, else None.
    """
    grpo = config["grpo"]
    deferred = dict(grpo.get("deferred_evaluation") or {})
    enabled = deferred.get("enabled", False)
    if not isinstance(enabled, (bool, int)) or enabled not in (0, 1):
        raise ValueError("deferred_evaluation.enabled must be a boolean or 0/1")
    launch_mode = os.environ.get("DEFERRED_OFFLINE_EVAL")
    if launch_mode is not None and bool(enabled) != (launch_mode == "1"):
        raise ValueError(
            "grpo.deferred_evaluation.enabled must match the DEFERRED_OFFLINE_EVAL launcher flag"
        )
    if not enabled:
        return None
    first = grpo.get("val_start_at")
    if type(first) is not int or first < 1:
        raise ValueError("grpo.val_start_at must be a positive integer")
    if type(grpo.get("max_num_steps")) is not int or grpo["max_num_steps"] < first:
        raise ValueError("deferred evaluation requires max_num_steps >= val_start_at")
    if not grpo.get("async_grpo", {}).get("enabled"):
        raise ValueError("deferred evaluation requires async GRPO")
    if config["policy"]["generation"]["backend"] != "vllm":
        raise ValueError("deferred evaluation requires vLLM generation")
    if not config["policy"].get("megatron_cfg", {}).get("enabled"):
        raise ValueError("deferred evaluation requires an MCore policy")
    if config.get("env", {}).get("nemo_gym", {}).get("is_trajectory_collection"):
        raise ValueError("deferred evaluation cannot collect trajectories only")
    threshold = config.get("logger", {}).get("mlperf", {}).get("target_accuracy")
    if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
        raise ValueError("deferred evaluation requires a threshold in [0, 1]")
    checkpointing = config["checkpointing"]
    if checkpointing.get("checkpoint_must_save_by") is not None:
        raise ValueError(
            "deferred evaluation requires checkpoint_must_save_by to be unset"
        )
    root = Path(checkpointing["checkpoint_dir"])
    if root.exists() and any(root.iterdir()):
        raise ValueError(
            f"deferred evaluation requires a fresh checkpoint directory: {root}"
        )

    deferred.update(enabled=True, threshold=float(threshold))
    grpo.update(
        deferred_evaluation=deferred,
        val_period=0,
        val_at_start=False,
        val_at_end=False,
    )
    # The trainer's MLPerf logger leaves the terminal events to the driver
    # (backdated train-block stop) and to this module's evaluator (run_stop).
    config.setdefault("logger", {}).setdefault("mlperf", {})["defer_run_stop"] = True
    checkpointing.update(
        enabled=True,
        metric_name=None,
        save_optimizer=False,
        save_period=first,
        keep_top_k=max(checkpointing.get("keep_top_k") or 0, 2),
    )
    return deferred


def wait_for_ray_resources_released(
    ray_module, placement_groups=(), *, timeout_seconds=600
):
    """Wait for asynchronous actor termination and placement-group removal."""
    expected = ray_module.cluster_resources().get("GPU", 0)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        states = [
            ray_module.util.placement_group_table(group).get("state", "REMOVED")
            for group in placement_groups
        ]
        available = ray_module.available_resources().get("GPU", 0)
        if available >= expected and all(
            state in {"REMOVED", "DEAD"} for state in states
        ):
            return
        time.sleep(1)
    raise RuntimeError(
        f"Ray resources still occupied: GPUs {available}/{expected}, placement groups {states}"
    )


def teardown_run_resources(nemo_gym=None, generation=None, policy=None, clusters=()):
    """Release Ray actors and placement groups so the next phase can reuse the allocation.

    The trainer's own epilogue already stops the actors it created, so the
    training-side caller passes only the clusters; the evaluator passes every
    resource its private setup() created.
    """
    import ray

    clusters = list({id(cluster): cluster for cluster in (clusters or ())}.values())
    placement_groups = [
        group
        for cluster in clusters
        for group in (getattr(cluster, "_node_placement_groups", None) or ())
    ]
    operations = []
    if nemo_gym is not None:
        operations.extend(
            (
                lambda: ray.get(nemo_gym.shutdown.remote(), timeout=120),
                lambda: ray.kill(nemo_gym, no_restart=True),
            )
        )
    for resource in (generation, policy):
        if resource is not None:
            operations.append(resource.shutdown)
    operations.extend(cluster.shutdown for cluster in clusters)
    failures = []
    for operation in operations:
        try:
            operation()
        except Exception as error:
            failures.append(str(error))
    if failures:
        raise RuntimeError("resource teardown failed: " + "; ".join(failures))
    wait_for_ray_resources_released(ray, placement_groups)


def read_checkpoint(checkpoint: Path, step: int) -> tuple[dict, dict]:
    """Load and validate a native checkpoint's config and training info."""
    with (checkpoint / "config.yaml").open() as stream:
        config = yaml.safe_load(stream)
    with (checkpoint / "training_info.json").open() as stream:
        info = json.load(stream)
    if info["current_step"] != step:
        raise ValueError(
            f"{checkpoint}: training_info.current_step differs from {step}"
        )
    if info["training_step_end_time_ms"] <= 0 or info["consumed_samples"] <= 0:
        raise ValueError(f"{checkpoint}: missing training timestamp or sample count")
    if not (checkpoint / "policy" / "weights").is_dir():
        raise FileNotFoundError(f"{checkpoint}: MCore weights directory is missing")
    return config, info


def evaluation_config(config: dict, checkpoint: Path, output: Path) -> dict:
    """Keep the saved policy/generation topology and enable validation only."""
    raw = copy.deepcopy(config)
    raw["logger"].update(
        mlperf_enabled=False,
        wandb_enabled=False,
        tensorboard_enabled=False,
        monitor_gpus=False,
        log_dir=str(output / "runtime"),
    )
    raw["grpo"].update(
        val_period=1,
        val_at_start=False,
        val_at_end=False,
    )
    # setup() selects the latest checkpoint. A private, single-entry root makes
    # its ordinary restore path select the requested endpoint without copying.
    view = output / "checkpoint-view"
    view.mkdir(parents=True)
    (view / checkpoint.name).symlink_to(checkpoint.resolve())
    raw["checkpointing"].update(
        checkpoint_dir=str(view),
        enabled=False,
        save_optimizer=False,
        metric_name=None,
    )
    return raw


def policy_without_optimizer(**kwargs: Any) -> Any:
    from nemo_rl.models.policy.lm_policy import Policy

    kwargs.update(init_optimizer=False, optimizer_path=None, init_reference_model=False)
    return Policy(**kwargs)


def evaluate_checkpoint(
    checkpoint: Path, config: dict, step: int, output: Path, samples: int, mllogger: Any
) -> float:
    """Restore one checkpoint without optimizer state, refit vLLM, and validate."""
    import numpy as np
    import torch

    from nemo_rl.algorithms.grpo import (
        MasterConfig,
        refit_policy_generation,
        setup,
        validate,
    )
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.data.utils import setup_response_data

    seed = int(config["grpo"]["seed"])
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    master_config = MasterConfig(**evaluation_config(config, checkpoint, output))
    tokenizer = get_tokenizer(master_config.policy["tokenizer"])
    train_dataset, val_dataset = setup_response_data(
        tokenizer, master_config.data, env_configs=None
    )
    policy = generation = nemo_gym = None
    clusters: tuple = ()
    try:
        (
            policy,
            generation,
            nemo_gym,
            clusters,
            _dataloader,
            val_dataloader,
            _loss_fn,
            _logger,
            _checkpointer,
            _state,
            materialized_config,
            _teachers,
            _aliases,
        ) = setup(
            master_config,
            tokenizer,
            train_dataset,
            val_dataset,
            policy_factory=policy_without_optimizer,
        )
        if val_dataloader is None:
            raise RuntimeError("setup did not create a validation dataloader")
        refit_policy_generation(
            policy,
            generation,
            bool(materialized_config.policy["generation"]["colocated"]["enabled"]),
        )
        if not generation.invalidate_kv_cache():
            raise RuntimeError("vLLM cache invalidation failed after MCore refit")

        metadata = {"samples_count": samples, "step": step}
        mllogger.start(key="eval_start", metadata=metadata)
        metrics, _timings = validate(
            generation,
            val_dataloader,
            tokenizer,
            {"nemo_gym": nemo_gym},
            step=step,
            master_config=materialized_config,
        )
        # With num_val_generations_per_prompt > 1 the accuracy metric is the
        # benchmark's pass@k convergence metric; with 1 it is plain pass@1.
        accuracy = float(metrics["accuracy"])
        if not 0 <= accuracy <= 1:
            raise ValueError(f"invalid validation accuracy: {accuracy}")
        mllogger.event(
            key="eval_accuracy", value=accuracy, metadata={"samples_count": samples}
        )
        mllogger.end(key="eval_stop", metadata=metadata)
        print(f"Deferred evaluation step {step}: accuracy={accuracy}", flush=True)
        return accuracy
    finally:
        teardown_run_resources(
            nemo_gym=nemo_gym, generation=generation, policy=policy, clusters=clusters
        )


def discover_checkpoints(checkpoint_root: Path) -> list[Path]:
    """List completed checkpoints in step order.

    NeMo-RL publishes completed checkpoints by renaming tmp_step_N to step_N.
    """
    checkpoints = sorted(
        (
            path
            for path in checkpoint_root.glob("step_*")
            if path.is_dir() and path.name.removeprefix("step_").isdigit()
        ),
        key=lambda path: int(path.name.removeprefix("step_")),
    )
    if not checkpoints:
        raise FileNotFoundError(f"no completed checkpoints in {checkpoint_root}")
    return checkpoints


def evaluate_endpoints(
    checkpoint_root: Path, log_dir: Path, config: dict, mllogger: Any
) -> None:
    """Evaluate the saved checkpoints in step order; stop at the first pass."""
    grpo = config["grpo"]
    threshold = float(grpo["deferred_evaluation"]["threshold"])
    checkpoints = discover_checkpoints(checkpoint_root)
    for checkpoint in checkpoints:
        step = int(checkpoint.name.removeprefix("step_"))
        saved_config, info = read_checkpoint(checkpoint, step)
        samples = (
            info["consumed_samples"]
            * saved_config["grpo"]["num_generations_per_prompt"]
        )
        accuracy = evaluate_checkpoint(
            checkpoint, saved_config, step, log_dir / f"step_{step}", samples, mllogger
        )
        passed = accuracy >= threshold
        if passed or checkpoint == checkpoints[-1]:
            mllogger.end(
                key="run_stop",
                metadata={
                    "status": "success" if passed else "aborted",
                    "samples_count": samples,
                },
                time_ms=int(info["training_step_end_time_ms"]),
            )
            break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()

    from mlperf_logging import mllog

    from nemo_rl.distributed.virtual_cluster import init_ray
    from nemo_rl.utils.config import register_omegaconf_resolvers

    checkpoints = discover_checkpoints(args.checkpoint_root)
    config, _info = read_checkpoint(
        checkpoints[0], int(checkpoints[0].name.removeprefix("step_"))
    )

    # Append to the training run's MLPerf log.
    log_file = (
        os.environ.get("MLPERF_MLLOG_FILE") or config["logger"]["mlperf"]["log_file"]
    )
    mllog.config(filename=log_file)
    register_omegaconf_resolvers()
    init_ray()
    evaluate_endpoints(args.checkpoint_root, args.log_dir, config, mllog.get_mllogger())


if __name__ == "__main__":
    main()
