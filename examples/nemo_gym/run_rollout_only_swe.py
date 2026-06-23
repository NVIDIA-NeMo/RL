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

"""Rollout-only latency harness for async SWE GRPO ("Plan Y").

Purpose
-------
Measure rollout (generation) end-to-end latency / throughput across vLLM
parallel + replica configurations WITHOUT running any training, while keeping
the *exact* rollout code path that ``async_grpo_train`` uses for bihu's run.

How it stays faithful to the async-training rollout path
--------------------------------------------------------
The trajectory generation here is driven by the very same
``AsyncTrajectoryCollector`` + ``ReplayBuffer`` + ``run_async_nemo_gym_rollout``
machinery as ``nemo_rl.algorithms.grpo.async_grpo_train``. In particular the
in-flight concurrency is identical: the collector caps in-flight prompt groups
at ``num_prompts_per_step * max_trajectory_age_steps`` (see
``AsyncTrajectoryCollector.__init__``).

The only difference from real async training is intentional and is the accepted
trade-off of Plan Y: there is **no policy and no in-flight weight refit**. vLLM
loads the model once from ``policy.model_name`` and weights stay frozen. We
therefore drop the refit-pause perturbation but reproduce everything else
(per-prompt streaming, concurrency, agent loop, vLLM config).

The collector throttles itself once it has generated for all currently-allowed
target weight versions (gated by ``max_trajectory_age_steps``). In real training
the trainer drains the buffer and advances the weight version, which unblocks
the collector. Here a lightweight "fake trainer" loop reproduces exactly that
buffer interaction (``sample`` + ``set_weight_version``) but does no training and
no refit -- it just records the rollout latency of each sampled group.

Usage
-----
Launched via ``test_assets/SWE/run_rollout_only_swe.sh``. Accepts the same
``--config`` + hydra-style overrides as ``run_grpo_nemo_gym.py``.
"""

import argparse
import gzip
import json
import os
import subprocess
import pprint
import statistics
import time

# Increase the W&B single object size warning threshold (mirrors run_grpo_nemo_gym.py).
import wandb.util

wandb.util.VALUE_BYTES_LIMIT = 10_000_000

import ray
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.async_utils import AsyncTrajectoryCollector, ReplayBuffer
from nemo_rl.algorithms.grpo import (
    MasterConfig,
    _should_use_nemo_gym,
)
from nemo_rl.algorithms.utils import get_tokenizer, log_generation_metrics_to_wandb
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.environments.nemo_gym import NemoGymConfig, setup_nemo_gym_config
from nemo_rl.environments.utils import create_env
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import Logger, get_next_experiment_dir

# Per-prompt-group end-to-end rollout latency is emitted by run_async_nemo_gym_rollout
# under this key (see nemo_rl/experience/rollouts.py: timer_prefix = "timing/rollout").
LATENCY_METRIC_KEY = "timing/rollout/total"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments (mirrors run_grpo_nemo_gym.py)."""
    parser = argparse.ArgumentParser(
        description="Rollout-only latency harness for async SWE GRPO"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        default=int(os.environ.get("TARGET_STEPS", "5")),
        help="Stop after this many fake-trainer steps (each step consumes PPS prompt groups).",
    )
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=int(os.environ.get("ROLLOUT_MAX_SECONDS", "0")) or None,
        help="Optional wall-clock cap (seconds). 0/unset disables it.",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def _build_master_config(args: argparse.Namespace, overrides: list[str]) -> MasterConfig:
    """Load YAML + apply hydra overrides, exactly like run_grpo_nemo_gym.py."""
    register_omegaconf_resolvers()
    assert args.config is not None, "--config is required"
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)
    config = OmegaConf.to_container(config, resolve=True)
    config = MasterConfig(**config)

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"Using log directory: {config.logger['log_dir']}")
    return config


def _write_run_manifest(config: MasterConfig, args: argparse.Namespace) -> None:
    """Dump a self-contained, reproducible record of this run's settings into
    the experiment log dir. Written EARLY (before vLLM init) so it exists even
    if the run later OOMs / fails — important for the TP sweep where some
    configs are expected to fail."""
    log_dir = config.logger["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    gen = config.policy["generation"]
    tp = gen["vllm_cfg"]["tensor_parallel_size"]
    pp = gen["vllm_cfg"].get("pipeline_parallel_size", 1)
    nodes = config.cluster["num_nodes"]
    gpn = config.cluster["gpus_per_node"]
    replica = (nodes * gpn) // (tp * pp) if tp * pp else None
    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
            ).stdout.strip()
            or None
        )
    except Exception:
        commit = None
    env_keys = ["SLURM_JOB_ID", "VLLM_TP", "VLLM_PP", "NUM_NODES",
                "TARGET_STEPS", "ROLLOUT_MAX_SECONDS"]
    manifest = {
        "git_commit": commit,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "model_name": config.policy["model_name"],
        "vllm_tp": tp,
        "vllm_pp": pp,
        "num_nodes": nodes,
        "gpus_per_node": gpn,
        "replica_count": replica,
        "train_global_batch_size": config.policy["train_global_batch_size"],
        "num_prompts_per_step": config.grpo["num_prompts_per_step"],
        "num_generations_per_prompt": config.grpo["num_generations_per_prompt"],
        "max_trajectory_age_steps": config.grpo["async_grpo"]["max_trajectory_age_steps"],
        "max_total_sequence_length": config.policy["max_total_sequence_length"],
        "target_steps": args.target_steps,
        "max_seconds": args.max_seconds,
        "launch_env": {k: os.environ.get(k) for k in env_keys},
        "relaunch_cmd": (
            f"VLLM_TP={tp} NUM_NODES={nodes} "
            f"TARGET_STEPS={args.target_steps} "
            "bash test_assets/SWE/run_rollout_only_swe.sh"
        ),
        "config": config.model_dump(),
    }
    with open(os.path.join(log_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"📄 Wrote run manifest to {log_dir}/run_manifest.json", flush=True)


def main() -> None:
    """Run rollout-only latency collection."""
    args, overrides = parse_args()
    config = _build_master_config(args, overrides)
    _write_run_manifest(config, args)

    # --- Tokenizer + generation/nemo-gym config (same prep as the GRPO entrypoint) ---
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    assert config.policy["generation"] is not None, "A generation config is required"
    # is_eval=True => vLLM load_format="auto" so it loads the REAL checkpoint from
    # model_name directly (exactly like run_eval.py). Without this, NeMo-RL sets
    # load_format="dummy" (random weights) expecting a training refit to populate
    # them — but this harness does no refit, so dummy weights make the model
    # generate garbage (zero reward, runaway-length rollouts, inflated latency).
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], tokenizer, is_eval=True
    )
    # vLLM needs the model name + hf overrides on the generation config (setup() does this).
    config.policy["generation"]["model_name"] = config.policy["model_name"]
    config.policy["generation"]["vllm_kwargs"]["hf_overrides"] = config.policy.get(
        "hf_config_overrides", {}
    )

    setup_nemo_gym_config(config, tokenizer)
    assert _should_use_nemo_gym(config), (
        "This harness only supports NeMo-Gym rollouts (env.should_use_nemo_gym=true)."
    )
    assert config.policy["generation"]["colocated"]["enabled"] is False, (
        "Rollout-only harness requires non-colocated generation (colocated.enabled=false)."
    )

    # --- Data: use the train dataset as the prompt source for the collector ---
    print("\n▶ Setting up data...")
    train_dataset, _ = setup_response_data(tokenizer, config.data, env_configs=None)

    print("Final config:")
    pprint.pprint(config)

    init_ray()

    # --- Inference-only cluster: ALL GPUs go to vLLM (no training cluster) ---
    gpus_per_node = config.cluster["gpus_per_node"]
    num_nodes = config.cluster["num_nodes"]
    inference_cluster = RayVirtualCluster(
        name="rollout_only_inference_cluster",
        bundle_ct_per_node_list=[gpus_per_node] * num_nodes,
        use_gpus=True,
        num_gpus_per_node=gpus_per_node,
        max_colocated_worker_groups=1,
    )
    print(
        f"  ✓ Inference cluster: {num_nodes} nodes x {gpus_per_node} GPUs "
        f"= {num_nodes * gpus_per_node} GPUs",
        flush=True,
    )

    # --- vLLM generation worker group (loads model_name directly; no policy/refit) ---
    policy_generation = VllmGeneration(
        cluster=inference_cluster, config=config.policy["generation"]
    )
    replica_count = policy_generation.dp_size
    print(
        f"  ✓ vLLM generation initialized: dp_size (replica count) = {replica_count}, "
        f"tp={config.policy['generation']['vllm_cfg']['tensor_parallel_size']}",
        flush=True,
    )
    policy_generation.prepare_for_generation()
    # Reset vLLM's internal metrics logger so generation_metrics start clean.
    policy_generation.clear_logger_metrics()

    # --- NeMo-Gym env (needs the vLLM OpenAI server URLs) ---
    is_trajectory_collection = config.env["nemo_gym"].pop("is_trajectory_collection", False)
    del is_trajectory_collection  # not used in this harness
    nemo_gym_config = NemoGymConfig(
        model_name=policy_generation.cfg["model_name"],
        base_urls=policy_generation.dp_openai_server_base_urls,
        initial_global_config_dict=config.env["nemo_gym"],
    )
    nemo_gym = create_env(env_name="nemo_gym", env_config=nemo_gym_config)
    ray.get(nemo_gym.health_check.remote())
    # Hardcoded key to match run_async_nemo_gym_rollout.
    task_to_env = {"nemo_gym": nemo_gym}

    logger = Logger(config.logger)
    logger.log_hyperparams(config.model_dump())

    # --- Prompt dataloader (mirrors setup()'s init_train_dataloader) ---
    num_prompts_per_step = config.grpo["num_prompts_per_step"]
    dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=num_prompts_per_step,
        shuffle=config.data["shuffle"],
        collate_fn=rl_collate_fn,
        drop_last=True,
        num_workers=config.data["num_workers"],
    )

    # --- ReplayBuffer + AsyncTrajectoryCollector (identical to async_grpo_train) ---
    max_trajectory_age_steps = config.grpo["async_grpo"]["max_trajectory_age_steps"]
    late_arrival_slack = 2
    optimal_buffer_size = (
        num_prompts_per_step * max_trajectory_age_steps * late_arrival_slack
    )

    replay_buffer = ReplayBuffer.options(
        runtime_env=_actor_runtime_env("nemo_rl.algorithms.async_utils.ReplayBuffer")
    ).remote(max_size=optimal_buffer_size)

    trajectory_collector = AsyncTrajectoryCollector.options(
        runtime_env=_actor_runtime_env(
            "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector"
        )
    ).remote(
        policy_generation=policy_generation,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        master_config=config,
        replay_buffer=replay_buffer,
        start_step=0,
    )

    weight_version = 0
    trajectory_collector.start_collection.remote(dataloader)
    trajectory_collector.set_weight_version.remote(weight_version)
    print("📦 Started background trajectory collection (frozen weights)")

    # --- Wait for the initial buffer fill ---
    min_trajectories_needed = num_prompts_per_step
    while ray.get(replay_buffer.size.remote()) < min_trajectories_needed:
        print(
            f"  ⏳ buffer {ray.get(replay_buffer.size.remote())}/{min_trajectories_needed}"
        )
        time.sleep(2.0)
    print("✅ Buffer ready, starting fake-trainer drain loop...")

    # --- Fake-trainer loop: drain buffer + advance weight version (no training) ---
    _drain_and_measure(
        args=args,
        config=config,
        replay_buffer=replay_buffer,
        trajectory_collector=trajectory_collector,
        policy_generation=policy_generation,
        logger=logger,
        replica_count=replica_count,
        start_weight_version=weight_version,
    )

    policy_generation.finish_generation()
    print("✅ Rollout-only run complete.")


def _actor_runtime_env(actor_fqn: str) -> dict:
    """Build the dedicated-venv runtime env for the replay/collector actors.

    Mirrors the lazy per-node venv bootstrap done in async_grpo_train.
    """
    from nemo_rl.utils.venvs import create_local_venv_on_each_node

    py_exec = get_actor_python_env(actor_fqn)
    if py_exec.startswith("uv"):
        py_exec = create_local_venv_on_each_node(py_exec, actor_fqn)
    py_venv = os.path.dirname(os.path.dirname(py_exec))  # strip bin/python
    return {
        "py_executable": py_exec,
        "env_vars": {
            **os.environ,
            "VIRTUAL_ENV": py_venv,
            "UV_PROJECT_ENVIRONMENT": py_venv,
        },
    }


def _drain_and_measure(
    *,
    args: argparse.Namespace,
    config: MasterConfig,
    replay_buffer,
    trajectory_collector,
    policy_generation,
    logger: Logger,
    replica_count: int,
    start_weight_version: int,
) -> None:
    """Repeatedly sample full prompt-group batches and record rollout latency.

    Each ``sample`` returns ``num_prompts_per_step`` prompt groups intended for
    the current weight version (exactly like async_grpo_train). We then bump the
    weight version, which unblocks the collector to generate the next target.
    """
    num_prompts_per_step = config.grpo["num_prompts_per_step"]
    max_trajectory_age_steps = config.grpo["async_grpo"]["max_trajectory_age_steps"]

    # Per-trajectory records are streamed gzip-compressed (one JSON line per
    # sample: timestamps, rollout latency, agent full_result incl. request/
    # response + phase timings + reward). Writing straight to .gz avoids ever
    # producing the multi-GB uncompressed file, and per-step flush keeps it
    # readable up to the last step even if the job is cancelled.
    traj_path = os.path.join(config.logger["log_dir"], "trajectories.jsonl.gz")
    os.makedirs(config.logger["log_dir"], exist_ok=True)
    # compresslevel=1: JSON text still compresses ~5x but at ~100 MB/s, so the
    # per-step compression cost is ~seconds (negligible vs the ~800s rollout),
    # unlike the default level 9 which would be ~minutes/step.
    traj_file = gzip.open(traj_path, "at", compresslevel=1, encoding="utf-8")
    print(f"📝 Writing full trajectories (gzip) to {traj_path}", flush=True)

    weight_version = start_weight_version
    collected = 0  # total prompt groups consumed (PPS per step), for logging only
    step = 0
    all_latencies: list[float] = []
    wall_start = time.time()

    try:
        while step < args.target_steps:
            if args.max_seconds and (time.time() - wall_start) >= args.max_seconds:
                print(f"⏱️ Hit wall-clock cap of {args.max_seconds}s, stopping.")
                break

            sample_result = ray.get(
                replay_buffer.sample.remote(
                    num_prompt_groups=num_prompts_per_step,
                    current_weight_version=weight_version,
                    max_age_steps=max_trajectory_age_steps,
                )
            )
            if (
                sample_result is None
                or len(sample_result["trajectories"]) != num_prompts_per_step
            ):
                time.sleep(0.5)
                continue

            trajectories = sample_result["trajectories"]

            # Per-group end-to-end rollout latency (one run_async_nemo_gym_rollout each).
            step_latencies = [
                float(t["rollout_metrics"][LATENCY_METRIC_KEY])
                for t in trajectories
                if LATENCY_METRIC_KEY in t["rollout_metrics"]
            ]
            all_latencies.extend(step_latencies)

            # Persist full trajectory content for every sample in every group.
            for group_idx, t in enumerate(trajectories):
                _persist_trajectory_group(traj_file, step, group_idx, t)
            traj_file.flush()

            # Aggregate the full rollout metric set the same way the trainer does,
            # then keep only scalar values for W&B (drop pass-through Tables/lists).
            per_group_metrics: dict = {}
            for t in trajectories:
                for k, v in t["rollout_metrics"].items():
                    per_group_metrics.setdefault(k, []).append(v)
            rollout_metrics = _aggregate_scalar_means(per_group_metrics)

            collected += len(trajectories)
            elapsed = time.time() - wall_start

            # Log ONLY bihu's rollout metric set (a subset of it) under the
            # "train/" prefix so keys line up exactly for cross-checking.
            # No custom/extra metrics are added.
            metrics = dict(rollout_metrics)
            logger.log_metrics(metrics, step, prefix="train")

            # ISL/OSL token-length histograms (non-scalar; logged separately to
            # match the training loop's generation_metrics/histogram/* keys).
            for k, vals in per_group_metrics.items():
                if not k.startswith("histogram/"):
                    continue
                merged: list = []
                for v in vals:
                    if isinstance(v, list):
                        merged.extend(v)
                if merged:
                    logger.log_histogram(merged, step, f"generation_metrics/{k}")

            # vLLM generation metrics (KV cache usage, pending samples, inflight
            # batch sizes, generation tokens) -> generation_metrics/* for parity
            # with the training runs. Reset the logger after each step.
            gen_cfg = config.policy["generation"]
            if gen_cfg.get("vllm_cfg", {}).get("enable_vllm_metrics_logger", False):
                try:
                    glm = policy_generation.get_logger_metrics()
                    if glm:
                        # (a) per-worker timeline plot images: generation_metrics/*
                        log_generation_metrics_to_wandb(
                            glm,
                            step,
                            gen_cfg["vllm_cfg"]["vllm_metrics_logger_interval"],
                            logger,
                        )
                        # (b) raw per-worker timeseries as RETRIEVABLE scalars/lists,
                        # exactly like the training loop -> train/generation_logger_metrics.*
                        # (the plot in (a) is only an image; this is what's queryable
                        # and what bihu's runs log, so we can numerically compare).
                        logger.log_metrics(
                            {"generation_logger_metrics": glm}, step, prefix="train"
                        )
                except Exception as e:
                    print(f"⚠️ generation metrics logging failed: {e}", flush=True)
                policy_generation.clear_logger_metrics()

            step_lat_mean = (
                statistics.mean(step_latencies) if step_latencies else float("nan")
            )
            print(
                f"step={step + 1}/{args.target_steps} groups_collected={collected} "
                f"step_lat_mean={step_lat_mean:.1f}s "
                f"throughput={collected / max(elapsed, 1e-6):.3f} groups/s",
                flush=True,
            )

            step += 1
            weight_version += 1
            trajectory_collector.set_weight_version.remote(weight_version)
    finally:
        traj_file.close()

    # Final summary
    if all_latencies:
        print("=" * 60)
        print(f"Rollout latency summary over {len(all_latencies)} prompt groups:")
        print(f"  mean = {statistics.mean(all_latencies):.2f}s")
        print(f"  p50  = {_percentile(all_latencies, 50):.2f}s")
        print(f"  p90  = {_percentile(all_latencies, 90):.2f}s")
        print(f"  replica_count = {replica_count}")
        print(f"  trajectories saved to: {traj_path}")
        print("=" * 60)


def _aggregate_scalar_means(per_group_metrics: dict) -> dict:
    """Aggregate per-group metric lists into scalars for W&B logging.

    Numeric metrics are averaged (min/max for keys ending in /min, /max);
    non-numeric values (e.g. the wandb Table of full_result) are dropped — they
    are persisted to the trajectories jsonl instead. This is a local stand-in
    for the upstream aggregate_rollout_metrics, which is absent at this code
    version.
    """
    out: dict = {}
    for k, vals in per_group_metrics.items():
        nums = [
            v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        if not nums or len(nums) != len(vals):
            continue
        if k.endswith("/min"):
            out[k] = min(nums)
        elif k.endswith("/max"):
            out[k] = max(nums)
        else:
            out[k] = sum(nums) / len(nums)
    return out


def _extract_full_results(rollout_metrics: dict) -> list:
    """Parse per-sample agent full_result dicts out of the ``*/full_result`` Table."""
    out: list = []
    for k, v in rollout_metrics.items():
        if not k.endswith("/full_result"):
            continue
        data = getattr(v, "data", None)
        if not data:
            continue
        for row in data:
            try:
                out.append(json.loads(row[0]))
            except Exception:
                out.append({"_raw": str(row[0])})
    return out


def _persist_trajectory_group(traj_file, step: int, group_idx: int, t: dict) -> None:
    """Write one jsonl line per sample in a prompt-group trajectory.

    The full conversation (request + trajectory) lives inside ``full_result``
    (``responses_create_params.input`` + ``response``); the NeMo-RL message_log
    carries only token_ids with empty text, so it is not persisted.
    """
    rm = t.get("rollout_metrics", {})
    full_results = _extract_full_results(rm)
    n = max(len(full_results), 1)
    for sample_idx in range(n):
        record = {
            "step": step,
            "group_idx": group_idx,
            "sample_idx": sample_idx,
            "group_timestamp": t.get("timestamp"),
            "rollout_total_s": rm.get(LATENCY_METRIC_KEY),
            "full_result": full_results[sample_idx]
            if sample_idx < len(full_results)
            else None,
        }
        traj_file.write(json.dumps(record, default=str) + "\n")


def _percentile(values: list[float], pct: float) -> float:
    """Simple nearest-rank percentile (no numpy dependency needed here)."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[k]


if __name__ == "__main__":
    main()
