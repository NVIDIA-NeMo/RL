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

"""Pipelined OSWorld REINFORCE with LoGRA and finite-difference KL control."""

import argparse
import json
import math
import os
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, cast

import ray
import torch
from logra.actor_environments import register_actor_environments
from logra.config import LoGRAConfig
from logra.cua import CuaConfig, TaskBaseline, make_cua_loss
from logra.evaluation_queue import enqueue_checkpoint, protocol_digest
from logra.osworld import (
    DesktopTask,
    IsolatedDesktopRollouts,
    OSWorldConfig,
    RollingTaskPool,
    RolloutPoolState,
    TaskSampler,
    make_training_batch,
    read_tasks,
    sandbox_credentials,
    summarize_episodes,
)
from logra.policy import make_policy_factory
from omegaconf import OmegaConf
from pydantic import BaseModel, Field

from nemo_rl.algorithms.grpo import refit_policy_generation
from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.distributed.virtual_cluster import (
    ClusterConfig,
    RayVirtualCluster,
    init_ray,
)
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.policy import AutomodelConfig, PolicyConfig
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import Logger, LoggerConfig
from nemo_rl.utils.outdated_config_checks import check_outdated_config


class TrainingConfig(BaseModel, extra="forbid"):
    max_steps: Annotated[int, Field(ge=0)] = 600
    batch_size: Annotated[int, Field(gt=0)] = 128
    rollout_pool_size: Annotated[int, Field(gt=0)] = 256
    seed: int = 43
    eval_period: Annotated[int, Field(gt=0)] = 20
    eval_at_start: bool = True
    eval_limit: Annotated[int, Field(gt=0)] | None = None
    resume: Path | None = None
    eval_only: bool = False
    external_evaluation_dir: Path | None = None


class ResearchConfig(BaseModel, extra="allow"):
    policy: PolicyConfig
    loss_fn: ClippedPGLossConfig
    logger: LoggerConfig
    checkpointing: CheckpointingConfig
    cluster: ClusterConfig
    logra: LoGRAConfig
    cua: CuaConfig
    osworld: OSWorldConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)


def record_metrics(output: Path, record: dict) -> None:
    """Append committed metrics for the live publisher and retain stdout logging."""
    line = json.dumps(record, default=float)
    with (output / "metrics.jsonl").open("a") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def evaluate(
    rollouts: IsolatedDesktopRollouts,
    tasks: list[DesktopTask],
    config: ResearchConfig,
    logger: Logger,
    step: int,
    training_prompts: set[str],
) -> None:
    if config.training.external_evaluation_dir is not None:
        request = enqueue_checkpoint(
            config.training.external_evaluation_dir,
            Path(config.checkpointing["checkpoint_dir"]) / f"step_{step}",
            step=step,
            requesting_run=Path(config.logger.log_dir),
            resume=config.training.resume,
        )
        record_metrics(
            Path(config.logger.log_dir),
            {"step": step, "external_evaluation_request": str(request)},
        )
        return
    episodes = rollouts.run(
        tasks, step=step, seed=config.training.seed, evaluation=True
    )
    metrics = summarize_episodes(episodes)
    nonduplicate = [
        episode
        for episode in episodes
        if " ".join(episode.task.task["instruction"].split()) not in training_prompts
    ]
    if nonduplicate and len(nonduplicate) != len(episodes):
        metrics.update(
            {
                "nonduplicate/" + key: value
                for key, value in summarize_episodes(nonduplicate).items()
            }
        )
    logger.log_metrics(metrics, step=step, prefix="val")
    record_metrics(Path(config.logger.log_dir), {"step": step, "validation": metrics})


def main(config: ResearchConfig) -> None:
    run, policy_config = config.training, config.policy
    if run.external_evaluation_dir is not None:
        protocol_digest(run.external_evaluation_dir)
        period = config.checkpointing["save_period"]
        if (
            run.eval_only
            or (run.eval_at_start and run.resume is None)
            or not config.checkpointing["enabled"]
            or period is None
            or period <= 0
            or run.eval_period % period != 0
            or run.max_steps % period != 0
        ):
            raise ValueError(
                "External evaluation requires a checkpoint at every evaluation step"
            )
    if run.rollout_pool_size < run.batch_size:
        raise ValueError("training.rollout_pool_size must cover training.batch_size")
    if os.environ.get("AENV_API_KEY") or os.environ.get("E2B_API_KEY"):
        raise ValueError(
            "Keep sandbox keys in osworld.credentials_file, outside the Ray environment"
        )
    if config.osworld.credentials_file is None:
        raise ValueError(
            "Set osworld.credentials_file to a private endpoint/apikey JSON file"
        )
    # Validate the file before allocating workers; do not propagate its contents.
    with sandbox_credentials(config.osworld.credentials_file):
        pass
    generation_config = cast(VllmConfig, policy_config["generation"])
    if (
        generation_config["backend"] != "vllm"
        or not generation_config["vllm_cfg"]["async_engine"]
        or not generation_config["vllm_cfg"].get("expose_http_server")
    ):
        raise ValueError("CUA requires native vLLM async HTTP generation")
    if generation_config["colocated"]["enabled"]:
        raise ValueError("This CUA recipe uses separate training and rollout GPUs")
    inference_nodes = generation_config["colocated"]["resources"]["num_nodes"]
    if inference_nodes is None:
        raise ValueError("CUA requires an explicit rollout node count")
    train_nodes = config.cluster.num_nodes - inference_nodes
    gpus = config.cluster.gpus_per_node
    if min(train_nodes, inference_nodes) <= 0:
        raise ValueError("Allocate at least one training node and one rollout node")
    dtensor = policy_config["automodel_cfg"]
    if not dtensor["enabled"]:
        raise ValueError("CUA requires the AutoModel backend")
    if cast(AutomodelConfig, dtensor)["expert_parallel_size"] != train_nodes * gpus:
        raise ValueError("Expert parallelism must span the training allocation")
    if not config.checkpointing["save_optimizer"]:
        raise ValueError(
            "CUA resume requires LoGRA master weights and projection state"
        )
    loss = make_cua_loss(config.cua, config.loss_fn)
    set_seed(run.seed)
    train_tasks = read_tasks(config.osworld.train_data, config.osworld.train_task_base)
    val_tasks = read_tasks(
        config.osworld.validation_data, config.osworld.validation_task_base
    )
    if run.eval_limit is not None:
        val_tasks = val_tasks[: run.eval_limit]
    training_prompts = {
        " ".join(task.task["instruction"].split()) for task in train_tasks
    }
    sampler = TaskSampler(len(train_tasks), run.seed)
    pool_state: RolloutPoolState | None = None
    baseline = TaskBaseline(config.cua)
    checkpointer = CheckpointManager(config.checkpointing)
    start = 0
    if run.resume is not None:
        info = checkpointer.load_training_info(run.resume)
        if info is None:
            raise ValueError("Resume checkpoint has no training state")
        start = info["step"]
        # Only load our own training state; model state uses the native loader.
        saved = torch.load(
            run.resume / "cua.pt", map_location="cpu", weights_only=False
        )
        if saved["cua"] != config.cua.model_dump() or saved[
            "osworld"
        ] != config.osworld.model_dump(mode="json"):
            raise ValueError("CUA/OSWorld settings changed on resume")
        baseline.load_state_dict(saved["baseline"])
        sampler.load_state_dict(saved["sampler"])
        pool_state = saved.get("rollout_pool")
        if pool_state is not None and saved["rollout_settings"] != {
            "batch_size": run.batch_size,
            "pool_size": run.rollout_pool_size,
            "seed": run.seed,
        }:
            raise ValueError("Rollout batch, capacity or seed changed on resume")
    output_dir = Path(config.logger.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    register_actor_environments(serial_refit=True)
    init_ray()
    logger = Logger(config.logger)
    logger.log_hyperparams(config.model_dump(mode="json"))
    processor = get_tokenizer(policy_config["tokenizer"], get_processor=True)
    tokenizer = processor.tokenizer
    generation_config = cast(
        VllmConfig, configure_generation_config(generation_config, tokenizer)
    )
    generation_config["model_name"] = policy_config["model_name"]
    policy_config["generation"] = generation_config
    generation_config["vllm_kwargs"]["enable_return_routed_experts"] = True
    train_cluster = RayVirtualCluster(
        name="logra_cua_train",
        bundle_ct_per_node_list=[gpus] * train_nodes,
        use_gpus=True,
        num_gpus_per_node=gpus,
        max_colocated_worker_groups=1,
    )
    generation_cluster = RayVirtualCluster(
        name="logra_cua_rollout",
        bundle_ct_per_node_list=[gpus] * inference_nodes,
        use_gpus=True,
        num_gpus_per_node=gpus,
        max_colocated_worker_groups=1,
    )
    policy = generation = None
    try:
        generation = VllmGeneration(
            cluster=generation_cluster, config=generation_config
        )
        generation.finish_generation()
        weights, optimizer = checkpointer.get_resume_paths(run.resume)
        policy = make_policy_factory(config.logra, config.cua)(
            cluster=train_cluster,
            config=policy_config,
            tokenizer=tokenizer,
            processor=processor,
            init_reference_model=False,
            weights_path=weights,
            optimizer_path=optimizer,
        )
        manifests = policy.run_all_workers_single_data("logra_manifest")
        (output_dir / f"projection-targets-step-{start}.json").write_text(
            json.dumps(manifests, indent=2)
        )
        host, port = train_cluster.get_master_address_and_port()
        training_gpus = train_cluster.world_size()
        world = training_gpus + generation_cluster.world_size()
        futures = policy.init_collective(
            host, port, world, train_world_size=training_gpus
        )
        futures += generation.init_collective(
            host, port, world, train_world_size=training_gpus
        )
        ray.get(futures)
        refit_info = policy.prepare_refit_info()
        generation.prepare_refit_info(refit_info or {})
        endpoints = generation.dp_openai_server_base_urls
        if any(url is None for url in endpoints):
            raise RuntimeError("A rollout worker did not start its HTTP endpoint")
        desktop_workers = IsolatedDesktopRollouts(
            config.osworld,
            processor,
            policy_config["model_name"],
            cast(list[str], endpoints),
            output_dir / "episodes",
            workers=max(run.rollout_pool_size, config.osworld.concurrency),
        )
        refit_policy_generation(policy, generation, colocated_inference=False)
        generation.prepare_for_generation()
        pool: RollingTaskPool | None = None
        batch_dir = output_dir / "batches"
        batch_dir.mkdir(exist_ok=True)
        step_started = time.perf_counter()
        with checkpointer, desktop_workers as rollouts, ExitStack() as streaming:
            step_started = time.perf_counter()
            for step in range(start, run.max_steps + 1):
                if (
                    run.eval_only
                    or (step == start and run.eval_at_start)
                    or (
                        step > start
                        and (step % run.eval_period == 0 or step == run.max_steps)
                    )
                ):
                    if pool is not None and run.external_evaluation_dir is None:
                        # Inline evaluation shares the rollout workers. External
                        # evaluation uses saved weights and needs no pool drain.
                        streaming.close()
                        pool_state = pool.state_dict()
                        pool = None
                    if run.external_evaluation_dir is not None:
                        # The external process must see the finalized directory.
                        checkpointer.finalize_pending()
                    evaluate(
                        rollouts, val_tasks, config, logger, step, training_prompts
                    )
                    step_started = time.perf_counter()
                if run.eval_only or step == run.max_steps:
                    break
                if pool is None:
                    pool = streaming.enter_context(
                        rollouts.stream(
                            train_tasks,
                            sampler,
                            batch_size=run.batch_size,
                            inflight=run.rollout_pool_size,
                            seed=run.seed,
                            start_step=step,
                            resume=pool_state,
                        )
                    )
                waiting = time.perf_counter()
                collected = pool.next_batch(
                    step=step, prefetch_next=step + 1 < run.max_steps
                )
                timings = {
                    "timing/actor_idle_wait": time.perf_counter() - waiting,
                    "timing/generation": collected.generation_seconds,
                }
                episodes = [episode for _, episode in collected.episodes]
                adjusted_scores = [
                    ep.error_adjusted_score
                    for ep in episodes
                    if ep.error_adjusted_score is not None
                ]
                if len(adjusted_scores) != len(episodes):
                    raise ValueError(
                        "A training rollout omitted its adjusted diagnostic score"
                    )
                prepared = time.perf_counter()
                batch = make_training_batch(
                    episodes, baseline, processor, training_gpus
                )
                timings["timing/make_experience"] = time.perf_counter() - prepared
                trained = time.perf_counter()
                policy.prepare_for_training()
                result = policy.train(
                    batch, loss, gbs=len(batch["input_lengths"]), mbs=1
                )
                timings["timing/policy_train"] = time.perf_counter() - trained
                transferred = time.perf_counter()
                # Let each active model turn finish before pausing native vLLM.
                # This avoids preempting a turn whose expert routes are captured.
                with rollouts.pause():
                    timings["timing/broadcast_lock_wait"] = (
                        time.perf_counter() - transferred
                    )
                    generation.pause_generation_for_refit(clear_cache=True)
                    try:
                        refit_policy_generation(
                            policy, generation, colocated_inference=False
                        )
                    finally:
                        generation.resume_generation_after_refit()
                timings["timing/broadcast"] = time.perf_counter() - transferred
                timings["timing/broadcast_transfer"] = (
                    timings["timing/broadcast"] - timings["timing/broadcast_lock_wait"]
                )
                timings["timing/step_total"] = time.perf_counter() - step_started
                metrics = {
                    "loss": result["loss"],
                    "grad_norm": result["grad_norm"],
                    "score": sum(ep.score for ep in episodes) / len(episodes),
                    "infrastructure_failure_rate": sum(not ep.valid for ep in episodes)
                    / len(episodes),
                    "segments": len(batch["input_lengths"]),
                    "tasks": len(episodes),
                    "error_adjusted_score": sum(adjusted_scores) / len(episodes),
                    "context_exhaustion_rate": sum(
                        ep.context_exhausted for ep in episodes
                    )
                    / len(episodes),
                    "mean_generated_tokens": sum(
                        int(segment.token_mask.sum())
                        for ep in episodes
                        for segment in ep.segments
                    )
                    / len(episodes),
                    "rollout/dropped_no_action": collected.dropped_no_action,
                    **timings,
                }
                # Match native GRPO: global counts and LR repeat in every
                # microbatch; loss diagnostics are already globally normalized.
                for key, values in result["all_mb_metrics"].items():
                    values = [float(value) for value in values]
                    if key in {"lr", "wd", "global_valid_seqs", "global_valid_toks"}:
                        metrics[key] = sum(values) / len(values)
                    elif key in {"probs_ratio_min", "probs_ratio_clamped_min"}:
                        metrics[key] = min(
                            (x for x in values if not math.isinf(x)), default=-1.0
                        )
                    elif key in {"probs_ratio_max", "probs_ratio_clamped_max"}:
                        metrics[key] = max(
                            (x for x in values if not math.isinf(x)), default=-1.0
                        )
                    else:
                        metrics[key] = sum(values)
                (batch_dir / f"update-{step + 1}.json").write_text(
                    json.dumps(
                        {
                            "step": step + 1,
                            "rollouts": [
                                {
                                    **asdict(ticket),
                                    "artifact_dir": str(episode.artifact_dir),
                                }
                                for ticket, episode in collected.episodes
                            ],
                        },
                        indent=2,
                    )
                )
                logger.log_metrics(metrics, step=step + 1, prefix="train")
                record_metrics(output_dir, {"step": step + 1, "train": metrics})
                # Release the finished batch before materializing the next one.
                del collected, episodes, batch, result
                # As in the original trainer, the next interval includes this checkpoint.
                step_started = time.perf_counter()
                if config.checkpointing["enabled"] and (
                    (step + 1) % config.checkpointing["save_period"] == 0
                    or step + 1 == run.max_steps
                ):
                    checkpoint_started = time.perf_counter()
                    checkpointer.finalize_pending()
                    path = Path(
                        checkpointer.init_tmp_checkpoint(
                            step + 1, {"step": step + 1}, config
                        )
                    )
                    policy.save_checkpoint(
                        str(path / "policy/weights"),
                        str(path / "policy/optimizer"),
                        str(path / "policy/tokenizer"),
                        is_final_checkpoint=step + 1 == run.max_steps,
                    )
                    saved_pool = pool.state_dict()
                    torch.save(
                        {
                            "baseline": baseline.state_dict(),
                            "sampler": saved_pool.sampler,
                            "rollout_pool": saved_pool,
                            "rollout_settings": {
                                "batch_size": run.batch_size,
                                "pool_size": run.rollout_pool_size,
                                "seed": run.seed,
                            },
                            "cua": config.cua.model_dump(),
                            "osworld": config.osworld.model_dump(mode="json"),
                        },
                        path / "cua.pt",
                    )
                    checkpointer.begin_finalization(
                        path, wait_fn=policy.finalize_async_save
                    )
                    logger.log_metrics(
                        {"timing/checkpoint": time.perf_counter() - checkpoint_started},
                        step=step + 1,
                        prefix="train",
                    )
    finally:
        checkpointer.shutdown()
        if generation is not None:
            generation.shutdown()
        if policy is not None:
            policy.shutdown()
        train_cluster.shutdown()
        generation_cluster.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cua_osworld.yaml")
    args, overrides = parser.parse_known_args()
    register_omegaconf_resolvers()
    raw = parse_hydra_overrides(load_config(args.config), overrides)
    resolved = cast(dict[str, Any], OmegaConf.to_container(raw, resolve=True))
    check_outdated_config(resolved)
    main(ResearchConfig.model_validate(resolved))
