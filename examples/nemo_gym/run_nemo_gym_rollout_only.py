#!/usr/bin/env python3
"""Collect NeMo Gym training rollouts without creating a trainer."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--num-steps",
        type=int,
        required=True,
        help="Rollout steps this driver runs (the recipe's grpo.max_num_steps "
        "sizes a training run, not a rollout-only one).",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        required=True,
        help="Inference nodes (this allocation). Overrides "
        "policy.generation.colocated.resources.num_nodes, or cluster.num_nodes "
        "when the recipe colocates generation with training; GPUs per node "
        "come from the recipe.",
    )
    parser.add_argument("--profile-step", type=int)
    parser.add_argument(
        "--train-time-s",
        type=float,
        default=0.0,
        help="Seconds each step holds where training's policy.train() would "
        "run, between sampling the batch and the refit stand-in, so the "
        "collector paces as in an e2e run. The hold ends early once no rollout "
        "is in flight, and the skipped seconds are reported (default 0: no "
        "training stand-in).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("num_steps", "num_nodes"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.train_time_s < 0:
        raise ValueError("--train-time-s must be non-negative")
    if args.profile_step is not None and not 1 <= args.profile_step <= args.num_steps:
        raise ValueError("--profile-step must be within --num-steps")


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a config field from a dict or a pydantic model (current MasterConfig)."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def load_training_config(args: argparse.Namespace):  # noqa: ANN201
    """Load the training recipe as-is, overriding only what a rollout-only run must.

    Everything else -- the backend, the rollout shape (grpo.num_prompts_per_step
    x num_generations_per_prompt), the generation engines and the PD
    disaggregation layout, sampling, and the Gym / agent settings -- is the
    recipe's, so a rollout-only run serves exactly what the training run would.
    Overridden: the inference node count (this allocation) and
    data.validation (a rollout-only run has no validation pass).

    Every node of a rollout-only run serves generation. A non-colocated recipe
    sizes generation by policy.generation.colocated.resources (cluster.num_nodes
    is the training + inference total there), so that is what --num-nodes
    overrides; a colocated recipe's generation spans the whole cluster.
    """
    from omegaconf import OmegaConf

    from nemo_rl.algorithms.grpo import MasterConfig
    from nemo_rl.utils.config import load_config

    raw_config = load_config(str(args.config))
    raw_config = OmegaConf.to_container(raw_config, resolve=True)

    colocated = raw_config["policy"]["generation"]["colocated"]
    if colocated["enabled"]:
        raw_config["cluster"]["num_nodes"] = args.num_nodes
    else:
        colocated["resources"]["num_nodes"] = args.num_nodes
    raw_config["data"]["validation"] = None
    # A NeMo-RL control-flow knob that run_grpo_nemo_gym.py pops before setup
    # so it never reaches Gym's global config.
    raw_config["env"]["nemo_gym"].pop("is_trajectory_collection", None)

    return MasterConfig(**raw_config)


def reject_unsupported_batching(master_config) -> None:  # noqa: ANN001
    """Refuse the GRPO batching modes that change a step's rollout shape.

    Dynamic sampling (with batch_multiplier) and multiple dataloaders make
    grpo_train draw a different number of prompts per step, or draw them from
    several datasets; this driver runs exactly num_prompts_per_step prompts
    from one dataloader, so it would silently measure a different shape.
    """
    grpo = master_config.grpo
    unsupported = []
    if grpo.use_dynamic_sampling:
        unsupported.append("grpo.use_dynamic_sampling=true")
    if grpo.batch_multiplier != 1:
        unsupported.append(f"grpo.batch_multiplier={grpo.batch_multiplier}")
    if master_config.data.get("use_multiple_dataloader", False):
        unsupported.append("data.use_multiple_dataloader=true")
    if unsupported:
        raise NotImplementedError(
            "Rollout-only does not reproduce these GRPO batching modes, which "
            f"change the per-step rollout shape: {', '.join(unsupported)}"
        )


def inference_gpus_per_node(master_config) -> int:  # noqa: ANN001
    """GPUs per inference node, as the recipe sizes generation."""
    colocated = master_config.policy["generation"]["colocated"]
    if colocated["enabled"]:
        return int(_cfg_get(master_config.cluster, "gpus_per_node"))
    return int(colocated["resources"]["gpus_per_node"])


def inference_topology(
    master_config,  # noqa: ANN001
    num_nodes: int,
    gpus_per_node: int,
) -> tuple[int | None, list[dict[str, float]] | None]:
    """The inference cluster's segment size and NVLink-domain constraints.

    grpo.setup's own rule (prepare_inference_segment_topology): a generation
    instance spanning several nodes -- a TRT-LLM replica, prefill plus decode
    under PD disaggregation, or a vLLM TP x PP engine -- is pinned to one
    NVLink domain. Every node of a rollout-only run is an inference node.
    Without cluster.segment_size the topology logic is off, as in training.

    Returns:
        (segment_size, node_resource_constraints) for RayVirtualCluster.
    """
    from nemo_rl.algorithms.grpo import prepare_inference_segment_topology
    from nemo_rl.distributed.virtual_cluster import get_ray_cluster_topology

    if _cfg_get(master_config.cluster, "segment_size") is None:
        return None, None
    constraints, segment_size, _ = prepare_inference_segment_topology(
        master_config.policy["generation"],
        master_config.policy,
        num_nodes,
        gpus_per_node,
        topology=get_ray_cluster_topology(),
    )
    return segment_size, constraints


def warn_gym_shard_placement(master_config, num_nodes: int) -> None:  # noqa: ANN001
    """Warn when the recipe's Gym shards cannot be placed on this allocation.

    STRICT_SPREAD (the shard default) pins every Gym replica to a distinct
    node. Training spreads them over its training and inference nodes; a
    rollout-only run has only the inference nodes, so a plan sized for the
    training allocation may ask for more nodes than exist here. The run is
    not stopped: build_nemo_gym_actors raises ShardSetupError once the
    placement group fails to become ready.
    """
    from nemo_rl.environments.nemo_gym import DEFAULT_SHARD_PG_READY_TIMEOUT_SECONDS
    from nemo_rl.environments.nemo_gym_shards import parse_shard_plan

    plan = parse_shard_plan(master_config.env["nemo_gym"])
    if plan is None:
        return
    replicas = sum(shard.replicas for shard in plan.shards)
    print(
        f"GEN_ONLY_GYM_SHARDS shards={len(plan.shards)} replicas={replicas} "
        f"placement_strategy={plan.placement_strategy} nodes={num_nodes}",
        flush=True,
    )
    if plan.placement_strategy == "STRICT_SPREAD" and replicas > num_nodes:
        print(
            f"WARNING: env.nemo_gym.shards asks for {replicas} Gym replicas with "
            f"STRICT_SPREAD, which puts each replica on its own node, but this "
            f"rollout-only run has {num_nodes} node(s) -- only the inference "
            f"nodes; training also counts its training nodes. The Gym placement "
            f"group cannot be satisfied, so Gym startup will fail after "
            f"{DEFAULT_SHARD_PG_READY_TIMEOUT_SECONDS:g}s, once the engines are "
            f"already up. Allocate at least {replicas} nodes, or use a recipe "
            f"with fewer replicas or a packing placement_strategy.",
            flush=True,
        )


def start_nemo_gym(master_config, base_urls: list[str], tokenizer, train_dataset):  # noqa: ANN001, ANN201
    """Start NeMo Gym exactly as GRPO setup does (grpo.py _spinup_nemo_gym).

    build_nemo_gym_actors makes the single actor of an unsharded recipe, or one
    actor per shard replica when env.nemo_gym.shards is set, and returns a
    NemoGymShardSet either way. The dataset coverage check fails startup if a
    row's agent route is hosted by no shard (a no-op without shards).
    """
    from nemo_rl.environments.nemo_gym import (
        build_nemo_gym_actors,
        validate_dataset_agent_coverage,
    )
    from nemo_rl.models.megatron.router_replay import router_replay_enabled

    shard_set = build_nemo_gym_actors(
        master_config.env,
        base_urls=base_urls,
        model_name=master_config.policy["generation"]["model_name"],
        tokenizer=tokenizer,
        enable_router_replay=router_replay_enabled(master_config.policy),
        use_fastokens=bool(master_config.policy["tokenizer"].get("use_fastokens")),
    )
    try:
        validate_dataset_agent_coverage(shard_set, {"train": train_dataset})
    except BaseException:
        shard_set.shutdown()
        raise
    return shard_set


def reject_online_quantization(generation_config: dict[str, Any]) -> None:
    """Refuse online quantization (fp8 precision, vLLM NVFP4 per-token or
    ModelOpt quant_cfg/real_quant): it quantizes at refit, and a rollout-only
    run has no refit."""
    backend = generation_config["backend"]
    if backend not in ("vllm", "trtllm"):
        return
    name = f"{backend}_cfg"
    precision = str((generation_config.get(name) or {}).get("precision", "")).lower()
    online = []
    if precision == "fp8":
        online.append(f"{name}.precision=fp8")
    if backend == "vllm":
        from nemo_rl.models.generation.vllm.config import parse_nvfp4_pertoken_rollout

        if parse_nvfp4_pertoken_rollout(generation_config) is not None:
            online.append("generation.nvfp4_pertoken_rollout")
        if generation_config.get("quant_cfg") is not None:
            online.append("generation.quant_cfg")
        if generation_config.get("real_quant"):
            online.append("generation.real_quant")
    if online:
        raise ValueError(
            f"{', '.join(online)} quantizes the engine online from the weights a "
            "trainer refit sends; a rollout-only run has no refit. Use an "
            "unquantized engine, and for a quantized one point the model at a "
            "pre-quantized checkpoint."
        )


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def iter_training_batches(dataloader, num_steps: int):  # noqa: ANN001, ANN201
    """Yield (step, batch) for num_steps steps, epoch after epoch, like grpo_train.

    grpo_train starts the next epoch when the dataloader runs out (re-shuffled
    when data.shuffle is set) and keeps going until it reaches its step limit.
    """
    step = 0
    epoch = 0
    while step < num_steps:
        if epoch > 0:
            print(f"Dataset exhausted; starting epoch {epoch + 1}", flush=True)
        batches_this_epoch = 0
        for batch in dataloader:
            batches_this_epoch += 1
            step += 1
            yield step, batch
            if step >= num_steps:
                return
        if batches_this_epoch == 0:
            raise RuntimeError(
                "The dataset holds no full batch of num_prompts_per_step prompts"
            )
        epoch += 1


def simulate_training(
    train_time_s: float,
    check_health=None,  # noqa: ANN001
    rollouts_idle=None,  # noqa: ANN001
) -> float:
    """Stand in for policy.train(): hold the step for up to train_time_s seconds.

    check_health, when given, is polled so a dead collector fails the step instead of
    surfacing only after the wait.

    The wait only matters while rollouts run beside training. Once
    rollouts_idle() reports that nothing is generating, the rest of the wait
    is idle on both sides and a rollout-only run continues identically after
    it, so it is skipped.

    Returns:
        The seconds of training time skipped as idle.
    """
    if train_time_s <= 0:
        return 0.0
    deadline = time.perf_counter() + train_time_s
    while (remaining := deadline - time.perf_counter()) > 0:
        if rollouts_idle is not None and rollouts_idle():
            return remaining
        time.sleep(min(remaining, 5.0))
        if check_health is not None:
            check_health()
    return 0.0


def run_sync_rollout_steps(
    master_config,  # noqa: ANN001
    generation,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    task_to_env: dict[str, Any],
    dataloader,  # noqa: ANN001
    num_steps: int,
    profile_step: int | None,
    train_time_s: float = 0.0,
) -> None:
    """Run grpo_train's rollout step by step: one whole batch per step.

    Each step sends num_prompts_per_step x num_generations_per_prompt rows
    through run_nemo_gym_rollout_sync, the call GRPO's synchronous NeMo-Gym
    training step makes, and waits for all of them before the next step.
    train_time_s is where grpo_train trains; nothing generates then, so it is
    skipped whole and reported, keeping e2e step time = elapsed + skipped.
    """
    from nemo_rl.experience.rollouts import (
        get_effort_config,
        get_nemo_gym_thinking_tags,
        run_nemo_gym_rollout_sync,
        should_mask_flagged_samples,
    )
    from nemo_rl.utils.logger import should_log_nemo_gym_full_result_tables

    generation_config = master_config.policy["generation"]
    backend = generation_config["backend"]
    num_generations_per_prompt = int(master_config.grpo.num_generations_per_prompt)
    # NeMo-Gym owns the stop criteria; clear the EOS auto-fill as grpo.py does.
    rollout_generation_config = {
        **generation_config,
        "stop_token_ids": None,
        "stop_strings": None,
    }
    vllm_metrics_enabled = backend == "vllm" and generation_config.get(
        "vllm_cfg", {}
    ).get("enable_vllm_metrics_logger", False)

    for step, batch in iter_training_batches(dataloader, num_steps):
        if step > 1 and not generation.invalidate_kv_cache():
            raise RuntimeError(
                f"Failed to reset the {backend} KV cache before rollout step {step}"
            )
        repeated_batch = batch.repeat_interleave(num_generations_per_prompt)
        if vllm_metrics_enabled:
            generation.clear_logger_metrics()
        # GPU profiling hooks exist on both vllm and trtllm generation.
        should_profile = step == profile_step and hasattr(
            generation, "start_gpu_profiling"
        )
        if should_profile:
            print(f"Starting TP-worker Nsight capture for rollout step {step}")
            generation.start_gpu_profiling()

        started = time.perf_counter()
        try:
            # The call GRPO's synchronous NeMo-Gym training step makes (grpo.py).
            rollout_result = run_nemo_gym_rollout_sync(
                policy_generation=generation,
                input_batch=repeated_batch,
                tokenizer=tokenizer,
                task_to_env=task_to_env,
                max_seq_len=master_config.policy["max_total_sequence_length"],
                generation_config=rollout_generation_config,
                num_generations_per_prompt=num_generations_per_prompt,
                log_full_result_tables=should_log_nemo_gym_full_result_tables(
                    wandb_enabled=master_config.logger.wandb_enabled,
                    wandb_config=master_config.logger.wandb,
                ),
                max_rollout_turns=None,
                greedy=False,
                effort_config=get_effort_config(master_config.env),
                reward_penalty_config=master_config.reward_penalties,
                length_penalty_config=master_config.grpo.length_penalty,
                thinking_tags=get_nemo_gym_thinking_tags(master_config.env),
                mask_env_flagged_samples=should_mask_flagged_samples(
                    master_config.env
                ),
                deduplicate_multimodal_data=(
                    master_config.grpo.deduplicate_multimodal_data
                ),
                debug_payload_metrics=master_config.grpo.debug_payload_metrics,
            )
        finally:
            if should_profile:
                generation.stop_gpu_profiling()
                print(f"Stopped TP-worker Nsight capture after rollout step {step}")
        # Synchronous GRPO runs no rollout while it trains.
        skipped = simulate_training(train_time_s, rollouts_idle=lambda: True)

        elapsed = time.perf_counter() - started
        if vllm_metrics_enabled:
            logger_metrics = generation.get_logger_metrics()
            print(
                f"vLLM logger metrics for rollout step {step}: "
                f"{json.dumps(logger_metrics, default=json_default)}",
                flush=True,
            )
        rows = rollout_result.final_batch.size
        if rows != repeated_batch.size:
            raise RuntimeError(
                f"Rollout step {step} returned {rows} trajectories; expected "
                f"{repeated_batch.size} (num_prompts_per_step x "
                f"num_generations_per_prompt)"
            )
        reward = rollout_result.rollout_metrics.get("total_reward/mean")
        turns = rollout_result.rollout_metrics.get("turns_per_sample/mean")
        print(
            f"Rollout step {step}/{num_steps}: {rows} trajectory row(s), "
            f"reward_mean={reward}, turns_mean={turns}, elapsed={elapsed:.2f}s, "
            f"simulated_training={train_time_s:.0f}s "
            f"(idle_skipped={skipped:.1f}s, e2e_step={elapsed + skipped:.2f}s)",
            flush=True,
        )


def run_async_rollout_steps(
    master_config,  # noqa: ANN001
    generation,  # noqa: ANN001
    tokenizer,  # noqa: ANN001
    task_to_env: dict[str, Any],
    dataloader,  # noqa: ANN001
    num_steps: int,
    profile_step: int | None,
    train_time_s: float = 0.0,
) -> None:
    """Drive async GRPO's own collector and replay buffer with no trainer.

    The rollouts come from the AsyncTrajectoryCollector + ReplayBuffer pair
    async_grpo_train runs (grpo.py), so prompt groups stream into the buffer
    and up to max_trajectory_age_steps + 1 batches are in flight. This loop
    stands in for the trainer: it samples step v's groups the way training
    does, holds the step train_time_s seconds where policy.train() would run
    (generation keeps going, as with non-colocated training), then advances
    the weight version (prepare_for_refit -> set_weight_version ->
    resume_after_refit) without a weight transfer, which releases the
    collector's next target.

    The hold ends early once every target the age window allows has a full
    batch in the buffer: the collector is then paused on its generation
    limit and no rollout is in flight, so waiting out the rest changes
    nothing. The skipped seconds are reported, e2e step = step + skipped.
    """
    import ray

    from nemo_rl.algorithms.async_utils import (
        AsyncTrajectoryCollector,
        ReplayBuffer,
    )
    from nemo_rl.data.dataloader import CyclingDataLoader
    from nemo_rl.utils.venvs import make_actor_runtime_env

    num_prompts_per_step = int(master_config.grpo.num_prompts_per_step)
    num_generations_per_prompt = int(master_config.grpo.num_generations_per_prompt)
    max_age_steps = master_config.grpo.async_grpo.max_trajectory_age_steps

    # Same buffer sizing as async_grpo_train.
    replay_buffer = ReplayBuffer.options(
        runtime_env=make_actor_runtime_env("nemo_rl.algorithms.async_utils.ReplayBuffer")
    ).remote(
        max_size=num_prompts_per_step * max_age_steps * 2,
        drop_incomplete_targets_on_restore=False,
    )
    collector = AsyncTrajectoryCollector.options(
        runtime_env=make_actor_runtime_env(
            "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector",
            extra_env_vars={"NRL_WORKER_GROUP": "trajectory_collector"},
        )
    ).remote(
        policy_generation=generation,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        master_config=master_config,
        replay_buffer=replay_buffer,
        start_step=0,
    )
    try:
        ray.get(collector.set_weight_version.remote(0))
        ray.get(collector.start_collection.remote(CyclingDataLoader(dataloader)))
        print(
            f"Async rollout: max_trajectory_age_steps={max_age_steps}, "
            f"replay buffer size={num_prompts_per_step * max_age_steps * 2}",
            flush=True,
        )
        step_started = time.perf_counter()
        total_skipped = 0.0
        for weight_version in range(num_steps):
            step = weight_version + 1
            should_profile = step == profile_step and hasattr(
                generation, "start_gpu_profiling"
            )
            if should_profile:
                print(f"Starting TP-worker Nsight capture for rollout step {step}")
                generation.start_gpu_profiling()
            wait_started = time.perf_counter()
            # The training step's sample(): exactly num_prompts_per_step
            # groups targeted at this weight version, or None to wait.
            while True:
                sample = ray.get(
                    replay_buffer.sample.remote(
                        num_prompt_groups=num_prompts_per_step,
                        current_weight_version=weight_version,
                        max_age_steps=max_age_steps,
                    )
                )
                if sample is not None:
                    break
                ray.get(collector.check_health.remote())
                time.sleep(1.0)
            exposed = time.perf_counter() - wait_started

            groups = sample["trajectories"]
            rows = sum(group["batch"].size for group in groups)
            expected_rows = num_prompts_per_step * num_generations_per_prompt
            if rows != expected_rows:
                raise RuntimeError(
                    f"Rollout step {step} sampled {rows} trajectories; expected "
                    f"{expected_rows} (num_prompts_per_step x "
                    f"num_generations_per_prompt)"
                )
            durations = sorted(
                group["rollout_metrics"].get("trajectory_duration_s", 0.0)
                for group in groups
            )
            # The reward a training step would see: total_reward per trajectory.
            rewards = [
                float(reward)
                for group in groups
                for reward in group["batch"]["total_reward"].tolist()
            ]
            reward_mean = sum(rewards) / len(rewards)
            num_solved = sum(reward > 0 for reward in rewards)
            # Every group has num_generations_per_prompt rows, so the mean of
            # the per-group means is the step mean.
            group_turns = [
                group["rollout_metrics"]["turns_per_sample/mean"]
                for group in groups
                if "turns_per_sample/mean" in group["rollout_metrics"]
            ]
            turns_mean = (
                sum(group_turns) / len(group_turns) if group_turns else float("nan")
            )

            # The collector targets v+1..v+max_age while step v trains.
            def rollouts_idle() -> bool:
                return all(
                    ray.get(
                        replay_buffer.has_complete_batch.remote(
                            target, num_prompts_per_step, max_age_steps
                        )
                    )
                    for target in range(
                        weight_version + 1, weight_version + max_age_steps + 1
                    )
                )

            # Stand-in for the refit: the same collector calls training makes
            # around refit_policy_generation, with no weights moved.
            skipped = 0.0
            try:
                skipped = simulate_training(
                    train_time_s,
                    check_health=lambda: ray.get(collector.check_health.remote()),
                    rollouts_idle=rollouts_idle,
                )
                total_skipped += skipped
                ray.get(collector.prepare_for_refit.remote())
                # An in-flight refit keeps the reusable prefix cache unless KV
                # is recomputed (vLLM's pause clear_cache, TRT-LLM's
                # finish_weight_update), so reset it only in that case.
                if (
                    master_config.policy["generation"]["backend"] == "trtllm"
                    and master_config.grpo.async_grpo.recompute_kv_cache_after_weight_updates
                    and not generation.invalidate_kv_cache()
                ):
                    raise RuntimeError(
                        f"Failed to reset the prefix cache after rollout step {step}"
                    )
                ray.get(collector.set_weight_version.remote(weight_version + 1))
                ray.get(collector.resume_after_refit.remote())
            finally:
                if should_profile:
                    generation.stop_gpu_profiling()
                    print(f"Stopped TP-worker Nsight capture after rollout step {step}")
            now = time.perf_counter()
            print(
                f"Rollout step {step}/{num_steps}: {len(groups)} group(s), "
                f"{rows} trajectory row(s), step={now - step_started:.2f}s, "
                f"exposed_generation={exposed:.2f}s, "
                f"simulated_training={train_time_s:.0f}s "
                f"(idle_skipped={skipped:.1f}s, "
                f"e2e_step={now - step_started + skipped:.2f}s), "
                f"reward_mean={reward_mean:.4f}, "
                f"solved={num_solved}/{len(rewards)}, "
                f"turns_mean={turns_mean:.1f}, "
                f"avg_trajectory_age={sample['avg_trajectory_age']:.2f}, "
                f"group_duration_s(min/median/max)="
                f"{durations[0]:.1f}/{durations[len(durations) // 2]:.1f}/"
                f"{durations[-1]:.1f}",
                flush=True,
            )
            step_started = now
        if train_time_s > 0:
            print(
                f"GEN_ONLY_TRAIN_IDLE_SKIPPED_S {total_skipped:.1f} "
                f"(of {num_steps} x {train_time_s:.0f}s simulated training)",
                flush=True,
            )
    finally:
        # Groups for later targets are still in flight; drop them with the
        # collector, as async_grpo_train does at the end of a run.
        try:
            ray.get(collector.flush_telemetry.remote(quiesce_timeout_s=3.0), timeout=15)
        except Exception as error:
            print(f"Warning: collector telemetry flush failed: {error}", flush=True)
        ray.kill(collector)
        ray.kill(replay_buffer)


def shutdown_resources(nemo_gym, generation, cluster) -> None:  # noqa: ANN001
    import ray

    if nemo_gym is not None:
        # NemoGymShardSet: stops every replica and releases its placement group.
        try:
            nemo_gym.shutdown()
        except Exception as error:
            print(f"Warning: NeMo Gym shutdown failed: {error}", flush=True)
    if generation is not None:
        generation.shutdown()
    if cluster is not None:
        cluster.shutdown()
    ray.shutdown()


def main() -> None:
    args = parse_args()
    validate_args(args)

    # Imports follow argument validation so --help works outside the workload
    # environment. NRL_NSYS_* must already be set by the launcher before these.
    from torchdata.stateful_dataloader import StatefulDataLoader

    from nemo_rl.algorithms.utils import get_tokenizer, set_seed
    from nemo_rl.data.collate_fn import rl_collate_fn
    from nemo_rl.data.utils import setup_response_data
    from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
    from nemo_rl.environments.nemo_gym import setup_nemo_gym_config
    from nemo_rl.models.generation import configure_generation_config
    from nemo_rl.utils.config import register_omegaconf_resolvers

    register_omegaconf_resolvers()
    master_config = load_training_config(args)
    reject_unsupported_batching(master_config)
    # Same seed as GRPO setup, so data.shuffle draws the training run's order.
    set_seed(_cfg_get(master_config.grpo, "seed"))
    num_prompts_per_step = int(
        _cfg_get(master_config.grpo, "num_prompts_per_step")
    )
    num_generations_per_prompt = int(
        _cfg_get(master_config.grpo, "num_generations_per_prompt")
    )
    backend = master_config.policy["generation"]["backend"]
    # Only these two have a rollout-only engine path below; the recipe's other
    # backends (megatron, sglang, dynamo) are set up differently in grpo.setup.
    assert backend in ("vllm", "trtllm"), (
        f"Rollout-only supports policy.generation.backend vllm or trtllm, got "
        f"{backend!r}"
    )
    trtllm_cfg = master_config.policy["generation"].get("trtllm_cfg") or {}
    gpus_per_node = inference_gpus_per_node(master_config)
    disaggregated = False
    if backend == "trtllm":
        from nemo_rl.models.generation.trtllm.config import resolve_disagg_layout

        # Log only: TrtllmGeneration resolves the PD layout from the same
        # generation.disaggregation block itself.
        disaggregated = resolve_disagg_layout(master_config.policy["generation"]).enabled
    # The resolved layout (after the recipe's defaults chain), for the log.
    print(
        f"GEN_ONLY_LAYOUT backend={backend} disaggregated={int(disaggregated)} "
        f"nodes={args.num_nodes} gpus_per_node={gpus_per_node} "
        f"tp={trtllm_cfg.get('tensor_parallel_size')} "
        f"ep={trtllm_cfg.get('moe_expert_parallel_size')}",
        flush=True,
    )
    tokenizer = get_tokenizer(master_config.policy["tokenizer"])

    # Before configure_generation_config, whose is_eval validation would reject
    # nvfp4_pertoken_rollout first with a message that does not say why.
    reject_online_quantization(master_config.policy["generation"])
    # Unlike GRPO training, there is no trainer to refit dummy weights.
    # is_eval=True makes the engine load the real HF checkpoint directly.
    generation_config = configure_generation_config(
        master_config.policy["generation"], tokenizer, is_eval=True
    )
    master_config.policy["generation"] = generation_config
    setup_nemo_gym_config(master_config, tokenizer)
    # Before any engine starts, so a shard plan this allocation cannot place
    # is reported up front rather than after the engines are loaded.
    warn_gym_shard_placement(master_config, args.num_nodes)
    generation_config["model_name"] = master_config.policy["model_name"]
    if backend == "vllm":
        from nemo_rl.models.generation.vllm.config import (
            normalize_nvfp4_pertoken_policy_config,
            normalize_vllm_refit_config,
        )
        from nemo_rl.models.megatron.router_replay import (
            configure_vllm_for_router_replay,
        )

        # grpo.setup's vLLM config normalization.
        normalize_nvfp4_pertoken_policy_config(master_config.policy, entry_point="grpo")
        normalize_vllm_refit_config(generation_config)
        generation_config.setdefault("vllm_kwargs", {})["hf_overrides"] = (
            master_config.policy.get("hf_config_overrides", {})
        )
        configure_vllm_for_router_replay(master_config.policy)

    train_dataset, _ = setup_response_data(
        tokenizer, master_config.data, env_configs=None
    )
    dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=num_prompts_per_step,
        shuffle=master_config.data["shuffle"],
        collate_fn=rl_collate_fn,
        drop_last=True,
        num_workers=master_config.data["num_workers"],
    )

    init_ray()

    cluster = None
    generation = None
    nemo_gym = None
    try:
        cluster_cfg = master_config.cluster
        segment_size, node_resource_constraints = inference_topology(
            master_config, args.num_nodes, gpus_per_node
        )
        cluster = RayVirtualCluster(
            name="grpo_rollout_cluster",
            bundle_ct_per_node_list=[gpus_per_node] * args.num_nodes,
            use_gpus=True,
            num_gpus_per_node=gpus_per_node,
            max_colocated_worker_groups=1,
            port_range_low=_cfg_get(cluster_cfg, "master_port_range_low"),
            port_range_high=_cfg_get(cluster_cfg, "master_port_range_high"),
            # As grpo.setup's inference cluster: one generation instance per
            # NVLink-domain segment, and nodes pinned to their domains.
            segment_size=segment_size,
            node_resource_constraints=node_resource_constraints,
        )
        if node_resource_constraints is not None:
            # Claim the domain-aligned nodes eagerly, as grpo.setup does.
            if backend == "trtllm":
                from nemo_rl.models.generation.trtllm import TrtllmGeneration

                TrtllmGeneration.init_cluster_placement_groups(cluster, generation_config)
            else:
                from nemo_rl.models.generation.vllm import VllmGeneration

                VllmGeneration.init_cluster_placement_groups(cluster, generation_config)

        # Same NeMo-RL outer generation actor + nested TP worker path as training.
        if backend == "vllm":
            from nemo_rl.models.generation.vllm import VllmGeneration

            # vLLM can defer the HF load, so Gym startup overlaps it.
            generation = VllmGeneration(
                cluster=cluster,
                config=generation_config,
                defer_model_load=True,
            )
            base_urls = [
                url
                for url in generation.dp_openai_server_base_urls
                if url is not None
            ]
            with ThreadPoolExecutor(max_workers=2) as executor:
                load_future = executor.submit(generation.load_and_start)
                gym_future = executor.submit(
                    start_nemo_gym, master_config, base_urls, tokenizer, train_dataset
                )
                load_future.result()
                nemo_gym = gym_future.result()
            generation.finish_generation()
        elif backend == "trtllm":
            from nemo_rl.models.generation.trtllm import TrtllmGeneration

            # TRT-LLM loads the engine and starts the HTTP server inside
            # __init__ (no deferred load), so base_urls are ready right after
            # construction. Gym is started once the server URLs are known.
            generation = TrtllmGeneration(
                cluster=cluster,
                config=generation_config,
            )
            base_urls = [
                url
                for url in generation.dp_openai_server_base_urls
                if url is not None
            ]
            nemo_gym = start_nemo_gym(
                master_config, base_urls, tokenizer, train_dataset
            )
        print(f"{backend} server URLs: {generation.dp_openai_server_base_urls}")
        print(
            "Rollout shape: "
            f"{num_prompts_per_step} training prompt(s) x "
            f"{num_generations_per_prompt} generation(s), "
            f"{args.num_steps} step(s)",
            flush=True,
        )

        task_to_env = {"nemo_gym": nemo_gym}

        # The recipe's grpo.async_grpo.enabled picks the rollout path, as it
        # picks grpo_train vs async_grpo_train in training.
        async_enabled = bool(master_config.grpo.async_grpo.enabled)
        print(f"GEN_ONLY_ROLLOUT_MODE {'async' if async_enabled else 'sync'}", flush=True)
        if async_enabled:
            run_async_rollout_steps(
                master_config,
                generation,
                tokenizer,
                task_to_env,
                dataloader,
                args.num_steps,
                args.profile_step,
                args.train_time_s,
            )
        else:
            run_sync_rollout_steps(
                master_config,
                generation,
                tokenizer,
                task_to_env,
                dataloader,
                args.num_steps,
                args.profile_step,
                args.train_time_s,
            )

        print(
            "NeMo Gym rollouts (trajectories, model calls) are under "
            f"env.nemo_gym.results_dir="
            f"{master_config.env['nemo_gym'].get('results_dir', '<Gym default>')}",
            flush=True,
        )
    finally:
        shutdown_resources(nemo_gym, generation, cluster)


if __name__ == "__main__":
    main()
