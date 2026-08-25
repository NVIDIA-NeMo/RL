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
"""Flow-GRPO training loop.

Mirrors :func:`nemo_rl.algorithms.grpo.grpo_train` in phase ordering
(cluster→policy→env→dataset→loop[rollout→reward→advantage→train→validate→checkpoint])
but drops token-only concepts (vLLM refit, message-log rollouts, token KL).

The advantage estimator reuses
:func:`nemo_rl.algorithms.utils.calculate_baseline_and_std_per_prompt` by
encoding each unique prompt as a ``(1,)`` integer tensor row, then
broadcasting the resulting per-sample advantage to ``[B*K, T]``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable

import torch
from pydantic import BaseModel, Field, model_validator

from nemo_rl.algorithms.loss.flow_grpo import FlowGRPOLossConfig
from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig
from nemo_rl.environments.image_reward_environment import (
    ImageRewardEnvConfig,
    ImageRewardEnvironment,
)
from nemo_rl.models.diffusion.flow_grpo_policy import (
    FlowGRPOPolicy,
    aggregate_worker_metrics,
)
from nemo_rl.models.diffusion.interfaces import (
    DiffusionPolicyConfig,
    DiffusionTrainDataSpec,
    DiffusionTrajectorySpec,
)
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import Logger, LoggerConfig
from nemo_rl.utils.timer import Timer


class FlowGRPOValGenerationCfg(BaseModel, extra="allow"):
    """Validation-time generation overrides.

    Validation always samples with the deterministic ODE (no SDE window, no
    logprob collection).
    """

    num_inference_steps: int = 40
    # Draw every validation sample's initial latent from a fresh generator
    # holding the same seed (verl-omni val_kwargs.seed semantics: every val
    # request is seeded identically), so val rewards are independent of batch
    # position, batch size and DP worker layout. Default keeps the current
    # behavior: one generator per rollout batch, per-worker seed offsets.
    single_seed: bool = False


class FlowGRPOAlgoConfig(BaseModel, extra="allow"):
    """Top-level flow-GRPO training-loop config."""

    num_prompts_per_step: int = 8
    num_generations_per_prompt: int = 16
    max_num_steps: int = 5000
    # 0 disables periodic validation.
    val_period: int = 50
    seed: int = 42
    ppo_epochs: int = 1
    val_at_start: bool = False
    val_at_end: bool = False
    # 0 validates on the full val dataloader.
    max_val_samples: int = 0
    use_leave_one_out_baseline: bool = True
    # Normalize advantages by the whole-batch reward std (verl-omni
    # `global_std`) instead of per-group std, which explodes on
    # near-constant groups under sparse rewards.
    use_global_std: bool = True
    # Actor-update mini-batch size in samples (verl-omni ppo_mini_batch_size
    # semantics, which counts prompts and scales by rollout.n). None keeps a
    # single optimizer update over the whole rollout batch. When set, each
    # step slices the rollout batch in original order into
    # (num_prompts_per_step * num_generations_per_prompt) / ppo_mini_batch_size
    # sequential optimizer updates; updates after the first are intentionally
    # off-policy, exactly as in verl-omni.
    ppo_mini_batch_size: int | None = None
    val_generation: FlowGRPOValGenerationCfg = Field(
        default_factory=FlowGRPOValGenerationCfg
    )

    @model_validator(mode="after")
    def _mini_batch_keeps_groups_whole(self) -> "FlowGRPOAlgoConfig":
        mini = self.ppo_mini_batch_size
        if mini is None:
            return self
        K = self.num_generations_per_prompt
        total = self.num_prompts_per_step * K
        if mini <= 0 or mini % K != 0:
            raise ValueError(
                f"flow_grpo.ppo_mini_batch_size={mini} must be a positive multiple "
                f"of num_generations_per_prompt={K} so contiguous slicing "
                "never splits a GRPO group across mini-batches"
            )
        if total % mini != 0:
            raise ValueError(
                f"flow_grpo.ppo_mini_batch_size={mini} must divide the rollout "
                f"batch of {total} samples "
                f"(num_prompts_per_step * num_generations_per_prompt)"
            )
        return self


class DiffusionDataSplitConfig(BaseModel, extra="allow"):
    """One split of the prompt dataset (.txt or .jsonl file)."""

    prompt_file: str


class DiffusionDataConfig(BaseModel, extra="allow"):
    train: DiffusionDataSplitConfig
    val: DiffusionDataSplitConfig | None = None


class DiffusionEnvConfig(BaseModel, extra="allow"):
    image_reward: ImageRewardEnvConfig


class MasterConfig(BaseModel, extra="allow"):
    """Schema for `examples/configs/flow_grpo_qwen_image*.yaml`."""

    policy: DiffusionPolicyConfig
    loss_fn: FlowGRPOLossConfig
    flow_grpo: FlowGRPOAlgoConfig
    data: DiffusionDataConfig
    env: DiffusionEnvConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig

    @model_validator(mode="after")
    def _kl_requires_lora(self) -> "MasterConfig":
        if self.loss_fn.beta > 0 and not self.policy.lora_cfg.enabled:
            raise ValueError(
                "loss_fn.beta > 0 (Gaussian KL vs the reference policy) requires "
                "policy.lora_cfg.enabled=true — the reference is the base model "
                "with the LoRA adapter disabled. Set beta to 0 or enable LoRA."
            )
        return self

    @model_validator(mode="after")
    def _per_element_logprob_pairing(self) -> "MasterConfig":
        if (
            self.policy.per_element_logprob
            != self.loss_fn.aggregate_logprobs_per_sample
        ):
            raise ValueError(
                "policy.per_element_logprob must equal "
                "loss_fn.aggregate_logprobs_per_sample: both keep per-latent-element "
                "log-probs and are only correct when set together."
            )
        if self.policy.per_element_logprob and self.loss_fn.beta > 0:
            raise ValueError(
                "policy.per_element_logprob is incompatible with loss_fn.beta > 0; "
                "the KL reference log-prob is reduced per sample."
            )
        return self


def _prompt_ids_for_baseline(rep_prompts: list[str]) -> torch.Tensor:
    """Encode prompts as a `(B*K, 1)` integer tensor; identical prompts share an id."""
    seen: dict[str, int] = {}
    ids: list[int] = []
    for p in rep_prompts:
        if p not in seen:
            seen[p] = len(seen)
        ids.append(seen[p])
    return torch.tensor(ids, dtype=torch.long).unsqueeze(-1)


def _compute_advantages(
    rep_prompts: list[str],
    rewards: torch.Tensor,
    *,
    use_leave_one_out_baseline: bool,
    use_global_std: bool,
) -> torch.Tensor:
    """Group-relative advantage per the GRPO recipe.

    ``use_global_std=True`` normalizes by the whole-batch reward std
    (verl-omni ``global_std`` semantics): with sparse rewards most groups are
    near-constant, and per-group std amplifies the few informative groups to
    the advantage clamp, destabilizing training.
    """
    prompt_ids = _prompt_ids_for_baseline(rep_prompts)
    valid_mask = torch.ones_like(rewards)
    baseline, std = calculate_baseline_and_std_per_prompt(
        prompts=prompt_ids,
        rewards=rewards,
        valid_mask=valid_mask,
        leave_one_out_baseline=use_leave_one_out_baseline,
    )
    if use_global_std:
        return (rewards - baseline) / (rewards.std() + 1e-4)
    return (rewards - baseline) / std.clamp_min(1e-6)


def _build_train_data(
    traj: DiffusionTrajectorySpec,
    advantages_per_sample: torch.Tensor,
) -> BatchedDataDict[DiffusionTrainDataSpec]:
    T = traj["timesteps"].shape[-1]
    # Keep only the columns inside the SDE window so the training-side
    # recompute never forwards out-of-window steps. Window starts may differ
    # per sample (sampled per worker), but the widths must match.
    mask = traj["timestep_mask"]
    widths = mask.sum(dim=1).long()
    w = int(widths.max().item())
    if 0 < w < T:
        assert torch.all(widths == w), (
            f"mixed window widths in one batch: {widths.tolist()}"
        )
        starts = mask.argmax(dim=1)  # first 1 in each row
        cols = starts.unsqueeze(1) + torch.arange(w, device=mask.device)
        cols_lat = starts.unsqueeze(1) + torch.arange(w + 1, device=mask.device)

        def take(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            flat = idx.view(idx.shape[0], idx.shape[1], *([1] * (x.ndim - 2)))
            return torch.gather(x, 1, flat.expand(-1, -1, *x.shape[2:]))

        traj = traj.copy()  # do not mutate the caller's trajectory
        traj["latents"] = take(traj["latents"], cols_lat)
        traj["timesteps"] = torch.gather(traj["timesteps"], 1, cols)
        traj["generation_logprobs"] = torch.gather(traj["generation_logprobs"], 1, cols)
        traj["timestep_mask"] = torch.ones_like(cols, dtype=mask.dtype)
        T = w
    advantages = advantages_per_sample.unsqueeze(-1).expand(-1, T)
    data: DiffusionTrainDataSpec = {
        "latents": traj["latents"],
        "timesteps": traj["timesteps"],
        "generation_logprobs": traj["generation_logprobs"],
        "advantages": advantages,
        "timestep_mask": traj["timestep_mask"],
        "prompt_embeds": traj["prompt_embeds"],
        "prompt_embeds_mask": traj["prompt_embeds_mask"],
        "negative_prompt_embeds": traj["negative_prompt_embeds"],
        "negative_prompt_embeds_mask": traj["negative_prompt_embeds_mask"],
    }
    return BatchedDataDict(data)


def flow_grpo_train(
    policy: FlowGRPOPolicy,
    env: ImageRewardEnvironment,
    train_dataloader: Iterable[BatchedDataDict[Any]],
    val_dataloader: Iterable[BatchedDataDict[Any]] | None,
    *,
    master_config: MasterConfig,
    logger: Logger,
    checkpointer: CheckpointManager,
    val_image_dir: str | None = None,
    num_val_images_to_save: int = 0,
) -> None:
    timer = Timer()
    algo_cfg = master_config.flow_grpo
    K = algo_cfg.num_generations_per_prompt
    max_steps = algo_cfg.max_num_steps
    seed_base = algo_cfg.seed
    checkpointing_enabled = master_config.checkpointing["enabled"]
    save_period = checkpointer.save_period
    # The loss config crosses the Ray boundary into train_step as a dict.
    loss_cfg_dict = master_config.loss_fn.model_dump()

    def run_validation(step: int) -> None:
        _run_validation(
            policy,
            env,
            val_dataloader,
            step=step,
            logger=logger,
            seed=seed_base,
            max_val_samples=algo_cfg.max_val_samples,
            image_dir=val_image_dir,
            num_images_to_save=num_val_images_to_save,
            generation_overrides=algo_cfg.val_generation.model_dump(),
        )

    start_step = 0
    if checkpointing_enabled:
        last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
        training_info = checkpointer.load_training_info(last_checkpoint_path)
        if training_info is not None:
            start_step = training_info["step"]
            policy.load_checkpoint(os.path.join(last_checkpoint_path, "policy"))
            print(
                f"[flow_grpo] resuming from {last_checkpoint_path} "
                f"(start_step={start_step}); note: the dataloader position "
                "is not restored, prompt order restarts from the beginning",
                flush=True,
            )

    if val_dataloader is not None and algo_cfg.val_at_start and start_step == 0:
        # step=-1 → images land in `step_0/` as the pre-training baseline.
        run_validation(step=-1)

    train_iter = iter(train_dataloader)
    for step in range(start_step, max_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)

        with timer.time("rollout"):
            traj = policy.sample_trajectory(
                prompts=batch["prompts"],
                negative_prompts=batch["negative_prompts"],
                metadata=batch["metadata"],
                K=K,
                seed=seed_base + step,
            )

        with timer.time("reward"):
            # The reward pool is CPU by default in the tiny configs; move
            # the GPU-side images to CPU before the Ray hop to keep CUDA
            # tensors local to the trainer worker.
            images_cpu = traj["images"].detach().to("cpu")
            rewards, reward_metrics = env.score_images(
                images_cpu, traj["prompts"], traj["metadata"]
            )

        with timer.time("advantage"):
            advantages_per_sample = _compute_advantages(
                traj["prompts"],
                rewards,
                use_leave_one_out_baseline=algo_cfg.use_leave_one_out_baseline,
                use_global_std=algo_cfg.use_global_std,
            )

        if traj["metadata"]:
            # sample_trajectory replicates metadata K times alongside latents;
            # guard the contract so a mismatch fails here, not inside the loss.
            assert len(traj["metadata"]) == rewards.shape[0], (
                f"metadata length {len(traj['metadata'])} != rollout batch "
                f"{rewards.shape[0]}"
            )

        train_data = _build_train_data(traj, advantages_per_sample)

        ppo_epochs = algo_cfg.ppo_epochs
        total_samples = int(train_data["generation_logprobs"].shape[0])
        # None → single optimizer update over the whole rollout batch.
        mini_size = algo_cfg.ppo_mini_batch_size or total_samples
        with timer.time("train"):
            per_epoch_metrics = []
            for _epoch in range(ppo_epochs):
                # Contiguous in-order mini-batches: sample_trajectory lays out
                # each prompt's K generations contiguously and the config
                # validator makes mini_size a multiple of K, so no GRPO group
                # is ever split. Each mini-batch is one full optimizer update;
                # updates after the first are off-policy by design (verl-omni
                # ppo_mini_batch_size semantics).
                per_mini = [
                    policy.train(
                        train_data.slice(start, start + mini_size), loss_cfg_dict
                    )
                    for start in range(0, total_samples, mini_size)
                ]
                per_epoch_metrics.append(
                    per_mini[0]
                    if len(per_mini) == 1
                    # Cross-mini reduction mirrors verl-omni's reduce_metrics
                    # (mean over all mini-batch updates in the step).
                    else aggregate_worker_metrics(per_mini)
                )
            # Aggregate: keep last-epoch values for ratio/loss (those are
            # the most informative since they reflect the largest policy drift),
            # but record per-epoch losses too.
            train_metrics = dict(per_epoch_metrics[-1])
            if ppo_epochs > 1:
                for i, m in enumerate(per_epoch_metrics):
                    train_metrics[f"epoch_{i}/loss"] = m.get("loss", 0.0)
                    train_metrics[f"epoch_{i}/mean_ratio"] = m.get("mean_ratio", 1.0)

        metrics = {
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **reward_metrics,
            "train/advantage_mean": float(advantages_per_sample.mean().item()),
            "train/advantage_std": float(advantages_per_sample.std().item()),
            "train/reward_mean": float(rewards.mean().item()),
        }
        if policy.num_workers > 1:
            # All-reduced DP ranks must hold identical weights; a non-zero
            # spread means gradient sync is broken.
            checksums = policy.trainable_checksums()
            metrics["train/dp_checksum_spread"] = float(max(checksums) - min(checksums))
        for tag, dur in timer.get_timing_metrics(reduction_op="mean").items():
            if isinstance(dur, float):
                metrics[f"timing/{tag}_s"] = dur
        timer.reset()
        logger.log_metrics(metrics, step=step)
        # Console echo so sanity + interactive runs always have observable
        # progress even when every logger backend is disabled. Report both
        # the final-epoch loss (signal under our reporting convention) and
        # the average-across-inner-epochs loss (matches verl-omni's console
        # aggregation, so the two stacks can be diffed directly).
        loss_val = metrics.get("train/loss")
        ratio_val = metrics.get("train/mean_ratio")
        rew_val = metrics.get("train/reward_mean")
        epoch_losses = [
            metrics[f"train/epoch_{i}/loss"]
            for i in range(ppo_epochs)
            if f"train/epoch_{i}/loss" in metrics
        ]
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else loss_val
        spread = metrics.get("train/dp_checksum_spread")
        print(
            f"[flow_grpo] step={step} "
            f"train/loss_last={loss_val} train/loss_avg={avg_loss} "
            f"train/mean_ratio={ratio_val} train/reward_mean={rew_val}"
            + (f" train/dp_checksum_spread={spread}" if spread is not None else ""),
            flush=True,
        )

        if checkpointing_enabled and save_period > 0 and (step + 1) % save_period == 0:
            # tmp_step_N -> step_N rename is what makes a checkpoint count as
            # complete; a run killed mid-save leaves tmp_step_N, which
            # get_latest_checkpoint_path() ignores.
            tmp_path = checkpointer.init_tmp_checkpoint(
                step + 1, {"step": step + 1, **metrics}, master_config
            )
            policy.save_checkpoint(
                os.path.join(tmp_path, "policy"),
                save_optimizer=checkpointer.save_optimizer,
            )
            checkpointer.finalize_checkpoint(tmp_path)

        if val_dataloader is not None:
            val_period = algo_cfg.val_period
            if val_period > 0 and (step + 1) % val_period == 0:
                run_validation(step=step)

    if val_dataloader is not None and algo_cfg.val_at_end:
        run_validation(step=max_steps - 1)


def _run_validation(
    policy: FlowGRPOPolicy,
    env: ImageRewardEnvironment,
    val_dataloader: Iterable[BatchedDataDict[Any]] | None,
    *,
    step: int,
    logger: Logger,
    seed: int,
    max_val_samples: int = 0,
    image_dir: str | None = None,
    num_images_to_save: int = 0,
    generation_overrides: dict[str, Any] | None = None,
) -> None:
    """Score the val set with K=1 and a fixed seed.

    The fixed seed keeps initial latents identical across successive
    validations, so `val/reward_mean` and the saved images are comparable
    over training steps. `generation_overrides` (e.g. the val-time
    `num_inference_steps`) switches the rollout to the deterministic ODE.
    """
    if val_dataloader is None:
        return
    rewards_acc: list[torch.Tensor] = []
    n_prompts = 0
    saved = 0
    out_dir = (
        os.path.join(image_dir, f"step_{step + 1}") if image_dir is not None else None
    )
    for batch in val_dataloader:
        traj = policy.sample_trajectory(
            prompts=batch["prompts"],
            negative_prompts=batch["negative_prompts"],
            metadata=batch["metadata"],
            K=1,
            seed=seed,
            generation_overrides=generation_overrides,
        )
        # Reward workers may be CPU-only; keep CUDA tensors trainer-local
        # (same convention as the training path).
        images_cpu = traj["images"].detach().to("cpu")
        rewards, _ = env.score_images(images_cpu, traj["prompts"], traj["metadata"])
        rewards_acc.append(rewards.float())
        if out_dir is not None and saved < num_images_to_save:
            saved += _save_val_images(
                images_cpu,
                traj["prompts"],
                rewards,
                out_dir=out_dir,
                offset=saved,
                limit=num_images_to_save - saved,
            )
        n_prompts += len(batch["prompts"])
        if max_val_samples > 0 and n_prompts >= max_val_samples:
            break
    if rewards_acc:
        all_rewards = torch.cat(rewards_acc)
        logger.log_metrics(
            {"val/reward_mean": float(all_rewards.mean().item())},
            step=max(step, 0),
        )


def _save_val_images(
    images: torch.Tensor,
    prompts: list[str],
    rewards: torch.Tensor,
    *,
    out_dir: str,
    offset: int,
    limit: int,
) -> int:
    """Save up to `limit` images (NCHW float [0,1]) as PNGs; returns the count."""
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    n = min(limit, images.shape[0])
    with open(os.path.join(out_dir, "prompts.jsonl"), "a", encoding="utf-8") as f:
        for i in range(n):
            arr = (images[i] * 255).round().clamp(0, 255).to(torch.uint8)
            Image.fromarray(arr.permute(1, 2, 0).numpy()).save(
                os.path.join(out_dir, f"{offset + i:03d}.png")
            )
            f.write(
                json.dumps(
                    {
                        "idx": offset + i,
                        "prompt": prompts[i],
                        "reward": float(rewards[i]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return n
