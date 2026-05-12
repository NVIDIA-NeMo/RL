# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Step-level REINFORCE on a NeMo-Gym environment.

Per-token advantage = step_reward - batch_mean(step_rewards), broadcast over
each assistant turn's tokens. Loss is ClippedPGLossFn; G=1 (no GRPO group).

Run:
    uv run examples/research/run_step_reinforce.py \\
        --config examples/research/configs/step_reinforce_step_arithmetic.yaml
"""
import argparse
import os
import pprint
from typing import Any

import ray
import torch
from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import (
    MasterConfig,
    refit_policy_generation,
    setup,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.nemo_gym import NemoGymConfig, setup_nemo_gym_config
from nemo_rl.environments.utils import create_env
from nemo_rl.experience.rollouts import run_async_nemo_gym_rollout
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Step-level REINFORCE on NeMo-Gym")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    args, overrides = parser.parse_known_args()
    return args, overrides


def _set_message_loss_metadata(message_log_batch: list[list[dict]]) -> None:
    """Add token_loss_mask and zero generation_logprobs (matches grpo_train preflight)."""
    for ml in message_log_batch:
        for m in ml:
            if m["role"] == "assistant":
                m["token_loss_mask"] = torch.ones_like(m["token_ids"])
            else:
                m["token_loss_mask"] = torch.zeros_like(m["token_ids"])
            if "generation_logprobs" not in m:
                m["generation_logprobs"] = torch.zeros_like(
                    m["token_ids"], dtype=torch.float32
                )


def _per_token_step_advantages(
    message_log: list[dict],
    step_rewards: list[float],
    baseline: float,
) -> torch.Tensor:
    """Per-token advantages: (step_reward - baseline) on assistant tokens, 0 elsewhere."""
    parts = []
    step_idx = 0
    for m in message_log:
        n = int(m["token_ids"].shape[0])
        if m["role"] == "assistant":
            r = step_rewards[step_idx] if step_idx < len(step_rewards) else 0.0
            parts.append(torch.full((n,), float(r) - baseline, dtype=torch.float32))
            step_idx += 1
        else:
            parts.append(torch.zeros(n, dtype=torch.float32))
    return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.float32)


def _pad_advantages(
    per_traj: list[torch.Tensor], target_len: int
) -> torch.Tensor:
    out = torch.zeros(len(per_traj), target_len, dtype=torch.float32)
    for i, a in enumerate(per_traj):
        n = min(a.shape[0], target_len)
        out[i, :n] = a[:n]
    return out


def step_reinforce_train(
    policy,
    policy_generation,
    dataloader,
    tokenizer,
    loss_fn,
    task_to_env: dict,
    logger,
    master_config: MasterConfig,
) -> None:
    """Step-level REINFORCE training loop. Replaces grpo_train."""
    max_num_steps = master_config["grpo"]["max_num_steps"]
    colocated = master_config["policy"]["generation"]["colocated"]["enabled"]
    skip_ref = master_config["grpo"].get("skip_reference_policy_logprobs_calculation", True)
    pad_id = tokenizer.pad_token_id

    NEED_REFIT = policy_generation is not None and policy_generation is not policy
    if policy_generation is None:
        policy_generation = policy

    step = 0
    for batch in dataloader:
        if step >= max_num_steps:
            break

        print(f"\n{'=' * 25} Step {step + 1}/{max_num_steps} {'=' * 25}", flush=True)

        if NEED_REFIT:
            refit_policy_generation(policy, policy_generation, colocated)
        else:
            policy_generation.prepare_for_generation()

        rollout = run_async_nemo_gym_rollout(
            policy_generation=policy_generation,
            input_batch=batch,
            tokenizer=tokenizer,
            task_to_env=task_to_env,
            generation_config=master_config["policy"]["generation"],
            max_seq_len=None,
            max_rollout_turns=None,
            greedy=False,
        )
        repeated_batch = rollout.final_batch
        policy_generation.finish_generation()
        logger.log_metrics(rollout.rollout_metrics, step + 1, prefix="train")

        all_step_rewards: list[list[float]] = []
        for full in repeated_batch["full_result"]:
            srs = full.get("step_rewards")
            if srs is None:
                srs = [float(full.get("reward", 0.0))]
            all_step_rewards.append([float(r) for r in srs])

        flat = torch.tensor(
            [r for srs in all_step_rewards for r in srs], dtype=torch.float32
        )
        baseline = float(flat.mean()) if flat.numel() > 0 else 0.0
        step_reward_mean = baseline
        step_reward_std = float(flat.std()) if flat.numel() > 1 else 0.0

        _set_message_loss_metadata(repeated_batch["message_log"])
        flat_messages, input_lengths = batched_message_log_to_flat_message(
            repeated_batch["message_log"],
            pad_value_dict={"token_ids": pad_id},
            make_sequence_length_divisible_by=master_config["policy"][
                "make_sequence_length_divisible_by"
            ],
        )
        target_len = int(flat_messages["token_ids"].shape[1])

        per_traj_advs = [
            _per_token_step_advantages(ml, srs, baseline)
            for ml, srs in zip(repeated_batch["message_log"], all_step_rewards)
        ]
        advantages = _pad_advantages(per_traj_advs, target_len)

        batch_size = int(flat_messages["token_ids"].shape[0])
        train_data = BatchedDataDict(
            {
                "input_ids": flat_messages["token_ids"],
                "input_lengths": input_lengths,
                "generation_logprobs": flat_messages["generation_logprobs"],
                "token_mask": flat_messages["token_loss_mask"],
                "sample_mask": torch.ones(batch_size, dtype=torch.float32),
                "advantages": advantages,
            }
        )
        train_data.update(flat_messages.get_multimodal_dict(as_tensors=False))
        train_data.to("cpu")

        policy.prepare_for_lp_inference()
        logprob_data = BatchedDataDict(
            {
                "input_ids": train_data["input_ids"],
                "input_lengths": train_data["input_lengths"],
                "token_mask": train_data["token_mask"],
                "sample_mask": train_data["sample_mask"],
            }
        )
        train_data["prev_logprobs"] = policy.get_logprobs(logprob_data)["logprobs"]
        if not skip_ref:
            train_data["reference_policy_logprobs"] = policy.get_reference_policy_logprobs(
                logprob_data
            )["reference_logprobs"]

        policy.prepare_for_training()
        result = policy.train(train_data, loss_fn)
        policy.finish_training()

        logger.log_metrics(
            {
                "loss": float(result["loss"]),
                "grad_norm": float(result.get("grad_norm", 0.0)),
                "step_reward_mean": step_reward_mean,
                "step_reward_std": step_reward_std,
                "num_step_samples": int(flat.numel()),
                "advantage_abs_mean": float(advantages.abs().mean()),
            },
            step + 1,
            prefix="train",
        )

        step += 1


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "step_reinforce_step_arithmetic.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )
    setup_nemo_gym_config(config, tokenizer)

    print("\n▶ Setting up data...")
    train_dataset, val_dataset = setup_response_data(
        tokenizer, config["data"], env_configs=None
    )

    print("Final config:")
    pprint.pprint(config)

    init_ray()

    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    nemo_gym_config = NemoGymConfig(
        model_name=policy_generation.cfg["model_name"],
        base_urls=policy_generation.dp_openai_server_base_urls,
        initial_global_config_dict=config["env"]["nemo_gym"],
    )
    nemo_gym = create_env(env_name="nemo_gym", env_config=nemo_gym_config)
    ray.get(nemo_gym.health_check.remote())
    task_to_env: dict[str, Any] = {"nemo_gym": nemo_gym}

    print("🚀 Running step-level REINFORCE training")
    step_reinforce_train(
        policy=policy,
        policy_generation=policy_generation,
        dataloader=dataloader,
        tokenizer=tokenizer,
        loss_fn=loss_fn,
        task_to_env=task_to_env,
        logger=logger,
        master_config=master_config,
    )


if __name__ == "__main__":
    main()
