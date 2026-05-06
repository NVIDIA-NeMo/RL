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

import argparse
import inspect
import os
import pprint
from typing import Any

import ray
import torch
from omegaconf import OmegaConf

import nemo_rl.algorithms.grpo as grpo_module
from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.algorithms.utils import (
    calculate_baseline_and_std_per_prompt,
    get_tokenizer,
    set_seed,
)
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed import ray_actor_environment_registry as actor_env_registry
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.experience.rollouts import calculate_rewards
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.interfaces import GenerationDatumSpec
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run routed GRPO training")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "configs", "grpo_math_1B_routed.yaml"
        ),
        help="Path to YAML config file",
    )
    return parser.parse_known_args()


def _to_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().mean().item())
    return float(value)


def _install_worker_extension_setup_patch(policy_config: dict[str, Any]) -> None:
    """Make this script tolerant of older setup() code that drops worker extensions."""
    worker_fqn = policy_config.get("worker_extension_cls_fqn")
    if worker_fqn is None:
        return

    policy_cls = grpo_module.Policy
    if "worker_extension_cls_fqn" not in inspect.signature(policy_cls).parameters:
        raise RuntimeError(
            "This NeMo RL runtime cannot construct policy worker extensions. "
            "Update nemo_rl.models.policy.lm_policy.Policy to accept "
            "worker_extension_cls_fqn, or run this script from the patched checkout."
        )

    def routed_policy_factory(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("worker_extension_cls_fqn", worker_fqn)
        return policy_cls(*args, **kwargs)

    grpo_module.Policy = routed_policy_factory


def _install_actor_environment_patch(policy_config: dict[str, Any]) -> None:
    worker_fqn = policy_config.get("worker_extension_cls_fqn")
    if worker_fqn is None:
        return

    megatron_worker_fqn = (
        "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
    )
    registry = actor_env_registry.ACTOR_ENVIRONMENT_REGISTRY
    if worker_fqn not in registry:
        registry[worker_fqn] = registry[megatron_worker_fqn]


def _assert_routed_policy_worker(policy: Any) -> None:
    try:
        futures = policy.worker_group.run_all_workers_single_data("is_routed_worker")
        results = ray.get(futures)
    except Exception as exc:
        raise RuntimeError(
            "Routed GRPO expected RoutedMegatronPolicyWorker, but the policy was "
            "constructed with the stock Megatron worker. Make sure "
            "policy.worker_extension_cls_fqn points to "
            "nemo_rl.models.policy.workers.routed_megatron_policy_worker."
            "RoutedMegatronPolicyWorker and that grpo.setup passes it into Policy."
        ) from exc

    if not all(bool(result) for result in results):
        raise RuntimeError("Not all policy workers reported routed-worker support.")


def _compute_advantages(
    prompt_ids: torch.Tensor,
    rewards: torch.Tensor,
    *,
    normalize_rewards: bool,
    leave_one_out: bool,
) -> torch.Tensor:
    baseline, std = calculate_baseline_and_std_per_prompt(
        prompt_ids,
        rewards,
        torch.ones_like(rewards),
        leave_one_out_baseline=leave_one_out,
    )
    advantages = rewards - baseline
    if normalize_rewards:
        non_zero_std = std > 0
        advantages[non_zero_std] = advantages[non_zero_std] / (
            std[non_zero_std] + 1.0e-6
        )
    return advantages


def _generate_routed_responses(
    policy: Any,
    generation_input_data: BatchedDataDict[GenerationDatumSpec],
    batch: BatchedDataDict[Any],
    tokenizer: Any,
    *,
    input_lengths: torch.Tensor,
    include_logprobs: bool = True,
    greedy: bool = False,
) -> tuple[BatchedDataDict[Any], list[torch.Tensor], dict[str, float | int]]:
    if "stop_strings" in batch:
        generation_input_data["stop_strings"] = batch["stop_strings"]
    else:
        generation_input_data["stop_strings"] = [None] * len(input_lengths)

    generation_outputs = policy.generate(generation_input_data, greedy=greedy)
    output_ids = generation_outputs["output_ids"]
    generation_lengths = generation_outputs["generation_lengths"]
    unpadded_sequence_lengths = generation_outputs["unpadded_sequence_lengths"]
    response_truncated = generation_outputs.get("truncated")

    for key in (
        "route_masks",
        "route_logprobs",
        "route_entropies",
        "route_compute_fractions",
        "route_keep_fractions",
    ):
        if key in generation_outputs:
            batch[key] = generation_outputs[key]

    generated_ids = []
    for idx in range(len(input_lengths)):
        input_len = input_lengths[idx].item()
        total_length = unpadded_sequence_lengths[idx].item()
        generated_ids.append(output_ids[idx, input_len:total_length])

    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    for idx, (text, input_length, total_length) in enumerate(
        zip(generated_texts, input_lengths, unpadded_sequence_lengths)
    ):
        assistant_message = {
            "role": "assistant",
            "content": text,
            "token_ids": output_ids[idx, input_length:total_length],
        }
        if include_logprobs and "logprobs" in generation_outputs:
            assistant_message["generation_logprobs"] = generation_outputs["logprobs"][
                idx, input_length:total_length
            ]
        batch["message_log"][idx].append(assistant_message)

    gen_metrics = {
        "mean_generation_length": generation_lengths.float().mean().item(),
        "total_generated_tokens": generation_lengths.sum().item(),
    }
    if response_truncated is not None:
        gen_metrics["_response_truncated"] = response_truncated

    return batch, generated_ids, gen_metrics


def _train_router(policy: Any, data: BatchedDataDict[Any]) -> dict[str, Any]:
    if hasattr(policy, "train_router"):
        return policy.train_router(data)

    dp_size = policy.sharding_annotations.get_axis_size("data_parallel")
    sharded_data = data.shard_by_batch_size(dp_size, batch_size=None)
    futures = policy.worker_group.run_all_workers_sharded_data(
        "train_router",
        data=sharded_data,
        in_sharded_axes=["data_parallel"],
        replicate_on_axes=[
            "context_parallel",
            "tensor_parallel",
            "pipeline_parallel",
        ],
        output_is_replicated=[
            "context_parallel",
            "tensor_parallel",
            "pipeline_parallel",
        ],
    )
    results = policy.worker_group.get_all_worker_results(futures)

    aggregated: dict[str, Any] = {}
    for key in results[0]:
        values = [result[key] for result in results]
        if torch.is_tensor(values[0]):
            aggregated[key] = torch.stack([v.detach().cpu() for v in values]).mean()
        elif isinstance(values[0], (int, float)):
            aggregated[key] = sum(float(v) for v in values) / len(values)
        else:
            aggregated[key] = values[0]
    return aggregated


def routed_grpo_train(
    policy,
    wrapped_dataloader,
    tokenizer,
    task_to_env,
    logger,
    master_config: MasterConfig,
) -> None:
    if master_config["data"]["use_multiple_dataloader"]:
        raise NotImplementedError("Routed MVP does not support multiple dataloaders")

    set_seed(master_config["grpo"]["seed"])
    routing_cfg = master_config["policy"]["routing"]
    compute_penalty = float(routing_cfg.get("compute_penalty", 0.0))
    target_compute_fraction = routing_cfg.get("target_compute_fraction")
    total_steps = 0
    max_steps = int(master_config["grpo"]["max_num_steps"])
    max_epochs = int(master_config["grpo"]["max_num_epochs"])
    generations_per_prompt = int(master_config["grpo"]["num_generations_per_prompt"])

    for epoch in range(max_epochs):
        for batch in wrapped_dataloader:
            if total_steps >= max_steps:
                print("Max number of routed steps reached; stopping.", flush=True)
                return

            repeated_batch = batch.repeat_interleave(generations_per_prompt)
            flat_messages, input_lengths = batched_message_log_to_flat_message(
                repeated_batch["message_log"],
                pad_value_dict={"token_ids": tokenizer.pad_token_id},
            )
            input_ids = flat_messages["token_ids"]
            generation_input = BatchedDataDict[GenerationDatumSpec](
                {
                    "input_ids": input_ids,
                    "input_lengths": input_lengths,
                    "stop_strings": repeated_batch.get(
                        "stop_strings", [None] * repeated_batch.size
                    ),
                }
            )
            generation_input.update(flat_messages.get_multimodal_dict(as_tensors=False))

            print(
                f"\n===== Routed Step {total_steps + 1}/{max_steps} "
                f"(epoch {epoch + 1}/{max_epochs}) =====",
                flush=True,
            )
            print("Generating routed responses...", flush=True)
            repeated_batch, generated_ids, gen_metrics = _generate_routed_responses(
                policy,
                generation_input,
                repeated_batch,
                tokenizer,
                input_lengths=input_lengths,
                include_logprobs=True,
                greedy=False,
            )

            print("Scoring math rewards...", flush=True)
            env_output = calculate_rewards(repeated_batch, task_to_env)
            task_rewards = (
                env_output.rewards.sum(dim=1)
                if env_output.rewards.ndim >= 2
                else env_output.rewards
            ).to(torch.float32)
            task_rewards = task_rewards * env_output.terminateds.to(torch.float32)

            compute_fraction = repeated_batch["route_compute_fractions"].to(
                torch.float32
            )
            if target_compute_fraction is None:
                cost = compute_fraction
            else:
                cost = torch.clamp(
                    compute_fraction - float(target_compute_fraction), min=0.0
                )
            routed_rewards = task_rewards - compute_penalty * cost

            advantages = _compute_advantages(
                input_ids,
                routed_rewards,
                normalize_rewards=master_config["grpo"]["normalize_rewards"],
                leave_one_out=master_config["grpo"]["use_leave_one_out_baseline"],
            )

            route_train_data = BatchedDataDict(
                {
                    "input_ids": input_ids,
                    "input_lengths": input_lengths,
                    "route_masks": repeated_batch["route_masks"],
                    "old_route_logprobs": repeated_batch["route_logprobs"],
                    "advantages": advantages,
                    "sample_mask": torch.ones_like(advantages),
                }
            )

            print("Training router...", flush=True)
            train_results = _train_router(policy, route_train_data)

            metrics = {
                "task_reward": float(task_rewards.mean().item()),
                "task_reward_sum": float(task_rewards.sum().item()),
                "routed_reward": float(routed_rewards.mean().item()),
                "compute_fraction": float(compute_fraction.mean().item()),
                "compute_fraction_min": float(compute_fraction.min().item()),
                "compute_fraction_max": float(compute_fraction.max().item()),
                "keep_fraction": float(
                    repeated_batch["route_keep_fractions"].float().mean().item()
                ),
                "keep_fraction_min": float(
                    repeated_batch["route_keep_fractions"].float().min().item()
                ),
                "keep_fraction_max": float(
                    repeated_batch["route_keep_fractions"].float().max().item()
                ),
                "route_entropy": float(
                    repeated_batch["route_entropies"].float().mean().item()
                ),
                "route_logprob": float(
                    repeated_batch["route_logprobs"].float().mean().item()
                ),
                "mean_generation_length": float(
                    gen_metrics["mean_generation_length"]
                ),
                "total_generated_tokens": float(gen_metrics["total_generated_tokens"]),
                "advantage_mean": float(advantages.mean().item()),
                "advantage_std": float(advantages.std(unbiased=False).item()),
            }
            metrics.update({key: _to_float(value) for key, value in train_results.items()})
            logger.log_metrics(
                metrics,
                total_steps + 1,
                prefix="train",
                step_finished=True,
            )

            print(
                "Routed results: "
                f"task_reward={metrics['task_reward']:.4f}, "
                f"task_sum={metrics['task_reward_sum']:.1f}, "
                f"routed_reward={metrics['routed_reward']:.4f}, "
                f"compute={metrics['compute_fraction']:.4f} "
                f"[{metrics['compute_fraction_min']:.4f}, {metrics['compute_fraction_max']:.4f}], "
                f"keep={metrics['keep_fraction']:.4f} "
                f"[{metrics['keep_fraction_min']:.4f}, {metrics['keep_fraction_max']:.4f}], "
                f"expected_compute={metrics.get('router_expected_compute_fraction', 0.0):.4f}, "
                f"route_entropy={metrics['route_entropy']:.4f}, "
                f"router_entropy={metrics.get('router_entropy', 0.0):.4f}, "
                f"ratio={metrics.get('router_ratio_mean', 0.0):.4f}, "
                f"grad_norm={metrics.get('router_grad_norm', 0.0):.4f}, "
                f"adv_std={metrics['advantage_std']:.4f}, "
                f"gen_len={metrics['mean_generation_length']:.1f}, "
                f"router_loss={metrics['router_loss']:.4f}",
                flush=True,
            )
            total_steps += 1


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print("Final config:")
    pprint.pprint(config)

    init_ray()

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"],
        tokenizer,
        has_refit_draft_weights=bool(config["policy"]["draft"]["enabled"]),
    )

    dataset, val_dataset, task_to_env, _ = setup_response_data(
        tokenizer,
        config["data"],
        config["env"],
    )
    _install_actor_environment_patch(config["policy"])
    _install_worker_extension_setup_patch(config["policy"])
    (
        policy,
        policy_generation,
        _cluster,
        dataloader,
        _val_dataloader,
        _loss_fn,
        logger,
        _checkpointer,
        _grpo_state,
        master_config,
    ) = grpo_module.setup(config, tokenizer, dataset, val_dataset)

    if policy_generation is not None:
        raise ValueError(
            "Routed MVP expects policy.generation.backend=megatron so generation "
            "runs through the routed Megatron policy worker."
        )
    _assert_routed_policy_worker(policy)

    routed_grpo_train(
        policy=policy,
        wrapped_dataloader=dataloader,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        logger=logger,
        master_config=master_config,
    )


if __name__ == "__main__":
    main()
