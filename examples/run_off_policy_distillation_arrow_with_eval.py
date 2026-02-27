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

"""Off-policy distillation on arrow data with inline MATH/MMLU evaluation.

Extends run_off_policy_distillation_arrow.py with periodic generation-based
evaluation (MATH, MMLU) using a colocated vLLM generation engine, following
the same pattern as run_sft_arrow_with_eval.py.

Usage:
    uv run examples/run_off_policy_distillation_arrow_with_eval.py \
        --config examples/configs/llama_off_policy_arrow.yaml
"""

import argparse
import os
import pprint
import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, cast

import numpy as np
import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import refit_policy_generation
from nemo_rl.algorithms.loss_functions import (
    DistillationLossConfig,
    DistillationLossDataDict,
    DistillationLossFn,
)
from nemo_rl.algorithms.off_policy_distillation import (
    OffPolicyDistillationSaveState,
    OffPolicyMasterConfig,
    _default_distillation_save_state,
    check_vocab_equality,
)
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset, load_eval_dataset
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.math_environment import MathEnvironment
from nemo_rl.experience.rollouts import run_multi_turn_rollout
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import Logger, get_next_experiment_dir, print_message_log_samples
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer

import ray

OmegaConf.register_new_resolver("mul", lambda a, b: a * b, replace=True)
OmegaConf.register_new_resolver("max", lambda a, b: max(a, b), replace=True)


# =========================================================================
# CLI
# =========================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Off-policy distillation on arrow data with inline MATH/MMLU eval"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    args, overrides = parser.parse_known_args()
    return args, overrides


# =========================================================================
# Arrow-text training data (reused from run_off_policy_distillation_arrow.py)
# =========================================================================
def _sft_preprocessor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
    add_bos: bool = True,
    add_eos: bool = True,
    add_generation_prompt: bool = False,
    datum_preprocessor: Optional[Callable] = None,
) -> DatumSpec:
    from nemo_rl.data.llm_message_utils import get_formatted_message_log

    if datum_preprocessor is not None:
        datum_dict = datum_preprocessor(datum_dict)

    message_log = get_formatted_message_log(
        datum_dict["messages"],
        tokenizer,
        task_data_spec,
        add_bos_token=add_bos,
        add_eos_token=add_eos,
        add_generation_prompt=add_generation_prompt,
        tools=datum_dict.get("tools", None),
    )

    length = sum(len(m["token_ids"]) for m in message_log)
    loss_multiplier = 1.0
    if length > max_seq_length:
        for message in message_log:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    return {
        "message_log": message_log,
        "length": length,
        "extra_env_info": None,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }


def setup_train_data(tokenizer: AutoTokenizer, data_config: DataConfig, seed: int):
    from nemo_rl.data.datasets import load_response_dataset

    print("\n▶ Setting up training data...")
    data = load_response_dataset(data_config, seed)
    train_dataset_raw = data.formatted_ds["train"]
    task_spec = data.task_spec

    train_dataset = AllTaskProcessedDataset(
        train_dataset_raw,
        tokenizer,
        task_spec,
        partial(
            _sft_preprocessor,
            add_bos=data_config.get("add_bos", True),
            add_eos=data_config.get("add_eos", True),
            add_generation_prompt=data_config.get("add_generation_prompt", False),
        ),
        max_seq_length=data_config["max_input_seq_length"],
    )
    print(f"  ✓ Training dataset loaded with {len(train_dataset)} samples")
    return train_dataset, task_spec


# =========================================================================
# Eval data + environments (from run_sft_arrow_with_eval.py)
# =========================================================================
def setup_eval_data(
    tokenizer: AutoTokenizer,
    eval_config: dict[str, Any],
    max_seq_length: int,
) -> tuple[
    dict[str, StatefulDataLoader],
    dict[str, dict[str, EnvironmentInterface]],
]:
    print("\n▶ Setting up evaluation benchmarks...")
    eval_dataloaders: dict[str, StatefulDataLoader] = {}
    eval_envs: dict[str, dict[str, EnvironmentInterface]] = {}

    for bench_name, bench_cfg in eval_config["benchmarks"].items():
        dataset_name = bench_cfg["dataset_name"]
        prompt_file = bench_cfg.get("prompt_file")
        system_prompt_file = bench_cfg.get("system_prompt_file")
        env_cfg = bench_cfg.get("env", {"num_workers": 8})

        data_cfg = {
            "dataset_name": dataset_name,
            "prompt_file": prompt_file,
            "system_prompt_file": system_prompt_file,
        }
        base_dataset = load_eval_dataset(data_cfg)

        task_spec = TaskDataSpec(
            task_name=dataset_name,
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )

        dataset = AllTaskProcessedDataset(
            dataset=base_dataset.rekeyed_ds,
            tokenizer=tokenizer,
            default_task_data_spec=task_spec,
            task_data_processors=base_dataset.processor,
            max_seq_length=max_seq_length,
        )

        dataloader = StatefulDataLoader(
            dataset,
            batch_size=eval_config["val_batch_size"],
            shuffle=False,
            collate_fn=rl_collate_fn,
        )

        math_env = MathEnvironment.options(
            runtime_env={
                "py_executable": get_actor_python_env(
                    "nemo_rl.environments.math_environment.MathEnvironment"
                ),
                "env_vars": dict(os.environ),
            }
        ).remote(env_cfg)

        task_to_env: dict[str, EnvironmentInterface] = defaultdict(lambda: math_env)
        task_to_env[dataset_name] = math_env
        task_to_env[None] = math_env

        eval_dataloaders[bench_name] = dataloader
        eval_envs[bench_name] = task_to_env
        print(f"  ✓ {bench_name}: {len(dataset)} samples, env={dataset_name}")

    return eval_dataloaders, eval_envs


# =========================================================================
# Generation-based validation (from run_sft_arrow_with_eval.py)
# =========================================================================
def gen_validate(
    generation: GenerationInterface,
    eval_dataloaders: dict[str, StatefulDataLoader],
    eval_envs: dict[str, dict[str, EnvironmentInterface]],
    eval_config: dict[str, Any],
    master_config: dict[str, Any],
    step: int,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    timer = Timer()
    all_val_metrics: dict[str, Any] = {}

    max_val_samples = eval_config.get("max_val_samples", 512)
    val_batch_size = eval_config["val_batch_size"]
    max_batches = max_val_samples // val_batch_size
    max_rollout_turns = eval_config.get("max_rollout_turns", 1)
    max_seq_len = master_config["policy"]["max_total_sequence_length"]

    with timer.time("total_eval_time"):
        for bench_name, dataloader in eval_dataloaders.items():
            print(f"\n▶ Evaluating {bench_name} at step {step}...", flush=True)
            total_rewards = []
            total_lengths = []
            all_message_logs = []

            for batch_idx, val_batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break

                val_batch, gen_metrics = run_multi_turn_rollout(
                    generation,
                    val_batch,
                    tokenizer,
                    eval_envs[bench_name],
                    max_seq_len=max_seq_len,
                    max_rollout_turns=max_rollout_turns,
                    greedy=True,
                )

                rewards = val_batch["total_reward"]
                total_rewards.extend(rewards.tolist())
                total_lengths.append(gen_metrics["mean_gen_tokens_per_sample"])

                to_env = [
                    get_keys_from_message_log(
                        val_batch["message_log"][i], ["role", "content"]
                    )
                    for i in range(len(val_batch["message_log"]))
                ]
                all_message_logs.extend(to_env)

            accuracy = (
                sum(total_rewards) / len(total_rewards)
                if len(total_rewards) > 0
                else 0
            )
            avg_length = (
                sum(total_lengths) / len(total_lengths)
                if len(total_lengths) > 0
                else 0
            )

            all_val_metrics[f"{bench_name}_accuracy"] = accuracy
            all_val_metrics[f"{bench_name}_avg_length"] = avg_length

            print(f"\n📊 {bench_name} Results:")
            print(f"    • Accuracy: {accuracy:.4f}")
            print(f"    • Avg response length: {avg_length:.1f} tokens")
            print(f"    • Samples processed: {len(total_rewards)}", flush=True)

            try:
                num_to_print = master_config["logger"].get(
                    "num_val_samples_to_print", 3
                )
                print_message_log_samples(
                    all_message_logs,
                    total_rewards,
                    num_samples=min(num_to_print, len(all_message_logs)),
                    step=step,
                )
            except Exception as e:
                print(f"  ⚠️ Error displaying samples: {e}", flush=True)

    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    eval_time = timing_metrics.get("total_eval_time", 0)
    print(f"\n  ⏱️  Total eval time: {eval_time:.2f}s", flush=True)
    timer.reset()

    return all_val_metrics, timing_metrics


# =========================================================================
# Training loop (off-policy distillation + eval hooks)
# =========================================================================
def off_policy_distillation_train_with_eval(
    student_policy: Policy,
    teacher_policy: Policy,
    dataloader: StatefulDataLoader,
    tokenizer: AutoTokenizer,
    loss_fn: DistillationLossFn,
    master_config: dict[str, Any],
    logger: Logger,
    checkpointer: CheckpointManager,
    distillation_save_state: OffPolicyDistillationSaveState,
    generation: Optional[GenerationInterface],
    eval_dataloaders: Optional[dict[str, StatefulDataLoader]],
    eval_envs: Optional[dict[str, dict[str, EnvironmentInterface]]],
) -> None:
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    current_epoch = distillation_save_state["current_epoch"]
    current_step = distillation_save_state["current_step"]
    total_steps = distillation_save_state["total_steps"]
    consumed_samples = distillation_save_state["consumed_samples"]
    total_valid_tokens = distillation_save_state["total_valid_tokens"]
    max_epochs = master_config["distillation"]["max_num_epochs"]
    max_steps = master_config["distillation"]["max_num_steps"]

    eval_config = master_config.get("eval")
    eval_period = eval_config["val_period"] if eval_config else 0
    eval_at_start = eval_config.get("val_at_start", False) if eval_config else False
    has_eval = (
        eval_period > 0
        and generation is not None
        and eval_dataloaders is not None
        and eval_envs is not None
    )

    colocated_inference = (
        master_config["policy"]["generation"]["colocated"]["enabled"]
        if master_config["policy"].get("generation")
        else True
    )
    generation_stale = True

    # ── optional eval at start ──
    if has_eval and eval_at_start and total_steps == 0:
        print("\n🔍 Running initial evaluation...", flush=True)
        refit_policy_generation(student_policy, generation, colocated_inference)
        generation_stale = False
        val_metrics, val_timings = gen_validate(
            generation, eval_dataloaders, eval_envs, eval_config, master_config,
            step=0, tokenizer=tokenizer,
        )
        generation.finish_generation()
        logger.log_metrics(val_metrics, 0, prefix="validation")
        logger.log_metrics(val_timings, 0, prefix="timing/validation")

    while total_steps < max_steps and current_epoch < max_epochs:
        print(
            f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_epochs} {'=' * 25}",
            flush=True,
        )

        for batch in dataloader:
            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(dataloader), max_steps)} {'=' * 25}",
                flush=True,
            )
            maybe_gpu_profile_step(student_policy, total_steps + 1)
            val_metrics = None

            with timer.time("total_step_time"):
                # ==== Data Processing ====
                print("▶ Processing batch data (off-policy)...", flush=True)
                with timer.time("data_processing"):
                    for message_log in batch["message_log"]:
                        for message in message_log:
                            if "token_loss_mask" not in message:
                                if message["role"] == "assistant":
                                    message["token_loss_mask"] = torch.ones_like(
                                        message["token_ids"]
                                    )
                                else:
                                    message["token_loss_mask"] = torch.zeros_like(
                                        message["token_ids"]
                                    )

                    flat_messages, input_lengths = batched_message_log_to_flat_message(
                        batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config["policy"][
                            "make_sequence_length_divisible_by"
                        ],
                    )

                    train_data = BatchedDataDict(
                        {
                            "input_ids": flat_messages["token_ids"],
                            "input_lengths": input_lengths,
                            "token_mask": flat_messages["token_loss_mask"],
                            "sample_mask": batch["loss_multiplier"],
                        }
                    )
                    train_data.update(
                        flat_messages.get_multimodal_dict(as_tensors=False)
                    )
                    train_data.to("cpu")

                # ==== Teacher Logprob Inference ====
                print("▶ Computing teacher logprobs...", flush=True)
                with timer.time("teacher_logprob_inference_prep"):
                    teacher_policy.prepare_for_lp_inference()

                with timer.time("teacher_logprob_inference"):
                    teacher_topk = teacher_policy.get_topk_logits(
                        train_data, k=master_config["distillation"]["topk_logits_k"]
                    )
                    if isinstance(teacher_topk, list):
                        train_data["teacher_topk_ipc_handles"] = teacher_topk
                    else:
                        train_data["teacher_topk_logits"] = teacher_topk["topk_logits"]
                        train_data["teacher_topk_indices"] = teacher_topk["topk_indices"]

                # ==== Student Training ====
                print("▶ Training student policy...", flush=True)
                with timer.time("training_prep"):
                    teacher_policy.offload_after_refit()
                    student_policy.prepare_for_training()

                with timer.time("policy_training"):
                    train_results = student_policy.train(train_data, loss_fn)

                generation_stale = True

                is_last_step = (total_steps + 1 >= max_steps) or (
                    (current_epoch + 1 == max_epochs)
                    and (current_step + 1 == len(dataloader))
                )

                # ==== Generation-based eval ====
                if has_eval and eval_period > 0 and (total_steps + 1) % eval_period == 0:
                    print(
                        f"\n🔍 Running generation-based eval at step {total_steps + 1}...",
                        flush=True,
                    )
                    with timer.time("gen_eval"):
                        if generation_stale:
                            refit_policy_generation(
                                student_policy, generation, colocated_inference
                            )
                            generation_stale = False
                        val_metrics, val_timings = gen_validate(
                            generation,
                            eval_dataloaders,
                            eval_envs,
                            eval_config,
                            master_config,
                            step=total_steps + 1,
                            tokenizer=tokenizer,
                        )
                        generation.finish_generation()
                    logger.log_metrics(
                        val_timings, total_steps + 1, prefix="timing/validation"
                    )
                    logger.log_metrics(
                        val_metrics, total_steps + 1, prefix="validation"
                    )
                    student_policy.prepare_for_training()

                # ==== Metrics ====
                metrics = {
                    "loss": train_results["loss"].numpy(),
                    "grad_norm": train_results["grad_norm"].numpy(),
                    "mean_seq_length": batch["length"].numpy().mean(),
                    "total_num_tokens": input_lengths.numpy().sum(),
                }
                metrics.update(train_results["all_mb_metrics"])
                for k, v in metrics.items():
                    if k in {
                        "lr",
                        "wd",
                        "global_valid_seqs",
                        "global_valid_toks",
                        "mean_seq_length",
                    }:
                        metrics[k] = np.mean(v).item()
                    else:
                        metrics[k] = np.sum(v).item()
                total_valid_tokens += metrics["global_valid_toks"]

                # ==== Checkpointing ====
                consumed_samples += master_config["distillation"]["num_prompts_per_step"]
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config["checkpointing"]["save_period"]
                    == 0
                )
                should_save_by_timeout = timeout.check_save()

                if master_config["checkpointing"]["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    student_policy.prepare_for_training()

                    distillation_save_state["current_epoch"] = current_epoch
                    distillation_save_state["current_step"] = current_step + 1
                    distillation_save_state["total_steps"] = total_steps + 1
                    distillation_save_state["total_valid_tokens"] = total_valid_tokens
                    distillation_save_state["consumed_samples"] = consumed_samples

                    full_metric_name = master_config["checkpointing"]["metric_name"]
                    if full_metric_name is not None:
                        assert full_metric_name.startswith(
                            "train:"
                        ) or full_metric_name.startswith("val:"), (
                            f"metric_name={full_metric_name} must start with 'val:' or 'train:'"
                        )
                        prefix, metric_name = full_metric_name.split(":", 1)
                        metrics_source = metrics if prefix == "train" else val_metrics
                        if not metrics_source:
                            warnings.warn(
                                f"Checkpointing metric {metric_name} requested but no {prefix} metrics collected.",
                                stacklevel=2,
                            )
                            if full_metric_name in distillation_save_state:
                                del distillation_save_state[full_metric_name]
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            distillation_save_state[full_metric_name] = metrics_source[
                                metric_name
                            ]

                    with timer.time("checkpointing"):
                        print(
                            f"Saving checkpoint for step {total_steps + 1}...",
                            flush=True,
                        )
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, distillation_save_state, master_config
                        )
                        student_policy.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=os.path.join(
                                checkpoint_path, "policy", "optimizer"
                            ),
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                            checkpointing_cfg=master_config["checkpointing"],
                        )
                        torch.save(
                            dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            # ── Logging ──
            log_data = {"content": flat_messages["content"]}
            log_data["input_lengths"] = input_lengths.tolist()
            logger.log_batched_dict_as_jsonl(
                log_data, f"train_data_step{total_steps + 1}.jsonl"
            )

            timing_metrics: dict[str, float] = timer.get_timing_metrics(
                reduction_op="sum"
            )

            print("\n📊 Training Results:")
            print(f"  • Loss: {metrics['loss']:.4f}")
            print(f"  • Grad Norm: {metrics['grad_norm']:.4f}")
            print(f"  • Mean Sequence Length: {metrics['mean_seq_length']:.1f}")

            if "total_flops" in train_results:
                total_time = timing_metrics.get("total_step_time", 0)
                total_tflops = (
                    train_results["total_flops"]
                    / timing_metrics["policy_training"]
                    / 1e12
                )
                num_ranks = train_results["num_ranks"]
                print(
                    f"  • Training FLOPS: {total_tflops:.2f} TFLOPS ({total_tflops / num_ranks:.2f} TFLOPS per rank)",
                    flush=True,
                )
                if "theoretical_tflops" in train_results:
                    theoretical_tflops = train_results["theoretical_tflops"]
                    print(
                        f"  • Training Model Floating Point Utilization: {100 * total_tflops / theoretical_tflops:.2f}%",
                        flush=True,
                    )
                    metrics["train_fp_utilization"] = total_tflops / theoretical_tflops

            print("\n⏱️  Timing:", flush=True)
            total_time = timing_metrics.get("total_step_time", 0)

            total_num_gpus = (
                master_config["cluster"]["num_nodes"]
                * master_config["cluster"]["gpus_per_node"]
            )
            metrics.update(
                {
                    "tokens_per_sec_per_gpu": metrics["total_num_tokens"]
                    / total_time
                    / total_num_gpus
                }
            )

            print(f"  • Total step time: {total_time:.2f}s", flush=True)
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)", flush=True)

            timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                metrics["global_valid_toks"] / total_time / total_num_gpus
            )
            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")

            timer.reset()
            current_step += 1
            total_steps += 1
            if should_save_by_timeout:
                print("Timeout reached, stopping training early", flush=True)
                return
            if total_steps >= max_steps:
                print("Max steps reached, stopping training early", flush=True)
                return

        current_epoch += 1
        current_step = 0


# =========================================================================
# Main
# =========================================================================
def main():
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "llama_off_policy_arrow.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: OffPolicyMasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Final config:")
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")

    init_ray()

    # ── Tokenizer ──
    from nemo_rl.algorithms.utils import get_tokenizer

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])

    # ── Configure generation (for eval only) ──
    generation_config = config["policy"].get("generation")
    if generation_config is not None:
        config["policy"]["generation"] = configure_generation_config(
            generation_config, tokenizer
        )

    # ── Training data (arrow) ──
    train_dataset, task_spec = setup_train_data(
        tokenizer, config["data"], config["distillation"]["seed"]
    )

    # ── Core setup ──
    set_seed(config["distillation"]["seed"])

    policy_config = config["policy"]
    teacher_config = config["teacher"]
    distillation_config = config["distillation"]
    data_config = config["data"]
    cluster_config = config["cluster"]

    logger = Logger(config["logger"])
    logger.log_hyperparams(config)

    checkpointer = CheckpointManager(config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    distillation_save_state: Optional[OffPolicyDistillationSaveState] = cast(
        Optional[OffPolicyDistillationSaveState],
        checkpointer.load_training_info(last_checkpoint_path),
    )
    if distillation_save_state is None:
        distillation_save_state = _default_distillation_save_state()

    # ── Dataloader ──
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=distillation_config["num_prompts_per_step"],
        shuffle=data_config.get("shuffle", True),
        collate_fn=rl_collate_fn,
        drop_last=True,
    )
    if last_checkpoint_path:
        train_dataloader.load_state_dict(
            torch.load(os.path.join(last_checkpoint_path, "train_dataloader.pt"))
        )

    has_generation = generation_config is not None
    max_colocated = 4 if has_generation else 3

    # ── Cluster ──
    print("\n▶ Setting up compute cluster...")
    cluster = RayVirtualCluster(
        name="off_policy_distillation_eval_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=max_colocated,
    )
    print(
        f"  ✓ Cluster: {cluster_config['num_nodes']} nodes, max_colocated={max_colocated}"
    )

    # ── Vocab check ──
    if not bool(os.getenv("NRL_SKIP_DISTILLATION_TOKENIZER_CHECK", False)):
        check_vocab_equality(
            tokenizer, policy_config["model_name"], teacher_config["model_name"]
        )

    # ── Teacher Policy ──
    print("\n▶ Setting up teacher policy...")
    if teacher_config.get("megatron_cfg", {}).get("enabled", False):
        total_train_iters = min(
            distillation_config["max_num_steps"],
            distillation_config["max_num_epochs"] * len(train_dataloader),
        )
        teacher_config["megatron_cfg"]["train_iters"] = total_train_iters

    teacher_policy = Policy(
        name_prefix="teacher",
        cluster=cluster,
        config=teacher_config,
        tokenizer=tokenizer,
        weights_path=None,
        optimizer_path=None,
        init_optimizer=False,
        init_reference_model=False,
    )
    teacher_policy.offload_after_refit()

    # ── Student Policy ──
    print("\n▶ Setting up student policy...")
    weights_path = None
    optimizer_path = None
    if last_checkpoint_path:
        weights_path = Path(last_checkpoint_path) / "policy" / "weights"
        optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"

    if policy_config.get("megatron_cfg", {}).get("enabled", False):
        total_train_iters = min(
            distillation_config["max_num_steps"],
            distillation_config["max_num_epochs"] * len(train_dataloader),
        )
        policy_config["megatron_cfg"]["train_iters"] = total_train_iters

    student_policy = Policy(
        name_prefix="student",
        cluster=cluster,
        config=policy_config,
        tokenizer=tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=False,
    )

    loss_fn = DistillationLossFn(config["loss_fn"])

    # ── vLLM Generation (colocated, for eval only) ──
    generation: Optional[GenerationInterface] = None
    if has_generation:
        print("\n▶ Setting up vLLM generation (colocated, for eval)...")
        gen_cfg = config["policy"]["generation"]
        gen_cfg["model_name"] = policy_config["model_name"]
        if "vllm_cfg" in gen_cfg:
            gen_cfg["vllm_cfg"]["hf_overrides"] = policy_config.get(
                "hf_config_overrides", {}
            )

        generation = VllmGeneration(
            cluster=cluster, config=cast(VllmConfig, gen_cfg)
        )
        generation.finish_generation()

        state_dict_info = student_policy.prepare_refit_info()
        generation.prepare_refit_info(state_dict_info)
        print(f"  ✓ vLLM generation ready (model={policy_config['model_name']})")

    # ── Eval datasets + environments ──
    eval_dataloaders: Optional[dict[str, StatefulDataLoader]] = None
    eval_envs: Optional[dict[str, dict[str, EnvironmentInterface]]] = None

    eval_config = config.get("eval")
    if eval_config and has_generation:
        eval_dataloaders, eval_envs = setup_eval_data(
            tokenizer,
            eval_config,
            max_seq_length=policy_config["max_total_sequence_length"],
        )

    print("\n" + "=" * 60)
    print(" " * 10 + "OFF-POLICY DISTILLATION + EVAL SETUP COMPLETE")
    print("=" * 60 + "\n")

    # ── Train ──
    off_policy_distillation_train_with_eval(
        student_policy=student_policy,
        teacher_policy=teacher_policy,
        dataloader=train_dataloader,
        tokenizer=tokenizer,
        loss_fn=loss_fn,
        master_config=config,
        logger=logger,
        checkpointer=checkpointer,
        distillation_save_state=distillation_save_state,
        generation=generation,
        eval_dataloaders=eval_dataloaders,
        eval_envs=eval_envs,
    )


if __name__ == "__main__":
    main()
