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

"""SFT training on arrow data with inline generation-based MATH/MMLU evaluation.

This script extends the standard SFT flow with periodic generation-based
evaluation using vLLM (colocated).  It does NOT modify sft.py or run_sft.py;
instead it builds its own setup / training loop on top of the existing SFT
primitives.

Usage:
    uv run examples/run_sft_arrow_with_eval.py \
        --config examples/configs/llama_sft_arrow.yaml
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
from nemo_rl.algorithms.loss.loss_functions import NLLLoss
from nemo_rl.algorithms.sft import SFTSaveState, _default_sft_save_state
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset, load_eval_dataset
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.data.llm_message_utils import (
    add_loss_mask_to_message_log,
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

OmegaConf.register_new_resolver("mul", lambda a, b: a * b, replace=True)
OmegaConf.register_new_resolver("max", lambda a, b: max(a, b), replace=True)


# =========================================================================
# CLI
# =========================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="SFT on arrow data with inline MATH/MMLU eval"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    args, overrides = parser.parse_known_args()
    return args, overrides


# =========================================================================
# Arrow-text SFT data (reused from run_sft.py pattern)
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
) -> DatumSpec:
    from nemo_rl.data.llm_message_utils import get_formatted_message_log

    message_log = get_formatted_message_log(
        datum_dict["messages"],
        tokenizer,
        task_data_spec,
        add_bos_token=add_bos,
        add_eos_token=add_eos,
        add_generation_prompt=add_generation_prompt,
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


def setup_train_data(
    tokenizer: AutoTokenizer, data_config: DataConfig, seed: int
):
    from nemo_rl.data.datasets import load_response_dataset

    print("\n▶ Setting up training data...")
    data = load_response_dataset(data_config, seed)
    train_dataset_raw = data.formatted_ds["train"]
    sft_task_spec = data.task_spec

    train_dataset = AllTaskProcessedDataset(
        train_dataset_raw,
        tokenizer,
        sft_task_spec,
        partial(
            _sft_preprocessor,
            add_bos=data_config.get("add_bos", True),
            add_eos=data_config.get("add_eos", True),
            add_generation_prompt=data_config.get("add_generation_prompt", False),
        ),
        max_seq_length=data_config["max_input_seq_length"],
    )
    print(f"  ✓ Training dataset loaded with {len(train_dataset)} samples")
    return train_dataset, sft_task_spec


# =========================================================================
# Eval data + environments (MATH, MMLU, etc.)
# =========================================================================
def setup_eval_data(
    tokenizer: AutoTokenizer,
    eval_config: dict[str, Any],
    max_seq_length: int,
) -> tuple[
    dict[str, StatefulDataLoader],
    dict[str, dict[str, EnvironmentInterface]],
]:
    """Load eval benchmark datasets and create scoring environments.

    Returns:
        eval_dataloaders: {benchmark_name: StatefulDataLoader}
        eval_envs:        {benchmark_name: {task_name: EnvironmentInterface}}
    """
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
# Generation-based validation (ported from distillation.py)
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
    """Run generation-based evaluation on all configured benchmarks."""
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
# Training loop (based on sft_train, with eval hooks)
# =========================================================================
def sft_train_with_eval(
    policy: Policy,
    train_dataloader: StatefulDataLoader,
    tokenizer: AutoTokenizer,
    loss_fn: NLLLoss,
    master_config: dict[str, Any],
    logger: Logger,
    sft_task_spec: TaskDataSpec,
    checkpointer: CheckpointManager,
    sft_save_state: Optional[SFTSaveState],
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

    if sft_save_state is None:
        sft_save_state = _default_sft_save_state()
        current_epoch = 0
        current_step = 0
        total_steps = 0
        total_valid_tokens = 0
    else:
        current_epoch = sft_save_state["epoch"]
        current_step = sft_save_state["step"]
        total_steps = sft_save_state["total_steps"]
        total_valid_tokens = sft_save_state.get("total_valid_tokens", 0)

    sft_config = master_config["sft"]
    max_num_epochs = sft_config["max_num_epochs"]
    max_num_steps = sft_config["max_num_steps"]

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
    need_refit = True
    generation_stale = True

    # ── optional eval at start ──
    if has_eval and eval_at_start and total_steps == 0:
        print("\n🔍 Running initial evaluation...", flush=True)
        refit_policy_generation(policy, generation, colocated_inference)
        generation_stale = False
        val_metrics, val_timings = gen_validate(
            generation, eval_dataloaders, eval_envs, eval_config, master_config, step=0,
            tokenizer=tokenizer,
        )
        generation.finish_generation()
        logger.log_metrics(val_metrics, 0, prefix="validation")
        logger.log_metrics(val_timings, 0, prefix="timing/validation")

    policy.prepare_for_training()

    while current_epoch < max_num_epochs and total_steps < max_num_steps:
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")

        for batch in train_dataloader:
            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(train_dataloader), max_num_steps)} {'=' * 25}"
            )
            maybe_gpu_profile_step(policy, total_steps + 1)
            val_metrics = None

            with timer.time("total_step_time"):
                # ── data prep ──
                print("▶ Preparing batch...")
                with timer.time("data_processing"):
                    add_loss_mask_to_message_log(
                        batch["message_log"], roles_to_train_on=["assistant"]
                    )
                    cat_and_padded, input_lengths = batched_message_log_to_flat_message(
                        batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config["policy"][
                            "make_sequence_length_divisible_by"
                        ],
                    )
                    train_data: BatchedDataDict = BatchedDataDict(
                        {
                            "input_ids": cat_and_padded["token_ids"],
                            "input_lengths": input_lengths,
                            "token_mask": cat_and_padded["token_loss_mask"],
                            "sample_mask": batch["loss_multiplier"],
                        }
                    )
                    train_data.update(
                        cat_and_padded.get_multimodal_dict(as_tensors=False)
                    )

                # ── NaN debug: check input data before training ──
                if current_step < 3:
                    _ids = train_data["input_ids"]
                    _mask = train_data["token_mask"]
                    _smask = train_data["sample_mask"]
                    print(
                        f"  [NaN debug] input_ids shape={_ids.shape}, "
                        f"min={_ids.min().item()}, max={_ids.max().item()}, "
                        f"token_mask sum={_mask.sum().item():.0f}, "
                        f"sample_mask sum={_smask.sum().item():.0f}/{_smask.numel()}, "
                        f"input_lengths range=[{train_data['input_lengths'].min().item()}, "
                        f"{train_data['input_lengths'].max().item()}]",
                        flush=True,
                    )

                # ── train step ──
                print("▶ Taking a training step...")
                with timer.time("policy_training"):
                    train_results = policy.train(train_data, loss_fn)

                # ── NaN debug: check loss ──
                if current_step < 3:
                    import numpy as _np
                    _loss_val = train_results["loss"].numpy()
                    if _np.isnan(_loss_val).any():
                        print(
                            f"  [NaN debug] Loss is NaN at step {total_steps + 1}! "
                            f"grad_norm={train_results['grad_norm'].numpy()}, "
                            f"all_mb_metrics={train_results.get('all_mb_metrics', {})}",
                            flush=True,
                        )

                generation_stale = True

                is_last_step = total_steps + 1 >= max_num_steps or (
                    current_epoch + 1 == max_num_epochs
                    and current_step + 1 == len(train_dataloader)
                )

                # ── generation-based eval ──
                if has_eval and eval_period > 0 and (total_steps + 1) % eval_period == 0:
                    print(
                        f"\n🔍 Running generation-based eval at step {total_steps + 1}...",
                        flush=True,
                    )
                    with timer.time("gen_eval"):
                        if generation_stale:
                            refit_policy_generation(
                                policy, generation, colocated_inference
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
                    policy.prepare_for_training()

                # ── metrics ──
                metrics = {
                    "loss": train_results["loss"].numpy(),
                    "grad_norm": train_results["grad_norm"].numpy(),
                }
                if "moe_metrics" in train_results:
                    metrics.update(
                        {f"moe/{k}": v for k, v in train_results["moe_metrics"].items()}
                    )
                metrics.update(train_results["all_mb_metrics"])
                for k, v in metrics.items():
                    if k in {"lr", "wd", "global_valid_seqs", "global_valid_toks"}:
                        metrics[k] = np.mean(v).item()
                    else:
                        metrics[k] = np.sum(v).item()
                total_valid_tokens += metrics["global_valid_toks"]

                # ── checkpointing ──
                sft_save_state["consumed_samples"] += master_config["policy"][
                    "train_global_batch_size"
                ]
                timeout.mark_iteration()
                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1)
                    % master_config["checkpointing"]["save_period"]
                    == 0
                )
                should_save_by_timeout = timeout.check_save()

                if master_config["checkpointing"]["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    sft_save_state["step"] = (current_step + 1) % len(train_dataloader)
                    sft_save_state["total_steps"] = total_steps + 1
                    sft_save_state["epoch"] = current_epoch
                    sft_save_state["total_valid_tokens"] = total_valid_tokens

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
                            if full_metric_name in sft_save_state:
                                del sft_save_state[full_metric_name]
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            sft_save_state[full_metric_name] = metrics_source[
                                metric_name
                            ]

                    with timer.time("checkpointing"):
                        print(f"Saving checkpoint for step {total_steps + 1}...")
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, sft_save_state, master_config
                        )
                        policy.save_checkpoint(
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
                            train_dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            # ── logging ──
            timing_metrics = timer.get_timing_metrics(reduction_op="sum")

            print("\n📊 Training Results:")
            print(f"  • Loss: {float(metrics['loss']):.4f}")
            if "total_flops" in train_results:
                total_tflops = (
                    train_results["total_flops"]
                    / timing_metrics["policy_training"]
                    / 1e12
                )
                num_ranks = train_results["num_ranks"]
                print(
                    f"  • Training FLOPS: {total_tflops:.2f} TFLOPS ({total_tflops / num_ranks:.2f} TFLOPS per rank)"
                )
                if "theoretical_tflops" in train_results:
                    theoretical_tflops = train_results["theoretical_tflops"]
                    print(
                        f"  • Training Model Floating Point Utilization: {100 * total_tflops / theoretical_tflops:.2f}%"
                    )
                    metrics["train_fp_utilization"] = total_tflops / theoretical_tflops

            print("\n⏱️  Timing:")
            total_time = timing_metrics.get("total_step_time", 0)
            print(f"  • Total step time: {total_time:.2f}s")
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

            total_num_gpus = (
                master_config["cluster"]["num_nodes"]
                * master_config["cluster"]["gpus_per_node"]
            )
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
            if total_steps >= max_num_steps:
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
            os.path.dirname(__file__), "configs", "llama_sft_arrow.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config = OmegaConf.to_container(config, resolve=True)
    print("Final config:")
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")

    init_ray()

    # ── Tokenizer ──
    from nemo_rl.algorithms.utils import get_tokenizer

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])

    # ── Configure generation ──
    generation_config = config["policy"].get("generation")
    if generation_config is not None:
        config["policy"]["generation"] = configure_generation_config(
            generation_config, tokenizer
        )

    # ── Training data (arrow) ──
    train_dataset, sft_task_spec = setup_train_data(
        tokenizer, config["data"], config["sft"]["seed"]
    )

    # ── Core SFT setup (cluster, policy, dataloader, loss, logger, checkpointer) ──
    set_seed(config["sft"]["seed"])

    policy_config = config["policy"]
    data_config = config["data"]
    logger_config = config["logger"]
    cluster_config = config["cluster"]
    sft_config = config["sft"]

    logger = Logger(logger_config)
    logger.log_hyperparams(config)

    checkpointer = CheckpointManager(config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    sft_save_state: Optional[SFTSaveState] = cast(
        Optional[SFTSaveState], checkpointer.load_training_info(last_checkpoint_path)
    )

    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=policy_config["train_global_batch_size"],
        shuffle=data_config.get("shuffle", True),
        collate_fn=rl_collate_fn,
        drop_last=True,
        num_workers=data_config.get("num_workers", 1),
    )
    if last_checkpoint_path is not None:
        train_dataloader.load_state_dict(
            torch.load(os.path.join(last_checkpoint_path, "train_dataloader.pt"))
        )

    has_generation = generation_config is not None
    max_colocated = 3 if has_generation else 1

    print("\n▶ Setting up compute cluster...")
    cluster = RayVirtualCluster(
        name="sft_eval_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=max_colocated,
    )
    print(f"  ✓ Cluster: {cluster_config['num_nodes']} nodes, max_colocated={max_colocated}")

    # ── Policy ──
    print("\n▶ Setting up model...")
    if policy_config.get("megatron_cfg", {}).get("enabled", False):
        total_train_iters = min(
            sft_config["max_num_steps"],
            sft_config["max_num_epochs"] * len(train_dataloader),
        )
        policy_config["megatron_cfg"]["train_iters"] = total_train_iters

    processor = None
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        processor = tokenizer
        tokenizer = processor.tokenizer

    policy = Policy(
        cluster=cluster,
        config=policy_config,
        tokenizer=tokenizer,
        processor=processor,
        weights_path=Path(last_checkpoint_path) / "policy" / "weights"
        if last_checkpoint_path
        else None,
        optimizer_path=Path(last_checkpoint_path) / "policy" / "optimizer"
        if last_checkpoint_path
        else None,
        init_optimizer=True,
        init_reference_model=False,
    )
    policy.print_node_ip_and_gpu_id()
    loss_fn = NLLLoss()
    print("  ✓ Model initialized")

    # ── vLLM Generation (colocated) ──
    generation: Optional[GenerationInterface] = None
    if has_generation:
        print("\n▶ Setting up vLLM generation (colocated)...")
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

        state_dict_info = policy.prepare_refit_info()
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
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    # ── Train ──
    sft_train_with_eval(
        policy=policy,
        train_dataloader=train_dataloader,
        tokenizer=tokenizer,
        loss_fn=loss_fn,
        master_config=config,
        logger=logger,
        sft_task_spec=sft_task_spec,
        checkpointer=checkpointer,
        sft_save_state=sft_save_state,
        generation=generation,
        eval_dataloaders=eval_dataloaders,
        eval_envs=eval_envs,
    )


if __name__ == "__main__":
    main()
