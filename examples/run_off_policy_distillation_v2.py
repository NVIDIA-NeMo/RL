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

"""Off-policy distillation v2: single dual-model Policy with GPU-local IPC.

Uses DTensorDistillationWorker which holds both teacher and student models
in the same Ray actor. Teacher logprobs stay on GPU via IPC buffers —
no Ray object store transfer.

Usage:
    uv run examples/run_off_policy_distillation_v2.py \
        --config examples/configs/llama_off_policy_arrow.yaml
"""

import argparse
import os
import pprint
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, cast

import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_rl.algorithms.loss_functions import DistillationLossFn
from nemo_rl.algorithms.off_policy_distillation import (
    OffPolicyDistillationSaveState,
    OffPolicyMasterConfig,
    _default_distillation_save_state,
    check_vocab_equality,
)
from nemo_rl.algorithms.off_policy_distillation_v2 import (
    off_policy_distillation_train_v2,
)
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset, load_eval_dataset
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.data.llm_message_utils import get_keys_from_message_log
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.math_environment import MathEnvironment
from nemo_rl.experience.rollouts import run_multi_turn_rollout
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import Logger, get_next_experiment_dir, print_message_log_samples
from nemo_rl.utils.timer import Timer

import ray

OmegaConf.register_new_resolver("mul", lambda a, b: a * b, replace=True)
OmegaConf.register_new_resolver("max", lambda a, b: max(a, b), replace=True)

DISTILLATION_WORKER_CLS = "nemo_rl.models.policy.workers.dtensor_distillation_worker.DTensorDistillationWorker"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Off-policy distillation v2 (dual-model worker, GPU-local IPC)"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    args, overrides = parser.parse_known_args()
    return args, overrides


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

    print("\n Setting up training data...")
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
    print(f"  Training dataset loaded with {len(train_dataset)} samples")
    return train_dataset, task_spec


def setup_eval_data(
    tokenizer: AutoTokenizer,
    eval_config: dict[str, Any],
    max_seq_length: int,
):
    print("\n Setting up evaluation benchmarks...")
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
        print(f"  {bench_name}: {len(dataset)} samples, env={dataset_name}")

    return eval_dataloaders, eval_envs


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
            print(f"\n Evaluating {bench_name} at step {step}...", flush=True)
            total_rewards = []
            total_lengths = []
            all_message_logs = []

            for batch_idx, val_batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break

                val_batch, gen_metrics = run_multi_turn_rollout(
                    generation, val_batch, tokenizer,
                    eval_envs[bench_name],
                    max_seq_len=max_seq_len,
                    max_rollout_turns=max_rollout_turns,
                    greedy=True,
                )

                rewards = val_batch["total_reward"]
                total_rewards.extend(rewards.tolist())
                total_lengths.append(gen_metrics["mean_gen_tokens_per_sample"])

                to_env = [
                    get_keys_from_message_log(val_batch["message_log"][i], ["role", "content"])
                    for i in range(len(val_batch["message_log"]))
                ]
                all_message_logs.extend(to_env)

            accuracy = sum(total_rewards) / len(total_rewards) if total_rewards else 0
            avg_length = sum(total_lengths) / len(total_lengths) if total_lengths else 0

            all_val_metrics[f"{bench_name}_accuracy"] = accuracy
            all_val_metrics[f"{bench_name}_avg_length"] = avg_length

            print(f"\n {bench_name} Results:")
            print(f"    Accuracy: {accuracy:.4f}")
            print(f"    Avg response length: {avg_length:.1f} tokens")
            print(f"    Samples processed: {len(total_rewards)}", flush=True)

            try:
                num_to_print = master_config["logger"].get("num_val_samples_to_print", 3)
                print_message_log_samples(
                    all_message_logs, total_rewards,
                    num_samples=min(num_to_print, len(all_message_logs)),
                    step=step,
                )
            except Exception as e:
                print(f"  Error displaying samples: {e}", flush=True)

    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    eval_time = timing_metrics.get("total_eval_time", 0)
    print(f"\n  Total eval time: {eval_time:.2f}s", flush=True)
    timer.reset()

    return all_val_metrics, timing_metrics


def make_gen_eval_hook(generation, eval_dataloaders, eval_envs,
                       eval_config, master_config, tokenizer, colocated_inference):
    generation_stale = True

    def hook(step, student_policy, teacher_policy, logger):
        nonlocal generation_stale
        from nemo_rl.algorithms.grpo import refit_policy_generation

        if generation_stale:
            refit_policy_generation(student_policy, generation, colocated_inference)
            generation_stale = False

        val_metrics, val_timings = gen_validate(
            generation, eval_dataloaders, eval_envs,
            eval_config, master_config, step=step, tokenizer=tokenizer,
        )
        generation.finish_generation()
        logger.log_metrics(val_timings, step, prefix="timing/validation")
        logger.log_metrics(val_metrics, step, prefix="validation")
        generation_stale = True
        return val_metrics

    return hook


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
    print(f"Using log directory: {config['logger']['log_dir']}")

    init_ray()

    from nemo_rl.algorithms.utils import get_tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])

    generation_config = config["policy"].get("generation")
    if generation_config is not None:
        config["policy"]["generation"] = configure_generation_config(
            generation_config, tokenizer
        )

    train_dataset, task_spec = setup_train_data(
        tokenizer, config["data"], config["distillation"]["seed"]
    )

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
    max_colocated = 3 if has_generation else 2

    print("\n Setting up compute cluster...")
    cluster = RayVirtualCluster(
        name="off_policy_distillation_v2_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=max_colocated,
    )
    print(f"  Cluster: {cluster_config['num_nodes']} nodes, max_colocated={max_colocated}")

    if not bool(os.getenv("NRL_SKIP_DISTILLATION_TOKENIZER_CHECK", False)):
        check_vocab_equality(
            tokenizer, policy_config["model_name"], teacher_config["model_name"]
        )

    # Single Policy with both student + teacher models in each worker
    print("\n Setting up dual-model policy (student + teacher)...")
    weights_path = None
    optimizer_path = None
    if last_checkpoint_path:
        weights_path = Path(last_checkpoint_path) / "policy" / "weights"
        optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"

    policy = Policy(
        name_prefix="distillation",
        cluster=cluster,
        config=policy_config,
        tokenizer=tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=False,
        worker_builder_cls_override=DISTILLATION_WORKER_CLS,
        extra_worker_kwargs={"teacher_config": teacher_config},
    )

    loss_fn = DistillationLossFn(config["loss_fn"])

    # vLLM Generation (colocated, for eval only)
    generation: Optional[GenerationInterface] = None
    if has_generation:
        print("\n Setting up vLLM generation (colocated, for eval)...")
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
        print(f"  vLLM generation ready (model={policy_config['model_name']})")

    eval_dataloaders = None
    eval_envs = None
    eval_config = config.get("eval")
    if eval_config and has_generation:
        eval_dataloaders, eval_envs = setup_eval_data(
            tokenizer, eval_config,
            max_seq_length=policy_config["max_total_sequence_length"],
        )

    print("\n" + "=" * 60)
    print(" " * 10 + "OFF-POLICY DISTILLATION V2 SETUP COMPLETE")
    print("=" * 60 + "\n")

    # Build eval hook
    eval_hook = None
    eval_hook_period = 0
    eval_hook_at_start = False
    if has_generation and eval_config and eval_dataloaders and eval_envs:
        colocated_inference = (
            config["policy"]["generation"]["colocated"]["enabled"]
            if config["policy"].get("generation")
            else True
        )
        eval_hook = make_gen_eval_hook(
            generation, eval_dataloaders, eval_envs,
            eval_config, config, tokenizer, colocated_inference,
        )
        eval_hook_period = eval_config["val_period"]
        eval_hook_at_start = eval_config.get("val_at_start", False)

    # Train
    off_policy_distillation_train_v2(
        policy=policy,
        dataloader=train_dataloader,
        tokenizer=tokenizer,
        loss_fn=loss_fn,
        logger=logger,
        checkpointer=checkpointer,
        distillation_save_state=distillation_save_state,
        master_config=config,
        eval_hook=eval_hook,
        eval_hook_period=eval_hook_period,
        eval_hook_at_start=eval_hook_at_start,
    )


if __name__ == "__main__":
    main()
