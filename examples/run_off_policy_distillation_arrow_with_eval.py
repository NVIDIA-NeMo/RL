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
import sys
import pprint

# Force unbuffered stdout/stderr so logs appear immediately in SLURM output files
os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, cast

import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_rl.algorithms.loss.loss_functions import (
    CrossTokenizerDistillationLossFn,
    DistillationLossFn,
)
from nemo_rl.algorithms.loss.interfaces import LossInputType
from nemo_rl.algorithms.off_policy_distillation import (
    OffPolicyDistillationSaveState,
    OffPolicyMasterConfig,
    _default_distillation_save_state,
    check_vocab_equality,
    off_policy_distillation_train,
)
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset, load_eval_dataset
from nemo_rl.data.datasets.utils import load_dataset_from_path
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


class DistillationLossFnCompat(DistillationLossFn):
    """Runner-local compatibility layer for distillation loss input naming."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.input_type = LossInputType.LOGIT

    def __call__(self, *args, **kwargs):
        # Keep positional calling convention untouched:
        #   loss_fn(logits, data, global_valid_seqs, global_valid_toks, ...)
        if args:
            return super().__call__(*args, **kwargs)

        # Also support keyword-based calling convention with `logits=...`.
        next_token_logits = kwargs.pop("next_token_logits", None)
        if next_token_logits is None and "logits" in kwargs:
            next_token_logits = kwargs.pop("logits")
        return super().__call__(next_token_logits=next_token_logits, **kwargs)


class CrossTokenizerDistillationLossFnCompat(CrossTokenizerDistillationLossFn):
    """Runner-local compatibility layer for cross-tokenizer loss input naming."""

    def __init__(self, cfg, token_aligner):
        super().__init__(cfg, token_aligner)
        self.input_type = LossInputType.LOGIT

    def __call__(self, *args, **kwargs):
        if args:
            return super().__call__(*args, **kwargs)

        next_token_logits = kwargs.pop("next_token_logits", None)
        if next_token_logits is None and "logits" in kwargs:
            next_token_logits = kwargs.pop("logits")
        return super().__call__(next_token_logits=next_token_logits, **kwargs)


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


def _kd_preprocessor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Knowledge-distillation preprocessor: raw text, no chat template, loss on all tokens.

    Both student and teacher tokenize the same raw text, matching the
    train_distillation_ddp.py pipeline. The raw text is stored in
    extra_env_info so the teacher can tokenize it directly.
    """
    raw_text = "\n".join(
        msg["content"]
        for msg in datum_dict["messages"]
        if isinstance(msg.get("content"), str)
    )

    token_ids = tokenizer(
        raw_text,
        return_tensors="pt",
        add_special_tokens=True,
        max_length=max_seq_length,
        truncation=True,
    )["input_ids"][0]

    length = len(token_ids)
    loss_multiplier = 1.0
    if length > max_seq_length:
        loss_multiplier = 0.0

    message_log = [
        {
            "role": "assistant",
            "content": raw_text,
            "token_ids": token_ids,
            "token_loss_mask": torch.ones_like(token_ids),
        }
    ]

    return {
        "message_log": message_log,
        "length": length,
        "extra_env_info": {"raw_text": raw_text},
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }


def setup_train_data(tokenizer: AutoTokenizer, data_config: DataConfig, seed: int):
    from nemo_rl.data.datasets import load_response_dataset

    if "train" not in data_config:
        raise ValueError(
            "The dataset config structure is updated. Please use data.train/default style."
        )

    train_cfg = data_config["train"]
    if isinstance(train_cfg, list):
        if len(train_cfg) != 1:
            raise ValueError(
                "Off-policy distillation currently supports exactly one train dataset config."
            )
        train_cfg = train_cfg[0]
    train_data_config: dict[str, Any] = dict(train_cfg)
    default_cfg = data_config.get("default")
    if isinstance(default_cfg, dict):
        merged_cfg = dict(default_cfg)
        merged_cfg.update(train_data_config)
        train_data_config = merged_cfg
    train_data_config["max_input_seq_length"] = data_config["max_input_seq_length"]

    print("\n▶ Setting up training data...")
    arrow_files = train_data_config.get("arrow_files")
    if arrow_files:
        data = load_response_dataset(train_data_config, seed)
        train_dataset_raw = data.formatted_ds["train"]
        task_spec = data.task_spec
        print(f"  ✓ Using Arrow dataset input: {arrow_files}")
    else:
        # Fallback for user-provided non-arrow path or HF dataset id.
        dataset_path = train_data_config.get(
            "dataset_path", train_data_config.get("hf_dataset_name", "allenai/c4")
        )
        hf_dataset_subset_raw = train_data_config.get("hf_dataset_subset")
        hf_dataset_subset = (
            None
            if hf_dataset_subset_raw in (None, "", "null")
            else hf_dataset_subset_raw
        )
        hf_split = train_data_config.get("hf_split", "train")
        text_key = train_data_config.get("text_key", "text")

        print(
            "  ↪ No data.arrow_files provided. "
            f"Loading dataset_path='{dataset_path}', subset='{hf_dataset_subset}', split='{hf_split}'"
        )
        hf_dataset = load_dataset_from_path(
            dataset_path, data_subset=hf_dataset_subset, data_split=hf_split
        )
        if text_key not in hf_dataset.column_names:
            raise ValueError(
                f"text_key='{text_key}' not found in HF dataset columns: {hf_dataset.column_names}"
            )

        def _to_messages(entry: dict[str, Any]) -> dict[str, Any]:
            text = entry[text_key]
            if not isinstance(text, str):
                text = str(text)
            return {
                "messages": [{"role": "assistant", "content": text}],
                "task_name": "off_policy_distillation",
            }

        train_dataset_raw = hf_dataset.map(
            _to_messages, remove_columns=hf_dataset.column_names
        )
        task_spec = TaskDataSpec(
            task_name="off_policy_distillation",
            prompt_file=train_data_config.get("prompt_file"),
            system_prompt_file=train_data_config.get("system_prompt_file"),
        )

    train_dataset = AllTaskProcessedDataset(
        train_dataset_raw,
        tokenizer,
        task_spec,
        _kd_preprocessor,
        max_seq_length=train_data_config["max_input_seq_length"],
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
            "num_few_shot": bench_cfg.get("num_few_shot", 0),
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
# Eval hook factory for generation-based evaluation (MATH/MMLU)
# =========================================================================
def make_gen_eval_hook(generation, eval_dataloaders, eval_envs,
                       eval_config, master_config, tokenizer, colocated_inference):
    """Create a closure that wraps gen_validate for use as an eval_hook callback.

    The returned function manages vLLM weight refitting and generation lifecycle
    so the shared training loop in off_policy_distillation.py doesn't need to
    know about generation-based evaluation details.
    """
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

    # ── Cross-tokenizer setup ──
    token_aligner_cfg = config.get("token_aligner", {})
    cross_tokenizer_enabled = token_aligner_cfg.get("enabled", False)
    token_aligner = None
    teacher_tokenizer = None

    if cross_tokenizer_enabled:
        from nemo_rl.algorithms.x_token import TokenAligner

        print("\n▶ Setting up cross-tokenizer distillation (TokenAligner)...")
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_config["model_name"])
        if teacher_tokenizer.pad_token is None:
            teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

        token_aligner = TokenAligner(
            teacher_tokenizer_name=teacher_config["model_name"],
            student_tokenizer_name=policy_config["model_name"],
            max_comb_len=token_aligner_cfg.get("max_comb_len", 4),
            projection_matrix_multiplier=token_aligner_cfg.get(
                "projection_matrix_multiplier", 1.0
            ),
        )
        token_aligner._load_logits_projection_map(
            file_path=token_aligner_cfg["projection_matrix_path"],
            use_sparse_format=token_aligner_cfg.get("use_sparse_format", True),
            learnable=token_aligner_cfg.get("learnable", False),
            device="cpu",
        )
        if token_aligner_cfg.get("project_teacher_to_student", False):
            token_aligner.create_reverse_projection_matrix(device="cpu")

        token_aligner.precompute_canonical_maps()

        print(f"  ✓ TokenAligner initialized ({policy_config['model_name']} → {teacher_config['model_name']})")
    else:
        # ── Vocab check (same-tokenizer mode only) ──
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
        tokenizer=teacher_tokenizer if cross_tokenizer_enabled else tokenizer,
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

    if cross_tokenizer_enabled:
        loss_fn = CrossTokenizerDistillationLossFnCompat(config["loss_fn"], token_aligner)
    else:
        loss_fn = DistillationLossFnCompat(config["loss_fn"])

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

    # ── Build eval hook ──
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

    # ── Train ──
    off_policy_distillation_train(
        student_policy=student_policy,
        teacher_policy=teacher_policy,
        dataloader=train_dataloader,
        val_dataloader=None,
        tokenizer=tokenizer,
        loss_fn=loss_fn,
        logger=logger,
        checkpointer=checkpointer,
        distillation_save_state=distillation_save_state,
        master_config=config,
        eval_hook=eval_hook,
        eval_hook_period=eval_hook_period,
        eval_hook_at_start=eval_hook_at_start,
        token_aligner=token_aligner,
        teacher_tokenizer=teacher_tokenizer,
    )


if __name__ == "__main__":
    main()
