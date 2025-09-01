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
import json
import os
import warnings
from pathlib import Path
from typing import Any, Optional, TypedDict, TypeVar, cast

import numpy as np
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.algorithms.loss_functions import (
    ClippedPGLossConfig,
    ClippedPGLossDataDict,
    ClippedPGLossFn,
)
from nemo_rl.algorithms.utils import calculate_grpo_advantages_per_prompt
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, rl_collate_fn
from nemo_rl.data.interfaces import (
    DatumSpec,
)
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    ClusterConfig,
    RayVirtualCluster,
)
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
)
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import (
    GenerationInterface,
)
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import (
    Logger,
    LoggerConfig,
    print_message_log_samples,
)
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import Timer

# ===============================================================================
# Configuration
# ===============================================================================
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


class GRPOConfig(TypedDict):
    num_prompts_per_step: int
    num_generations_per_prompt: int
    max_num_steps: int
    max_rollout_turns: int
    normalize_rewards: bool
    use_leave_one_out_baseline: bool
    val_period: int
    val_batch_size: int
    val_at_start: bool
    max_val_samples: int


class GRPOSaveState(TypedDict):
    step: int
    val_reward: float
    consumed_samples: int


def _default_grpo_save_state() -> GRPOSaveState:
    return {
        "step": 0,
        "val_reward": -99999999.0,
        "consumed_samples": 0,
    }


class GRPOLoggerConfig(LoggerConfig):
    num_val_samples_to_print: int  # number of val samples to print to stdout


class MasterConfig(TypedDict):
    policy: PolicyConfig
    loss_fn: ClippedPGLossConfig
    env: dict[str, Any]
    data: DataConfig
    grpo: GRPOConfig
    logger: GRPOLoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


# ===============================================================================
# Setup & Initialization
# ===============================================================================


def setup(
    master_config: MasterConfig,
    tokenizer: TokenizerType,
    dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
) -> tuple[
    ColocatablePolicyInterface,
    Optional[GenerationInterface],
    tuple[RayVirtualCluster, RayVirtualCluster],
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    ClippedPGLossFn,
    Logger,
    CheckpointManager,
    GRPOSaveState,
    MasterConfig,
]:
    """Main entry point for running GRPO algorithm.

    Returns:
        tuple of policy, cluster, dataloader, tokenizer, loss_fn, math_env, logger, master_config, val_dataloader
    """
    # Extract individual configs for easier access
    policy_config = master_config["policy"]
    generation_config = master_config["policy"]["generation"]
    loss_config = master_config["loss_fn"]
    grpo_config = master_config["grpo"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]

    assert generation_config is not None, (
        "A generation config in the PolicyConfig is required for GRPO"
    )

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    print("\n▶ Print master config...")
    print(json.dumps(master_config, indent=4))

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    grpo_save_state: Optional[GRPOSaveState] = checkpointer.load_training_info(
        last_checkpoint_path
    )
    if grpo_save_state is None:
        grpo_save_state = _default_grpo_save_state()

    # ==========================
    #           Data
    # ==========================
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=grpo_config["num_prompts_per_step"],
        shuffle=False,
        collate_fn=rl_collate_fn,
        drop_last=True,
    )
    if last_checkpoint_path is not None:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        dataloader.load_state_dict(dataloader_state_dict)

    print(f"  ✓ Training dataloader loaded with {len(dataset)} samples")

    # Load validation dataset if provided
    val_dataloader: Optional[StatefulDataLoader] = None
    # If validation is enabled, load the validation dataloader
    if grpo_config["val_period"] > 0 or grpo_config["val_at_start"]:
        assert val_dataset is not None, (
            "Validation dataset is required if validation is enabled"
        )
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=grpo_config["val_batch_size"],
            shuffle=False,
            collate_fn=rl_collate_fn,
        )
        print(f"  ✓ Validation dataloader loaded with {len(val_dataset)} samples")

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    colocated_inference = generation_config["colocated"]["enabled"]

    if colocated_inference:
        cluster = RayVirtualCluster(
            name="grpo_policy_cluster",
            bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
            * cluster_config["num_nodes"],
            use_gpus=True,
            num_gpus_per_node=cluster_config["gpus_per_node"],
            max_colocated_worker_groups=1
            if generation_config["backend"] == "megatron"
            else 2,
        )
        train_cluster = cluster
        inference_cluster = cluster
        print(f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes")

    else:
        assert generation_config["backend"] != "megatron", (
            "Non-colocated inference is not supported for Megatron generation backends. "
            "Please use vLLM backend for generation."
        )

        # train resources will be updated through overall and inference resources below
        train_gpus_per_node = cluster_config["gpus_per_node"]
        train_nodes = cluster_config["num_nodes"]

        inference_resources = generation_config["colocated"]["resources"]
        inference_gpus_per_node = inference_resources["gpus_per_node"]
        inference_nodes = inference_resources["num_nodes"]

        # validate and configure resources
        if cluster_config["num_nodes"] == 1:
            assert inference_gpus_per_node > 0, (
                "policy.generation.colocated.resources.gpus_per_node must be > 0 "
                "when cluster.num_nodes = 1 and inference is non-colocated, "
                f"but got {inference_gpus_per_node}."
            )
            assert inference_nodes is None or inference_nodes == 1, (
                "policy.generation.colocated.resources.num_nodes must be 1 or set to null "
                "when cluster.num_nodes = 1 and inference is non-colocated, "
                f"but got {inference_nodes}."
            )
            inference_nodes = 1
            train_gpus_per_node -= inference_gpus_per_node
        else:
            assert inference_nodes > 0, (
                "policy.generation.colocated.resources.num_nodes must be > 0 "
                "when cluster.num_nodes > 1 and inference is non-colocated, "
                f"but got {inference_nodes}."
            )
            assert (
                inference_gpus_per_node is None
                or inference_gpus_per_node == cluster_config["gpus_per_node"]
            ), (
                "policy.generation.colocated.resources.gpus_per_node must be equal to cluster.gpus_per_node or set to null "
                "when cluster.num_nodes > 1 and inference is non-colocated, "
                f"but got {inference_gpus_per_node}."
            )
            inference_gpus_per_node = cluster_config["gpus_per_node"]
            train_nodes -= inference_nodes

        # initialize train cluster
        train_cluster = RayVirtualCluster(
            name="grpo_train_cluster",
            bundle_ct_per_node_list=[train_gpus_per_node] * train_nodes,
            use_gpus=True,
            num_gpus_per_node=train_gpus_per_node,
            max_colocated_worker_groups=1,
        )
        print(
            f"  ✓ Ray train cluster initialized with {train_nodes} nodes with {train_gpus_per_node} GPUs per node"
        )

        # initialize inference cluster
        inference_cluster = RayVirtualCluster(
            name="grpo_inference_cluster",
            bundle_ct_per_node_list=[inference_gpus_per_node] * inference_nodes,
            use_gpus=True,
            num_gpus_per_node=inference_gpus_per_node,
            max_colocated_worker_groups=1,
        )
        print(
            f"  ✓ Ray inference cluster initialized with {inference_nodes} nodes with {inference_gpus_per_node} GPUs per node"
        )

    # ==========================
    #   Training and Inference
    # ==========================
    print("\n▶ Setting up model and training...")

    # vllm model loading prefers clean environment, initialize policy_generation before policy (#52 will fix this)
    backend = generation_config["backend"]
    generation_config["model_name"] = policy_config["model_name"]  # Needed for vLLM

    if backend == "megatron":
        policy_generation = None
        print(
            f"  ✓ Using {backend} backend for generation with {policy_config['model_name']}"
        )
    elif backend == "vllm":
        generation_config = cast(VllmConfig, generation_config)
        policy_generation = VllmGeneration(
            cluster=inference_cluster, config=generation_config
        )
        # Worker groups are not initialized until the first call to run something on workergroups.
        # vllm 0.8 fails in initialization if its called in the first training step since it has no clean view of the GPU memory (HF is sharing the same memory).
        policy_generation.finish_generation()
        print(
            f"  ✓ Using vLLM backend for generation with {policy_config['model_name']}"
        )

    if last_checkpoint_path:
        weights_path = Path(last_checkpoint_path) / "policy" / "weights"
        optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"
    else:
        weights_path = None
        optimizer_path = None

    policy = Policy(
        cluster=train_cluster,
        config=policy_config,
        tokenizer=tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
    )

    # if it is not colocated inference, initialize collective communication for update weights
    if not colocated_inference:
        ip, port = train_cluster.get_master_address_and_port()
        print(f"Using ip: {ip}, port: {port} for collective communication")
        # inference cluster + head node of the train cluster
        world_size = inference_nodes * inference_gpus_per_node + 1
        # init collective
        futures_train = policy.init_collective(ip, port, world_size)
        futures_inference = policy_generation.init_collective(ip, port, world_size)  # type: ignore
        # wait for all futures to complete
        ray.get(futures_train + futures_inference)

    # prepare refit info
    state_dict_info = policy.prepare_refit_info()
    policy_generation.prepare_refit_info(state_dict_info)

    loss_fn = ClippedPGLossFn(loss_config)

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        policy,
        policy_generation,
        (train_cluster, inference_cluster),
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_save_state,
        master_config,
    )


# ===============================================================================
# Core Algorithm Functions
# ===============================================================================

PARALLELIZER_PROMPT = """
You are an AI agent working to coordinate a team of agents to solve a problem. Given a problem, your task is to rewrite the problem several times. Your goal is to make the rewritten problems easier to solve but retain the same information as the orignal problem.

You must put the rewrites in the following format: the rewrites should be in-between <rewrite> and </rewrite> tags. Each rewrite should appear on a new line within the xml tags.

Here is an example of the format you must follow:

<rewrite>
{{rewrite_1}}
</rewrite>
<rewrite>
{{rewrite_2}}
</rewrite>

Now, provide the rewrites for this problem:
<problem>
{problem}
</problem>

Please follow these rules in the rewrites:
    1. Make sure EACH REWRITE CAN BE SOLVED INDIVIDUALLY. Make sure ALL RELEVANT information from the original is present in each rewrite. There is no exception.
    2. When solving the rewrite, the person will have NO ACCESS TO THE ORIGINAL QUESTION and TO EACH REWRITE, so each one should be independent.
    3. Make sure the rewrites ARE DIVERSE. They must be able to be solved independently. 

DO NOT SOLVE THE PROBLEM IN THE REWRITES. YOU JUST PROVIDE THE REWRITES.
Please provide the rewrites in the specified format. You should provide 2 rewrites exactly. MAKE SURE TO GIVE ME 2 REWRITES!! Ensure to follow the XML format exactly.
"""

EXECUTOR_PROMPT = """
Please solve the following problem. Make sure to put the answer (and only answer) inside \\boxed{{}}.

<problem>
{problem}
</problem>

"""


def _should_use_async_rollouts(master_config: MasterConfig) -> bool:
    """Determine if async rollouts should be used based on the configuration.

    Returns True if vLLM backend is used with async_engine enabled.
    """
    generation_config = master_config["policy"]["generation"]
    if generation_config is None:
        return False

    backend = generation_config.get("backend", "")
    if backend != "vllm":
        return False

    vllm_cfg = generation_config.get("vllm_cfg", {})
    return vllm_cfg.get("async_engine", False)


def refit_policy_generation(
    policy: ColocatablePolicyInterface,
    policy_generation: GenerationInterface,
    colocated_inference: bool,
    _refit_buffer_size_gb: Optional[int] = None,
) -> None:
    """Refit the policy generation interface with the latest policy weights.

    Args:
        policy: The policy to provide weights to the inference engine.
        policy_generation: The inference engine to refit.
        _refit_buffer_size_gb: The size of the buffer to use for refitting.
            If it is None, the buffer size will be computed by the remaining memory.
            This parameter is primarily used for testing.
    """
    if colocated_inference:
        policy.offload_before_refit()
        policy_generation.prepare_for_generation(tags=["weights"])

    # update weights
    update_success = False
    if colocated_inference:
        # get model param keys, which is grouped by size
        grouped_param_keys = policy.prepare_weights_for_ipc(
            _refit_buffer_size_gb=_refit_buffer_size_gb
        )
        total_num_keys = sum(len(k) for k in grouped_param_keys)
        print(
            f"[Refit] Split {total_num_keys} keys into {len(grouped_param_keys)} groups"
        )
        # do update
        for keys in grouped_param_keys:
            ipc_handles = policy.get_weights_ipc_handles(keys)
            update_success = policy_generation.update_weights_from_ipc_handles(
                ipc_handles
            )
            if not update_success:
                break
    else:
        # update weights through nccl
        futures_train = policy.broadcast_weights_for_collective()
        futures_inference = policy_generation.update_weights_from_collective()
        # wait for all futures to complete
        ray.get(futures_train)
        results = ray.get(futures_inference)
        update_success = all(result for result in results if result is not None)

    # check if update is successful
    if not update_success:
        error_tag = "cuda-ipc" if colocated_inference else "nccl"
        error_message = (
            "❌ Error: Updating weights for the generation policy failed during refit.\n"
            f"This often indicates an issue with {error_tag} or "
            "a problem within the generation backend (e.g., vLLM worker).\n"
        )
        raise RuntimeError(error_message)

    if colocated_inference:
        policy.offload_after_refit()
        policy_generation.prepare_for_generation(tags=["kv_cache"])


# ===============================================================================
# Training & Validation
# ===============================================================================

def grpo_train(
    policy: ColocatablePolicyInterface,
    policy_generation: Optional[GenerationInterface],
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    loss_fn: LossFunction,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    grpo_save_state: GRPOSaveState,
    master_config: MasterConfig,
) -> None:
    """
    Run GRPO training algorithm with optional executor training.

    NEW CONFIGS in master_config["grpo"]:
      - executor_samples_per_rewrite: int = 1
          Number of executor samples per rewrite (de-noises stage-1 value).
      - executor_backprop_on_original: bool = False
          If True, executor samples using the REWRITE but PPO loss is computed
          conditioning on the ORIGINAL problem prompt (swap user prompt before logprob calc).
    """
    timer = Timer()
    NEED_REFIT = True
    # If policy_generation is None, use the policy as the generation interface (megatron framework backend)
    if policy_generation is None:
        policy_generation = policy  # type: ignore
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True  # tracks if generation needs a refit before running
    assert policy_generation is not None  # for mypy type check

    # what to train: 'rewriter' | 'executor' | 'both'
    train_target = str(master_config["grpo"].get("train_target", "executor")).lower()
    assert train_target in {"rewriter", "executor", "both"}, \
        f"Unsupported grpo.train_target={train_target}. Use 'rewriter', 'executor', or 'both'."

    # ---- NEW: executor sampling & backprop control
    samples_per_rewrite = int(master_config["grpo"].get("executor_samples_per_rewrite", 1))
    backprop_on_original = bool(master_config["grpo"].get("executor_backprop_on_original", False))
    print (f"USING {samples_per_rewrite} samples per rewrite!")

    # common config/state items
    step = grpo_save_state["step"]
    consumed_samples = grpo_save_state["consumed_samples"]
    val_period = master_config["grpo"]["val_period"]
    val_at_start = master_config["grpo"]["val_at_start"]
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]

    # Run validation at the start if configured
    if val_at_start and step == 0:
        print("\n🔍 Running initial validation...")
        if NEED_REFIT and POLICY_GENERATION_STALE:
            refit_policy_generation(policy, policy_generation, colocated_inference)
            POLICY_GENERATION_STALE = False
        else:
            policy_generation.prepare_for_generation()
        val_metrics, validation_timings = validate(
            policy_generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            step=0,
            master_config=master_config,
        )
        policy_generation.finish_generation()
        logger.log_metrics(val_metrics, step, prefix="validation")
        logger.log_metrics(validation_timings, step, prefix="timing/validation")

    batch: BatchedDataDict[DatumSpec]
    for batch in dataloader:
        print(
            f"\n{'=' * 25} Step {step + 1}/{min(len(dataloader), master_config['grpo']['max_num_steps'])} {'=' * 25}"
        )
        maybe_gpu_profile_step(policy, step + 1)
        if policy != policy_generation:
            maybe_gpu_profile_step(policy_generation, step + 1)
        val_metrics, validation_timings = None, None

        # ==== Stage 1: build parallelizer prompts (controller/rewriter) ====
        text2save = []
        for i, message_log in enumerate(batch["message_log"]):
            original_problem = batch["original_problem"][i]
            parallelizer_prompt = PARALLELIZER_PROMPT.format(problem=original_problem)

            parallel_message = tokenizer.apply_chat_template(  # type: ignore
                [{"role": "user", "content": parallelizer_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
            )
            message_log[0]["token_ids"] = tokenizer(parallel_message, return_tensors="pt")["input_ids"][0]
            text2save.append(parallel_message)

        with open("{}/parallelizer_prompts_{}.jsonl".format(master_config["logger"]["log_dir"], step), "w") as f:
            for i, prompt in enumerate(text2save):
                json_entry = {
                    "step": step,
                    "prompt_idx": i,
                    "parallelizer_prompt": prompt,
                    "original_problem": batch["original_problem"][i]
                }
                f.write(json.dumps(json_entry) + "\n")

        with timer.time("total_step_time"):
            # -----------------------
            # Stage 1 Generation
            # -----------------------
            print("▶ Preparing batch...")
            with timer.time("data_processing"):
                repeated_parallel_batch: BatchedDataDict[DatumSpec] = batch.repeat_interleave(
                    master_config["grpo"]["num_generations_per_prompt"]
                )
                batched_flat_stage1, input_lengths_stage1 = batched_message_log_to_flat_message(
                    repeated_parallel_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                )
                # Stage-1 advantages will be computed later using these prompt ids
                input_ids_stage1 = batched_flat_stage1["token_ids"]

            print(f"▶ Generating responses for batch of size {repeated_parallel_batch.size} (Stage 1)...")
            with timer.time("prepare_for_generation"):
                if NEED_REFIT and POLICY_GENERATION_STALE:
                    refit_policy_generation(policy, policy_generation, colocated_inference)
                    POLICY_GENERATION_STALE = False
                else:
                    policy_generation.prepare_for_generation()

            with timer.time("generation"):
                if _should_use_async_rollouts(master_config):
                    (
                        repeated_parallel_batch,
                        stage1_rollout_metrics,
                    ) = run_async_multi_turn_rollout(
                        policy_generation=policy_generation,
                        input_batch=repeated_parallel_batch,
                        tokenizer=tokenizer,
                        task_to_env=task_to_env,
                        max_seq_len=master_config["policy"]["max_total_sequence_length"],
                        max_rollout_turns=master_config["grpo"]["max_rollout_turns"],
                        greedy=False,
                    )
                else:
                    repeated_parallel_batch, stage1_rollout_metrics = run_multi_turn_rollout(
                        policy_generation=policy_generation,
                        input_batch=repeated_parallel_batch,
                        tokenizer=tokenizer,
                        task_to_env=task_to_env,
                        max_seq_len=master_config["policy"]["max_total_sequence_length"],
                        max_rollout_turns=master_config["grpo"]["max_rollout_turns"],
                        greedy=False,
                    )
                policy_generation.finish_generation()

            # -----------------------
            # Stage 2: build executor prompts from Stage 1 outputs
            # -----------------------
            stage2_data = {
                "message_log": [],
                "controller_idx": [],
                "extra_env_info": [],
                "task_name": [],
                "tasks": [],
                "problems": [],
                "stage1_responses": [],
                # NEW:
                "rewrite_idx": [],
                "sample_rep": [],
            }
            text2save = []
            print("\n=== BEGIN Stage 1 RESPONSES ===")
            max_tasks = master_config["grpo"].get("max_tasks", 2)
            use_identity = bool(master_config["grpo"].get("identity_parallelizer", True))

            for i, message_log in enumerate(repeated_parallel_batch["message_log"]):
                original_problem = repeated_parallel_batch["original_problem"][i]
                for j, message in enumerate(message_log):
                    if message["role"] != "assistant":
                        continue

                    generated_text = tokenizer.decode(message["token_ids"], skip_special_tokens=True)

                    if use_identity:
                        print("USING IDENTITY PARALLELIZER!")
                        rewrites = [original_problem]
                    else:
                        print("USING REAL PARALLELIZER")
                        if "<rewrite" in generated_text and "</rewrite" in generated_text:
                            import re
                            if "</think>" in generated_text:
                                processed_text = generated_text.split("</think>")[1]
                            else:
                                processed_text = generated_text

                            print (processed_text)
                            rewrites = re.findall(
                                r"<rewrite>\s*(.*?)\s*</rewrite>",
                                processed_text,
                                flags=re.IGNORECASE | re.DOTALL,
                            )
                            if not rewrites:
                                wrapper = re.search(
                                    r"<rewrites>(.*?)</rewrites>",
                                    processed_text,
                                    flags=re.IGNORECASE | re.DOTALL,
                                )
                                if wrapper:
                                    rewrites = [line.strip() for line in wrapper.group(1).splitlines() if line.strip()]
                            rewrites = [r.strip() for r in rewrites if r.strip()]
                            print (rewrites)
                            if len(rewrites) < 2:
                                rewrites.extend([original_problem] * (2 - len(rewrites)))
                            else:
                                rewrites = rewrites[:2]
                        else:
                            rewrites = [original_problem] * 2

                    rewrites = rewrites[:max_tasks]

                    for rewrite_idx, rewrite in enumerate(rewrites):
                        for rep in range(samples_per_rewrite):
                            executor_prompt = EXECUTOR_PROMPT.format(problem=rewrite)
                            executor_message = tokenizer.apply_chat_template(
                                [{"role": "user", "content": executor_prompt}],
                                tokenize=False,
                                add_generation_prompt=True,
                                add_special_tokens=False,
                            )
                            text2save.append(executor_message)
                            token_ids = tokenizer(executor_message, return_tensors="pt")["input_ids"][0]

                            stage2_data["message_log"].append([{
                                "role": "user",
                                "content": executor_message,
                                "token_ids": token_ids,
                            }])
                            stage2_data["controller_idx"].append(i)
                            stage2_data["extra_env_info"].append(repeated_parallel_batch["extra_env_info"][i])
                            stage2_data["task_name"].append(repeated_parallel_batch["task_name"][i])
                            stage2_data["tasks"].append(rewrite)
                            stage2_data["problems"].append(original_problem)
                            stage2_data["stage1_responses"].append(generated_text)
                            # NEW:
                            stage2_data["rewrite_idx"].append(rewrite_idx)
                            stage2_data["sample_rep"].append(rep)

            with open("{}/subtask_prompts_{}.jsonl".format(master_config["logger"]["log_dir"], step), "w") as f:
                for i_, prompt in enumerate(text2save):
                    json_entry = {
                        "step": step,
                        "prompt_idx": i_,
                        "subtask_prompt": prompt,
                        "original_problem": stage2_data["problems"][i_],
                        "task": stage2_data["tasks"][i_],
                        "controller_idx": stage2_data["controller_idx"][i_],
                        "stage1_response": stage2_data["stage1_responses"][i_],
                        # NEW:
                        "rewrite_idx": stage2_data["rewrite_idx"][i_],
                        "sample_rep": stage2_data["sample_rep"][i_],
                    }
                    f.write(json.dumps(json_entry) + "\n")

            stage2_batch_size = len(stage2_data["message_log"])
            stage2_batch = BatchedDataDict[DatumSpec]({
                "message_log": stage2_data["message_log"],
                "length": torch.tensor([len(msg[0]["token_ids"]) for msg in stage2_data["message_log"]]),
                "extra_env_info": stage2_data["extra_env_info"],
                "idx": list(range(stage2_batch_size)),
                "loss_multiplier": torch.ones(stage2_batch_size),
                "task_name": stage2_data["task_name"],
                "controller_idx": stage2_data["controller_idx"],
                "tasks": stage2_data["tasks"],
                "problems": stage2_data["problems"],
                "rewrite_idx": stage2_data["rewrite_idx"],
                "sample_rep": stage2_data["sample_rep"],
            })

            repeated_batch_stage2: BatchedDataDict[DatumSpec] = stage2_batch.repeat_interleave(1)

            flat_stage2_prompt, _ = batched_message_log_to_flat_message(
                repeated_batch_stage2["message_log"],
                pad_value_dict={"token_ids": tokenizer.pad_token_id},
            )
            input_ids_stage2_prompt = flat_stage2_prompt["token_ids"]

            print(f"▶ Generating responses for batch of size {repeated_batch_stage2.size} (Stage 2)...")
            with timer.time("prepare_for_generation"):
                if NEED_REFIT and POLICY_GENERATION_STALE:
                    refit_policy_generation(policy, policy_generation, colocated_inference)
                    POLICY_GENERATION_STALE = False
                else:
                    policy_generation.prepare_for_generation()

            with timer.time("generation"):
                if _should_use_async_rollouts(master_config):
                    (
                        repeated_batch_stage2,
                        stage2_rollout_metrics,
                    ) = run_async_multi_turn_rollout(
                        policy_generation=policy_generation,
                        input_batch=repeated_batch_stage2,
                        tokenizer=tokenizer,
                        task_to_env=task_to_env,
                        max_seq_len=master_config["policy"]["max_total_sequence_length"],
                        max_rollout_turns=master_config["grpo"]["max_rollout_turns"],
                        greedy=False,
                    )
                else:
                    repeated_batch_stage2, stage2_rollout_metrics = run_multi_turn_rollout(
                        policy_generation=policy_generation,
                        input_batch=repeated_batch_stage2,
                        tokenizer=tokenizer,
                        task_to_env=task_to_env,
                        max_seq_len=master_config["policy"]["max_total_sequence_length"],
                        max_rollout_turns=master_config["grpo"]["max_rollout_turns"],
                        greedy=False,
                    )
                policy_generation.finish_generation()

            print("▶ Processing rewards...")

            with timer.time("reward_calculation"):
                # Build map: controller_idx -> rewrite_idx -> list[reward]
                rewards_by_controller_and_rewrite: dict[int, dict[int, list[torch.Tensor]]] = {}
                for i_ in range(len(repeated_batch_stage2["total_reward"])):
                    reward = repeated_batch_stage2["total_reward"][i_]
                    if not torch.is_tensor(reward):
                        reward = torch.tensor(reward)
                    ctrl = int(repeated_batch_stage2["controller_idx"][i_])
                    r_idx = int(repeated_batch_stage2["rewrite_idx"][i_])
                    rewards_by_controller_and_rewrite.setdefault(ctrl, {}).setdefault(r_idx, []).append(reward)

                # ---- Stage 1 (rewriter) reward aggregation from Stage 2
                for ctrl_idx in range(len(repeated_parallel_batch["total_reward"])):
                    if ctrl_idx in rewards_by_controller_and_rewrite:
                        per_rewrite_means = []
                        for r_idx, r_list in rewards_by_controller_and_rewrite[ctrl_idx].items():
                            per_rewrite_means.append(torch.stack(r_list).mean())
                        controller_mean = (
                            torch.stack(per_rewrite_means).mean() if per_rewrite_means else torch.tensor(0.0)
                        )
                        repeated_parallel_batch["total_reward"][ctrl_idx] = controller_mean
                    else:
                        repeated_parallel_batch["total_reward"][ctrl_idx] = torch.tensor(0.0, dtype=torch.float32)

                # Advantages for rewriter (stage 1)
                stage1_rewards = repeated_parallel_batch["total_reward"]
                advantages_stage1 = calculate_grpo_advantages_per_prompt(
                    input_ids_stage1,
                    stage1_rewards,
                    torch.ones_like(stage1_rewards),
                    leave_one_out_baseline=master_config["grpo"]["use_leave_one_out_baseline"],
                    normalize_rewards=master_config["grpo"]["normalize_rewards"],
                )

                # Advantages for executor (stage 2) using PRE-GENERATION prompt ids
                stage2_rewards = repeated_batch_stage2["total_reward"]
                advantages_stage2 = calculate_grpo_advantages_per_prompt(
                    input_ids_stage2_prompt,
                    stage2_rewards,
                    torch.ones_like(stage2_rewards),
                    leave_one_out_baseline=master_config["grpo"]["use_leave_one_out_baseline"],
                    normalize_rewards=master_config["grpo"]["normalize_rewards"],
                )

            # Build training tensors (stage 1)
            with timer.time("data_processing"):
                # Attach masks + advantages to Stage 1 messages
                for i_, message_log in enumerate(repeated_parallel_batch["message_log"]):
                    for msg in message_log:
                        if msg["role"] == "assistant":
                            msg["token_loss_mask"] = torch.ones_like(msg["token_ids"])
                        else:
                            msg["token_loss_mask"] = torch.zeros_like(msg["token_ids"])
                        if "generation_logprobs" not in msg:
                            msg["generation_logprobs"] = torch.zeros_like(
                                msg["token_ids"], dtype=torch.float32
                            )
                        msg["advantages"] = advantages_stage1[i_].expand(msg["token_ids"].shape)

                flat_messages_stage1, input_lengths_stage1 = batched_message_log_to_flat_message(
                    repeated_parallel_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    make_sequence_length_divisible_by=master_config["policy"]["make_sequence_length_divisible_by"],
                )

                train_data_stage1 = BatchedDataDict[ClippedPGLossDataDict](
                    {
                        "input_ids": flat_messages_stage1["token_ids"],
                        "input_lengths": input_lengths_stage1,
                        "advantages": flat_messages_stage1["advantages"],
                        "generation_logprobs": flat_messages_stage1["generation_logprobs"],
                        "token_mask": flat_messages_stage1["token_loss_mask"],
                        "sample_mask": repeated_parallel_batch["loss_multiplier"],
                    }
                )
                train_data_stage1.to("cpu")

            # Build training tensors (stage 2)
            with timer.time("data_processing"):
                # Optionally backprop against the ORIGINAL problem prompt (swap only the user message)
                if backprop_on_original:
                    new_lengths = []
                    for i_, message_log in enumerate(repeated_batch_stage2["message_log"]):
                        orig_problem = repeated_batch_stage2["problems"][i_]
                        bp_prompt = orig_problem 
                        bp_message = tokenizer.apply_chat_template(
                            [{"role": "user", "content": bp_prompt}],
                            tokenize=False,
                            add_generation_prompt=True,
                            add_special_tokens=False,
                        )
                        message_log[0]["content"] = bp_message
                        message_log[0]["token_ids"] = tokenizer(bp_message, return_tensors="pt")["input_ids"][0]
                        new_lengths.append(len(message_log[0]["token_ids"]))
                    if new_lengths:
                        repeated_batch_stage2["length"] = torch.tensor(new_lengths)
                else:
                    print ("You must enable backprop_on_original otherwise the groups won't work.")
                    raise ValueError()

                # Attach masks + advantages to Stage 2 messages
                for i_, message_log in enumerate(repeated_batch_stage2["message_log"]):
                    for msg in message_log:
                        if msg["role"] == "assistant":
                            msg["token_loss_mask"] = torch.ones_like(msg["token_ids"])
                        else:
                            msg["token_loss_mask"] = torch.zeros_like(msg["token_ids"])
                        if "generation_logprobs" not in msg:
                            msg["generation_logprobs"] = torch.zeros_like(
                                msg["token_ids"], dtype=torch.float32
                            )
                        msg["advantages"] = advantages_stage2[i_].expand(msg["token_ids"].shape)

                flat_messages_stage2, input_lengths_stage2 = batched_message_log_to_flat_message(
                    repeated_batch_stage2["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    make_sequence_length_divisible_by=master_config["policy"]["make_sequence_length_divisible_by"],
                )

                train_data_stage2 = BatchedDataDict[ClippedPGLossDataDict](
                    {
                        "input_ids": flat_messages_stage2["token_ids"],
                        "input_lengths": input_lengths_stage2,
                        "advantages": flat_messages_stage2["advantages"],
                        "generation_logprobs": flat_messages_stage2["generation_logprobs"],
                        "token_mask": flat_messages_stage2["token_loss_mask"],
                        "sample_mask": repeated_batch_stage2["loss_multiplier"],
                    }
                )
                train_data_stage2.to("cpu")

            print("▶ Preparing for logprob inference...")
            with timer.time("logprob_inference_prep"):
                policy.prepare_for_lp_inference()

            # Well compute logprobs only for the branch(es) we train
            def _fill_logprobs(td):
                fprop_logprobs = policy.get_logprobs(td)["logprobs"]
                reference_logprobs = policy.get_reference_policy_logprobs(td)["reference_logprobs"]
                td["prev_logprobs"] = fprop_logprobs
                td["reference_policy_logprobs"] = reference_logprobs

            if train_target in {"rewriter", "both"}:
                print("▶ Computing logprobs (Stage 1)...")
                with timer.time("policy_and_reference_logprobs"):
                    _fill_logprobs(train_data_stage1)

            if train_target in {"executor", "both"}:
                print("▶ Computing logprobs (Stage 2)...")
                with timer.time("policy_and_reference_logprobs"):
                    _fill_logprobs(train_data_stage2)

            print("▶ Preparing for training...")
            with timer.time("training_prep"):
                policy.prepare_for_training()  # set model train and reload optim to GPU
                POLICY_GENERATION_STALE = True

            metrics_collected = {}
            if train_target == "rewriter":
                print("▶ Training policy on REWRITER (Stage 1)...")
                with timer.time("policy_training"):
                    train_results = policy.train(train_data_stage1, loss_fn)

                active_rewards = stage1_rewards
                active_flat_messages = flat_messages_stage1
                active_input_lengths = input_lengths_stage1
                metrics_collected["stage1"] = (train_results, stage1_rollout_metrics, active_rewards)
            elif train_target == "executor":
                print("▶ Training policy on EXECUTOR (Stage 2)...")
                with timer.time("policy_training"):
                    train_results = policy.train(train_data_stage2, loss_fn)
                active_rewards = stage2_rewards
                active_flat_messages = flat_messages_stage2
                active_input_lengths = input_lengths_stage2
                metrics_collected["stage2"] = (train_results, stage2_rollout_metrics, active_rewards)
            else:  # both
                print("▶ Training policy on REWRITER (Stage 1) then EXECUTOR (Stage 2)...")
                with timer.time("policy_training"):
                    train_results_stage1 = policy.train(train_data_stage1, loss_fn)
                with timer.time("policy_training"):
                    train_results_stage2 = policy.train(train_data_stage2, loss_fn)
                # last active vars just for generic logging below
                active_rewards = stage2_rewards
                active_flat_messages = flat_messages_stage2
                active_input_lengths = input_lengths_stage2
                metrics_collected["stage1"] = (train_results_stage1, stage1_rollout_metrics, stage1_rewards)
                metrics_collected["stage2"] = (train_results_stage2, stage2_rollout_metrics, stage2_rewards)

            is_last_step = step + 1 == min(master_config["grpo"]["max_num_steps"], len(dataloader))

            # Validation
            if val_period > 0 and (step + 1) % val_period == 0:
                if NEED_REFIT and POLICY_GENERATION_STALE:
                    refit_policy_generation(policy, policy_generation, colocated_inference)
                    POLICY_GENERATION_STALE = False
                else:
                    policy_generation.prepare_for_generation()
                val_metrics, validation_timings = validate(
                    policy_generation,
                    val_dataloader,
                    tokenizer,
                    val_task_to_env,
                    step=step + 1,
                    master_config=master_config,
                )
                policy_generation.finish_generation()
                logger.log_metrics(validation_timings, step + 1, prefix="timing/validation")
                logger.log_metrics(val_metrics, step + 1, prefix="validation")

            # Checkpointing
            consumed_samples += master_config["grpo"]["num_prompts_per_step"]
            if master_config["checkpointing"]["enabled"] and (
                is_last_step or (step + 1) % master_config["checkpointing"]["save_period"] == 0
            ):  # +1 because step is 0-indexed
                policy.prepare_for_training()

                grpo_save_state["step"] = step + 1
                if val_metrics is not None:
                    grpo_save_state["val_reward"] = val_metrics["accuracy"]
                elif "val_reward" in grpo_save_state:
                    del grpo_save_state["val_reward"]
                grpo_save_state["consumed_samples"] = consumed_samples

                if master_config["checkpointing"]["metric_name"] is not None:
                    if master_config["checkpointing"]["metric_name"] not in grpo_save_state:
                        warnings.warn(
                            f"You asked to save checkpoints based on {master_config['checkpointing']['metric_name']} but the metric is not found in the save state. "
                            "Saving most recent k checkpoints instead."
                        )
                        master_config["checkpointing"]["metric_name"] = None

                with timer.time("checkpointing"):
                    print(f"Saving checkpoint for step {step + 1}...")
                    checkpoint_path = checkpointer.init_tmp_checkpoint(step + 1, grpo_save_state, master_config)
                    policy.save_checkpoint(
                        weights_path=os.path.join(checkpoint_path, "policy", "weights"),
                        optimizer_path=os.path.join(checkpoint_path, "policy", "optimizer"),
                        tokenizer_path=os.path.join(checkpoint_path, "policy", "tokenizer"),
                    )
                    torch.save(dataloader.state_dict(), os.path.join(checkpoint_path, "train_dataloader.pt"))
                    checkpointer.finalize_checkpoint(checkpoint_path)
                policy.offload_after_refit()

        # ========= END of timed step =========
        # Write JSONL dumps for whichever branch(es) we trained
        if train_target in {"rewriter", "both"}:
            log_data1 = {"content": flat_messages_stage1["content"]}
            log_data1["rewards"] = stage1_rewards.tolist()
            log_data1["generation_logprobs"] = train_data_stage1["generation_logprobs"].tolist()
            log_data1["prev_logprobs"] = train_data_stage1["prev_logprobs"].tolist()
            log_data1["input_lengths"] = input_lengths_stage1.tolist()
            logger.log_batched_dict_as_jsonl(log_data1, f"train_data_stage1_step{step}.jsonl")

        if train_target in {"executor", "both"}:
            log_data2 = {"content": flat_messages_stage2["content"]}
            log_data2["rewards"] = stage2_rewards.tolist()
            log_data2["generation_logprobs"] = train_data_stage2["generation_logprobs"].tolist()
            log_data2["prev_logprobs"] = train_data_stage2["prev_logprobs"].tolist()
            log_data2["input_lengths"] = input_lengths_stage2.tolist()
            logger.log_batched_dict_as_jsonl(log_data2, f"train_data_stage2_step{step}.jsonl")

        # Numeric metrics
        def _pack_train_metrics(tr, rewards_array):
            metrics = {
                "loss": tr["loss"].numpy(),
                "reward": rewards_array.numpy(),
                "grad_norm": tr["grad_norm"].numpy(),
            }
            metrics.update(tr["all_mb_metrics"])
            for k, v in metrics.items():
                if k in {"lr", "wd", "reward", "global_valid_seqs", "global_valid_toks"}:
                    metrics[k] = np.mean(v).item()
                else:
                    metrics[k] = np.sum(v).item()
            return metrics

        timing_metrics: dict[str, float] = timer.get_timing_metrics(reduction_op="sum")  # totals are now valid

        if train_target == "rewriter":
            tr, rollout_m, rewards_arr = metrics_collected["stage1"]
            metrics = _pack_train_metrics(tr, rewards_arr)
            metrics.update(rollout_m)

            if metrics.get("token_mult_prob_error", 1.0) > 1.05:
                logger.log_plot_token_mult_prob_error(
                    {
                        "prompt_lengths": repeated_parallel_batch["length"],
                        "full_lengths": input_lengths_stage1,
                        "generation_logprobs": train_data_stage1["generation_logprobs"],
                        "prev_logprobs": train_data_stage1["prev_logprobs"],
                        "token_mask": train_data_stage1["token_mask"],
                        "sample_mask": train_data_stage1["sample_mask"],
                    },
                    step + 1,
                    name="train/token_mult_prob_error_plot_sample",
                )

            print("\n Training Results (Stage 1 / Rewriter):")
            print(f"   Loss: {metrics['loss']:.4f}")
            print(f"   Avg Reward: {np.mean(rewards_arr.numpy()):.4f}")
            print("\n  Timing:")
            total_time = timing_metrics.get("total_step_time", 0)
            print(f"   Total step time: {total_time:.2f}s")
            for k, v in sorted(timing_metrics.items(), key=lambda item: item[1], reverse=True):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"   {k}: {v:.2f}s ({percent:.1f}%)")

            logger.log_metrics(metrics, step + 1, prefix="train")
            logger.log_metrics(timing_metrics, step + 1, prefix="timing/train")

        elif train_target == "executor":
            tr, rollout_m, rewards_arr = metrics_collected["stage2"]
            metrics = _pack_train_metrics(tr, rewards_arr)
            metrics.update(rollout_m)

            if metrics.get("token_mult_prob_error", 1.0) > 1.05:
                logger.log_plot_token_mult_prob_error(
                    {
                        "prompt_lengths": repeated_batch_stage2["length"],
                        "full_lengths": input_lengths_stage2,
                        "generation_logprobs": train_data_stage2["generation_logprobs"],
                        "prev_logprobs": train_data_stage2["prev_logprobs"],
                        "token_mask": train_data_stage2["token_mask"],
                        "sample_mask": train_data_stage2["sample_mask"],
                    },
                    step + 1,
                    name="train/token_mult_prob_error_plot_sample",
                )

            print("\n Training Results (Stage 2 / Executor):")
            print(f"   Loss: {metrics['loss']:.4f}")
            print(f"   Avg Reward: {np.mean(rewards_arr.numpy()):.4f}")
            print("\n  Timing:")
            total_time = timing_metrics.get("total_step_time", 0)
            print(f"   Total step time: {total_time:.2f}s")
            for k, v in sorted(timing_metrics.items(), key=lambda item: item[1], reverse=True):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

            logger.log_metrics(metrics, step + 1, prefix="train")
            logger.log_metrics(timing_metrics, step + 1, prefix="timing/train")

        else:  # both
            tr1, rollout_m1, rewards_arr1 = metrics_collected["stage1"]
            tr2, rollout_m2, rewards_arr2 = metrics_collected["stage2"]
            metrics1 = _pack_train_metrics(tr1, rewards_arr1); metrics1.update(rollout_m1)
            metrics2 = _pack_train_metrics(tr2, rewards_arr2); metrics2.update(rollout_m2)

            # token mult prob error plots (optional)
            if metrics1.get("token_mult_prob_error", 1.0) > 1.05:
                logger.log_plot_token_mult_prob_error(
                    {
                        "prompt_lengths": repeated_parallel_batch["length"],
                        "full_lengths": input_lengths_stage1,
                        "generation_logprobs": train_data_stage1["generation_logprobs"],
                        "prev_logprobs": train_data_stage1["prev_logprobs"],
                        "token_mask": train_data_stage1["token_mask"],
                        "sample_mask": train_data_stage1["sample_mask"],
                    },
                    step + 1,
                    name="train/rewriter_token_mult_prob_error_plot_sample",
                )
            if metrics2.get("token_mult_prob_error", 1.0) > 1.05:
                logger.log_plot_token_mult_prob_error(
                    {
                        "prompt_lengths": repeated_batch_stage2["length"],
                        "full_lengths": input_lengths_stage2,
                        "generation_logprobs": train_data_stage2["generation_logprobs"],
                        "prev_logprobs": train_data_stage2["prev_logprobs"],
                        "token_mask": train_data_stage2["token_mask"],
                        "sample_mask": train_data_stage2["sample_mask"],
                    },
                    step + 1,
                    name="train/executor_token_mult_prob_error_plot_sample",
                )

            print("\n Training Results:")
            print(f"   Rewriter Loss: {metrics1['loss']:.4f} | Avg Reward: {np.mean(rewards_arr1.numpy()):.4f}")
            print(f"   Executor Loss: {metrics2['loss']:.4f} | Avg Reward: {np.mean(rewards_arr2.numpy()):.4f}")
            print("\n  Timing:")
            total_time = timing_metrics.get("total_step_time", 0)
            print(f"  • Total step time: {total_time:.2f}s")
            for k, v in sorted(timing_metrics.items(), key=lambda item: item[1], reverse=True):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

            logger.log_metrics(metrics1, step + 1, prefix="train/rewriter")
            logger.log_metrics(metrics2, step + 1, prefix="train/executor")
            logger.log_metrics(timing_metrics, step + 1, prefix="timing/train")

        # prepare for next step
        timer.reset()
        step += 1
        if step >= master_config["grpo"]["max_num_steps"]:
            break

def validate(
    policy_generation: GenerationInterface,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer,
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    step: int,
    master_config: MasterConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run validation on the validation dataset."""
    if val_dataloader is None:
        print("  ⚠️ No validation dataloader provided, skipping validation")
        return {}, {}

    timer = Timer()
    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...")

        total_rewards = []
        total_lengths = []
        all_message_logs = []  # Collect all message logs

        max_batches = (
            master_config["grpo"]["max_val_samples"]
            // master_config["grpo"]["val_batch_size"]
        )
        val_num_generations_per_prompt = master_config["grpo"].get("val_num_generations_per_prompt", 1)

        for batch_idx, val_batch in enumerate(val_dataloader):
            if batch_idx >= max_batches:
                break

            for _ in range(val_num_generations_per_prompt):
                # Generate responses (updates the LLMMessageLogType in batch_with_msg_logs)
                # Use async rollouts if vLLM async engine is enabled
                if _should_use_async_rollouts(master_config):
                    val_batch, gen_metrics = run_async_multi_turn_rollout(
                        policy_generation,
                        val_batch,
                        tokenizer,
                        val_task_to_env,
                        max_seq_len=master_config["policy"]["max_total_sequence_length"],
                        max_rollout_turns=master_config["grpo"]["max_rollout_turns"],
                        greedy=False,
                    )
                else:
                    val_batch, gen_metrics = run_multi_turn_rollout(
                        policy_generation,
                        val_batch,
                        tokenizer,
                        val_task_to_env,
                        max_seq_len=master_config["policy"]["max_total_sequence_length"],
                        max_rollout_turns=master_config["grpo"]["max_rollout_turns"],
                        greedy=False,
                    )
                rewards = val_batch["total_reward"]

                total_rewards.extend(rewards.tolist())
                total_lengths.append(gen_metrics["mean_gen_tokens_per_sample"])

                # Collect message logs for later display
                to_env = [
                    get_keys_from_message_log(
                        val_batch["message_log"][i], ["role", "content"]
                    )
                    for i in range(len(val_batch["message_log"]))
                ]

                all_message_logs.extend(to_env)

        # Calculate validation metrics
        accuracy = float(np.mean(total_rewards))
        median_length = float(np.median(total_lengths))
        avg_length = float(np.mean(total_lengths))
        std_length = float(np.std(total_lengths))
        max_length = float(np.amax(total_lengths))
        # min_length = float(np.amin(total_lengths))

        val_metrics = {
            "accuracy": accuracy,
            "median_length": median_length,
            "mean_length": avg_length,
            "std_length": std_length,
            "max_length": max_length,
        }

        # Print sample conversations only once at the end of validation
        try:
            print_message_log_samples(
                all_message_logs,
                total_rewards,
                num_samples=min(
                    master_config["logger"]["num_val_samples_to_print"],
                    len(all_message_logs),
                ),
                step=step,
            )
        except Exception as e:
            print(f"\n  ⚠️ Error displaying message samples: {str(e)}")
            print("  ⚠️ Continuing validation without displaying samples...")

    # Get timing metrics
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    # Print summary of validation results
    print("\n📊 Validation Results:")
    print(f"    • Accuracy: {accuracy:.4f}")
    print(f"    • Average response length: {avg_length:.1f} tokens")
    print(f"    • Samples processed: {len(total_rewards)}")

    # Print timing information
    print("\n  ⏱️  Validation Timing:")
    validation_time = timing_metrics.get("total_validation_time", 0)
    print(f"    • Total validation time: {validation_time:.2f}s")

    # Make sure to reset the timer after validation
    timer.reset()

    return val_metrics, timing_metrics

