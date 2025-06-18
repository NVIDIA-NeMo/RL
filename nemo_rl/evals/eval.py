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

import os
from pathlib import Path
from typing import Optional, Tuple, TypedDict

import ray
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from nemo_rl.algorithms.grpo import refit_policy_generation
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data import MathDataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, eval_collate_fn
from nemo_rl.data.llm_message_utils import get_keys_from_message_log
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.environments.math_environment import MathEnvConfig
from nemo_rl.models.generation.interfaces import GenerationConfig
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.policy.hf_policy import HfPolicy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager

# ===============================================================================
# Configuration
# ===============================================================================


class EvalConfig(TypedDict):
    metric: str
    num_tests_per_prompt: int
    seed: int


class MasterConfig(TypedDict):
    eval: EvalConfig
    generation: GenerationConfig
    data: MathDataConfig
    env: MathEnvConfig
    cluster: ClusterConfig
    checkpointing: Optional[CheckpointingConfig] = None


# ===============================================================================
# Setup & Initialization
# ===============================================================================


def setup(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
    dataset: AllTaskProcessedDataset,
) -> Tuple[
    VllmGeneration,
    DataLoader,
    MasterConfig,
]:
    """Set up components for model evaluation.

    Initializes the VLLM model and data loader.

    Args:
        master_config: Configuration settings.
        dataset: Dataset to evaluate on.

    Returns:
        VLLM model, data loader, and config.
    """
    # Extract individual configs for easier access
    eval_config = master_config["eval"]
    generation_config = master_config["generation"]
    cluster_config = master_config["cluster"]
    checkpointing_config = master_config.get("checkpointing")

    # Set seed for reproducibility
    set_seed(eval_config["seed"])

    # Check settings
    metric = eval_config["metric"]
    num_tests_per_prompt = eval_config["num_tests_per_prompt"]
    temperature = generation_config["temperature"]
    top_k = generation_config["top_k"]
    # TODO @yukih: support pass@k and cons@k
    assert metric in ["pass@1"], f"Invalid metric: {metric}"
    if num_tests_per_prompt > 1:
        assert temperature > 0 and top_k != 1, (
            "temperature > 0 and top_k != 1 are required for multiple samples"
        )

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    cluster = RayVirtualCluster(
        name="eval_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=2,
    )
    print(f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes")

    # ==========================
    #           Model
    # ==========================
    print("\n▶ Setting up model...")
    backend = generation_config["backend"]
    assert backend == "vllm", "Only vLLM backend is supported for evaluation"

    vllm_model: VllmGeneration

    # Proceed with checkpoint refit only if checkpointing is enabled and a directory is provided
    if (
        checkpointing_config
        and checkpointing_config.get("enabled")
        and checkpointing_config.get("checkpoint_dir")
    ):
        print("\n▶ Loading from DCP checkpoint using HfPolicy and refit mechanism...")
        checkpointer = CheckpointManager(checkpointing_config)
        best_checkpoint_path_str = checkpointer.get_best_checkpoint_path()
        print(f"  ✓ Found checkpoint at: {best_checkpoint_path_str}")
        best_checkpoint_path = Path(best_checkpoint_path_str)
        dcp_weights_path = best_checkpoint_path / "policy" / "weights"
        dcp_tokenizer_path = best_checkpoint_path / "policy" / "tokenizer"

        if not dcp_weights_path.exists():
            raise FileNotFoundError(
                f"DCP weights path not found: {dcp_weights_path}. Cannot proceed with refit."
            )

        ckpt_cfg_path = best_checkpoint_path / "config.yaml"
        if ckpt_cfg_path.exists():
            with open(ckpt_cfg_path, "r") as f:
                _ckpt_master_cfg = yaml.safe_load(f)
            policy_config = _ckpt_master_cfg.get("policy")

        if policy_config is None:
            raise ValueError(
                "Policy configuration is required for refitting but was not provided in the evaluation YAML "
                "and could not be found in the checkpoint (config.yaml)."
            )

        base_model_name = policy_config["model_name"]
        tokenizer_path_for_vllm_and_hf = str(
            dcp_tokenizer_path if dcp_tokenizer_path.exists() else base_model_name
        )
        trust_remote_code_tokenizer = policy_config.get("tokenizer", {}).get(
            "trust_remote_code"
        )

        # 1. Initialize VllmGeneration with base model and load_format="dummy"
        print(
            f"  ▶ Initializing VllmGeneration with base model '{base_model_name}' (dummy load)..."
        )
        vllm_init_gen_cfg = generation_config.copy()
        vllm_init_gen_cfg["model_name"] = base_model_name
        # Ensure VllmGeneration uses the tokenizer from DCP if available
        vllm_init_gen_cfg["tokenizer_name"] = tokenizer_path_for_vllm_and_hf
        # Pass trust_remote_code for tokenizer (assuming GenerationConfig/VllmGeneration supports tokenizer_trust_remote_code)
        vllm_init_gen_cfg["tokenizer_trust_remote_code"] = trust_remote_code_tokenizer
        vllm_init_gen_cfg["trust_remote_code"] = policy_config.get("trust_remote_code")

        current_vllm_opts = vllm_init_gen_cfg.get("vllm_cfg", {}).copy()
        current_vllm_opts["load_format"] = "dummy"
        vllm_init_gen_cfg["vllm_cfg"] = current_vllm_opts
        vllm_model = VllmGeneration(cluster=cluster, config=vllm_init_gen_cfg)
        print(f"    ✓ VllmGeneration initialized for base '{base_model_name}'.")

        # 2. Initialize temporary HfPolicy to load DCP weights
        print(
            f"  ▶ Loading DCP weights into temporary HfPolicy from: {dcp_weights_path}..."
        )
        tokenizer_for_hf_policy = AutoTokenizer.from_pretrained(
            tokenizer_path_for_vllm_and_hf,
            trust_remote_code=trust_remote_code_tokenizer,
        )

        temp_hf_policy_config = policy_config.copy()
        temp_hf_policy = HfPolicy(
            cluster=cluster,
            config=temp_hf_policy_config,
            tokenizer=tokenizer_for_hf_policy,
            weights_path=str(dcp_weights_path),
            init_optimizer=False,
        )
        print(f"    ✓ Temporary HfPolicy loaded with weights from {dcp_weights_path}.")

        # 3. Perform Refit
        print("  ▶ Refitting weights from HfPolicy to VllmGeneration...")
        refit_buffer_size = policy_config.get("refit_buffer_size_gb")
        vllm_model.finish_generation()
        refit_policy_generation(temp_hf_policy, vllm_model, refit_buffer_size)
        print("    ✓ Weights refitted to VllmGeneration.")

        del temp_hf_policy
        print(
            f"  ✓ Using vLLM backend, with weights from DCP checkpoint: {dcp_weights_path}"
        )

    else:
        print(
            "\n▶ Checkpointing not configured or checkpoint_dir not set. Using model_name from generation_config for VLLM."
        )
        # Initialize VllmGeneration directly with model_name from config
        vllm_cfg_mut = generation_config.get("vllm_cfg", {}).copy()
        vllm_cfg_mut["load_format"] = "auto"  # Ensure auto load for direct HF model
        gen_config_for_direct_load = generation_config.copy()
        gen_config_for_direct_load["vllm_cfg"] = vllm_cfg_mut
        vllm_model = VllmGeneration(cluster=cluster, config=gen_config_for_direct_load)
        print(
            f"  ✓ Using vLLM backend, directly loading {generation_config['model_name']}"
        )

    # ==========================
    #           Data
    # ==========================
    if generation_config["num_prompts_per_step"] == -1:
        generation_config["num_prompts_per_step"] = len(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=generation_config["num_prompts_per_step"],
        shuffle=False,
        collate_fn=eval_collate_fn,
    )
    print(f"  ✓ Evaluation dataset loaded with {len(dataset)} samples")

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        vllm_model,
        dataloader,
        master_config,
    )


# ===============================================================================
# Evaluation
# ===============================================================================


def run_env_eval(vllm_model, dataloader, env, master_config):
    """Main entry point for running evaluation using environment.

    Generates model responses and evaluates them by env.

    Args:
        vllm_model: Model for generating responses.
        dataloader: Data loader with evaluation samples.
        env: Environment that scores responses.
        master_config: Configuration settings.
    """
    # Extract for easier access
    generation_config = master_config["generation"]
    eval_config = master_config["eval"]
    metric = eval_config["metric"]
    num_tests_per_prompt = eval_config["num_tests_per_prompt"]

    # Run evaluation loop
    score, count = 0.0, 0
    for batch in dataloader:
        # update stats
        count += batch.size * num_tests_per_prompt

        # measure multiple samples
        if num_tests_per_prompt > 1:
            batch = batch.repeat_interleave(num_tests_per_prompt)

        # get input prompt from message_log
        prompts = []
        for message_log in batch["message_log"]:
            content = [message["content"] for message in message_log]
            content = "\n".join(content)
            prompts.append(content)

        # generate by vllm
        inputs = BatchedDataDict({"prompts": prompts})
        outputs = vllm_model.generate_text(inputs)["texts"]

        # append to message_log
        for idx, output in enumerate(outputs):
            batch["message_log"][idx].append(
                {
                    "role": "assistant",
                    "content": output,
                }
            )

        # evaluate generations with the environment
        to_env = [
            get_keys_from_message_log(batch["message_log"][i], ["role", "content"])
            for i in range(len(batch["message_log"]))
        ]
        env_return = ray.get(env.step.remote(to_env, batch["extra_env_info"]))

        # update stats
        if metric == "pass@1":
            score += env_return.rewards.sum().item()
        else:
            raise ValueError(f"Invalid metric: {metric}")

    # Cleanup before printing results
    ray.get(env.shutdown.remote())
    vllm_model.shutdown()

    # Print results
    dataset_name = os.path.basename(master_config["data"]["dataset_name"])
    model_name = os.path.basename(generation_config["model_name"])
    max_new_tokens = generation_config["vllm_cfg"]["max_model_len"]
    temperature = generation_config["temperature"]
    top_p = generation_config["top_p"]
    top_k = generation_config["top_k"]
    average_score = score / count

    print("\n" + "=" * 60)
    print(f"{model_name=} {dataset_name=}")
    print(f"{max_new_tokens=} {temperature=} {top_p=} {top_k=}\n")
    print(f"{metric=} {num_tests_per_prompt=}\n")
    print(f"score={average_score:.4f} ({score}/{count})")
    print("=" * 60 + "\n")
