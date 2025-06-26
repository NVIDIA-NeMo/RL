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
from typing import TypedDict

import ray
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data import MathDataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, eval_collate_fn
from nemo_rl.data.llm_message_utils import get_keys_from_message_log
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.environments.math_environment import MathEnvConfig
from nemo_rl.models.generation.interfaces import GenerationConfig
from nemo_rl.models.generation.vllm import VllmGeneration

# ===============================================================================
# Configuration
# ===============================================================================


class EvalConfig(TypedDict):
    metric: str
    num_tests_per_prompt: int
    seed: int


class MasterConfig(TypedDict):
    eval: EvalConfig
    generate: GenerationConfig
    data: MathDataConfig
    env: MathEnvConfig
    cluster: ClusterConfig


# ===============================================================================
# Setup & Initialization
# ===============================================================================


def setup(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
    dataset: AllTaskProcessedDataset,
) -> tuple[
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
        max_colocated_worker_groups=1,
    )
    print(f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes")

    # ==========================
    #           Model
    # ==========================
    print("\n▶ Setting up model...")
    # check backend
    backend = generation_config["backend"]
    assert backend == "vllm", "Only vLLM backend is supported for evaluation"

    # initialize vllm generation
    vllm_generation = VllmGeneration(cluster=cluster, config=generation_config)
    print(
        f"  ✓ Using vLLM backend for generation with {generation_config['model_name']}"
    )

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        vllm_generation,
        dataloader,
        master_config,
    )


# ===============================================================================
# Evaluation
# ===============================================================================


def run_env_eval(vllm_generation, dataloader, env, master_config):
    """Main entry point for running evaluation using environment.

    Generates model responses and evaluates them by env.

    Args:
        vllm_generation: Model for generating responses.
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
        outputs = vllm_generation.generate_text(inputs)["texts"]

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
    vllm_generation.shutdown()

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
