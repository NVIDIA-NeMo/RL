from typing import List

import numpy as np
import pytest
import torch
from datasets import load_dataset
from tqdm import tqdm

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.environments.reward_model_environment import (
    RewardModelEnvironment,
    RewardModelEnvironmentConfig,
)
from nemo_rl.models.generation.vllm import VllmConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
REWARD_MODEL_NAME = "Skywork/Skywork-Reward-V2-Qwen3-8B"

basic_vllm_config: VllmConfig = {
    "backend": "vllm",
    "model_name": MODEL_NAME,
    "dtype": "bfloat16",
    "max_new_tokens": 100,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": None,
    "stop_token_ids": None,
    "stop_strings": None,
    "vllm_cfg": {
        "async_engine": False,
        "precision": "bfloat16",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "max_model_len": 1024,
        "disable_log_stats": True,
        "disable_log_requests": True,
        "gpu_memory_utilization": 0.6,
        "enforce_eager": "False",
    },
    "colocated": {
        "enabled": True,
        "resources": {
            "gpus_per_node": None,
            "num_nodes": None,
        },
    },
}


basic_env_config: RewardModelEnvironmentConfig = {
    "enabled": True,
    "model_name": REWARD_MODEL_NAME,
    "tokenizer": {"name": REWARD_MODEL_NAME},
    "precision": "bfloat16",
    "batch_size": 32,
    "checkpoint_path": None,
    "max_model_len": basic_vllm_config["vllm_cfg"]["max_model_len"],
    "resources": {"gpus_per_node": 8, "num_nodes": 1},
    "reward_model_cfg": {
        "enabled": True,
        "reward_model_type": "bradley_terry",
    },
    "dtensor_cfg": {
        "enabled": True,
        "cpu_offload": False,
        "sequence_parallel": False,
        "activation_checkpointing": False,
        "tensor_parallel_size": 8,
        "context_parallel_size": 1,
        "custom_parallel_plan": None,
    },
    "dynamic_batching": {"enabled": False},
    "sequence_packing": {"enabled": False},
    "max_grad_norm": None,
}


@pytest.fixture(scope="function")
def reward_model_env():
    """Create a reward model environment for testing."""
    try:
        env_actor = RewardModelEnvironment(basic_env_config)
        yield env_actor
    finally:
        if env_actor:
            env_actor.shutdown()


@pytest.fixture(scope="function")
def cluster():
    """Create a virtual cluster for testing."""
    cluster_instance = None
    cluster_name = f"test-reward-model-cluster-{id(cluster_instance)}"
    print(f"\nCreating virtual cluster '{cluster_name}'...")
    try:
        cluster_instance = RayVirtualCluster(
            name=cluster_name,
            bundle_ct_per_node_list=[1],
            use_gpus=True,
            num_gpus_per_node=8,
            max_colocated_worker_groups=2,
        )
        yield cluster_instance
    finally:
        print(f"\nCleaning up cluster '{cluster_name}'...")
        if cluster_instance:
            cluster_instance.shutdown()


EVAL_SET = "allenai/reward-bench"


# EVAL_SET = "THU-KEG/RM-Bench"
def test_reward_model_environment_performance(reward_model_env):
    """Test the reward model environment."""
    raw_dataset = load_dataset(EVAL_SET, split="filtered")
    subsets = raw_dataset["subset"]
    batch_size = 32
    dataloader = torch.utils.data.DataLoader(
        raw_dataset,
        batch_size=batch_size,
        collate_fn=None,
        shuffle=False,
        drop_last=False,
    )
    reward_data_chosen = []
    reward_data_rejected = []
    for idx, batch in enumerate(tqdm(dataloader, desc="Processing dataset")):
        message_log_batch_chosen: List[LLMMessageLogType] = []
        message_log_batch_rejected: List[LLMMessageLogType] = []
        for example in batch:
            message_log_chosen = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["chosen"]},
            ]
            message_log_rejected = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["rejected"]},
            ]
            message_log_batch_chosen.append(message_log_chosen)
            message_log_batch_rejected.append(message_log_rejected)
        reward_data_chosen_batch = reward_model_env.preprocess_data(
            message_log_batch_chosen
        )
        reward_data_rejected_batch = reward_model_env.preprocess_data(
            message_log_batch_rejected
        )
        reward_data_chosen_scores = reward_model_env.reward_model_policy.score(
            reward_data_chosen_batch
        )["scores"]
        reward_data_rejected_scores = reward_model_env.reward_model_policy.score(
            reward_data_rejected_batch
        )["scores"]
        reward_data_chosen.extend(reward_data_chosen_scores)
        reward_data_rejected.extend(reward_data_rejected_scores)
    results = []
    for chosen, rejected in zip(reward_data_chosen, reward_data_rejected):
        results.append(1) if chosen > rejected else results.append(0)
    print(results)
    print(len(results))
    out_dataset = raw_dataset.add_column("results", results)

    # add subsets back (removed so it's not handled by cuda)
    out_dataset = out_dataset.add_column("subset", subsets)
    present_subsets = np.unique(subsets)
    results_grouped = {}
    for subset in present_subsets:
        subset_dataset = out_dataset.filter(lambda example: example["subset"] == subset)
        num_correct = sum(subset_dataset["results"])
        num_total = len(subset_dataset["results"])
        print(f"{subset}: {num_correct}/{num_total} ({num_correct / num_total})")
        results_grouped[subset] = num_correct / num_total


def test_reward_model_environment_generate_responses(reward_model_env):
    """Test the reward model environment generate responses."""

    message_log_batch = [
        [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of Brazil is Brasilia."},
        ],
        [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ],
    ]

    output = reward_model_env.step(message_log_batch, [])
    assert output.rewards.shape == (2,)
    assert output.rewards.dtype == torch.float32
    assert output.rewards[0] < output.rewards[1]
