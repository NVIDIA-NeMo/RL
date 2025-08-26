import pytest
import torch

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


def test_reward_model_environment(reward_model_env, cluster):
    """Test the reward model environment."""
    pass


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
