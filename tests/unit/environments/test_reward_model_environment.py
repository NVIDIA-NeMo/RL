import pytest

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.environments.reward_model_environment import (
    RewardModelEnvironment,
    RewardModelEnvironmentConfig,
)
from nemo_rl.models.generation.vllm import VllmConfig

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
REWARD_MODEL_NAME = "nvidia/Llama-3.1-Nemotron-70B-Reward-HF"

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

basic_reward_model_generation_config: VllmConfig = {
    "backend": "vllm",
    "model_name": REWARD_MODEL_NAME,
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
        "tensor_parallel_size": 8,
        "pipeline_parallel_size": 1,
        "max_model_len": 1024,
        "disable_log_stats": True,
        "disable_log_requests": True,
        "gpu_memory_utilization": 0.6,
        "enforce_eager": "False",
        "task": "score",
    },
    "colocated": {
        "enabled": False,
        "resources": {
            "gpus_per_node": None,
            "num_nodes": None,
        },
    },
}

basic_env_config: RewardModelEnvironmentConfig = {
    "enabled": True,
    "model_name": REWARD_MODEL_NAME,
    "precision": "bfloat16",
    "batch_size": 1,
    "checkpoint_path": None,
    "logprob_batch_size": 1,
    "resources": {"gpus_per_node": 8, "num_nodes": 1},
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
    "generation": basic_reward_model_generation_config,
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

    message_log_batch: list[LLMMessageLogType] = [
        [
            {
                "role": "user",
                "content": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nThink step-by-step to solve the following problem. Output your answer inside of \\\\boxed{} tags.:\n[{'content': 'list all types of itag used by pytube library for specific video streams.', 'role': 'user'}]\n\nLet's think step-by-step<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            },
            {
                "role": "assistant",
                "content": "To solve this problem, we'll think step-by-step as follows:\n\n1. **Identify the problem**: The problem is asking to list all types of itag used by the pytube library for specific video streams.\n\n2. **Understand the context**: We know that pytube is a Python library used for downloading YouTube videos. It uses itags to refer to different types of video streams.\n\n3. **Locate the relevant information**: We'll need to use the pytube library documentation to find a list of itags supported by the library.\n\n4. **List the itags**: Unfortunately, the pytube library is built on top of the YouTube API, and it does not provide an official list of itags supported by the library. However, we can find this information by searching the library's code or online resources that have extracted this information.\n\nHowever, according to the unofficial documentation of the library it seems pytube works with the following itags: '410', '137', and '130' of the 1080p streams, then '136', and '298' of the 720p streams and lastly, the most basic '133' and '134', '135'.\n\nSo, based on my initial search, here are some results.\n\n \\\\boxed{We've found that pytube library works with itags '410', '137', '130' for 1080p streams; '136', '298' for 720p streams; and '133', '134', '135' for basic quality.}",
            },
        ]
    ]

    output = reward_model_env.step(message_log_batch, [])
    print(output)
