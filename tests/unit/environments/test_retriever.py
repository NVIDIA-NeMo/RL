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

import pytest
import ray
from transformers import AutoTokenizer

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.environments.tools.retriever import RAGEnvConfig, RAGEnvironment
from nemo_rl.experience.rollouts import run_multi_turn_rollout
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration

MODEL_NAME = "meta-llama/Llama-3.2-1B"

cfg: RAGEnvConfig = {
    "dataset_name": "rahular/simple-wikipedia",
    "dataset_split": "train",
    "text_column": "text",
    "num_results": 1,
    "k1": 1.5,
    "b": 0.75,
    "device": "cpu",
}

# Define basic vLLM test config
basic_vllm_test_config: VllmConfig = {
    "backend": "vllm",
    "model_name": MODEL_NAME,
    "tokenizer_name": None,
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


@pytest.fixture(scope="function")
def rag_env():
    """Create a RAG environment for testing."""
    try:
        env_actor = RAGEnvironment.remote(cfg)
        yield env_actor
    finally:
        if env_actor:
            ray.kill(env_actor)


@pytest.fixture(scope="function")
def tokenizer():
    """Loads the tokenizer for the tests."""
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(
        f"Tokenizer loaded. Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id}), EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})"
    )
    return tokenizer


@pytest.fixture(scope="function")
def cluster():
    """Create a virtual cluster for testing."""
    cluster_instance = None
    cluster_name = f"test-rag-cluster-{id(cluster_instance)}"
    print(f"\nCreating virtual cluster '{cluster_name}'...")
    try:
        cluster_instance = RayVirtualCluster(
            name=cluster_name,
            bundle_ct_per_node_list=[1],
            use_gpus=True,
            num_gpus_per_node=1,
            max_colocated_worker_groups=2,
        )
        yield cluster_instance
    finally:
        print(f"\nCleaning up cluster '{cluster_name}'...")
        if cluster_instance:
            cluster_instance.shutdown()


@pytest.mark.hf_gated
def test_vllm_retrieve(cluster, tokenizer, rag_env):
    """Test that vLLM can use the RAG environment for document retrieval."""
    # Prepare test data
    queries = [
        "<retrieve>Jen-Hsun Huang</retrieve>\n",
    ]
    expected_results = [
        "<result>\n<1>\n"
        "Nvidia was established in 1993 by Jen-Hsun Huang, Curtis Priem, and Chris Malachowsky. In 2000 Nvidia took intellectual possession of 3dfx, one of the biggest GPU producers in 1990s.\n"
        "</1>\n</result>\n",
    ]

    # Create message logs
    message_logs = []
    for query in queries:
        # Tokenize the message content
        prompt = query * 4
        token_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ][0]
        message_logs.append(
            [{"role": "user", "content": prompt, "token_ids": token_ids}]
        )

    # Create initial batch
    initial_batch = BatchedDataDict(
        {
            "message_log": message_logs,
            "extra_env_info": [{}] * len(queries),  # No metadata needed for RAG
            "task_name": ["document_retrieval"] * len(queries),
            "stop_strings": [["</retrieve>"]] * len(queries),
        }
    )

    # Create vLLM generation
    vllm_config = basic_vllm_test_config.copy()
    vllm_config = configure_generation_config(vllm_config, tokenizer, is_eval=True)
    vllm_generation = VllmGeneration(cluster, vllm_config)

    # Create RAG environment
    task_to_env = {"document_retrieval": rag_env}

    # Run rollout
    vllm_generation.prepare_for_generation()
    final_batch, _ = run_multi_turn_rollout(
        policy_generation=vllm_generation,
        input_batch=initial_batch,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        max_seq_len=256,
        max_rollout_turns=1,
        greedy=True,
    )
    vllm_generation.finish_generation()

    # Check results
    for i, msg_log in enumerate(final_batch["message_log"]):
        # Get the last message which should contain the result
        last_msg = msg_log[-1]
        assert last_msg["role"] == "environment"
        assert last_msg["content"] == expected_results[i], (
            f"Expected {expected_results[i]}, got {last_msg['content']}"
        )
