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

from tempfile import TemporaryDirectory

import pytest
import ray
from transformers import AutoTokenizer

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.environments.code_environment import (
    CodeEnvConfig,
    CodeEnvironment,
    CodeEnvMetadata,
)
from nemo_rl.experience.rollouts import run_multi_turn_rollout
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration

MODEL_NAME = "meta-llama/Llama-3.2-1B"

cfg: CodeEnvConfig = {
    "num_workers": 2,
    "terminate_on_evaluation": True,
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
def code_env():
    """Create a code environment for testing."""
    try:
        env_actor = CodeEnvironment.remote(cfg)
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
    cluster_name = f"test-code-cluster-{id(cluster_instance)}"
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


def test_untrusted_code(code_env):
    """Test whether the code environment can block untrusted code."""
    codes = [
        "with open('allowed_file.txt', 'w') as fout:\n"
        "    fout.write('some content')\n"
        "with open('allowed_file.txt') as fin:\n"
        "    content = fin.read()\n"
        "content",
        "with open('/etc/passwd', 'r') as fin:\n    fin.read()",
        "import math\nround(math.sqrt(8))",
        "import os",
    ]
    results = [
        "\n\n<result>\n'some content'\n</result>",
        "\n\n<result>\nPermissionError('Access beyond the temporary working directory is blocked')\n</result>",
        "\n\n<result>\n3\n</result>",
        "<result>PermissionError('Importing system and network modules is blocked')</result>",
    ]

    message_log_batch = [
        [{"role": "user", "content": f"<code>{code}</code>"}] for code in codes
    ]
    temp_dirs = [TemporaryDirectory() for _ in codes]
    metadata_batch = [
        CodeEnvMetadata(
            context={},
            working_dir=temp_dir.name,
        )
        for temp_dir in temp_dirs
    ]

    # Execute the code
    output = ray.get(code_env.step.remote(message_log_batch, metadata_batch))
    responses = [obs["content"] for obs in output.observations]

    assert responses == results, f"Got wrong output {responses}"


@pytest.mark.hf_gated
def test_vllm_execute_code(cluster, tokenizer, code_env):
    """Test that vLLM can call the code executor."""
    # Prepare test data
    codes = [
        "<code>x = 3; y = 4</code>\nThis is some regular text.\n<code>x + y</code>\n",
        "<code>\ndef f(x):\n    return x * x\n\nf(2)\n</code>\n",
    ]
    results = ["<result>7</result>", "\n<result>\n4\n</result>"]

    # Create message logs
    message_logs = []
    metadata_batch = []
    temp_dirs = []
    for code in codes:
        # Tokenize the message content
        prompt = code * 4
        token_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ][0]
        temp_dir = TemporaryDirectory()
        message_logs.append(
            [{"role": "user", "content": prompt, "token_ids": token_ids}]
        )
        metadata_batch.append(CodeEnvMetadata(context={}, working_dir=temp_dir.name))
        temp_dirs.append(temp_dir)

    # Create initial batch
    initial_batch = BatchedDataDict(
        {
            "message_log": message_logs,
            "extra_env_info": metadata_batch,
            "task_name": ["code_execution"] * len(codes),
            "stop_strings": [["</code>"]] * len(codes),
        }
    )

    # Create vLLM generation
    vllm_config = basic_vllm_test_config.copy()
    vllm_config = configure_generation_config(vllm_config, tokenizer, is_eval=True)
    vllm_generation = VllmGeneration(cluster, vllm_config)

    # Create code environment
    task_to_env = {"code_execution": code_env}

    # Run rollout
    vllm_generation.prepare_for_generation()
    final_batch, _ = run_multi_turn_rollout(
        policy_generation=vllm_generation,
        input_batch=initial_batch,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        max_seq_len=256,
        max_rollout_turns=2,
        greedy=True,
    )
    vllm_generation.finish_generation()

    # Check results
    for i, msg_log in enumerate(final_batch["message_log"]):
        # Get the last message which should contain the result
        last_msg = msg_log[-1]
        assert last_msg["role"] == "environment"
        assert last_msg["content"] == results[i], (
            f"Expected {results[i]}, got {last_msg['content']}"
        )
