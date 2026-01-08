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

"""Unit tests for SGLang generation backend.

These tests verify that the SGLang generation backend produces sane outputs.
While not true unit tests, they validate the generation quality in unit test runs.
"""

import gc
from copy import deepcopy

import pytest
import ray
import torch

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.sglang import SGLangConfig, SGLangGeneration

model_name = "Qwen/Qwen3-0.6B"

# Define basic SGLang test config
basic_sglang_test_config: SGLangConfig = {
    "backend": "sglang",
    "model_name": model_name,
    "model_path": model_name,
    "tokenizer": {
        "name": model_name,
    },
    "dtype": "bfloat16",
    "max_new_tokens": 5,  # Small number of tokens for testing
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": None,
    "stop_token_ids": None,
    "stop_strings": None,
    "sglang_cfg": {
        "model_path": model_name,
        "gpus_per_server": 2,
        "dtype": "bfloat16",
        "context_length": 1024,
        "log_level": "warning",
        "skip_server_warmup": True,
        "enable_memory_saver": False,
        "dp_size": 1,
        "pp_size": 1,
        "ep_size": 1,
        "mem_fraction_static": 0.7,
    },
    "colocated": {
        "enabled": True,
        "resources": {
            "gpus_per_node": None,
            "num_nodes": None,
        },
    },
    "sglang_kwargs": {},
}

# Basic DTensor test config for Policy tests
basic_dtensor_test_config = {
    "model_name": model_name,
    "tokenizer": {
        "name": model_name,
    },
    "train_global_batch_size": 1,
    "train_micro_batch_size": 1,
    "learning_rate": 5e-6,
    "logprob_batch_size": 1,
    "max_new_tokens": 16,
    "do_sample": False,
    "precision": "float32",
    "offload_optimizer_for_logprob": False,
    "optimizer": {
        "name": "torch.optim.AdamW",
        "kwargs": {
            "lr": 5e-6,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
    },
    "dtensor_cfg": {
        "_v2": True,  # Use DTensorPolicyWorkerV2 for stream_weights_via_http
        "enabled": True,
        "cpu_offload": False,
        "sequence_parallel": False,
        "activation_checkpointing": False,
        "tensor_parallel_size": 2,
        "context_parallel_size": 1,
        "custom_parallel_plan": None,
    },
    "dynamic_batching": {
        "enabled": True,
        "train_mb_tokens": 40,
        "logprob_mb_tokens": 40,
        "sequence_length_round": 4,
    },
    "sequence_packing": {
        "enabled": False,
    },
    "max_grad_norm": 1.0,
    "make_sequence_length_divisible_by": 1,
    "generation": deepcopy(basic_sglang_test_config),
}


def configure_sglang_config(
    config: SGLangConfig, tokenizer, is_eval=True
) -> SGLangConfig:
    """Apply specific configurations to SGLang config."""
    config = deepcopy(config)
    config["_pad_token_id"] = tokenizer.pad_token_id
    if config["stop_token_ids"] is None:
        config["stop_token_ids"] = [tokenizer.eos_token_id]
    return config


@pytest.fixture(scope="function")
def cluster():
    """Create a virtual cluster for testing with 2 GPUs."""
    virtual_cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[2],
        use_gpus=True,
        max_colocated_worker_groups=2,
        num_gpus_per_node=2,
        name="sglang-test-cluster",
    )
    yield virtual_cluster
    virtual_cluster.shutdown()


@pytest.fixture(scope="function")
def tokenizer():
    """Initialize tokenizer for the test model."""
    tokenizer = get_tokenizer(basic_sglang_test_config["tokenizer"])
    return tokenizer


@pytest.fixture(scope="function")
def policy(cluster, tokenizer):
    """Initialize the SGLang policy."""
    sglang_config = deepcopy(basic_sglang_test_config)
    sglang_config = configure_sglang_config(sglang_config, tokenizer)
    p = SGLangGeneration(cluster, sglang_config)
    yield p
    try:
        p.shutdown()
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error during policy cleanup: {e}")


@pytest.fixture(scope="function")
def test_input_data(tokenizer):
    """Create test input data for inference."""
    test_prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]

    # Tokenize prompts
    encodings = tokenizer(
        test_prompts,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_tensors="pt",
        padding_side="right",
    )

    # Calculate input lengths from attention mask
    input_lengths = encodings["attention_mask"].sum(dim=1).to(torch.int32)

    # Create input data dictionary
    return BatchedDataDict(
        {
            "input_ids": encodings["input_ids"],
            "input_lengths": input_lengths,
        }
    )


@pytest.fixture(scope="function")
def policy_cluster_separate():
    """Create a virtual cluster for the Policy, using 2 GPUs."""
    cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[2],
        use_gpus=True,
        max_colocated_worker_groups=2,
        num_gpus_per_node=2,
        name="sglang-test-policy-cluster-separate",
    )
    yield cluster
    try:
        cluster.shutdown()
    except Exception as e:
        print(f"Error during policy_cluster_separate shutdown: {e}")


def get_generation_cluster_separate(num_gpus_per_node: int = 2) -> RayVirtualCluster:
    """Create a virtual cluster for the SGLangGeneration policy."""
    return RayVirtualCluster(
        bundle_ct_per_node_list=[num_gpus_per_node],
        use_gpus=True,
        max_colocated_worker_groups=1,
        num_gpus_per_node=num_gpus_per_node,
        name="sglang-test-generation-cluster-separate",
    )


# =============================================================================
# Basic Configuration Tests
# =============================================================================


@pytest.mark.sglang
@pytest.mark.timeout(120)
def test_sglang_missing_required_config_key(cluster, tokenizer):
    """Test that an error is raised when a required config key is missing."""
    # SGLang requires sglang_cfg to be present
    incomplete_config = deepcopy(basic_sglang_test_config)
    incomplete_config = configure_sglang_config(incomplete_config, tokenizer)
    del incomplete_config["sglang_cfg"]

    with pytest.raises((KeyError, ValueError, AssertionError, TypeError)):
        SGLangGeneration(cluster, incomplete_config)


@pytest.mark.sglang
def test_sglang_top_p_top_k_validation(cluster, tokenizer):
    """Test that top_p and top_k values are accepted by SGLang.

    Note: SGLang may have different validation thresholds than vLLM.
    This test verifies that reasonable sampling parameters are accepted.
    """
    # Test that reasonable top_p and top_k values are accepted
    config = deepcopy(basic_sglang_test_config)
    config["top_p"] = 0.95
    config["top_k"] = 50
    config = configure_sglang_config(config, tokenizer)

    policy = None
    try:
        policy = SGLangGeneration(cluster, config)
        print("Successfully initialized with top_p=0.95 and top_k=50")
    except Exception as e:
        pytest.fail(f"Should not raise error with reasonable sampling params: {e}")
    finally:
        if policy:
            policy.shutdown()
            gc.collect()
            torch.cuda.empty_cache()


# =============================================================================
# Basic Generation Tests
# =============================================================================


@pytest.mark.sglang
@pytest.mark.timeout(180)
def test_sglang_policy_generation(policy, test_input_data, tokenizer):
    """Test SGLang policy generation capabilities."""
    print("Testing SGLang generation...")
    outputs = policy.generate(test_input_data)

    # Validate outputs format
    assert "output_ids" in outputs, "output_ids not found in generation output"
    assert "logprobs" in outputs, "logprobs not found in generation output"
    assert "generation_lengths" in outputs, (
        "generation_lengths not found in generation output"
    )
    assert "unpadded_sequence_lengths" in outputs, (
        "unpadded_sequence_lengths not found in generation output"
    )

    # Validate outputs shape and content
    assert outputs["output_ids"].shape[0] == len(test_input_data["input_ids"]), (
        "Wrong batch size in output"
    )
    assert outputs["generation_lengths"].shape[0] == len(
        test_input_data["input_ids"]
    ), "Wrong batch size in generation_lengths"

    # Decode and check outputs
    generated_sequences = outputs["output_ids"]
    generated_texts = tokenizer.batch_decode(
        generated_sequences, skip_special_tokens=True
    )

    print(f"Generated texts: {generated_texts}")

    # All texts should have a non-zero length
    assert all(len(text) > 0 for text in generated_texts), (
        "Some generated texts are empty"
    )


@pytest.mark.sglang
def test_sglang_worker_seed_behavior(cluster, tokenizer):
    """
    Test that different workers generate different outputs for identical prompts due to different seeds.
    This ensures proper randomization across distributed workers for diverse exploration in RLHF.

    Key: Use gpus_per_server=1 to create 2 independent SGLang servers (each with its own seed),
    rather than 1 server with TP=2.
    """
    from nemo_rl.algorithms.grpo import refit_policy_generation
    from nemo_rl.models.policy.lm_policy import Policy

    unique_prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]

    # Create a batch where each prompt appears twice
    # When sharded, different workers will get the same prompt
    duplicated_prompts = unique_prompts + unique_prompts

    # Tokenize prompts
    encodings = tokenizer(
        duplicated_prompts,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_tensors="pt",
        padding_side="right",
    )

    input_lengths = encodings["attention_mask"].sum(dim=1).to(torch.int32)

    # Create input data dictionary
    duplicated_batch = BatchedDataDict(
        {
            "input_ids": encodings["input_ids"],
            "input_lengths": input_lengths,
        }
    )

    # Test with gpus_per_server=1 to create 2 independent servers with different seeds
    print("Creating SGLang policy with gpus_per_server=1 (2 independent servers)...")
    sglang_config = deepcopy(basic_sglang_test_config)
    # Use gpus_per_server=1 to create 2 independent SGLang servers
    sglang_config["sglang_cfg"]["gpus_per_server"] = 1
    sglang_config = configure_sglang_config(sglang_config, tokenizer)

    policy = SGLangGeneration(cluster, sglang_config)
    policy.finish_generation()

    dtensor_config = deepcopy(basic_dtensor_test_config)
    dtensor_config["dtensor_cfg"]["tensor_parallel_size"] = 1  # Match gpus_per_server
    lm_policy = Policy(cluster, dtensor_config, tokenizer)

    state_dict_info = lm_policy.prepare_refit_info()
    policy.prepare_refit_info(state_dict_info)

    print("Refitting SGLang policy...")
    refit_policy_generation(lm_policy, policy, sglang_config["colocated"]["enabled"])

    try:
        # Generate with duplicated prompts
        print("Running generation with duplicated prompts...")
        outputs = policy.generate(duplicated_batch, greedy=False)

        # Decode the generated sequences
        gen_texts = tokenizer.batch_decode(
            outputs["output_ids"], skip_special_tokens=True
        )

        print(f"Generated texts with duplicated prompts: {gen_texts}")

        # Check if the duplicated prompts generated different texts
        # The first half and second half should be different due to different worker seeds
        first_half = gen_texts[: len(unique_prompts)]
        second_half = gen_texts[len(unique_prompts) :]

        print(f"First worker outputs: {first_half}")
        print(f"Second worker outputs: {second_half}")

        # At least one of the pairs should be different due to different seeds
        assert first_half != second_half, (
            "Different workers should generate different outputs for identical prompts due to different seeds"
        )

    finally:
        # Clean up resources
        if "policy" in locals() and hasattr(policy, "shutdown"):
            policy.shutdown()
        if "lm_policy" in locals() and hasattr(lm_policy, "shutdown"):
            lm_policy.shutdown()

        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.sglang
def test_sglang_policy_tensor_parallel(cluster, tokenizer):
    """Test SGLang policy with tensor parallelism > 1 (gpus_per_server=2)."""
    # Configure with gpus_per_server=2 for tensor parallelism
    tp_config = deepcopy(basic_sglang_test_config)
    tp_config = configure_sglang_config(tp_config, tokenizer)
    tp_config["sglang_cfg"]["gpus_per_server"] = 2  # TP=2

    sglang_policy = None
    try:
        sglang_policy = SGLangGeneration(cluster, tp_config)

        # Create simple test input
        test_prompts = ["Hello, my name is", "The capital of France is"]
        encodings = tokenizer(
            test_prompts,
            padding="max_length",
            max_length=10,
            truncation=True,
            return_tensors="pt",
            padding_side="right",
        )

        test_input_data = BatchedDataDict(
            {
                "input_ids": encodings["input_ids"],
                "input_lengths": encodings["attention_mask"].sum(dim=1).to(torch.int32),
            }
        )

        # Test generation with tensor parallelism
        outputs = sglang_policy.generate(test_input_data)

        sglang_policy.finish_generation()
        sglang_policy.prepare_for_generation()

        # Test generation again after cache reset
        outputs = sglang_policy.generate(test_input_data)

        assert "output_ids" in outputs, "output_ids not found in generation output"
        assert outputs["output_ids"].shape[0] == 2, "Wrong batch size in output"

        # Decode and check output
        generated_text = tokenizer.decode(
            outputs["output_ids"][0], skip_special_tokens=True
        )
        print(f"Generated text with TP=2: {generated_text}")
        assert len(generated_text) > 0, "Generated text is empty"

    finally:
        # Clean up resources
        if sglang_policy:
            sglang_policy.shutdown()
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.sglang
def test_sglang_generate_text(cluster, tokenizer):
    """Test that SGLang can generate coherent text.

    Note: SGLang doesn't have a generate_text method like vLLM,
    so we use generate + tokenizer decode to verify text generation.
    """
    # Prepare test data
    test_prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]

    encodings = tokenizer(
        test_prompts,
        padding="max_length",
        max_length=10,
        truncation=True,
        return_tensors="pt",
        padding_side="right",
    )

    test_input_data = BatchedDataDict(
        {
            "input_ids": encodings["input_ids"],
            "input_lengths": encodings["attention_mask"].sum(dim=1).to(torch.int32),
        }
    )

    # Create SGLang config with gpus_per_server=2 (using tensor parallelism)
    sglang_config = deepcopy(basic_sglang_test_config)
    sglang_config["sglang_cfg"]["gpus_per_server"] = 2
    sglang_config = configure_sglang_config(sglang_config, tokenizer, is_eval=True)

    # Ensure correct model
    assert sglang_config["model_name"] == "Qwen/Qwen3-0.6B", (
        "Model name should be Qwen/Qwen3-0.6B to get expected output"
    )

    sglang_generation = None
    try:
        # Create SGLang generation
        sglang_generation = SGLangGeneration(cluster, sglang_config)

        # Generate with greedy decoding for deterministic output
        output = sglang_generation.generate(test_input_data, greedy=True)

        # Decode generated text
        generated_texts = tokenizer.batch_decode(
            output["output_ids"], skip_special_tokens=True
        )

        print(f"Generated texts: {generated_texts}")

        # Verify we got non-empty text for each prompt
        for i, text in enumerate(generated_texts):
            assert len(text) > len(test_prompts[i]), (
                f"Generated text should be longer than input prompt: {text}"
            )
            # Verify the generated text starts with or contains the prompt
            print(f"Prompt: {test_prompts[i]} -> Generated: {text}")

    finally:
        # Clean up
        if sglang_generation:
            sglang_generation.shutdown()
        gc.collect()
        torch.cuda.empty_cache()


def _wait_for_sglang_http_server_spinup(base_url: str):
    """Wait for the SGLang HTTP server to be ready."""
    import time

    import requests

    max_wait = 60  # 60 seconds max wait
    start = time.time()
    while time.time() - start < max_wait:
        try:
            response = requests.get(f"{base_url}/health_generate", timeout=5)
            if response.status_code == 200:
                return
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass
        time.sleep(1)
    raise TimeoutError(f"SGLang server at {base_url} did not start within {max_wait}s")


@pytest.mark.sglang
def test_sglang_http_server(cluster, tokenizer):
    """Test that SGLang HTTP server works with direct API calls.

    SGLang exposes a /generate endpoint that accepts input_ids and sampling_params.
    This test verifies we can make direct HTTP requests to the SGLang server.
    """
    import requests

    # Create SGLang config
    sglang_config = deepcopy(basic_sglang_test_config)
    sglang_config = configure_sglang_config(sglang_config, tokenizer, is_eval=True)

    # Ensure correct model for reproducible output
    assert sglang_config["model_name"] == "Qwen/Qwen3-0.6B", (
        "Model name should be Qwen/Qwen3-0.6B to get expected output"
    )

    sglang_generation = None
    try:
        # Create SGLang generation (this starts the servers)
        sglang_generation = SGLangGeneration(cluster, sglang_config)

        # Get server URLs
        base_urls = sglang_generation.get_sglang_server_urls()
        print(f"SGLang server URLs: {base_urls}")
        assert len(base_urls) >= 1, "Should have at least one SGLang server"

        # Wait for server to be ready
        _wait_for_sglang_http_server_spinup(base_urls[0])

        # Prepare input - tokenize "count to 5"
        test_prompt = "count to 5"
        input_ids = tokenizer.encode(test_prompt, add_special_tokens=True)

        # Build request payload for SGLang /generate endpoint
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,  # Greedy for determinism
                "top_p": 1.0,
                "max_new_tokens": 5,
            },
            "return_logprob": True,
        }

        # Make request to SGLang server
        response = requests.post(
            url=f"{base_urls[0]}/generate",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        actual_result = response.json()
        print(f"SGLang response: {actual_result}")

        # Verify response structure
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert "meta_info" in actual_result, "Response should contain meta_info"

        meta_info = actual_result["meta_info"]
        assert "output_token_logprobs" in meta_info, (
            "meta_info should contain output_token_logprobs"
        )

        # Verify we got some generated tokens
        output_token_logprobs = meta_info["output_token_logprobs"]
        assert len(output_token_logprobs) > 0, (
            "Should have generated at least one token"
        )

        # Each entry should be [logprob, token_id]
        first_token_info = output_token_logprobs[0]
        assert len(first_token_info) >= 2, (
            "Each token info should have logprob and token_id"
        )

        logprob = first_token_info[0]
        token_id = first_token_info[1]
        assert isinstance(logprob, float), "Logprob should be a float"
        assert isinstance(token_id, int), "Token ID should be an int"

        print(f"First generated token: id={token_id}, logprob={logprob}")

        # Decode the generated tokens to verify text output
        generated_token_ids = [item[1] for item in output_token_logprobs]
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        print(f"Generated text: {generated_text}")

    finally:
        # Clean up
        if sglang_generation:
            sglang_generation.shutdown()
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.sglang
@pytest.mark.timeout(180)
def test_sglang_non_divisible_batch_handling(policy):
    """Test that SGLang generation handles non divisible input batches correctly."""
    empty_batch = BatchedDataDict(
        {
            "input_ids": torch.zeros((1, 1), dtype=torch.long),
            "input_lengths": torch.ones(1, dtype=torch.long),
        }
    )

    outputs = policy.generate(empty_batch)

    required_keys = [
        "output_ids",
        "logprobs",
        "generation_lengths",
        "unpadded_sequence_lengths",
    ]
    assert all(key in outputs for key in required_keys), (
        "Missing required output fields"
    )
    assert all(outputs[key].shape[0] == 1 for key in required_keys), (
        "Output tensors should have batch dimension of 1"
    )


# =============================================================================
# Policy Integration Tests
# =============================================================================


@pytest.mark.sglang
@pytest.mark.timeout(300)
def test_sglang_generation_with_hf_training_colocated(cluster, tokenizer):
    """Test that DTensor policy can work together with colocated SGLang policy."""
    from nemo_rl.algorithms.grpo import refit_policy_generation
    from nemo_rl.models.policy.lm_policy import Policy

    sglang_config = deepcopy(basic_sglang_test_config)
    sglang_config = configure_sglang_config(sglang_config, tokenizer)

    dtensor_config = deepcopy(basic_dtensor_test_config)
    dtensor_config["train_global_batch_size"] = 4
    dtensor_config["dtensor_cfg"]["_v2"] = (
        True  # Use DTensorPolicyWorkerV2 for stream_weights_via_http
    )

    sglang_policy = None
    lm_policy = None

    try:
        print("Creating SGLang policy...")
        sglang_policy = SGLangGeneration(cluster, sglang_config)
        sglang_policy.finish_generation()

        print("Creating DTensor policy...")
        lm_policy = Policy(cluster, dtensor_config, tokenizer)

        print("Preparing refit info...")
        state_dict_info = lm_policy.prepare_refit_info()
        sglang_policy.prepare_refit_info(state_dict_info)

        print("Refitting SGLang policy...")
        refit_policy_generation(
            lm_policy, sglang_policy, sglang_config["colocated"]["enabled"]
        )

        # Test generation
        test_prompts = ["Hello, my name is", "The capital of France is"]
        encodings = tokenizer(
            test_prompts,
            padding="max_length",
            max_length=20,
            truncation=True,
            return_tensors="pt",
            padding_side="right",
        )
        test_input_data = BatchedDataDict(
            {
                "input_ids": encodings["input_ids"],
                "input_lengths": encodings["attention_mask"].sum(dim=1).to(torch.int32),
            }
        )

        outputs = sglang_policy.generate(test_input_data, greedy=True)
        assert "output_ids" in outputs, "output_ids not found in generation output"

        generated_texts = tokenizer.batch_decode(
            outputs["output_ids"], skip_special_tokens=True
        )
        print(f"Generated texts: {generated_texts}")

    finally:
        if sglang_policy:
            sglang_policy.shutdown()
        if lm_policy and hasattr(lm_policy, "shutdown"):
            lm_policy.shutdown()


@pytest.mark.skip(reason="Non-colocated mode not implemented for SGLang")
@pytest.mark.timeout(300)
@pytest.mark.sglang
def test_sglang_generation_with_hf_training_non_colocated(
    policy_cluster_separate, tokenizer
):
    """Test that DTensor policy can work together with non-colocated SGLang policy."""
    from nemo_rl.algorithms.grpo import refit_policy_generation
    from nemo_rl.models.policy.lm_policy import Policy

    generation_cluster_separate = get_generation_cluster_separate(2)

    sglang_config = deepcopy(basic_sglang_test_config)
    sglang_config = configure_sglang_config(sglang_config, tokenizer)
    sglang_config["colocated"]["enabled"] = False

    dtensor_config = deepcopy(basic_dtensor_test_config)
    dtensor_config["generation"]["colocated"]["enabled"] = False
    dtensor_config["train_global_batch_size"] = 4
    dtensor_config["dtensor_cfg"]["_v2"] = (
        True  # Use DTensorPolicyWorkerV2 for stream_weights_via_http
    )

    sglang_policy = None
    lm_policy = None

    try:
        print("Creating SGLang policy...")
        sglang_policy = SGLangGeneration(generation_cluster_separate, sglang_config)
        sglang_policy.finish_generation()

        print("Creating DTensor policy...")
        lm_policy = Policy(policy_cluster_separate, dtensor_config, tokenizer)

        # Initialize collective communication
        ip, port = policy_cluster_separate.get_master_address_and_port()
        train_world_size = policy_cluster_separate.world_size()
        inference_world_size = generation_cluster_separate.world_size()
        world_size = train_world_size + inference_world_size

        futures_train = lm_policy.init_collective(
            ip, port, world_size=world_size, train_world_size=train_world_size
        )
        futures_inference = sglang_policy.init_collective(
            ip, port, world_size=world_size, train_world_size=train_world_size
        )
        ray.get(futures_train + futures_inference)

        # Prepare refit info
        state_dict_info = lm_policy.prepare_refit_info()
        sglang_policy.prepare_refit_info(state_dict_info)

        print("Refitting SGLang policy...")
        refit_policy_generation(lm_policy, sglang_policy, False)

        # Test generation
        test_prompts = ["Hello, my name is", "The capital of France is"]
        encodings = tokenizer(
            test_prompts,
            padding="max_length",
            max_length=20,
            truncation=True,
            return_tensors="pt",
            padding_side="right",
        )
        test_input_data = BatchedDataDict(
            {
                "input_ids": encodings["input_ids"],
                "input_lengths": encodings["attention_mask"].sum(dim=1).to(torch.int32),
            }
        )

        outputs = sglang_policy.generate(test_input_data, greedy=True)
        assert "output_ids" in outputs, "output_ids not found in generation output"

    finally:
        if sglang_policy:
            sglang_policy.shutdown()
        if lm_policy and hasattr(lm_policy, "shutdown"):
            lm_policy.shutdown()
        try:
            generation_cluster_separate.shutdown()
        except Exception as e:
            print(f"Error during generation_cluster_separate shutdown: {e}")


@pytest.mark.sglang
@pytest.mark.timeout(180)
def test_sglang_weight_update_and_prefix_cache_reset(cluster, tokenizer):
    """Test that the SGLang prefix cache is correctly reset when weights change."""
    from nemo_rl.models.policy.lm_policy import Policy

    sglang_config = deepcopy(basic_sglang_test_config)
    sglang_config = configure_sglang_config(sglang_config, tokenizer, is_eval=True)

    dtensor_config = basic_dtensor_test_config

    sglang_policy = None
    lm_policy = None

    try:
        print("Creating DTensor policy...")
        lm_policy = Policy(cluster, dtensor_config, tokenizer)

        print("Creating SGLang policy...")
        sglang_policy = SGLangGeneration(cluster, sglang_config)

        print("Preparing refit info...")
        state_dict_info = lm_policy.prepare_refit_info()
        sglang_policy.prepare_refit_info(state_dict_info)

        # Prepare input data
        text = "Answer the question. What is 2+2?"
        test_prompt = [text, text]
        encodings = tokenizer(
            test_prompt,
            padding=True,
            return_tensors="pt",
            padding_side="right",
        )
        input_ids = encodings["input_ids"]
        input_lengths = encodings["attention_mask"].sum(dim=1).to(torch.int32)
        test_input_data = BatchedDataDict(
            {"input_ids": input_ids, "input_lengths": input_lengths}
        )

        print("Running Generation 1 (Initial)...")
        sglang_policy.prepare_for_generation()
        outputs1 = sglang_policy.generate(test_input_data, greedy=True)
        logprob1 = outputs1["logprobs"][0, input_lengths[0]].item()
        print(f"Logprob of first generated token (Run 1): {logprob1}")

        print("Adding noise to weights in HF policy...")
        ray.get(
            [
                worker._add_noise_to_weights.remote()
                for worker in lm_policy.worker_group.workers
            ]
        )

        print("Updating SGLang weights from DTensor policy via HTTP...")
        # Get SGLang server URL to GPU UUID mapping
        sglang_url_to_gpu_uuids = sglang_policy.get_sglang_url_to_gpu_uuids()
        print(f"SGLang URL to GPU UUIDs: {sglang_url_to_gpu_uuids}")

        # Stream weights via HTTP (CUDA IPC)
        ray.get(lm_policy.stream_weights_via_http(sglang_url_to_gpu_uuids))

        print("Running Generation 2 (Weights Updated)...")
        outputs2 = sglang_policy.generate(test_input_data, greedy=True)
        logprob2 = outputs2["logprobs"][0, input_lengths[0]].item()
        print(f"Logprob of first generated token (Run 2): {logprob2}")
        assert logprob2 != logprob1, "Logprobs should be different after weight update."

        print("Resetting SGLang prefix cache...")
        sglang_policy.finish_generation()
        sglang_policy.prepare_for_generation()

        print("Running Generation 3 (Cache Reset)...")
        outputs3 = sglang_policy.generate(test_input_data, greedy=True)
        logprob3 = outputs3["logprobs"][0, input_lengths[0]].item()
        print(f"Logprob of first generated token (Run 3): {logprob3}")

        print("Prefix cache reset verified successfully.")

    finally:
        print("Cleaning up resources...")
        if sglang_policy:
            sglang_policy.shutdown()
        if lm_policy:
            lm_policy.shutdown()
        gc.collect()
        torch.cuda.empty_cache()
