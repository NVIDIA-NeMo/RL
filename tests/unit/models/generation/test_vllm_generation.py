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
from copy import deepcopy
from unittest import mock

import pytest
import ray
import torch

from nemo_rl.algorithms.grpo import refit_policy_generation
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.hf_policy import HfPolicy

model_name = "Qwen/Qwen3-0.6B"
# Define basic vLLM test config
basic_vllm_test_config: VllmConfig = {
    "backend": "vllm",
    "model_name": model_name,
    "tokenizer": {
        "name": model_name,
    },
    "dtype": "bfloat16",
    "max_new_tokens": 5,
    "temperature": 0.8,
    "top_p": 1.0,
    "top_k": None,
    "stop_token_ids": None,
    "stop_strings": None,
    "vllm_cfg": {
        "precision": "bfloat16",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "gpu_memory_utilization": 0.7,
        "max_model_len": 1024,
        "async_engine": False,  # Default to False for synchronous tests
        "skip_tokenizer_init": False,
        "load_format": "auto",
    },
    "vllm_kwargs": {},
}


def get_basic_hf_test_config(enable_dtensor: bool = False) -> PolicyConfig:
    # Create HF-specific config with required parameters
    return {
        "model_name": basic_vllm_test_config["model_name"],
        "tokenizer": {
            "name": basic_vllm_test_config["tokenizer"]["name"],
        },
        # Required training parameters
        "train_global_batch_size": 1,
        "train_micro_batch_size": 1,
        "learning_rate": 5e-6,
        "logprob_batch_size": 1,
        "max_new_tokens": 16,
        "do_sample": False,
        "precision": "float32",
        "fsdp_offload_enabled": False,
        "activation_checkpointing_enabled": False,
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
            "enabled": enable_dtensor,
            "cpu_offload": False,
            "sequence_parallel": False,
            "activation_checkpointing": False,
            "tensor_parallel_size": 1,
            "custom_parallel_plan": None,
        },
        "dynamic_batching": {
            "enabled": enable_dtensor,  # Dynamic batching is only supported with DTensor
            "train_mb_tokens": 40,
            "logprob_mb_tokens": 40,
            "sequence_length_round": 4,
        },
        "max_grad_norm": 1.0,
        "make_sequence_length_divisible_by": 1,
        "generation": {
            "temperature": 0.8,
        },
    }


@pytest.fixture(scope="function")
def cluster():
    """Create a virtual cluster for testing."""
    # Create a cluster with 1 node that has 2 GPU bundles
    virtual_cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[2],  # 1 node with 2 GPU bundle
        use_gpus=True,
        max_colocated_worker_groups=2,
        num_gpus_per_node=2,  # Use available GPUs
        name="vllm-test-cluster",
    )
    yield virtual_cluster
    virtual_cluster.shutdown()


@pytest.fixture(scope="function")
def tokenizer():
    """Initialize tokenizer for the test model."""
    tokenizer = get_tokenizer(basic_vllm_test_config["tokenizer"])
    return tokenizer


@pytest.fixture(scope="function")
def policy(cluster, tokenizer):
    """Initialize the vLLM policy (synchronous by default)."""
    vllm_config = deepcopy(basic_vllm_test_config)
    # Ensure async_engine is False for the standard policy fixture
    vllm_config["vllm_cfg"]["async_engine"] = False
    vllm_config = configure_generation_config(vllm_config, tokenizer)
    p = VllmGeneration(cluster, vllm_config)
    yield p
    try:
        p.shutdown()
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error during policy cleanup: {e}")


def _create_ray_virtual_cluster_for_test(name: str) -> RayVirtualCluster:
    """Helper function to create a standard RayVirtualCluster for tests."""
    return RayVirtualCluster(
        bundle_ct_per_node_list=[1],
        use_gpus=True,
        max_colocated_worker_groups=1,
        num_gpus_per_node=1,
        name=name,
    )


@pytest.fixture(scope="function")
def policy_cluster_separate():
    """Create a virtual cluster for the HfPolicy, using 1 GPU."""
    cluster = _create_ray_virtual_cluster_for_test("vllm-test-policy-cluster-separate")
    yield cluster
    try:
        cluster.shutdown()
    except Exception as e:
        print(f"Error during policy_cluster_separate shutdown: {e}")


@pytest.fixture(scope="function")
def generation_cluster_separate():
    """Create a virtual cluster for the VllmGeneration policy, using 1 GPU."""
    cluster = _create_ray_virtual_cluster_for_test(
        "vllm-test-generation-cluster-separate"
    )
    yield cluster
    try:
        cluster.shutdown()
    except Exception as e:
        print(f"Error during generation_cluster_separate shutdown: {e}")


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


@pytest.fixture(scope="module", autouse=True)
def skip_tied_weight_check_for_all():
    """Automatically skip tied weight check for all tests in this module."""
    os.environ["NRL_SKIP_TIED_WEIGHT_CHECK"] = "1"

    yield

    # Restore the original value
    os.environ.pop("NRL_SKIP_TIED_WEIGHT_CHECK", None)


def test_vllm_missing_required_config_key(cluster):
    """Test that an assertion error is raised when a required config key is missing."""
    # Create a config missing a required key by removing 'model_name'
    incomplete_config = basic_vllm_test_config.copy()
    del incomplete_config["model_name"]  # Remove a required key

    # Also need to ensure skip_tokenizer_init and load_format are there
    # since these are checked in VllmConfig.__annotations__
    incomplete_config["skip_tokenizer_init"] = True
    incomplete_config["load_format"] = "auto"

    # Attempt to initialize VllmGeneration with incomplete config - should raise AssertionError
    with pytest.raises(AssertionError) as excinfo:
        VllmGeneration(cluster, incomplete_config)

    # Verify the error message contains information about the missing key
    error_message = str(excinfo.value)
    assert "Missing required keys in VllmConfig" in error_message
    assert "model_name" in error_message, (
        "Error should mention the missing 'model_name' key"
    )
    print(f"Successfully caught missing config key with error: {error_message}")


def test_vllm_policy_generation(policy, test_input_data, tokenizer):
    """Test vLLM policy generation capabilities."""
    # Test generation
    print("Testing generation...")
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

    # All texts should have a non-zero length and be longer than inputs
    assert all(len(text) > 0 for text in generated_texts), (
        "Some generated texts are empty"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tensor_parallel_size,pipeline_parallel_size", [(2, 1), (1, 2)]
)
async def test_vllm_policy_generation_async(
    cluster, test_input_data, tokenizer, tensor_parallel_size, pipeline_parallel_size
):
    """Test vLLM policy async generation capabilities."""
    # Ensure the policy is configured for async generation
    # Create separate configs for each policy
    hf_policy = None
    async_policy = None
    try:
        vllm_config = deepcopy(basic_vllm_test_config)
        vllm_config["vllm_cfg"]["async_engine"] = True
        vllm_config = configure_generation_config(vllm_config, tokenizer)
        vllm_config["vllm_cfg"]["tensor_parallel_size"] = tensor_parallel_size
        vllm_config["vllm_cfg"]["pipeline_parallel_size"] = pipeline_parallel_size
        hf_config = get_basic_hf_test_config(enable_dtensor=True)
        from nemo_rl.models.policy.hf_policy import HfPolicy

        async_policy = VllmGeneration(cluster, vllm_config)
        async_policy.finish_generation()
        print("creating hf policy...")

        hf_policy = HfPolicy(cluster, hf_config, tokenizer)
        refit_policy_generation(hf_policy, async_policy)

        outputs = async_policy.generate_async(test_input_data)
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

        # All texts should have a non-zero length and be longer than inputs
        assert all(len(text) > 0 for text in generated_texts), (
            "Some generated texts are empty"
        )

    finally:
        # Clean up resources
        print("Cleaning up resources...")
        if async_policy:
            async_policy.shutdown()
        if hf_policy and hasattr(hf_policy, "shutdown"):
            hf_policy.shutdown()


@pytest.mark.skip(
    reason="Skipping for now, will be fixed in https://github.com/NVIDIA/NeMo-RL/issues/408"
)
def test_vllm_worker_seed_behavior(cluster, tokenizer):
    """
    1. Different workers generate different outputs for identical prompts due to different seeds
    2. When forced to use the same seed, workers generate identical outputs
    """
    from nemo_rl.models.generation.vllm import VllmGenerationWorker

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

    # Part 1: Test that different workers generate different outputs due to different seeds
    print("Creating vLLM policy with default seed behavior...")
    vllm_config = basic_vllm_test_config.copy()
    vllm_config = configure_generation_config(vllm_config, tokenizer)
    policy = VllmGeneration(cluster, vllm_config)
    policy.finish_generation()

    from nemo_rl.models.policy.hf_policy import HfPolicy

    hf_config = get_basic_hf_test_config(enable_dtensor=False)
    hf_policy = HfPolicy(cluster, hf_config, tokenizer)

    print("refitting vllm policy...")
    refit_policy_generation(hf_policy, policy)

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

        # Clean up before the second test
        policy.shutdown()

        # Part 2: Test with fixed seed to verify identical outputs
        print("\nNow testing with fixed seed...")

        # Store the original configure_worker method
        original_configure_worker = VllmGenerationWorker.configure_worker

        # Override the configure_worker method to always use the same seed
        def configure_worker_fixed_seed(num_gpus, bundle_indices=None):
            resources, env_vars, init_kwargs = original_configure_worker(
                num_gpus, bundle_indices
            )
            # Override with fixed seed
            init_kwargs["seed"] = 42
            return resources, env_vars, init_kwargs

        VllmGenerationWorker.configure_worker = configure_worker_fixed_seed

        # Create a new policy with fixed seed
        fixed_seed_policy = VllmGeneration(cluster, vllm_config)

        # Generate with the same duplicated prompts
        print("Running generation with fixed seed...")
        fixed_seed_outputs = fixed_seed_policy.generate(duplicated_batch, greedy=False)

        # Decode the generated sequences
        fixed_seed_gen_texts = tokenizer.batch_decode(
            fixed_seed_outputs["output_ids"], skip_special_tokens=True
        )

        print(f"Generated texts with fixed seed: {fixed_seed_gen_texts}")

        # Check if the duplicated prompts now generate the same texts
        fixed_seed_first_half = fixed_seed_gen_texts[: len(unique_prompts)]
        fixed_seed_second_half = fixed_seed_gen_texts[len(unique_prompts) :]

        print(f"First worker outputs (fixed seed): {fixed_seed_first_half}")
        print(f"Second worker outputs (fixed seed): {fixed_seed_second_half}")

        # With the same seed, outputs should be identical
        assert fixed_seed_first_half == fixed_seed_second_half, (
            "Workers with the same fixed seed should generate identical outputs for identical prompts"
        )

    finally:
        # Restore the original method if we patched it
        if "original_configure_worker" in locals():
            VllmGenerationWorker.configure_worker = original_configure_worker

        # Clean up resources
        if "policy" in locals() and hasattr(policy, "shutdown"):
            policy.shutdown()
        if "fixed_seed_policy" in locals() and hasattr(fixed_seed_policy, "shutdown"):
            fixed_seed_policy.shutdown()

        # Force garbage collection
        import gc

        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.timeout(140)
@pytest.mark.parametrize("async_engine", [True, False])
@pytest.mark.parametrize("enable_dtensor", [True, False])
def test_vllm_generation_with_hf_training(
    cluster, tokenizer, enable_dtensor, async_engine
):
    """1. Use vLLM for generation
    2. Use HF policy for training and logprob computation

    This test validates that the two policies can work together.
    """
    from nemo_rl.models.policy.hf_policy import HfPolicy
    from tests.unit.test_utils import SimpleNLLLoss

    # Create separate configs for each policy
    vllm_config = basic_vllm_test_config.copy()
    vllm_config["vllm_cfg"]["async_engine"] = async_engine
    vllm_config = configure_generation_config(vllm_config, tokenizer)

    hf_config = get_basic_hf_test_config(enable_dtensor=enable_dtensor)
    hf_config["train_global_batch_size"] = 4

    vllm_policy = None
    hf_policy = None

    try:
        prompts = [
            "Write a story about a magical forest",
            "Explain how photosynthesis works",
            "What are the benefits of exercise?",
            "Describe the water cycle",
            "What is the capital of France?",
            "Who is the president of the USA?",
            "What is the capital of the moon?",
            "Where is the sun?",
        ]

        # Tokenize the prompts the same way as in test_hf_ray_policy
        tokenized = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
            padding_side="right",
        )
        # Calculate input lengths from attention mask
        input_lengths = tokenized["attention_mask"].sum(dim=1).to(torch.int32)

        test_input_data = BatchedDataDict(
            {
                "input_ids": tokenized["input_ids"],
                "input_lengths": input_lengths,
            }
        )

        # Create both policies
        print("Creating vLLM policy...")
        vllm_policy = VllmGeneration(cluster, vllm_config)
        vllm_policy.finish_generation()

        print("Creating HF policy...")
        hf_policy = HfPolicy(cluster, hf_config, tokenizer)

        print("refitting vllm policy...")
        refit_policy_generation(hf_policy, vllm_policy)

        # Step 1: Use vLLM for generation
        print("Using vLLM policy for fast generation...")
        if async_engine:
            generation_results = vllm_policy.generate_async(
                test_input_data, greedy=True
            )
        else:
            generation_results = vllm_policy.generate(test_input_data, greedy=True)
        vllm_policy.finish_generation()
        # Validate generation outputs
        assert "output_ids" in generation_results, (
            "output_ids not found in vLLM generation output"
        )
        assert "logprobs" in generation_results, (
            "logprobs not found in vLLM generation output"
        )

        # Decode generations
        generated_texts = tokenizer.batch_decode(
            generation_results["output_ids"], skip_special_tokens=True
        )
        print(f"vLLM generated texts: {generated_texts}")

        # Run logprob calculation with HF policy to verify
        fprop_logprob_data = BatchedDataDict(
            {
                "input_ids": generation_results["output_ids"],
                "input_lengths": generation_results["unpadded_sequence_lengths"],
            }
        )
        # Get logprobs from HF policy
        hf_policy.prepare_for_lp_inference()
        fprop_results = hf_policy.get_logprobs(fprop_logprob_data)
        # Zero out logprobs for input tokens

        print(f"HF logprobs: {fprop_results['logprobs']}")
        print(f"vLLM logprobs: {generation_results['logprobs']}")

        # Validate that the logprobs are correct (comparing vLLM generation logprobs with HF computed logprobs)

        # Create a mask for padding tokens to only include tokens up to generation_lengths
        padding_mask = torch.zeros_like(
            generation_results["logprobs"], dtype=torch.bool
        )
        for i, (input_len, total_valid_len) in enumerate(
            zip(
                test_input_data.get("input_lengths"),
                generation_results["unpadded_sequence_lengths"],
            )
        ):
            padding_mask[i, input_len:total_valid_len] = True

        abs_diff = torch.abs(generation_results["logprobs"] - fprop_results["logprobs"])
        masked_abs_diff = abs_diff.masked_select(padding_mask)
        avg_prob_mult_error = (
            torch.mean(torch.exp(masked_abs_diff))
            if masked_abs_diff.numel() > 0
            else torch.tensor(0.0)
        )

        print(f"Average probability multiplicative error: {avg_prob_mult_error}")
        assert avg_prob_mult_error <= 1.043, "vLLM and HF logprobs should closely match"

        # Step 2: Prepare simplified training data (smaller and with padding removed to prevent OOM)
        # Use a very small sequence for training to ensure it works
        max_seq_len = min(40, generation_results["output_ids"].shape[1])
        # cap generation lengths to max_seq_len
        generation_results["unpadded_sequence_lengths"] = torch.clamp(
            generation_results["unpadded_sequence_lengths"], max=max_seq_len
        )

        train_input_ids = generation_results["output_ids"][:, :max_seq_len]
        token_loss_mask = torch.ones_like(train_input_ids)
        # Only compute loss on generated tokens, not input
        input_len = test_input_data.get("input_ids").size(1)
        token_loss_mask[:, :input_len] = 0

        for idx, length in enumerate(generation_results["unpadded_sequence_lengths"]):
            token_loss_mask[idx, length:] = 0

        train_data = BatchedDataDict(
            {
                "input_ids": train_input_ids,
                "input_lengths": generation_results["unpadded_sequence_lengths"],
                "token_loss_mask": token_loss_mask,
                "sample_mask": torch.ones(train_input_ids.shape[0]),
            }
        )

        # Step 3: Try a minimal training step with HF policy
        print("Training with HF policy (single step)...")
        hf_policy.prepare_for_training()

        # Just do one training step to verify it works
        results = hf_policy.train(train_data, SimpleNLLLoss())
        print(f"Training loss: {results['loss']}")

        hf_policy.finish_training()
        hf_policy.offload_after_refit()

        # Step 4: Use vLLM for generation again to complete the workflow
        print("Using vLLM for generation again...")
        vllm_policy.prepare_for_generation()
        if async_engine:
            final_generation = vllm_policy.generate_async(test_input_data)
        else:
            final_generation = vllm_policy.generate(test_input_data)

        assert "output_ids" in final_generation, (
            "Final generation should contain output_ids"
        )

        print("Successfully demonstrated vLLM generation + HF training workflow!")

    finally:
        # Clean up resources
        print("Cleaning up resources...")
        if vllm_policy:
            vllm_policy.shutdown()
        if hf_policy and hasattr(hf_policy, "shutdown"):
            hf_policy.shutdown()


def test_vllm_policy_tensor_parallel(cluster, tokenizer):
    """Test vLLM policy with tensor parallelism > 1."""
    # Configure with tensor_parallel_size=2
    tp_config = deepcopy(basic_vllm_test_config)
    tp_config = configure_generation_config(tp_config, tokenizer)
    tp_config["vllm_cfg"]["tensor_parallel_size"] = 2

    # Ensure we specify the distributed executor backend
    tp_config["vllm_kwargs"] = {"distributed_executor_backend": "ray"}

    vllm_policy = None
    try:
        vllm_policy = VllmGeneration(cluster, tp_config)

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
        outputs = vllm_policy.generate(test_input_data)

        vllm_policy.finish_generation()
        vllm_policy.prepare_for_generation()
        # Validate outputs
        # Test generation with tensor parallelism
        outputs = vllm_policy.generate(test_input_data)

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
        if vllm_policy:
            vllm_policy.shutdown()


def test_vllm_generate_text(cluster, tokenizer):
    """Test that vLLM can generate text."""
    # Prepare test data
    test_prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]
    test_prompts = BatchedDataDict({"prompts": test_prompts})

    # Create separate configs for each policy
    vllm_config = basic_vllm_test_config.copy()
    vllm_config = configure_generation_config(vllm_config, tokenizer, is_eval=True)

    # Ensure we can get same output
    assert vllm_config["model_name"] == "Qwen/Qwen3-0.6B", (
        "Model name should be Qwen/Qwen3-0.6B to get expected output"
    )
    assert vllm_config["vllm_cfg"]["tensor_parallel_size"] == 1, (
        "Tensor parallel size should be 1 to get expected output"
    )

    # Create vLLM generation
    vllm_generation = VllmGeneration(cluster, vllm_config)

    # Generate and check result
    output = vllm_generation.generate_text(test_prompts, greedy=True)
    assert output["texts"] == [
        " Lina. I'm",
        " Paris. The capital of",
    ], "Output should be the same as the expected output"

    # Clean up
    vllm_generation.shutdown()


@pytest.mark.timeout(180)
@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
@pytest.mark.parametrize("enable_dtensor", [True, False])
def test_vllm_weight_update_and_prefix_cache_reset(
    cluster, tokenizer, tensor_parallel_size, enable_dtensor
):
    """Test that the vLLM prefix cache is correctly reset when weights change."""
    from nemo_rl.models.policy.hf_policy import HfPolicy

    # Create configs
    vllm_config = deepcopy(basic_vllm_test_config)
    vllm_config = configure_generation_config(vllm_config, tokenizer, is_eval=True)
    vllm_config["vllm_cfg"]["tensor_parallel_size"] = tensor_parallel_size
    if tensor_parallel_size > 1:
        vllm_config["vllm_kwargs"] = {"distributed_executor_backend": "ray"}

    hf_config = get_basic_hf_test_config(enable_dtensor=enable_dtensor)

    # Create policies
    vllm_policy = None
    hf_policy = None
    try:
        print(f"Creating HF policy for TP={tensor_parallel_size}...")
        hf_policy = HfPolicy(cluster, hf_config, tokenizer)
        print(f"Creating vLLM policy for TP={tensor_parallel_size}...")
        vllm_policy = VllmGeneration(cluster, vllm_config)

        # Prepare input data (batch size 2)
        text = """Answer the question based on the context below. Keep the answer short and concise. Respond "Unsure about answer" if not sure about the answer. Context: Teplizumab traces its roots to a New Jersey drug company called Ortho Pharmaceutical. There, scientists generated an early version of the antibody, dubbed OKT3. Originally sourced from mice, the molecule was able to bind to the surface of T cells and limit their cell-killing potential. In 1986, it was approved to help prevent organ rejection after kidney transplants, making it the first therapeutic antibody allowed for human use.Question: What was OKT3 originally sourced from?Answer:"""
        test_prompt = [text, text]  # Use batch size 2
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
        vllm_policy.prepare_for_generation()
        outputs1 = vllm_policy.generate(test_input_data, greedy=True)
        generated_text = tokenizer.decode(
            outputs1["output_ids"][0], skip_special_tokens=True
        )
        print(f"Generated text (Run 1): {generated_text}")
        logprob1 = outputs1["logprobs"][0, input_lengths[0]].item()
        print(f"Logprob of first generated token (Run 1): {logprob1}")

        print("Adding noise to weights in HF policy...")
        ray.get(
            [
                worker._add_noise_to_weights.remote()
                for worker in hf_policy.worker_group.workers
            ]
        )

        print("Updating vLLM weights from HF policy...")
        grouped_param_keys = hf_policy.prepare_weights_for_ipc()
        for keys in grouped_param_keys:
            ipc_handles = hf_policy.get_weights_ipc_handles(keys)
            update_success = vllm_policy.update_weights(ipc_handles)
            assert update_success, "Weight update should succeed"
        print("vLLM weights successfully updated.")

        print("Running Generation 2 (Weights Updated, Cache Still Active)...")
        # Generate again *without* resetting the cache
        outputs2 = vllm_policy.generate(test_input_data, greedy=True)
        logprob2 = outputs2["logprobs"][0, input_lengths[0]].item()
        print(f"Logprob of first generated token (Run 2): {logprob2}")
        assert logprob2 != logprob1, "Logprobs should be different after weight update."

        print("Resetting vLLM prefix cache (via finish/prepare cycle)...")
        vllm_policy.finish_generation()  # Calls sleep() which resets cache
        vllm_policy.prepare_for_generation()  # Calls wake_up()

        print("Running Generation 3 (Weights updated, Cache Reset)...")
        outputs3 = vllm_policy.generate(test_input_data, greedy=True)
        logprob3 = outputs3["logprobs"][0, input_lengths[0]].item()
        print(f"Logprob of first generated token (Run 3): {logprob3}")
        assert logprob2 != logprob3, (
            "Logprobs should be different after cache reset and weight update."
        )

        print("Prefix cache reset verified successfully.")

    finally:
        # --- Cleanup ---
        print("Cleaning up resources...")
        if vllm_policy:
            vllm_policy.shutdown()
        if hf_policy:
            hf_policy.shutdown()
        # Force garbage collection to help release resources
        import gc

        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.parametrize("enable_dtensor", [True, False])
def test_vllm_weight_update_memory(cluster, tokenizer, enable_dtensor):
    """Test that vLLM streaming weight update and can save memory."""
    from nemo_rl.models.policy.hf_policy import HfPolicy

    if cluster.num_gpus_per_node < 2:
        pytest.skip("Need at least 2 GPUs per node for this test")

    # Create separate configs for each policy
    vllm_config = basic_vllm_test_config.copy()
    vllm_config = configure_generation_config(vllm_config, tokenizer, is_eval=False)

    # Ensure we can get same peak memory
    assert vllm_config["model_name"] == "Qwen/Qwen3-0.6B", (
        "Model name should be Qwen/Qwen3-0.6B to get expected peak memory"
    )

    # Create policies
    print("Creating vLLM policy...")
    vllm_policy = VllmGeneration(cluster, vllm_config)
    vllm_policy.finish_generation()

    print("Creating HF policy...")
    hf_config = get_basic_hf_test_config(enable_dtensor=enable_dtensor)
    hf_policy = HfPolicy(cluster, hf_config, tokenizer)

    print("refitting vllm policy...")
    # take it outside statistics to get clean peak memory during refit
    hf_policy.offload_before_refit()
    # reset peak memory stats before refit
    workers = hf_policy.worker_group.workers
    ray.get([w.reset_peak_memory_stats.remote() for w in workers])
    refit_policy_generation(hf_policy, vllm_policy, _refit_buffer_size_gb=1)
    gpu_infos = ray.get([w.get_gpu_info.remote() for w in workers])

    # Gather memory stats
    current_allocated = 0.0
    current_reserved = 0.0
    peak_allocated = 0.0
    peak_reserved = 0.0
    for status in gpu_infos:
        current_allocated = max(current_allocated, status["memory_allocated_mb"])
        current_reserved = max(current_reserved, status["memory_reserved_mb"])
        peak_allocated = max(peak_allocated, status["peak_memory_allocated_mb"])
        peak_reserved = max(peak_reserved, status["peak_memory_reserved_mb"])

    # Check memory stats
    assert current_allocated == 0.0, "Memory should be 0 after refit completed"
    assert current_reserved == 0.0, "Memory should be 0 after refit completed"
    # memory threshold: memory during non-streaming weight update on 0.6B model on 2 GPUs
    # memory during streaming weight update should less than this baseline threshold
    if enable_dtensor:
        assert peak_allocated < 4005, "Peak allocated memory should < 4005 MB"
        assert peak_reserved < 4016, "Peak reserved memory should < 4016 MB"
    else:
        assert peak_allocated < 5736, "Peak allocated memory should < 5736 MB"
        assert peak_reserved < 5748, "Peak reserved memory should < 5748 MB"

    # Clean up
    vllm_policy.shutdown()
    hf_policy.shutdown()


@pytest.mark.parametrize("is_eval", [True, False])
@pytest.mark.parametrize("enable_dtensor", [True, False])
def test_vllm_generation_with_stop(
    cluster, test_input_data, tokenizer, is_eval, enable_dtensor
):
    """Test vLLM generation with stop."""
    from nemo_rl.models.policy.hf_policy import HfPolicy

    # Create separate configs for each policy
    vllm_config = basic_vllm_test_config.copy()
    vllm_config["stop_token_ids"] = [6722]  # 'Ġcapital'
    vllm_config["stop_strings"] = ["I'm"]
    vllm_config = configure_generation_config(vllm_config, tokenizer, is_eval=is_eval)

    # Ensure we can get same output
    assert vllm_config["model_name"] == "Qwen/Qwen3-0.6B", (
        "Model name should be Qwen/Qwen3-0.6B to get expected output"
    )
    assert vllm_config["vllm_cfg"]["tensor_parallel_size"] == 1, (
        "Tensor parallel size should be 1 to get expected output"
    )

    # Create policies
    print("Creating vLLM policy...")
    vllm_generation = VllmGeneration(cluster, vllm_config)

    # Get weights from HF policy if not in eval mode
    if not is_eval:
        # set to sleep first if not in eval mode
        vllm_generation.finish_generation()

        print("Creating HF policy...")
        hf_config = get_basic_hf_test_config(enable_dtensor=enable_dtensor)
        hf_policy = HfPolicy(cluster, hf_config, tokenizer)

        print("refitting vllm policy...")
        refit_policy_generation(hf_policy, vllm_generation)

    # test generate
    outputs = vllm_generation.generate(test_input_data, greedy=True)
    output_ids = outputs["output_ids"]
    generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    assert generated_texts == [
        "Hello, my name is Lina. I'm",
        "The capital of France is Paris. The capital",
    ], "Output should be the same as the expected output"

    # test generate_text
    test_prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]
    test_prompts = BatchedDataDict({"prompts": test_prompts})
    output = vllm_generation.generate_text(test_prompts, greedy=True)
    assert output["texts"] == [
        " Lina. I'm",
        " Paris. The capital",
    ], "Output should be the same as the expected output"

    # Clean up
    vllm_generation.shutdown()
    if not is_eval:
        hf_policy.shutdown()


def test_vllm_non_divisible_batch_handling(policy):
    """Test that VLLM generation handles non divisible input batches correctly."""
    # This test runs on 2 GPUs but has a batch size of 1. The first GPU will run a batch
    # and the second will run a batch of size 0.

    # Create and run with non divisible batch
    empty_batch = BatchedDataDict(
        {
            "input_ids": torch.zeros((1, 1), dtype=torch.long),
            "input_lengths": torch.ones(1, dtype=torch.long),
        }
    )

    outputs = policy.generate(empty_batch)

    # Verify output structure and dimensions
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
        "Output tensors should have a batch dimension of 1"
    )


def test_vllm_refit_non_collocated_handles_update_failure(
    policy_cluster_separate,
    generation_cluster_separate,
    tokenizer,
    test_input_data,
):
    if (
        policy_cluster_separate.num_gpus_per_node < 1
        or generation_cluster_separate.num_gpus_per_node < 1
    ):
        pytest.skip(
            "Test requires at least two GPUs to run policies on separate clusters."
        )

    # Create HfPolicy on its own cluster
    hf_config = get_basic_hf_test_config(enable_dtensor=True)
    hf_config["dtensor_cfg"]["tensor_parallel_size"] = 1
    hf_policy = HfPolicy(policy_cluster_separate, hf_config, tokenizer)

    # Create VllmGeneration policy on its own cluster
    vllm_config = deepcopy(basic_vllm_test_config)
    vllm_config = configure_generation_config(vllm_config, tokenizer, is_eval=True)
    vllm_config["vllm_cfg"]["tensor_parallel_size"] = 1
    vllm_policy = VllmGeneration(generation_cluster_separate, vllm_config)

    hf_policy_instance = None
    vllm_policy_instance = None

    try:
        hf_policy_instance = hf_policy
        vllm_policy_instance = vllm_policy
        ray.get(
            [
                worker._add_noise_to_weights.remote()
                for worker in hf_policy_instance.worker_group.workers
            ]
        )
        print("Refitting vLLM policy from HF policy (non-collocated)")
        with mock.patch.object(
            vllm_policy_instance, "update_weights", return_value=False
        ):
            with pytest.raises(RuntimeError):
                refit_policy_generation(
                    hf_policy_instance,
                    vllm_policy_instance,
                )
        print("RuntimeError during refit correctly caught.")

    finally:
        print("Cleaning up non-collocated test resources...")
        if hf_policy_instance:
            try:
                hf_policy_instance.shutdown()
            except Exception as e:
                print(f"Error during HfPolicy cleanup: {e}")
        if vllm_policy_instance:
            try:
                vllm_policy_instance.shutdown()
            except Exception as e:
                print(f"Error during VllmPolicy cleanup: {e}")
        # Force garbage collection
        import gc

        gc.collect()
        torch.cuda.empty_cache()
