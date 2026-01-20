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
from copy import deepcopy
from pathlib import Path

import pytest
import ray
import requests
import torch

from nemo_rl.algorithms.grpo import refit_policy_generation
from nemo_rl.algorithms.loss_functions import NLLLoss
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
)
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    _replace_prefix_tokens,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.lm_policy import Policy

model_name = "Qwen/Qwen3-0.6B"
# Define basic vLLM test config
basic_vllm_test_config: VllmConfig = {
    "backend": "vllm",
    "model_name": model_name,
    "tokenizer": {
        "name": model_name,
    },
    "dtype": "bfloat16",
    "max_new_tokens": 5,  # Small number of tokens for testing
    # Set temperature=1.0 to ensure consistent probability scaling when comparing vLLM and HF policy outputs.
    # Note: greedy=True is only used in tests for deterministic behavior and not used in the real training.
    # In vLLM, enabling greedy=True disables temperature scaling (temperature is overridden to None).
    # The HF policy worker does not currently support greedy=True for get_logprobs.
    # Using temperature=1.0 allows us to meaningfully test the average probability multiplicative error between the two implementations,
    # while still maintaining the deterministic behavior.
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": None,
    "stop_token_ids": None,
    "stop_strings": None,
    "vllm_cfg": {
        "precision": "bfloat16",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "expert_parallel_size": 1,
        "gpu_memory_utilization": 0.7,
        "max_model_len": 1024,
        "async_engine": False,  # Default to False for synchronous tests
        "skip_tokenizer_init": False,
        "load_format": "auto",
        "enforce_eager": "False",
    },
    "colocated": {
        "enabled": True,
        "resources": {
            "gpus_per_node": None,
            "num_nodes": None,
        },
    },
    "vllm_kwargs": {},
}

basic_dtensor_test_config: PolicyConfig = {
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
        "enabled": True,
        "cpu_offload": False,
        "sequence_parallel": False,
        "activation_checkpointing": False,
        "tensor_parallel_size": 1,
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
    "generation": deepcopy(basic_vllm_test_config),
}


def get_basic_megatron_test_config(
    tp: int = 1,
    pp: int = 1,
    precision: str = "float32",
    activation_checkpointing: bool = False,
    sequence_parallel: bool = False,
    empty_unused_memory_level: int = 0,
) -> PolicyConfig:
    """Create a test config for Megatron policy worker."""
    # Use the exact same model as vLLM tests for perfect compatibility
    model_name = basic_vllm_test_config["model_name"]  # Use same model as vLLM config

    return {
        "model_name": model_name,
        "tokenizer": {"name": model_name},
        "generation_batch_size": 2,  # Small batch size for testing
        "train_global_batch_size": 4,
        "train_micro_batch_size": 2,
        "learning_rate": 5e-6,
        "logprob_batch_size": 2,
        "precision": precision,
        "offload_optimizer_for_logprob": False,
        "dtensor_cfg": {
            "enabled": False,  # Disabled for Megatron tests
        },
        "dynamic_batching": {
            "enabled": False,  # Start with simple batching
        },
        "sequence_packing": {
            "enabled": False,
        },
        "megatron_cfg": {
            "enabled": True,
            "empty_unused_memory_level": empty_unused_memory_level,
            "activation_checkpointing": activation_checkpointing,
            "converter_type": "Qwen2ForCausalLM",  # Use Qwen2 converter for Qwen3 models (compatible)
            "tensor_model_parallel_size": tp,
            "expert_tensor_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "pipeline_model_parallel_size": pp,
            "num_layers_in_first_pipeline_stage": None,
            "num_layers_in_last_pipeline_stage": None,
            "context_parallel_size": 1,
            "pipeline_dtype": precision,
            "sequence_parallel": sequence_parallel,
            "freeze_moe_router": True,
            "moe_router_dtype": "fp64",
            "moe_router_load_balancing_type": "none",
            "moe_router_bias_update_rate": 0.0,
            "moe_permute_fusion": False,
            "moe_enable_deepep": False,
            "moe_token_dispatcher_type": "allgather",
            "moe_shared_expert_overlap": False,
            "apply_rope_fusion": True,
            "bias_activation_fusion": True,
            "moe_per_layer_logging": False,
            "train_iters": 100,  # Required for Megatron training
            "optimizer": {
                "optimizer": "adam",
                "lr": 5.0e-6,
                "min_lr": 5.0e-7,
                "weight_decay": 0.01,
                "bf16": precision == "bfloat16",
                "fp16": precision == "float16",
                "params_dtype": "float32",
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_eps": 1e-8,
                "use_distributed_optimizer": True,
                "use_precision_aware_optimizer": True,
                "clip_grad": 1.0,
                "optimizer_cpu_offload": False,
                "optimizer_offload_fraction": 0.0,
            },
            "scheduler": {
                "start_weight_decay": 0.01,
                "end_weight_decay": 0.01,
                "weight_decay_incr_style": "constant",
                "lr_decay_style": "constant",
                "lr_decay_iters": None,
                "lr_warmup_iters": 50,
                "lr_warmup_init": 5.0e-7,
            },
            "distributed_data_parallel_config": {
                "grad_reduce_in_fp32": False,
                "overlap_grad_reduce": True,
                "overlap_param_gather": False,
                "data_parallel_sharding_strategy": "optim_grads_params",
            },
        },
        "optimizer": None,  # Remove default FSDP optimizer
        "scheduler": None,  # Remove default scheduler
        "max_grad_norm": 1.0,
        "generation": deepcopy(basic_vllm_test_config),
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
def moe_cluster():
    """Create a virtual cluster for testing MoE models."""
    virtual_cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[2],  # 1 node with 8 GPU bundle
        use_gpus=True,
        max_colocated_worker_groups=2,
        num_gpus_per_node=2,
        name="vllm-test-moe-cluster",
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
    """Create a virtual cluster for the Policy, using 1 GPU."""
    cluster = _create_ray_virtual_cluster_for_test("vllm-test-policy-cluster-separate")
    yield cluster
    try:
        cluster.shutdown()
    except Exception as e:
        print(f"Error during policy_cluster_separate shutdown: {e}")


def get_generation_cluster_separate(num_gpus_per_node: int = 1) -> RayVirtualCluster:
    """Create a virtual cluster for the VllmGeneration policy, using num_gpus_per_node GPU."""
    return RayVirtualCluster(
        bundle_ct_per_node_list=[num_gpus_per_node],
        use_gpus=True,
        max_colocated_worker_groups=1,
        num_gpus_per_node=num_gpus_per_node,
        name="vllm-test-generation-cluster-separate",
    )


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


def test_vllm_missing_required_config_key(cluster):
    """Test that an assertion error is raised when a required config key is missing."""
    # Create a config missing a required key by removing 'model_name'
    incomplete_config = deepcopy(basic_vllm_test_config)
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


def test_vllm_top_p_top_k_validation(cluster):
    """Test that top_p and top_k validation works correctly with threshold-based logic."""
    # Test that values above thresholds are allowed
    config_above_thresholds = deepcopy(basic_vllm_test_config)
    config_above_thresholds["top_p"] = 0.99  # Above TOP_P_THRESHOLD
    config_above_thresholds["top_k"] = 8000  # Above TOP_K_THRESHOLD

    # Should not raise an error
    try:
        VllmGeneration(cluster, config_above_thresholds)
        print("Successfully initialized with top_p=0.99 and top_k=8000")
    except Exception as e:
        pytest.fail(f"Should not raise error with values above thresholds: {e}")

    # Test that values below thresholds are rejected
    config_below_thresholds = deepcopy(basic_vllm_test_config)
    config_below_thresholds["top_p"] = 0.9  # Below TOP_P_THRESHOLD

    with pytest.raises(ValueError) as excinfo:
        VllmGeneration(cluster, config_below_thresholds)

    error_message = str(excinfo.value)
    assert "top_p sampling with values < 0.99 is not supported" in error_message
    print(f"Successfully caught low top_p value with error: {error_message}")

    # Test that low top_k values are rejected
    config_low_top_k = deepcopy(basic_vllm_test_config)
    config_low_top_k["top_k"] = 7999  # Below TOP_K_THRESHOLD

    with pytest.raises(ValueError) as excinfo:
        VllmGeneration(cluster, config_low_top_k)

    error_message = str(excinfo.value)
    assert "top_k sampling with values < 8000 is not supported" in error_message
    print(f"Successfully caught low top_k value with error: {error_message}")


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


async def _generate_async(vllm_policy, tokenizer, test_input_data, greedy=False):
    collected_indexed_outputs = []
    # generate_async is restricted to handle only single samples
    input_generator = test_input_data.make_microbatch_iterator(microbatch_size=1)
    for single_item_input in input_generator:
        async for original_idx, single_item_output in vllm_policy.generate_async(
            single_item_input, greedy=greedy
        ):
            collected_indexed_outputs.append((original_idx, single_item_output))

    # Sort by original_idx to ensure order matches generation_input_data
    collected_indexed_outputs.sort(key=lambda x: x[0])

    # Extract in correct order
    outputs = [item for _, item in collected_indexed_outputs]
    pad_token_id = vllm_policy.cfg.get("_pad_token_id", tokenizer.pad_token_id)
    outputs = BatchedDataDict.from_batches(
        outputs,
        pad_value_dict={"output_ids": pad_token_id, "logprobs": 0.0},
    )
    return outputs


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
    lm_policy = None
    async_policy = None
    try:
        vllm_config = deepcopy(basic_vllm_test_config)
        vllm_config = configure_generation_config(vllm_config, tokenizer)
        vllm_config["vllm_cfg"]["async_engine"] = True
        vllm_config["vllm_cfg"]["tensor_parallel_size"] = tensor_parallel_size
        vllm_config["vllm_cfg"]["pipeline_parallel_size"] = pipeline_parallel_size
        dtensor_config = basic_dtensor_test_config
        from nemo_rl.models.policy.lm_policy import Policy

        print("creating vllm policy...")
        async_policy = VllmGeneration(cluster, vllm_config)
        async_policy.finish_generation()

        print("creating lm policy...")
        lm_policy = Policy(cluster, dtensor_config, tokenizer)

        print("preparing refit info...")
        state_dict_info = lm_policy.prepare_refit_info()
        async_policy.prepare_refit_info(state_dict_info)

        print("refitting vllm policy...")
        refit_policy_generation(
            lm_policy, async_policy, vllm_config["colocated"]["enabled"]
        )

        outputs = await _generate_async(async_policy, tokenizer, test_input_data)

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
        if lm_policy and hasattr(lm_policy, "shutdown"):
            lm_policy.shutdown()


@pytest.mark.skip(
    reason="Skipping for now, will be fixed in https://github.com/NVIDIA-NeMo/RL/issues/408"
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
    vllm_config = deepcopy(basic_vllm_test_config)
    vllm_config = configure_generation_config(vllm_config, tokenizer)
    policy = VllmGeneration(cluster, vllm_config)
    policy.finish_generation()

    from nemo_rl.models.policy.lm_policy import Policy

    dtensor_config = basic_dtensor_test_config
    lm_policy = Policy(cluster, dtensor_config, tokenizer)

    state_dict_info = lm_policy.prepare_refit_info()
    policy.prepare_refit_info(state_dict_info)

    print("refitting vllm policy...")
    refit_policy_generation(lm_policy, policy, vllm_config["colocated"]["enabled"])

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


async def run_hf_train_process(
    lm_policy, vllm_policy, tokenizer, async_engine, colocated, vllm_precision
):
    """Validates that the two policies can work together.

    1. Use vLLM for generation
    2. Use HF policy for training and logprob computation
    """
    from tests.unit.test_utils import SimpleNLLLoss

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

        print("refitting vllm policy...")
        refit_policy_generation(lm_policy, vllm_policy, colocated)

        # Step 1: Use vLLM for generation
        print("Using vLLM policy for fast generation...")
        if async_engine:
            generation_results = await _generate_async(
                vllm_policy, tokenizer, test_input_data, greedy=True
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
        lm_policy.prepare_for_lp_inference()
        fprop_results = lm_policy.get_logprobs(fprop_logprob_data)
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
        if vllm_precision == "fp8":
            assert avg_prob_mult_error <= 1.080, (
                "vLLM and HF logprobs should closely match"
            )
        else:
            assert avg_prob_mult_error <= 1.043, (
                "vLLM and HF logprobs should closely match"
            )

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
        lm_policy.prepare_for_training()

        # Just do one training step to verify it works
        results = lm_policy.train(train_data, SimpleNLLLoss())
        print(f"Training loss: {results['loss']}")

        lm_policy.finish_training()
        refit_policy_generation(lm_policy, vllm_policy, colocated)

        # Step 4: Use vLLM for generation again to complete the workflow
        print("Using vLLM for generation again...")
        vllm_policy.prepare_for_generation()
        if async_engine:
            final_generation = await _generate_async(
                vllm_policy, tokenizer, test_input_data
            )
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
        if lm_policy and hasattr(lm_policy, "shutdown"):
            lm_policy.shutdown()


@pytest.mark.timeout(300)
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("async_engine", "cpu_offload", "vllm_precision"),
    [
        (True, False, "bfloat16"),
        (False, True, "bfloat16"),
        (True, False, "fp8"),
        (False, True, "fp8"),
    ],
)
async def test_vllm_generation_with_hf_training_colocated(
    cluster, tokenizer, async_engine, cpu_offload, vllm_precision
):
    """This test validates that DTensor policy can work together with colocated vLLM policy."""

    # Skip the fp8 tests if the GPU is not H100 or newer (compute capability < 9.0)
    if vllm_precision == "fp8":
        major_capability, _ = torch.cuda.get_device_capability()
        if major_capability < 9:
            pytest.skip(
                f"Skipping FP8 test. GPU compute capability {major_capability}.0 is < 9.0 (H100 required)."
            )

    # Create VllmGeneration Policy
    print("Creating vLLM policy...")
    vllm_config = deepcopy(basic_vllm_test_config)
    vllm_config["vllm_cfg"]["async_engine"] = async_engine
    vllm_config["vllm_cfg"]["precision"] = vllm_precision

    vllm_config = configure_generation_config(vllm_config, tokenizer)
    vllm_policy = VllmGeneration(cluster, vllm_config)
    vllm_policy.finish_generation()

    # Create Policy
    print("Creating DTensor policy...")
    dtensor_config = deepcopy(basic_dtensor_test_config)
    dtensor_config["dtensor_cfg"]["cpu_offload"] = cpu_offload
    dtensor_config["train_global_batch_size"] = 4
    lm_policy = Policy(cluster, dtensor_config, tokenizer)

    # Prepare refit info
    print("Preparing refit info...")
    state_dict_info = lm_policy.prepare_refit_info()
    vllm_policy.prepare_refit_info(state_dict_info)

    # Test
    await run_hf_train_process(
        lm_policy, vllm_policy, tokenizer, async_engine, True, vllm_precision
    )


@pytest.mark.timeout(300)
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("async_engine", "cpu_offload", "vllm_precision"),
    [
        (True, False, "bfloat16"),
        (False, True, "bfloat16"),
        (True, False, "fp8"),
        (False, True, "fp8"),
    ],
)
async def test_vllm_generation_with_hf_training_non_colocated(
    policy_cluster_separate, tokenizer, async_engine, cpu_offload, vllm_precision
):
    # Skip the fp8 tests if the GPU is not H100 or newer (compute capability < 9.0)
    if vllm_precision == "fp8":
        major_capability, _ = torch.cuda.get_device_capability()
        if major_capability < 9:
            pytest.skip(
                f"Skipping FP8 test. GPU compute capability {major_capability}.0 is < 9.0 (H100 required)."
            )

    """This test validates that DTensor policy can work together with non-colocated vLLM policy."""
    generation_cluster_separate = get_generation_cluster_separate(1)

    # Create VllmGeneration Policy
    print("Creating vLLM policy...")
    vllm_config = deepcopy(basic_vllm_test_config)
    vllm_config["vllm_cfg"]["async_engine"] = async_engine
    vllm_config["vllm_cfg"]["precision"] = vllm_precision
    vllm_config["colocated"]["enabled"] = False
    vllm_config = configure_generation_config(vllm_config, tokenizer)
    vllm_policy = VllmGeneration(generation_cluster_separate, vllm_config)
    vllm_policy.finish_generation()

    # Create Policy
    print("Creating DTensor policy...")
    dtensor_config = deepcopy(basic_dtensor_test_config)
    dtensor_config["generation"]["colocated"]["enabled"] = False
    dtensor_config["dtensor_cfg"]["cpu_offload"] = cpu_offload
    dtensor_config["train_global_batch_size"] = 4
    lm_policy = Policy(policy_cluster_separate, dtensor_config, tokenizer)

    # Refit
    # initialize collective communication for update weights
    ip, port = policy_cluster_separate.get_master_address_and_port()
    train_world_size = policy_cluster_separate.world_size()
    inference_world_size = generation_cluster_separate.world_size()
    world_size = train_world_size + inference_world_size
    futures_train = lm_policy.init_collective(
        ip, port, world_size=world_size, train_world_size=train_world_size
    )
    futures_inference = vllm_policy.init_collective(
        ip, port, world_size=world_size, train_world_size=train_world_size
    )
    ray.get(futures_train + futures_inference)

    # prepare refit info
    state_dict_info = lm_policy.prepare_refit_info()
    vllm_policy.prepare_refit_info(state_dict_info)

    # Test
    await run_hf_train_process(
        lm_policy, vllm_policy, tokenizer, async_engine, False, vllm_precision
    )


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
    vllm_config = deepcopy(basic_vllm_test_config)
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


def configure_http_server_config(tokenizer) -> VllmConfig:
    # Create separate configs for each policy
    generation_config = deepcopy(basic_vllm_test_config)
    generation_config = configure_generation_config(
        generation_config, tokenizer, is_eval=True
    )

    # Enable the http server. Requires both async engine and the expose_http_server flag
    generation_config["vllm_cfg"]["async_engine"] = True
    generation_config["vllm_cfg"]["expose_http_server"] = True

    return generation_config


def _wait_for_vllm_http_server_spinup(base_url: str):
    while True:
        try:
            requests.get(base_url, timeout=5)
            # We don't check the status code since there may not be a route at /
            break
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            Exception,
        ):
            pass


def test_vllm_http_server(cluster, tokenizer):
    """Test that vLLM http server works."""

    generation_config = configure_http_server_config(tokenizer)

    # Ensure we can get same output
    assert generation_config["model_name"] == "Qwen/Qwen3-0.6B", (
        "Model name should be Qwen/Qwen3-0.6B to get expected output"
    )
    assert generation_config["vllm_cfg"]["tensor_parallel_size"] == 1, (
        "Tensor parallel size should be 1 to get expected output"
    )

    # Set to greedy for test reproducibility.
    generation_config["temperature"] = 0.0

    # Create vLLM generation
    vllm_generation = VllmGeneration(cluster, generation_config)

    # We expect one server per vLLM DP rank.
    base_urls = vllm_generation.dp_openai_server_base_urls
    assert len(base_urls) == cluster.num_gpus_per_node

    body = dict(
        model=generation_config["model_name"],
        messages=[
            {"role": "user", "content": "count to 5"},
        ],
        temperature=generation_config["temperature"],
        top_p=generation_config["top_p"],
        # We want to test the actual train flow and how this is used. So we need to get logprobs here.
        logprobs=True,
        return_tokens_as_token_ids=True,
        max_tokens=1,
    )

    _wait_for_vllm_http_server_spinup(base_urls[0])

    # Generate and check result
    response = requests.post(url=f"{base_urls[0]}/chat/completions", json=body)
    actual_result = response.json()

    # This result assumes this exact model. The expected result here is what the full result looks like before we standardize.
    expected_result = {
        "id": "chatcmpl-7b8c0cdeeab34fd58ad260cf44b1a408",
        "object": "chat.completion",
        "created": 1756421711,
        "model": "Qwen/Qwen3-0.6B",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "<think>",
                    "refusal": None,
                    "annotations": None,
                    "audio": None,
                    "function_call": None,
                    "tool_calls": [],
                    "reasoning_content": None,
                    "reasoning": None,
                },
                "logprobs": {
                    "content": [
                        {
                            "token": "token_id:151667",
                            "logprob": -0.00023779425828251988,
                            "bytes": [60, 116, 104, 105, 110, 107, 62],
                            "top_logprobs": [],
                        }
                    ]
                },
                "finish_reason": "length",
                "stop_reason": None,
                "token_ids": None,
            }
        ],
        "service_tier": None,
        "system_fingerprint": None,
        "usage": {
            "prompt_tokens": 12,
            "total_tokens": 13,
            "completion_tokens": 1,
            "prompt_tokens_details": None,
        },
        "prompt_logprobs": None,
        "prompt_token_ids": None,
        "kv_transfer_params": None,
    }

    def _standardize(d: dict) -> dict:
        d = deepcopy(d)
        d.pop("id")
        d.pop("created")
        # We don't want to implicate log prob accuracy in this test.
        d["choices"][0]["logprobs"]["content"][0].pop("logprob")

        # Remove this fork when https://github.com/NVIDIA-NeMo/RL/pull/1563 is merged to NeMo RL main bumping to vLLM 0.11.2
        message = d["choices"][0]["message"]
        if "reasoning" in message:
            message.pop("reasoning")

        return d

    assert _standardize(expected_result) == _standardize(actual_result)

    # Check that tokenization route works
    response = requests.post(url=f"{base_urls[0]}/../tokenize", json=body)
    actual_result = response.json()
    expected_result = {
        "count": 12,
        "max_model_len": 1024,
        "tokens": [
            151644,
            872,
            198,
            1830,
            311,
            220,
            20,
            151645,
            198,
            151644,
            77091,
            198,
        ],
        "token_strs": None,
    }
    assert expected_result == actual_result

    # Clean up
    vllm_generation.shutdown()

    # We should not be able to connect after shutdown
    with pytest.raises(requests.ConnectionError):
        requests.post(
            url=f"{base_urls[0]}/chat/completions",
            json=dict(
                messages=[
                    {"role": "user", "content": "count to 5"},
                ],
                temperature=0.0,
                logprobs=True,
                return_tokens_as_token_ids=True,
                max_tokens=1,
            ),
        )


def test_VllmAsyncGenerationWorker_replace_prefix_tokens(tokenizer):
    # This test assumes the tokenizer model is for the Qwen 3 family
    eos_token_id = tokenizer.eos_token_id
    assert eos_token_id == 151645

    data_fpath = Path(__file__).with_name(
        "test_vllmasyncgenerationworker_replace_prefix_worker.json"
    )
    with data_fpath.open() as f:
        data = json.load(f)

    og_model_token_ids = data["og_model_token_ids"]
    model_token_ids = data["model_token_ids"]
    template_token_ids = data["template_token_ids"]

    og_model_str = tokenizer.decode(og_model_token_ids)
    model_str = tokenizer.decode(model_token_ids)
    template_str = tokenizer.decode(template_token_ids)
    assert og_model_str == template_str
    assert model_str != template_str

    model_prefix_token_ids = og_model_token_ids[:-16]
    assert model_prefix_token_ids[-1] == eos_token_id
    template_prefix_token_ids = template_token_ids[:-16]
    assert template_prefix_token_ids[-1] == eos_token_id
    result = _replace_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=model_prefix_token_ids,
        template_prefix_token_ids=template_prefix_token_ids,
        template_token_ids=template_token_ids,
    )
    assert result == og_model_token_ids

    # no EOS
    model_prefix_token_ids = og_model_token_ids[:-17]
    assert model_prefix_token_ids[-1] != eos_token_id
    template_prefix_token_ids = template_token_ids[:-16]
    assert template_prefix_token_ids[-1] == eos_token_id
    result = _replace_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=model_prefix_token_ids,
        template_prefix_token_ids=template_prefix_token_ids,
        template_token_ids=template_token_ids,
    )
    assert result == og_model_token_ids

    model_prefix_token_ids = og_model_token_ids[:-16]
    assert model_prefix_token_ids[-1] == eos_token_id
    # newline after EOS
    template_prefix_token_ids = template_token_ids[:-15]
    assert template_prefix_token_ids[-2] == eos_token_id
    assert template_prefix_token_ids[-1] != eos_token_id
    result = _replace_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=model_prefix_token_ids,
        template_prefix_token_ids=template_prefix_token_ids,
        template_token_ids=template_token_ids,
    )
    assert result == og_model_token_ids

    # no EOS
    model_prefix_token_ids = og_model_token_ids[:-17]
    assert model_prefix_token_ids[-1] != eos_token_id
    # newline after EOS
    template_prefix_token_ids = template_token_ids[:-15]
    assert template_prefix_token_ids[-2] == eos_token_id
    assert template_prefix_token_ids[-1] != eos_token_id
    result = _replace_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=model_prefix_token_ids,
        template_prefix_token_ids=template_prefix_token_ids,
        template_token_ids=template_token_ids,
    )
    assert result == og_model_token_ids

    model_prefix_token_ids = model_token_ids[:-16]
    assert model_prefix_token_ids[-1] == eos_token_id
    template_prefix_token_ids = template_token_ids[:-16]
    assert template_prefix_token_ids[-1] == eos_token_id
    result = _replace_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=model_prefix_token_ids,
        template_prefix_token_ids=template_prefix_token_ids,
        template_token_ids=template_token_ids,
    )
    assert result == model_token_ids

    # no EOS
    model_prefix_token_ids = model_token_ids[:-17]
    assert model_prefix_token_ids[-1] != eos_token_id
    template_prefix_token_ids = template_token_ids[:-16]
    assert template_prefix_token_ids[-1] == eos_token_id
    result = _replace_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=model_prefix_token_ids,
        template_prefix_token_ids=template_prefix_token_ids,
        template_token_ids=template_token_ids,
    )
    assert result == model_token_ids


def test_replace_prefix_tokens_empty_model_prefix_returns_template():
    class _T:
        eos_token_id = 2

    tokenizer = _T()
    model_prefix_token_ids = []
    template_prefix_token_ids = [9, 2]
    template_token_ids = [9, 2, 33, 44]
    result = _replace_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=model_prefix_token_ids,
        template_prefix_token_ids=template_prefix_token_ids,
        template_token_ids=template_token_ids,
    )
    assert result == template_token_ids


def test_replace_prefix_tokens_missing_eos_in_template_prefix_raises():
    class _T:
        eos_token_id = 2

        def decode(self, *args, **kwargs):
            pass

    tokenizer = _T()
    model_prefix_token_ids = [7, 2]
    template_prefix_token_ids = [9, 9, 9]  # no EOS inside prefix
    template_token_ids = [9, 9, 9, 2, 10]
    with pytest.raises(AssertionError):
        _replace_prefix_tokens(
            tokenizer=tokenizer,
            model_prefix_token_ids=model_prefix_token_ids,
            template_prefix_token_ids=template_prefix_token_ids,
            template_token_ids=template_token_ids,
        )


def test_replace_prefix_tokens_tokenizer_without_eos_raises():
    class _T:
        eos_token_id = None

    tokenizer = _T()
    with pytest.raises(AssertionError):
        _replace_prefix_tokens(
            tokenizer=tokenizer,
            model_prefix_token_ids=[1],
            template_prefix_token_ids=[1, 2],
            template_token_ids=[1, 2],
        )


def test_replace_prefix_tokens_uses_last_eos_in_template_prefix():
    class _T:
        eos_token_id = 2

    tokenizer = _T()
    model_prefix_token_ids = [100, 2]
    template_prefix_token_ids = [9, 2, 9, 2]  # two EOS; last at idx=3
    template_token_ids = [9, 2, 9, 2, 77, 88]
    result = _replace_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=model_prefix_token_ids,
        template_prefix_token_ids=template_prefix_token_ids,
        template_token_ids=template_token_ids,
    )
    assert result == [100, 2, 77, 88]


@pytest.mark.asyncio
async def test_vllm_http_server_correct_merged_tokens_matches_baseline(
    cluster, tokenizer
):
    """Test that vLLM http server works."""

    generation_config = configure_http_server_config(tokenizer)

    # Ensure we can get same output
    assert generation_config["model_name"] == "Qwen/Qwen3-0.6B", (
        "Model name should be Qwen/Qwen3-0.6B to get expected output"
    )
    assert generation_config["vllm_cfg"]["tensor_parallel_size"] == 1, (
        "Tensor parallel size should be 1 to get expected output"
    )

    # Set to greedy for test reproducibility.
    generation_config["temperature"] = 0.0

    # Create vLLM generation
    vllm_generation = VllmGeneration(cluster, generation_config)

    # We expect one server per vLLM DP rank.
    base_urls = vllm_generation.dp_openai_server_base_urls
    assert len(base_urls) == cluster.num_gpus_per_node

    detokenized_str = " Skinny"
    initial_tokenized_ids = [26951, 3834]
    re_tokenized_ids = [94224]

    body = dict(
        messages=[
            {"role": "user", "content": detokenized_str},
        ],
        temperature=generation_config["temperature"],
        top_p=generation_config["top_p"],
        # We want to test the actual train flow and how this is used. So we need to get logprobs here.
        logprobs=True,
        return_tokens_as_token_ids=True,
        max_tokens=1,
    )

    _wait_for_vllm_http_server_spinup(base_urls[0])

    # Check that the re-tokenized ids are the same with the reference and different without the reference.
    # WITHOUT reference token IDs
    response = requests.post(url=f"{base_urls[0]}/../tokenize", json=body)
    actual_result = response.json()
    expected_result = {
        "count": 9,
        "max_model_len": 1024,
        "tokens": [
            151644,
            872,
            198,
            *re_tokenized_ids,
            151645,
            198,
            151644,
            77091,
            198,
        ],
        "token_strs": None,
    }
    assert expected_result == actual_result

    # WITH reference token IDs
    initial_tokenized_query_ids_prefix = [151644, 872, 198, *initial_tokenized_ids]
    initial_tokenized_query_ids = [
        *initial_tokenized_query_ids_prefix,
        151645,
        198,
        151644,
        77091,
        198,
    ]
    body_with_reference_token_ids = body | {
        "required_prefix_token_ids": initial_tokenized_query_ids_prefix
    }
    response = requests.post(
        url=f"{base_urls[0]}/../tokenize", json=body_with_reference_token_ids
    )
    actual_result = response.json()
    expected_result = {
        "count": 10,
        "max_model_len": 1024,
        "tokens": initial_tokenized_query_ids,
        "token_strs": None,
    }
    assert expected_result == actual_result

    # Generate and check result
    response = requests.post(
        url=f"{base_urls[0]}/chat/completions", json=body_with_reference_token_ids
    )
    vllm_http_server_result = response.json()
    vllm_http_server_generated_token = vllm_http_server_result["choices"][0][
        "logprobs"
    ]["content"][0]
    vllm_http_server_generated_token_id = int(
        vllm_http_server_generated_token["token"].removeprefix("token_id:")
    )

    async for _, generate_result in vllm_generation.generate_async(
        BatchedDataDict[GenerationDatumSpec](
            {
                "input_ids": torch.tensor([initial_tokenized_query_ids]),
                "input_lengths": torch.tensor([len(initial_tokenized_query_ids)]),
            }
        )
    ):
        pass

    generate_generated_token_id = generate_result["output_ids"][0][
        len(initial_tokenized_query_ids)
    ].item()

    # We just check the first token here to check the alignment
    assert vllm_http_server_generated_token_id == generate_generated_token_id

    # Clean up
    vllm_generation.shutdown()


@pytest.mark.timeout(180)
@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
@pytest.mark.parametrize("vllm_precision", ["bfloat16", "fp8"])
def test_vllm_weight_update_and_prefix_cache_reset(
    cluster, tokenizer, tensor_parallel_size, vllm_precision
):
    """Test that the vLLM prefix cache is correctly reset when weights change."""

    if vllm_precision == "fp8":
        major_capability, _ = torch.cuda.get_device_capability()
        if major_capability < 9:
            pytest.skip(
                f"Skipping FP8 test. GPU compute capability {major_capability}.0 is < 9.0 (H100 required)."
            )

    from nemo_rl.models.policy.lm_policy import Policy

    # Create configs
    vllm_config = deepcopy(basic_vllm_test_config)
    vllm_config = configure_generation_config(vllm_config, tokenizer, is_eval=True)
    vllm_config["vllm_cfg"]["tensor_parallel_size"] = tensor_parallel_size
    vllm_config["vllm_cfg"]["precision"] = vllm_precision

    if tensor_parallel_size > 1:
        vllm_config["vllm_kwargs"] = {"distributed_executor_backend": "ray"}

    dtensor_config = basic_dtensor_test_config

    # Create policies
    vllm_policy = None
    lm_policy = None
    try:
        print(f"Creating DTensor policy for TP={tensor_parallel_size}...")
        lm_policy = Policy(cluster, dtensor_config, tokenizer)

        print(f"Creating vLLM policy for TP={tensor_parallel_size}...")
        vllm_policy = VllmGeneration(cluster, vllm_config)

        print("preparing refit info...")
        state_dict_info = lm_policy.prepare_refit_info()
        vllm_policy.prepare_refit_info(state_dict_info)

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
                for worker in lm_policy.worker_group.workers
            ]
        )

        print("Updating vLLM weights from HF policy...")

        buffer_size_bytes = int(lm_policy.get_free_memory_bytes() * 0.3)
        lm_policy.stream_weights_via_ipc_zmq(buffer_size_bytes=buffer_size_bytes)
        update_success = vllm_policy.update_weights_via_ipc_zmq()
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
        if lm_policy:
            lm_policy.shutdown()
        # Force garbage collection to help release resources
        import gc

        gc.collect()
        torch.cuda.empty_cache()


# megatron still holds little memory after refit, so we only test dtensor now
@pytest.mark.parametrize("train_backend", ["dtensor_v1", "dtensor_v2"])
def test_vllm_weight_update_memory(cluster, tokenizer, train_backend):
    """Test that vLLM streaming weight update and can save memory."""
    from nemo_rl.models.policy.lm_policy import Policy

    if cluster.num_gpus_per_node < 2:
        pytest.skip("Need at least 2 GPUs per node for this test")

    # Create separate configs for each policy
    vllm_config = deepcopy(basic_vllm_test_config)
    vllm_config = configure_generation_config(vllm_config, tokenizer, is_eval=False)

    # Ensure we can get same peak memory
    assert vllm_config["model_name"] == "Qwen/Qwen3-0.6B", (
        "Model name should be Qwen/Qwen3-0.6B to get expected peak memory"
    )

    # Create policies
    print("Creating vLLM policy...")
    vllm_policy = VllmGeneration(cluster, vllm_config)
    vllm_policy.finish_generation()

    print("Creating Training Policy...")
    if train_backend == "dtensor_v1":
        train_config = basic_dtensor_test_config
    elif train_backend == "dtensor_v2":
        train_config = deepcopy(basic_dtensor_test_config)
        train_config["dtensor_cfg"]["_v2"] = True
    elif train_backend == "megatron":
        train_config = get_basic_megatron_test_config(tp=1, pp=1, precision="float32")
    else:
        raise ValueError(f"Invalid train backend: {train_backend}")
    lm_policy = Policy(cluster, train_config, tokenizer)

    print("preparing refit info...")
    state_dict_info = lm_policy.prepare_refit_info()
    vllm_policy.prepare_refit_info(state_dict_info)

    print("refitting vllm policy...")
    # take it outside statistics to get clean peak memory during refit
    lm_policy.offload_before_refit()
    # reset peak memory stats before refit
    workers = lm_policy.worker_group.workers
    ray.get([w.reset_peak_memory_stats.remote() for w in workers])
    refit_policy_generation(
        lm_policy,
        vllm_policy,
        vllm_config["colocated"]["enabled"],
        _refit_buffer_size_gb=1.5,
    )
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
    assert peak_allocated < 4005, "Peak allocated memory should < 4005 MB"
    assert peak_reserved < 4016, "Peak reserved memory should < 4016 MB"

    # Clean up
    vllm_policy.shutdown()
    lm_policy.shutdown()


@pytest.mark.parametrize("is_eval", [True, False])
def test_vllm_generation_with_stop(cluster, test_input_data, tokenizer, is_eval):
    """Test vLLM generation with stop."""
    from nemo_rl.models.policy.lm_policy import Policy

    # Create separate configs for each policy
    vllm_config = deepcopy(basic_vllm_test_config)
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

        print("Creating DTensor policy...")
        dtensor_config = basic_dtensor_test_config
        lm_policy = Policy(cluster, dtensor_config, tokenizer)

        print("preparing refit info...")
        state_dict_info = lm_policy.prepare_refit_info()
        vllm_generation.prepare_refit_info(state_dict_info)

        print("refitting vllm policy...")
        refit_policy_generation(
            lm_policy, vllm_generation, vllm_config["colocated"]["enabled"]
        )

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
        lm_policy.shutdown()


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


@pytest.mark.asyncio
@pytest.mark.parametrize("async_engine", [True, False])
@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
@pytest.mark.parametrize("policy_type", ["dtensor", "megatron"])
async def test_vllm_refit_non_colocated_update_weights(
    policy_cluster_separate,
    tokenizer,
    test_input_data,
    async_engine,
    tensor_parallel_size,
    policy_type,
):
    # Skip tensor_parallel_size == 2 until we have resources in CI
    if tensor_parallel_size == 2:
        pytest.skip(
            "Test requires at least three GPUs to run with tensor_parallel_size == 2 on separate clusters."
        )

    generation_cluster_separate = get_generation_cluster_separate(tensor_parallel_size)

    if (
        policy_cluster_separate.num_gpus_per_node < 1
        or generation_cluster_separate.num_gpus_per_node < 1
    ):
        pytest.skip(
            "Test requires at least two GPUs to run policies on separate clusters."
        )

    # Get policy config
    if policy_type == "dtensor":
        lm_config = deepcopy(basic_dtensor_test_config)
    else:
        assert policy_type == "megatron"
        lm_config = get_basic_megatron_test_config(tp=1, pp=1, precision="float32")
    lm_config["generation"]["colocated"]["enabled"] = False

    # Get vllm config
    vllm_config = deepcopy(basic_vllm_test_config)
    vllm_config = configure_generation_config(vllm_config, tokenizer, is_eval=True)
    vllm_config["vllm_cfg"]["async_engine"] = async_engine
    vllm_config["vllm_cfg"]["tensor_parallel_size"] = tensor_parallel_size
    vllm_config["colocated"]["enabled"] = False

    # Megatron config with Qwen2.5-0.5B
    if policy_type == "megatron":
        model_name = "Qwen/Qwen2.5-0.5B"
        tokenizer = get_tokenizer({"name": model_name})

        lm_config["model_name"] = model_name
        lm_config["tokenizer"]["name"] = model_name

        vllm_config["model_name"] = model_name
        vllm_config["tokenizer"]["name"] = model_name

    # Create Policy and VllmGeneration
    lm_policy = Policy(policy_cluster_separate, lm_config, tokenizer)
    vllm_generation = VllmGeneration(generation_cluster_separate, vllm_config)

    # initialize collective communication for update weights
    ip, port = policy_cluster_separate.get_master_address_and_port()
    train_world_size = policy_cluster_separate.world_size()
    inference_world_size = generation_cluster_separate.world_size()
    world_size = train_world_size + inference_world_size
    futures_train = lm_policy.init_collective(
        ip, port, world_size=world_size, train_world_size=train_world_size
    )
    futures_inference = vllm_generation.init_collective(
        ip, port, world_size=world_size, train_world_size=train_world_size
    )
    ray.get(futures_train + futures_inference)

    # prepare refit info
    state_dict_info = lm_policy.prepare_refit_info()
    vllm_generation.prepare_refit_info(state_dict_info)

    print("refitting vllm policy...")
    refit_policy_generation(lm_policy, vllm_generation, False)

    # test generate
    if async_engine:
        outputs = await _generate_async(
            vllm_generation, tokenizer, test_input_data, greedy=True
        )
    else:
        outputs = vllm_generation.generate(test_input_data, greedy=True)

    output_ids = outputs["output_ids"]
    generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    if policy_type == "dtensor":
        expected_texts = [
            "Hello, my name is Lina. I'm",
            "The capital of France is Paris. The capital of",
        ]
    else:
        expected_texts = [
            "Hello, my name is Kaitlin and I",
            "The capital of France is Paris. It is the",
        ]
    assert generated_texts == expected_texts, (
        "Output should be the same as the expected output"
    )

    # Clean up
    vllm_generation.shutdown()
    lm_policy.shutdown()
    try:
        generation_cluster_separate.shutdown()
    except Exception as e:
        print(f"Error during generation_cluster_separate shutdown: {e}")


@pytest.mark.timeout(360)
@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
@pytest.mark.parametrize("vllm_precision", ["bfloat16", "fp8"])
@pytest.mark.parametrize("kv_cache_dtype", [None, "fp8"])
def test_vllm_generation_with_megatron_training(
    cluster, tokenizer, tensor_parallel_size, vllm_precision, kv_cache_dtype
):
    """Test that uses vLLM for generation and Megatron policy for training and logprob computation.

    This test validates that vLLM and Megatron policies can work together.
    """

    # Skip invalid configurations: kv_cache_dtype=fp8 requires precision=fp8
    if kv_cache_dtype == "fp8" and vllm_precision != "fp8":
        pytest.skip("kv_cache_dtype='fp8' requires precision='fp8'")

    # Skip the fp8 tests if the GPU is not H100 or newer (compute capability < 9.0)
    if vllm_precision == "fp8":
        major_capability, _ = torch.cuda.get_device_capability()
        if major_capability < 9:
            pytest.skip(
                f"Skipping FP8 test. GPU compute capability {major_capability}.0 is < 9.0 (H100 required)."
            )

    if cluster.num_gpus_per_node < tensor_parallel_size:
        pytest.skip(f"Need at least {tensor_parallel_size} GPUs for this test")

    # Both policies must use the same model (Qwen2.5-0.5B) for weight transfer compatibility
    model_name = "Qwen/Qwen2.5-0.5B"

    # Create tokenizer for both policies
    test_tokenizer = get_tokenizer({"name": model_name})

    # vLLM config with Qwen2.5-0.5B
    vllm_config = deepcopy(basic_vllm_test_config)
    vllm_config["model_name"] = model_name
    vllm_config["tokenizer"]["name"] = model_name
    vllm_config["vllm_cfg"]["async_engine"] = False
    vllm_config["vllm_cfg"]["precision"] = vllm_precision
    if kv_cache_dtype is not None:
        vllm_config["vllm_cfg"]["kv_cache_dtype"] = kv_cache_dtype
    vllm_config = configure_generation_config(vllm_config, test_tokenizer)

    # Megatron config with same model
    megatron_config = get_basic_megatron_test_config(
        tp=tensor_parallel_size, pp=1, precision="float32"
    )
    megatron_config["model_name"] = model_name
    megatron_config["tokenizer"]["name"] = model_name

    vllm_policy = None
    megatron_policy = None

    try:
        prompts = [
            "Hello, how are you?",
            "The capital of France is",
            "Write a short story about",
            "Explain quantum physics in simple terms:",
        ]

        # Tokenize the prompts with the shared tokenizer
        tokenized = test_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=32,  # Smaller for faster testing
            return_tensors="pt",
            padding_side="right",
        )
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

        print("Creating Megatron policy...")
        megatron_policy = Policy(cluster, megatron_config, test_tokenizer)

        print("preparing refit info...")
        state_dict_info = megatron_policy.prepare_refit_info()
        vllm_policy.prepare_refit_info(state_dict_info)

        print("Refitting vLLM policy with Megatron weights...")
        refit_policy_generation(
            megatron_policy, vllm_policy, vllm_config["colocated"]["enabled"]
        )

        # Step 1: Use vLLM for generation
        print("Using vLLM policy for fast generation...")
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
        generated_texts = test_tokenizer.batch_decode(
            generation_results["output_ids"], skip_special_tokens=True
        )
        print(f"vLLM generated texts: {generated_texts}")

        # Step 2: Prepare training data for Megatron (convert tokens to Megatron tokenizer space)
        # Re-tokenize with Megatron tokenizer for training
        megatron_tokenized = test_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt",
            padding_side="right",
        )

        max_seq_len = min(32, megatron_tokenized["input_ids"].shape[1])
        train_input_ids = megatron_tokenized["input_ids"][:, :max_seq_len]
        token_loss_mask = torch.ones_like(train_input_ids)

        # Only compute loss on generated tokens, not input
        input_len = megatron_tokenized["input_ids"].size(1)
        token_loss_mask[:, :input_len] = 0

        train_data = BatchedDataDict(
            {
                "input_ids": train_input_ids,
                "input_lengths": megatron_tokenized["attention_mask"]
                .sum(dim=1)
                .to(torch.int32),
                "token_mask": token_loss_mask,
                "sample_mask": torch.ones(train_input_ids.shape[0]),
            }
        )

        # Step 3: Train with Megatron policy
        print("Training with Megatron policy...")
        megatron_policy.prepare_for_training()

        # Do one training step to verify it works
        results = megatron_policy.train(train_data, NLLLoss())
        print(f"Training loss: {results['loss']}")

        megatron_policy.finish_training()
        megatron_policy.offload_after_refit()

        # Step 4: Use vLLM for generation again
        print("Using vLLM for generation again...")
        vllm_policy.prepare_for_generation()
        final_generation = vllm_policy.generate(test_input_data)

        assert "output_ids" in final_generation, (
            "Final generation should contain output_ids"
        )

        print("Successfully demonstrated vLLM generation + Megatron training workflow!")

    finally:
        # Clean up resources
        print("Cleaning up resources...")
        if vllm_policy:
            vllm_policy.shutdown()
        if megatron_policy and hasattr(megatron_policy, "shutdown"):
            megatron_policy.shutdown()


@pytest.mark.timeout(360)
@pytest.mark.parametrize("vllm_precision", ["bfloat16", "fp8"])
def test_vllm_generation_with_megatron_training_moe_model(
    moe_cluster, tokenizer, vllm_precision
):
    """Test that uses vLLM for generation and Megatron policy for training and logprob computation for a MoE model.

    This test validates that vLLM and Megatron policies can work together.
    """

    # Skip the fp8 tests if the GPU is not H100 or newer (compute capability < 9.0)
    if vllm_precision == "fp8":
        major_capability, _ = torch.cuda.get_device_capability()
        if major_capability < 9:
            pytest.skip(
                f"Skipping FP8 test. GPU compute capability {major_capability}.0 is < 9.0 (H100 required)."
            )

    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"
    expert_parallel_size = 8

    if moe_cluster.num_gpus_per_node < expert_parallel_size:
        pytest.skip(f"Need at least {expert_parallel_size} GPUs for this test")

    # Create tokenizer for both policies
    test_tokenizer = get_tokenizer({"name": model_name})

    # vLLM config with MoE model
    vllm_config = deepcopy(basic_vllm_test_config)
    vllm_config["model_name"] = model_name
    vllm_config["tokenizer"]["name"] = model_name
    vllm_config["vllm_cfg"]["precision"] = vllm_precision
    vllm_config["vllm_cfg"]["expert_parallel_size"] = expert_parallel_size
    vllm_config = configure_generation_config(vllm_config, test_tokenizer)

    # Megatron config with same model
    megatron_config = get_basic_megatron_test_config(tp=1, pp=1, precision="bfloat16")
    megatron_config["model_name"] = model_name
    megatron_config["tokenizer"]["name"] = model_name
    megatron_config["expert_model_parallel_size"] = expert_parallel_size

    vllm_policy = None
    megatron_policy = None

    try:
        prompts = [
            "Hello, how are you?",
            "The capital of France is",
            "Write a short story about",
            "Explain quantum physics in simple terms:",
        ]

        # Tokenize the prompts with the shared tokenizer
        tokenized = test_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=32,  # Smaller for faster testing
            return_tensors="pt",
            padding_side="right",
        )
        input_lengths = tokenized["attention_mask"].sum(dim=1).to(torch.int32)

        test_input_data = BatchedDataDict(
            {
                "input_ids": tokenized["input_ids"],
                "input_lengths": input_lengths,
            }
        )

        # Create both policies
        print("Creating vLLM policy...")
        vllm_policy = VllmGeneration(moe_cluster, vllm_config)
        vllm_policy.finish_generation()

        print("Creating Megatron policy...")
        megatron_policy = Policy(moe_cluster, megatron_config, test_tokenizer)

        print("preparing refit info...")
        state_dict_info = megatron_policy.prepare_refit_info()
        vllm_policy.prepare_refit_info(state_dict_info)

        print("Refitting vLLM policy with Megatron weights...")
        refit_policy_generation(
            megatron_policy, vllm_policy, vllm_config["colocated"]["enabled"]
        )

        # Step 1: Use vLLM for generation
        print("Using vLLM policy for fast generation...")
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
        generated_texts = test_tokenizer.batch_decode(
            generation_results["output_ids"], skip_special_tokens=True
        )
        print(f"vLLM generated texts: {generated_texts}")

        # Step 2: Prepare training data for Megatron (convert tokens to Megatron tokenizer space)
        # Re-tokenize with Megatron tokenizer for training
        megatron_tokenized = test_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt",
            padding_side="right",
        )

        max_seq_len = min(32, megatron_tokenized["input_ids"].shape[1])
        train_input_ids = megatron_tokenized["input_ids"][:, :max_seq_len]
        token_loss_mask = torch.ones_like(train_input_ids)

        # Only compute loss on generated tokens, not input
        input_len = megatron_tokenized["input_ids"].size(1)
        token_loss_mask[:, :input_len] = 0

        train_data = BatchedDataDict(
            {
                "input_ids": train_input_ids,
                "input_lengths": megatron_tokenized["attention_mask"]
                .sum(dim=1)
                .to(torch.int32),
                "token_mask": token_loss_mask,
                "sample_mask": torch.ones(train_input_ids.shape[0]),
            }
        )

        # Step 3: Train with Megatron policy
        print("Training with Megatron policy...")
        megatron_policy.prepare_for_training()

        # Do one training step to verify it works
        results = megatron_policy.train(train_data, NLLLoss())
        print(f"Training loss: {results['loss']}")

        megatron_policy.finish_training()
        megatron_policy.offload_after_refit()

        # Step 4: Use vLLM for generation again
        print("Using vLLM for generation again...")
        vllm_policy.prepare_for_generation()
        final_generation = vllm_policy.generate(test_input_data)

        assert "output_ids" in final_generation, (
            "Final generation should contain output_ids"
        )

        print("Successfully demonstrated vLLM generation + Megatron training workflow!")

    finally:
        # Clean up resources
        print("Cleaning up resources...")
        if vllm_policy:
            vllm_policy.shutdown()
        if megatron_policy and hasattr(megatron_policy, "shutdown"):
            megatron_policy.shutdown()


@pytest.mark.timeout(180)
def test_vllm_megatron_weight_update_memory(cluster, tokenizer):
    """Test that vLLM streaming weight update with Megatron can save memory."""

    if cluster.num_gpus_per_node < 2:
        pytest.skip("Need at least 2 GPUs per node for this test")

    # Both policies must use the same model (Qwen2.5-0.5B) for weight transfer compatibility
    model_name = "Qwen/Qwen2.5-0.5B"

    # Create tokenizer for both policies
    test_tokenizer = get_tokenizer({"name": model_name})

    # vLLM config with Qwen2.5-0.5B
    vllm_config = deepcopy(basic_vllm_test_config)
    vllm_config["model_name"] = model_name
    vllm_config["tokenizer"]["name"] = model_name
    vllm_config = configure_generation_config(
        vllm_config, test_tokenizer, is_eval=False
    )

    # Megatron config with same model
    megatron_config = get_basic_megatron_test_config(
        tp=1, pp=1, precision="float32", empty_unused_memory_level=1
    )
    megatron_config["model_name"] = model_name
    megatron_config["tokenizer"]["name"] = model_name

    # Create policies
    print("Creating vLLM policy...")
    vllm_policy = VllmGeneration(cluster, vllm_config)
    vllm_policy.finish_generation()

    print("Creating Megatron policy...")
    megatron_policy = Policy(cluster, megatron_config, test_tokenizer)

    print("preparing refit info...")
    state_dict_info = megatron_policy.prepare_refit_info()
    vllm_policy.prepare_refit_info(state_dict_info)

    print("Refitting vLLM policy with Megatron...")
    # Take it outside statistics to get clean peak memory during refit
    megatron_policy.offload_before_refit()
    # Reset peak memory stats before refit
    workers = megatron_policy.worker_group.workers
    ray.get([w.reset_peak_memory_stats.remote() for w in workers])

    refit_policy_generation(
        megatron_policy,
        vllm_policy,
        vllm_config["colocated"]["enabled"],
        _refit_buffer_size_gb=1.5,
    )

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

    # Check memory stats - should be minimal after refit
    assert current_allocated <= 0.1, "Memory should be minimal after refit completed"
    assert current_reserved <= 2.1, "Memory should be minimal after refit completed"

    # Memory thresholds for Qwen2.5-0.5B model on 2 GPUs with Megatron
    assert peak_allocated < 6000, (
        f"Peak allocated memory should < 6000 MB, got {peak_allocated}"
    )
    assert peak_reserved < 6000, (
        f"Peak reserved memory should < 6000 MB, got {peak_reserved}"
    )

    print(
        f"Peak memory usage: {peak_allocated:.1f}MB allocated, {peak_reserved:.1f}MB reserved"
    )

    # Clean up
    vllm_policy.shutdown()
    megatron_policy.shutdown()


@pytest.mark.timeout(120)
def test_vllm_megatron_pipeline_parallel(cluster, tokenizer):
    """Test vLLM generation with Megatron pipeline parallel training."""

    if cluster.num_gpus_per_node < 2:
        pytest.skip("Need at least 2 GPUs for pipeline parallel test")

    # Both policies must use the same model (Qwen2.5-0.5B) for weight transfer compatibility
    model_name = "Qwen/Qwen2.5-0.5B"

    # Create tokenizer for both policies
    test_tokenizer = get_tokenizer({"name": model_name})

    # vLLM config with Qwen2.5-0.5B
    vllm_config = deepcopy(basic_vllm_test_config)
    vllm_config["model_name"] = model_name
    vllm_config["tokenizer"]["name"] = model_name
    vllm_config = configure_generation_config(vllm_config, test_tokenizer)

    megatron_config = get_basic_megatron_test_config(
        tp=1,
        pp=2,  # Pipeline parallel
        precision="float32",
    )
    megatron_config["model_name"] = model_name
    megatron_config["tokenizer"]["name"] = model_name

    vllm_policy = None
    megatron_policy = None

    try:
        # Create simple test data
        prompts = ["Hello, world!", "How are you?"]
        tokenized = test_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=16,
            return_tensors="pt",
            padding_side="right",
        )
        test_input_data = BatchedDataDict(
            {
                "input_ids": tokenized["input_ids"],
                "input_lengths": tokenized["attention_mask"].sum(dim=1).to(torch.int32),
            }
        )

        print("Creating Megatron policy with PP=2...")
        megatron_policy = Policy(cluster, megatron_config, test_tokenizer)

        print("Creating vLLM policy...")
        vllm_policy = VllmGeneration(cluster, vllm_config)
        vllm_policy.finish_generation()

        print("preparing refit info...")
        state_dict_info = megatron_policy.prepare_refit_info()
        vllm_policy.prepare_refit_info(state_dict_info)

        print("Refitting vLLM with Megatron PP=2 weights...")
        refit_policy_generation(
            megatron_policy, vllm_policy, vllm_config["colocated"]["enabled"]
        )

        # Test generation
        print("Testing generation with PP=2 Megatron weights...")
        outputs = vllm_policy.generate(test_input_data, greedy=True)

        # Validate outputs
        assert "output_ids" in outputs, "output_ids not found in generation output"
        assert outputs["output_ids"].shape[0] == len(prompts), "Wrong batch size"

        generated_texts = test_tokenizer.batch_decode(
            outputs["output_ids"], skip_special_tokens=True
        )
        print(f"Generated texts with PP=2: {generated_texts}")

        # All texts should be non-empty
        assert all(len(text) > 0 for text in generated_texts), (
            "Some generated texts are empty"
        )

        print("Pipeline parallel test successful!")

    finally:
        if vllm_policy:
            vllm_policy.shutdown()
        if megatron_policy:
            megatron_policy.shutdown()


def test_vllm_megatron_weight_update_with_packing(cluster, test_input_data):
    megatron_policy = None
    vllm_generation = None

    try:
        # Enable packing during test
        os.environ["NEMO_RL_MEGATRON_IPC_TENSOR_PACKING_THRESHOLD"] = "1"

        # Both policies must use the same model for weight transfer compatibility
        # NOTE: We have tried using Qwen/Qwen2.5-0.5B, but some small models exhibit variance depending
        #  on which hardware it is run on.
        model_name = "Qwen/Qwen3-0.6B"
        tokenizer = get_tokenizer({"name": model_name})

        # Create Policy
        megatron_config = get_basic_megatron_test_config(
            tp=1, pp=1, precision="bfloat16"
        )
        megatron_config["model_name"] = model_name
        megatron_config["tokenizer"]["name"] = model_name
        megatron_policy = Policy(cluster, megatron_config, tokenizer)

        # Create VllmGeneration
        vllm_config = deepcopy(basic_vllm_test_config)
        vllm_config = configure_generation_config(vllm_config, tokenizer, is_eval=True)
        vllm_config["model_name"] = model_name
        vllm_config["tokenizer"]["name"] = model_name
        vllm_generation = VllmGeneration(cluster, vllm_config)

        # prepare refit info
        state_dict_info = megatron_policy.prepare_refit_info()
        vllm_generation.prepare_refit_info(state_dict_info)

        print("refitting vllm policy...")
        refit_policy_generation(
            megatron_policy, vllm_generation, vllm_config["colocated"]["enabled"]
        )

        # test generate
        outputs = vllm_generation.generate(test_input_data, greedy=True)
        output_ids = outputs["output_ids"]
        generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        assert generated_texts == [
            "Hello, my name is Lina. I'm",
            "The capital of France is Paris. The capital of",
        ], "Output should be the same as the expected output"

    finally:
        # Restore the original value
        os.environ.pop("NEMO_RL_MEGATRON_IPC_TENSOR_PACKING_THRESHOLD", None)
        # Clean up
        if megatron_policy:
            megatron_policy.shutdown()
        if vllm_generation:
            vllm_generation.shutdown()
