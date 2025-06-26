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
import pprint
from copy import deepcopy

import pytest
import ray
import torch

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.algorithms.loss_functions import ClippedPGLossFn, NLLLoss
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.lm_policy import Policy
from tests.unit.test_utils import SimpleLoss, SimpleNLLLoss

basic_llama_test_config: PolicyConfig = {
    "model_name": "Qwen/Qwen3-0.6B",
    "tokenizer": {
        "name": "Qwen/Qwen3-0.6B",
    },
    "generation_batch_size": 1,  # Small batch size for testing
    "train_global_batch_size": 4,
    "train_micro_batch_size": 1,
    "learning_rate": 5e-6,
    "logprob_batch_size": 1,
    "precision": "float32",
    "fsdp_offload_enabled": False,
    "activation_checkpointing_enabled": False,
    "generation": {
        "backend": "hf",
        "temperature": 1.0,
        "max_new_tokens": 16,  # Small number of tokens for testing
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
    },
    "dtensor_cfg": {
        "enabled": False,
        "cpu_offload": False,
        "sequence_parallel": False,
        "activation_checkpointing": False,
        "tensor_parallel_size": 1,
        "context_parallel_size": 1,
        "custom_parallel_plan": None,
    },
    "dynamic_batching": {
        "enabled": False,
    },
    "optimizer": {
        "name": "torch.optim.AdamW",
        "kwargs": {
            "lr": 5e-6,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
    },
    "scheduler": {
        "name": "torch.optim.lr_scheduler.CosineAnnealingLR",
        "kwargs": {
            "T_max": 100,
        },
    },
    "max_grad_norm": 1.0,
}


@pytest.fixture(scope="module", autouse=True)
def skip_tied_weight_check_for_all():
    """Automatically skip tied weight check for all tests in this module."""
    os.environ["NRL_SKIP_TIED_WEIGHT_CHECK"] = "1"

    yield

    # Restore the original value
    os.environ.pop("NRL_SKIP_TIED_WEIGHT_CHECK", None)


@pytest.fixture(scope="function")
def gc_collect():
    """Helper function to force garbage collection after a test"""
    import gc

    gc.collect()


@pytest.fixture(scope="function")
def tokenizer():
    """Initialize tokenizer for the test model."""
    tokenizer = get_tokenizer(basic_llama_test_config["tokenizer"])
    return tokenizer


@pytest.fixture(scope="function")
def test_input_data(tokenizer):
    """Create test input data for inference."""
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

    expected_generations = [
        "Write a story about a magical forest where the trees are made of stars and the ground is made of light. The",
        "Explain how photosynthesis works in the context of the environment and the role of the sun in it.\nAnswer",
        "What are the benefits of exercise? What are the risks of exercise? What are the benefits and risks of physical activity",
        "Describe the water cycle and its importance in the environment.\nAnswer:\nThe **water cycle** is a",
        "What is the capital of France? The capital of France is Paris. The answer is Paris. The answer is Paris",
        "Who is the president of the USA? The answer is the president of the United States of America, which is the president",
        "What is the capital of the moon? The answer is...? Let me think. I know that the moon is a",
        "Where is the sun? Where is the moon? Where is the earth? Where is the sun in the",
    ]

    # Tokenize the prompts
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

    data = BatchedDataDict(
        {
            "input_ids": tokenized["input_ids"],
            "input_lengths": input_lengths,
        }
    )

    return data, prompts, expected_generations


@pytest.fixture
def policy_setup(tokenizer, num_gpus):
    """Setup and teardown for policy tests - creates a virtual cluster and policy."""
    policy = None
    cluster = None

    cluster_name = f"test-init-{num_gpus}gpu"
    print(f"Creating virtual cluster '{cluster_name}' for {num_gpus} GPUs...")

    cluster = RayVirtualCluster(
        name=cluster_name,
        bundle_ct_per_node_list=[num_gpus],
        use_gpus=True,
        num_gpus_per_node=num_gpus,
        max_colocated_worker_groups=1,
    )

    config = basic_llama_test_config
    config["generation"] = configure_generation_config(config["generation"], tokenizer)

    print("Creating Policy...")
    policy = Policy(cluster=cluster, config=config, tokenizer=tokenizer)

    yield policy, cluster

    # Clean up after the test
    print("Cleaning up resources for test")
    cluster.shutdown()
    policy.worker_group.shutdown()


@pytest.mark.timeout(180)
@pytest.mark.parametrize("num_gpus", [1, 2], ids=["1gpu", "2gpu"])
def test_lm_policy_init(policy_setup, num_gpus):
    policy, cluster = policy_setup

    # Verify cluster and policy were properly created
    assert policy is not None, "Policy was not created properly"
    assert cluster is not None, "Cluster was not created properly"

    # Verify we have workers matching the GPU count
    assert len(policy.worker_group.workers) == num_gpus, (
        f"Should have {num_gpus} worker(s), one per GPU"
    )

    # Check workers are alive
    worker_alive = ray.get([w.is_alive.remote() for w in policy.worker_group.workers])
    assert all(worker_alive), f"Not all workers are alive: {worker_alive}"

    # Get GPU info from both workers to verify GPU usage
    print("\nGetting GPU information from workers...")
    gpu_infos = ray.get([w.get_gpu_info.remote() for w in policy.worker_group.workers])
    print("\nGPU Information:")
    for i, info in enumerate(gpu_infos):
        print(f"\nWorker {i} GPU Info:")
        pprint.pprint(info)

    # Check 1: Verify workers have different ranks
    gpu_ranks = [info["rank"] for info in gpu_infos]
    assert len(set(gpu_ranks)) == num_gpus, (
        f"Expected {num_gpus} different ranks, got {gpu_ranks}"
    )
    assert set(gpu_ranks) == set(range(num_gpus)), (
        f"Expected ranks {set(range(num_gpus))}, got {gpu_ranks}"
    )

    # Check 2: Verify workers have different local_ranks
    local_ranks = [info["local_rank"] for info in gpu_infos]
    assert len(set(local_ranks)) == num_gpus, (
        f"Expected {num_gpus} different local_ranks, got {local_ranks}"
    )
    assert set(local_ranks) == set(range(num_gpus)), (
        f"Expected local_ranks {set(range(num_gpus))}, got {local_ranks}"
    )

    # Check 3: Verify workers have different CUDA_VISIBLE_DEVICES
    cuda_visible_devices = [
        info["env_vars"].get("CUDA_VISIBLE_DEVICES") for info in gpu_infos
    ]
    if num_gpus > 1:
        assert len(set(cuda_visible_devices)) == num_gpus, (
            f"Expected different CUDA_VISIBLE_DEVICES, got {cuda_visible_devices}"
        )
    else:
        assert len(set(cuda_visible_devices)) == 1, (
            f"Expected one CUDA_VISIBLE_DEVICES for 1 GPU, got {cuda_visible_devices}"
        )

    # Check 4: Verify all workers report correct world_size
    for info in gpu_infos:
        assert info["world_size"] == num_gpus, (
            f"Expected world_size={num_gpus}, got {info['world_size']}"
        )
        assert info["env_vars"]["WORLD_SIZE"] == str(num_gpus), (
            f"Expected WORLD_SIZE={num_gpus}, got {info['env_vars']['WORLD_SIZE']}"
        )

    # Check 5: Verify significant GPU memory is allocated (at least 1GB) on all GPUs
    for info in gpu_infos:
        assert info["memory_allocated_mb"] > 1000, (
            f"Not enough memory allocated on GPU for rank {info['rank']}: {info['memory_allocated_mb']:.2f} MB"
        )

    # Check 6: Verify model parameters are on CUDA devices for all workers
    for info in gpu_infos:
        param_sample = list(info["parameter_sample"].values())[0]
        assert "cuda" in param_sample["device"], (
            f"Parameter not on CUDA device: {param_sample['device']}"
        )

    # Check 8: Verify same model parameters are being tracked across workers
    param_names = [list(info["parameter_sample"].keys())[0] for info in gpu_infos]
    assert len(set(param_names)) == 1, (
        f"Workers are not tracking the same parameter: {param_names}"
    )

    # Check 9: Both workers should see their device as cuda:0 (correct distributed behavior)
    for info in gpu_infos:
        param_device = list(info["parameter_sample"].values())[0]["device"]
        assert param_device == "cuda:0", (
            f"Expected parameter device to be cuda:0, got {param_device}"
        )


@pytest.fixture
def training_setup(tokenizer, request, num_gpus):
    """
    Setup and teardown specifically for training tests.

    When used without parameterization, uses the default config.
    When parameterized, takes any config updates as a dictionary in request.param
    and applies them to the basic config.
    """
    policy = None
    cluster = None
    data = None
    loss_fn = None

    # Get config updates from request.param if available
    config_updates = {}
    config_suffix = ""
    if hasattr(request, "param") and request.param is not None:
        config_updates = request.param
        config_suffix = "-" + "-".join([f"{k}={v}" for k, v in config_updates.items()])

    try:
        # Create resources with unique name
        cluster_name = f"test-train-{num_gpus}gpu{config_suffix}"
        print(
            f"Creating training virtual cluster '{cluster_name}' for {num_gpus} GPUs"
            f"{' with config updates: ' + str(config_updates) if config_updates else ''}"
        )

        cluster = RayVirtualCluster(
            name=cluster_name,
            bundle_ct_per_node_list=[num_gpus],
            use_gpus=True,
            num_gpus_per_node=num_gpus,
            max_colocated_worker_groups=1,
        )

        # Create a config with optional modifications
        config = deepcopy(basic_llama_test_config)
        if config_updates:
            config.update(config_updates)

        print("Creating training Policy...")
        policy = Policy(
            cluster=cluster,
            config=config,
            init_reference_model=False,
            tokenizer=tokenizer,
        )

        # Create a test batch
        print("Creating test batch...")
        # set random seed
        torch.manual_seed(42)

        # Create test input_ids and attention_mask
        input_ids = torch.randint(0, 32000, (8, 128))  # 8 sequences, each of length 128
        attention_mask = torch.ones(8, 128)

        # Calculate input_lengths (all sequences are full length in this test)
        input_lengths = attention_mask.sum(dim=1).to(torch.int32)

        data = BatchedDataDict(
            {
                "input_ids": input_ids,
                "input_lengths": input_lengths,
                "attention_mask": attention_mask,  # Keep for compatibility with loss functions
                "labels": torch.randint(0, 32000, (8, 128)),
                "sample_mask": torch.ones(8),
            }
        )

        # Create loss function
        loss_fn: LossFunction = SimpleLoss()

        # Provide the resources to the test
        yield policy, cluster, data, loss_fn

    except Exception as e:
        print(f"Error during training setup: {e}")
        pytest.skip(f"Training setup failed: {e}")
    finally:
        # Clean up after the test
        print("Cleaning up resources for test")
        policy.worker_group.shutdown()
        cluster.shutdown()


def get_max_gpu_utilization(policy):
    max_memory_allocated = 0
    max_memory_reserved = 0
    gpu_infos = ray.get([w.get_gpu_info.remote() for w in policy.worker_group.workers])
    for info in gpu_infos:
        max_memory_allocated = max(max_memory_allocated, info["memory_allocated_mb"])
        max_memory_reserved = max(max_memory_reserved, info["memory_reserved_mb"])
    return max_memory_allocated, max_memory_reserved


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "num_gpus, training_setup, config_name",
    [
        (1, None, "default"),
        (2, None, "default"),
        (2, {"fsdp_offload_enabled": True}, "fsdp_offload"),
        (2, {"activation_checkpointing_enabled": True}, "activation_checkpointing"),
    ],
    indirect=["training_setup"],
    ids=[
        "1gpu_default",
        "2gpu_default",
        "2gpu_fsdp_offload",
        "2gpu_activation_checkpointing",
    ],
)
def test_lm_policy_training(training_setup, tracker, num_gpus, config_name):
    def verify_loss_tensor(loss_tensor):
        assert not torch.isnan(loss_tensor).any(), "Loss should not be NaN"
        assert not torch.isinf(loss_tensor).any(), "Loss should not be Inf"
        return loss_tensor

    policy, cluster, data, loss_fn = training_setup

    # Verify resources were created properly
    assert policy is not None, "Training policy was not created properly"
    assert cluster is not None, "Training cluster was not created properly"
    assert data is not None, "Test data was not created properly"
    assert loss_fn is not None, "Loss function was not created properly"

    # Call prepare_for_training if available
    print(
        f"\nPreparing for training with {num_gpus} GPU(s) and {config_name} config..."
    )
    policy.prepare_for_training()

    losses = []
    for steps in range(4):
        results = policy.train(data, loss_fn)

        # Verify results
        assert "loss" in results, "Training results should contain 'loss'"
        loss_tensor = results["loss"]
        verify_loss_tensor(loss_tensor)
        losses.append(loss_tensor[-1].item())

        print(
            f"Training loss with {num_gpus} GPU(s) and {config_name} config: {results['loss']}"
        )

    policy.finish_training()
    assert losses[0] > losses[-1], "Loss should decrease over training iterations"

    after_training_mem_allocated, after_training_mem_reserved = get_max_gpu_utilization(
        policy
    )
    print(
        f"Max GPU Utilization after training with {num_gpus} GPU(s) and {config_name} config: {after_training_mem_allocated:,.1f} MB allocated, "
        f"{after_training_mem_reserved:,.1f} MB reserved"
    )
    tracker.track(
        f"{num_gpus}gpu_{config_name}_after_training_mem_allocated",
        after_training_mem_allocated,
    )
    tracker.track(
        f"{num_gpus}gpu_{config_name}_after_training_mem_reserved",
        after_training_mem_reserved,
    )

    policy.offload_after_refit()
    after_offload_mem_allocated, after_offload_mem_reserved = get_max_gpu_utilization(
        policy
    )
    print(
        f"Max GPU Utilization after offload with {num_gpus} GPU(s) and {config_name} config: {after_offload_mem_allocated:,.1f} MB allocated, "
        f"{after_offload_mem_reserved:,.1f} MB reserved"
    )
    tracker.track(
        f"{num_gpus}gpu_{config_name}_after_offload_mem_allocated",
        after_offload_mem_allocated,
    )
    tracker.track(
        f"{num_gpus}gpu_{config_name}_after_offload_mem_reserved",
        after_offload_mem_reserved,
    )

    # Compare memory after offload to memory after training
    if config_name == "fsdp_offload":
        # With FSDP offload, memory usage after training should already be low
        assert after_training_mem_allocated < 1_200, (
            "FSDP offload after training should be less than 1.2GB)"
        )
    else:
        assert after_training_mem_allocated > 5_000, (
            f"Memory after training with {config_name} config should be more than 5GB"
        )

    assert after_offload_mem_allocated < 1_200, (
        "Memory after offload should be less than 1.2GB"
    )


@pytest.fixture
def generation_setup(request, test_input_data, tokenizer, num_gpus):
    """Setup and teardown specifically for generation tests."""
    policy = None
    cluster = None
    data = None
    init_reference_model = request.param

    try:
        # Create resources with unique name
        cluster_name = f"test-gen-{num_gpus}gpu-ref{init_reference_model}"
        print(
            f"Creating generation virtual cluster '{cluster_name}' for {num_gpus} GPUs "
            f"(ref_model={init_reference_model})..."
        )

        cluster = RayVirtualCluster(
            name=cluster_name,
            bundle_ct_per_node_list=[num_gpus],
            use_gpus=True,
            num_gpus_per_node=num_gpus,
            max_colocated_worker_groups=1,
        )

        config = basic_llama_test_config
        config["generation"] = configure_generation_config(
            config["generation"], tokenizer
        )

        print("Creating generation Policy...")
        policy = Policy(
            cluster=cluster,
            config=config,
            tokenizer=tokenizer,
            init_reference_model=request.param,
        )

        # Create a test batch
        print("Creating test batch...")
        torch.manual_seed(42)  # For reproducibility

        # Prepare test data
        data, prompts, expected_generations = test_input_data

        # Provide the resources to the test
        yield policy, cluster, data, prompts, expected_generations

    except Exception as e:
        print(f"Error during generation setup: {e}")
        pytest.skip(f"Generation setup failed: {e}")
    finally:
        # Clean up after the test
        print("Cleaning up resources for test")
        policy.worker_group.shutdown()
        cluster.shutdown()


@pytest.mark.timeout(180)
@pytest.mark.parametrize("num_gpus", [1, 2], ids=["1gpu", "2gpu"])
@pytest.mark.parametrize("generation_setup", [False], indirect=True)
def test_lm_policy_generation(generation_setup, tokenizer, num_gpus, tracker):
    policy, cluster, data, prompts, expected_generations = generation_setup

    # Verify resources were created properly
    assert policy is not None, "Generation policy was not created properly"
    assert cluster is not None, "Generation cluster was not created properly"
    assert data is not None, "Test data was not created properly"

    # Call prepare_for_generation if available
    print("Preparing for generation...")
    policy.prepare_for_generation()

    # Generate text
    print("Generating text...")
    results = policy.generate(data, greedy=True)

    # Verify results
    assert "output_ids" in results, "Generation results should contain 'output_ids'"
    output_ids = results["output_ids"]

    # run logprob calculation manually to verify
    fprop_logprob_data = BatchedDataDict(
        {
            "input_ids": results.get("output_ids"),
            "input_lengths": results.get("unpadded_sequence_lengths"),
        }
    )
    fprop_results = policy.get_logprobs(fprop_logprob_data)
    for i, length in enumerate(data["input_lengths"]):
        fprop_results["logprobs"][i, :length] = 0

    for i, valid_seq_len in enumerate(results["unpadded_sequence_lengths"]):
        fprop_results["logprobs"][i, valid_seq_len:] = 0

    # Basic validation of output shape and content
    assert isinstance(output_ids, torch.Tensor), "Output should be a tensor"
    assert output_ids.dim() == 2, (
        "Output should be 2-dimensional [batch_size, seq_length]"
    )
    assert output_ids.size(0) == data.get("input_ids").size(0), (
        "Output batch size should match input"
    )
    assert output_ids.size(1) > data.get("input_ids").size(1), (
        "Output should be longer than input"
    )

    # validate that the logprobs are correct
    avg_prob_mult_error = torch.mean(
        torch.exp(torch.abs(results["logprobs"] - fprop_results["logprobs"]))
    )
    print(f"avg prob mult error: {avg_prob_mult_error}")
    tracker.track(f"avg_prob_mult_error_{num_gpus}gpu", float(avg_prob_mult_error))
    assert avg_prob_mult_error <= 1.025

    # get logprobs for the expected generations
    expected_tokenized = tokenizer(
        expected_generations,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
        padding_side="right",
    )

    # Calculate input_lengths for expected generations
    expected_lengths = expected_tokenized["attention_mask"].sum(dim=1).to(torch.int32)

    expected_data = BatchedDataDict(
        {
            "input_ids": expected_tokenized["input_ids"],
            "input_lengths": expected_lengths,
        }
    )

    expected_logprobs = policy.get_logprobs(expected_data)["logprobs"]
    mean_lps = torch.mean(expected_logprobs * expected_tokenized["attention_mask"])
    tracker.track(f"mean_lps_{num_gpus}gpu", float(mean_lps))
    assert mean_lps > -1.7, "Expected logprobs should be greater than -1.7"
    assert mean_lps < -1.4, "Expected logprobs should be less than -1.4"

    # Call finish_generation if available
    print("Finishing generation...")
    policy.finish_generation()


@pytest.mark.timeout(180)
@pytest.mark.parametrize("num_gpus", [1, 2], ids=["1gpu", "2gpu"])
@pytest.mark.parametrize("generation_setup", [True], indirect=True)
def test_all_lm_policy_generation_lps_ref_training(generation_setup):
    policy, cluster, data, prompts, expected_generations = generation_setup

    # Verify resources were created properly
    assert policy is not None, "Generation policy was not created properly"
    assert cluster is not None, "Generation cluster was not created properly"
    assert data is not None, "Test data was not created properly"

    # Create reference data by generating with the model
    print("creating some data")
    ref_results = policy.generate(data, greedy=True)

    # Create training data with reference outputs
    token_loss_mask = torch.ones_like(ref_results["output_ids"])
    token_loss_mask[:, : data.get("input_ids").size(1)] = 0

    for idx, length in enumerate(ref_results["unpadded_sequence_lengths"]):
        token_loss_mask[idx, length:] = 0

    train_data = BatchedDataDict(
        {
            "input_ids": ref_results["output_ids"],
            "input_lengths": ref_results["unpadded_sequence_lengths"],
            "token_loss_mask": token_loss_mask,
            "sample_mask": torch.ones(data.get("input_ids").size(0)),
        }
    )

    fprop_logprobs = policy.get_logprobs(train_data)["logprobs"]

    loss_fn: LossFunction = SimpleNLLLoss()

    # Train for a few steps
    policy.prepare_for_training()
    losses = []
    for step in range(8):
        results = policy.train(train_data, loss_fn)

        # Verify results
        assert "loss" in results, "Training results should contain 'loss'"
        loss_tensor = results["loss"]
        assert not torch.isnan(loss_tensor).any(), "Loss should not be NaN"
        assert not torch.isinf(loss_tensor).any(), "Loss should not be Inf"
        losses.append(loss_tensor[-1].item())

        print(f"Training loss at step {step}: {results['loss']}")

    policy.finish_training()

    post_train_reference_logprobs = policy.get_reference_policy_logprobs(train_data)[
        "reference_logprobs"
    ]
    post_train_fprop_logprobs = policy.get_logprobs(train_data)["logprobs"]

    # Verify that the reference policy logprobs match the original policy logprobs
    assert torch.allclose(fprop_logprobs, post_train_reference_logprobs), (
        "Logprobs from policy before training and reference policy after training should match"
    )

    # Calculate NLL before and after training
    pre_train_nll = -torch.sum(fprop_logprobs * token_loss_mask, dim=-1)
    post_train_nll = -torch.sum(post_train_fprop_logprobs * token_loss_mask, dim=-1)
    print(f"Pre-training NLL: {pre_train_nll.mean().item()}")
    print(f"Post-training NLL: {post_train_nll.mean().item()}")

    # Verify that training improved the model's predictions on every sample
    assert torch.all(post_train_nll < pre_train_nll), (
        "Model should improve at predicting its own generations after training"
    )
    assert torch.all(post_train_nll < 5), (
        "Model should improve at predicting its own generations after training"
    )

    # Verify loss decreased during training
    assert losses[0] > losses[-1], "Loss should decrease over training iterations"


def test_lm_policy_generation_with_stop(test_input_data, tokenizer):
    # Create resources with unique name
    cluster_name = "test-generate-with-stop"
    print(f"Creating training virtual cluster '{cluster_name}'...")

    cluster = RayVirtualCluster(
        name=cluster_name,
        bundle_ct_per_node_list=[2],  # Single node, 2 gpus
        use_gpus=True,
        num_gpus_per_node=2,  # Using both GPUs
        max_colocated_worker_groups=1,  # Only one worker group
    )

    # Create separate configs for each policy
    config = deepcopy(basic_llama_test_config)
    config["generation"] = configure_generation_config(config["generation"], tokenizer)
    # Add stop strings for testing
    config["generation"]["stop_token_ids"] = [12095, 1112]  # ["ĠParis", "..."]
    config["generation"]["stop_strings"] = ["the"]

    # Ensure we can get same output
    assert config["model_name"] == "Qwen/Qwen3-0.6B", (
        "Model name should be Qwen/Qwen3-0.6B to get expected output"
    )

    # Create policy
    policy = Policy(cluster=cluster, config=config, tokenizer=tokenizer)

    # Call prepare_for_generation if available
    print("Preparing for generation...")
    policy.prepare_for_generation()

    # Generate text
    print("Generating text...")
    data, _, _ = test_input_data
    results = policy.generate(data, greedy=True)
    output_ids = results["output_ids"]

    # Call finish_generation if available
    print("Finishing generation...")
    policy.finish_generation()

    # Check result
    generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    assert (
        generated_texts
        == [
            "Write a story about a magical forest where the",  # trees are made of stars and the ground is made of light. The
            "Explain how photosynthesis works in the",  # context of the environment and the role of the sun in it.\nAnswer
            "What are the benefits of exercise? What are the",  # risks of exercise? What are the benefits and risks of physical activity
            "Describe the water cycle and its importance in the",  # environment.\nAnswer:\nThe **water cycle** is a
            "What is the capital of France? The capital of France is Paris",  # . The answer is Paris. The answer is Paris
            "Who is the president of the USA? The answer is the",  # president of the United States of America, which is the president
            "What is the capital of the moon? The answer is...",  # ? Let me think. I know that the moon is a
            "Where is the sun? Where is the",  # moon? Where is the earth? Where is the sun in the
        ]
    ), "Output should be the same as the expected output"

    # Clean up after the test
    print("Cleaning up resources for test")
    policy.worker_group.shutdown()
    cluster.shutdown()


@pytest.mark.timeout(180)
@pytest.mark.parametrize("num_gpus", [2], ids=["2gpu"])
def test_loss_independent_of_microbatch_size(num_gpus, tokenizer):
    """Tests that changing microbatch size while keeping global batch size constant does not affect loss values."""

    # Create test batch with global batch size of 8
    global_batch_size = 8
    seq_len = 128
    vocab_size = 32000

    # Create test input_ids and attention_mask
    input_ids = torch.randint(0, vocab_size, (global_batch_size, seq_len))
    attention_mask = torch.ones(global_batch_size, seq_len)
    input_lengths = attention_mask.sum(dim=1).to(torch.int32)

    # Create data dictionary
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "attention_mask": attention_mask,
            "token_mask": torch.triu(
                torch.ones(global_batch_size, seq_len), diagonal=1
            ),  ## give different examples different numbers of valid tokens
            "sample_mask": torch.ones((global_batch_size,)),
            "labels": torch.randint(0, vocab_size, (global_batch_size, seq_len)),
            "num_valid_tokens_in_batch": torch.tensor(
                [seq_len] * global_batch_size, dtype=torch.float32
            ),
            "advantages": torch.randn(global_batch_size, seq_len),
            "prev_logprobs": torch.randn(global_batch_size, seq_len),
            "reference_policy_logprobs": torch.randn(global_batch_size, seq_len),
            "generation_logprobs": torch.randn(global_batch_size, seq_len),
        }
    )

    # Compute loss with microbatching
    cluster = RayVirtualCluster(
        name=f"test-{num_gpus}gpu",
        bundle_ct_per_node_list=[num_gpus],
        use_gpus=True,
        num_gpus_per_node=num_gpus,
        max_colocated_worker_groups=1,
    )

    config = basic_llama_test_config

    print("Creating training Policy...")
    policy_mbs1 = Policy(
        cluster=cluster,
        config=config,
        init_reference_model=False,
        tokenizer=tokenizer,
    )
    # Test NLLLoss and ClippedPGLossFn with mbs=1
    nll_loss_fn = NLLLoss()
    pg_loss_fn = ClippedPGLossFn(
        {
            "ratio_clip_min": 0.2,
            "ratio_clip_max": 0.2,
            "ratio_clip_c": None,
            "reference_policy_kl_penalty": 0.1,
            "disable_ppo_ratio": False,
            "use_on_policy_kl_approximation": False,
            "use_importance_sampling_correction": False,
            "token_level_loss": True,
        }
    )

    # Compute loss with mbs1
    policy_mbs1.prepare_for_training()
    mbs1_results = policy_mbs1.train(data, nll_loss_fn)
    mbs1_nll_loss = mbs1_results["loss"]

    mbs1_results = policy_mbs1.train(data, pg_loss_fn)
    mbs1_pg_loss = mbs1_results["loss"]

    policy_mbs1.worker_group.shutdown()

    # Compute loss with mbs2
    config = basic_llama_test_config
    config["train_micro_batch_size"] = 2
    config["generation"] = configure_generation_config(config["generation"], tokenizer)

    print("Creating training Policy...")
    policy_mbs2 = Policy(
        cluster=cluster,
        config=config,
        init_reference_model=False,
        tokenizer=tokenizer,
    )

    # Compute loss with mbs2
    policy_mbs2.prepare_for_training()
    mbs2_results = policy_mbs2.train(data, nll_loss_fn)
    mbs2_nll_loss = mbs2_results["loss"]

    mbs2_results = policy_mbs2.train(data, pg_loss_fn)
    mbs2_pg_loss = mbs1_results["loss"]

    # Verify NLLLoss is independent of microbatch size
    torch.testing.assert_close(mbs1_nll_loss, mbs2_nll_loss)
    torch.testing.assert_close(mbs1_pg_loss, mbs2_pg_loss)

    cluster.shutdown()
    policy_mbs2.worker_group.shutdown()
