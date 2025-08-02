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

import pytest
import ray
import torch

# Define a custom marker for model configuration tests
pytestmark = pytest.mark.modelconfig

from transformers import AutoModelForCausalLM

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.algorithms.loss_functions import ClippedPGLossFn, NLLLoss
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.lm_policy import Policy
from tests.unit.test_utils import SimpleLoss


def create_test_config(
    model_name: str,
    tp: int = 1,
    cp: int = 1,
    sequence_parallel: bool = False,
    cpu_offload: bool = False,
    activation_checkpointing: bool = False,
    custom_parallel_plan: str | None = None,
) -> PolicyConfig:
    return {
        "model_name": model_name,
        "tokenizer": {"name": model_name},
        "generation_batch_size": 1,  # Small batch size for testing
        "train_global_batch_size": 4,
        "train_micro_batch_size": 1,
        "learning_rate": 5e-6,
        "logprob_batch_size": 1,
        "precision": "float32",
        "generation": {
            "backend": "hf",
            "temperature": 1.0,
            "max_new_tokens": 16,  # Small number of tokens for testing
            "top_p": 1.0,
            "top_k": None,
            "stop_token_ids": None,
            "stop_strings": None,
            "colocated": {
                "enabled": True,
                "resources": {
                    "gpus_per_node": None,
                    "num_nodes": None,
                },
            },
        },
        "dtensor_cfg": {
            "enabled": True,
            "cpu_offload": cpu_offload,
            "sequence_parallel": sequence_parallel,
            "activation_checkpointing": activation_checkpointing,
            "tensor_parallel_size": tp,
            "context_parallel_size": cp,
            "custom_parallel_plan": custom_parallel_plan,
        },
        "dynamic_batching": {
            "enabled": True,
            "train_mb_tokens": 128,
            "logprob_mb_tokens": 128,
            "sequence_length_round": 4,
        },
        "sequence_packing": {
            "enabled": False,
        },
        "optimizer": {
            "name": "torch.optim.AdamW",
            "kwargs": {
                "lr": 5e-6,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "foreach": False,
                "fused": False,
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


@pytest.fixture(scope="module")
def two_gpu_virtual_cluster():
    cluster_name = "test"
    print(f"Creating virtual cluster '{cluster_name}'...")
    cluster = RayVirtualCluster(
        name=cluster_name,
        bundle_ct_per_node_list=[2],  # Use tp bundles, one per GPU
        use_gpus=True,
        num_gpus_per_node=2,  # Using tp GPUs
        max_colocated_worker_groups=1,  # Only one worker group
    )
    yield cluster
    print("Shutting down virtual cluster...")
    cluster.shutdown()


@pytest.fixture(scope="function")
def gc_collect():
    """Helper function to force garbage collection after a test"""
    import gc

    yield
    gc.collect()


@pytest.fixture
def policy_setup(two_gpu_virtual_cluster, tiny_llama_model_path):
    """Setup and teardown for policy tests - creates a virtual cluster and policy."""
    config = create_test_config(tiny_llama_model_path)
    tokenizer = get_tokenizer(config["tokenizer"])
    config["generation"] = configure_generation_config(config["generation"], tokenizer)

    print("Creating Policy...")
    policy = Policy(cluster=two_gpu_virtual_cluster, config=config, tokenizer=tokenizer)

    yield policy

    print("Shutting down policy...")
    policy.shutdown()


@pytest.mark.hf_gated
@pytest.mark.timeout(360)
def test_lm_policy_init(policy_setup):
    policy = policy_setup

    # Verify we have two workers, one per GPU
    assert len(policy.worker_group.workers) == 2, "Should have 2 workers, one per GPU"

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
    assert len(set(gpu_ranks)) == 2, f"Expected 2 different ranks, got {gpu_ranks}"
    assert set(gpu_ranks) == {0, 1}, f"Expected ranks 0 and 1, got {gpu_ranks}"

    # Check 2: Verify workers have different local_ranks
    local_ranks = [info["local_rank"] for info in gpu_infos]
    assert len(set(local_ranks)) == 2, (
        f"Expected 2 different local_ranks, got {local_ranks}"
    )
    assert set(local_ranks) == {0, 1}, (
        f"Expected local_ranks 0 and 1, got {local_ranks}"
    )

    # Check 3: Verify workers have different CUDA_VISIBLE_DEVICES
    cuda_visible_devices = [
        info["env_vars"].get("CUDA_VISIBLE_DEVICES") for info in gpu_infos
    ]
    assert len(set(cuda_visible_devices)) == 2, (
        f"Expected different CUDA_VISIBLE_DEVICES, got {cuda_visible_devices}"
    )

    # Check 4: Verify all workers report correct world_size
    for info in gpu_infos:
        assert info["world_size"] == 2, (
            f"Expected world_size=2, got {info['world_size']}"
        )
        assert info["env_vars"]["WORLD_SIZE"] == "2", (
            f"Expected WORLD_SIZE=2, got {info['env_vars']['WORLD_SIZE']}"
        )

    # Check 5: Verify GPU memory is allocated on both GPUs
    for info in gpu_infos:
        assert info["memory_allocated_mb"] > 10, (
            f"Not enough memory allocated on GPU for rank {info['rank']}: {info['memory_allocated_mb']:.2f} MB"
        )

    # Check 6: Verify model parameters are on CUDA devices for both workers
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
def training_setup(request, two_gpu_virtual_cluster):
    """Setup and teardown specifically for training tests."""
    (
        model_fixture_name,
        tp,
        cp,
        sequence_parallel,
        cpu_offload,
        activation_checkpointing,
    ) = request.param

    # Get the actual model path from the requested fixture
    model_name = request.getfixturevalue(model_fixture_name)
    policy = None
    data = None
    loss_fn = None

    try:
        config = create_test_config(
            model_name, tp, cp, sequence_parallel, cpu_offload, activation_checkpointing
        )
        tokenizer = get_tokenizer(config["tokenizer"])
        print(
            f"Creating training Policy with tp={tp}, cpu_offload={cpu_offload}, sequence_parallel={sequence_parallel}, activation_checkpointing={activation_checkpointing}..."
        )
        policy = Policy(
            cluster=two_gpu_virtual_cluster,
            config=config,
            tokenizer=tokenizer,
            init_reference_model=False,
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
        yield policy, data, loss_fn

    except Exception as e:
        print(f"Error during training setup: {e}")
        pytest.skip(f"Training setup failed: {e}")
    finally:
        # Clean up after the test
        print("Cleaning up resources for test")
        policy.shutdown()


@pytest.mark.hf_gated
@pytest.mark.timeout(360)
@pytest.mark.parametrize(
    "training_setup",
    [
        # model_fixture_name        tp cp  sp     cpu    act
        ("tiny_llama_model_path", 1, 1, False, False, False),
        ("tiny_llama_model_path", 1, 1, True, False, False),
        ("tiny_llama_model_path", 1, 1, False, True, False),
        ("tiny_llama_model_path", 1, 1, False, False, True),
        ("tiny_llama_model_path", 1, 2, False, False, False),
        ("tiny_qwen2_model_path", 1, 1, True, True, False),
        ("tiny_qwen2_model_path", 1, 1, True, False, True),
        ("tiny_qwen2_model_path", 1, 1, False, True, True),
        ("tiny_qwen2_model_path", 1, 1, True, True, True),
        ("tiny_qwen2_model_path", 1, 2, False, False, False),
        ("tiny_qwen3_model_path", 1, 1, True, True, False),
        ("tiny_qwen3_model_path", 1, 1, True, False, True),
        ("tiny_qwen3_model_path", 1, 1, False, True, True),
        ("tiny_qwen3_model_path", 1, 1, True, True, True),
        ("tiny_qwen3_model_path", 1, 2, False, False, False),
        (
            "tiny_gemma3_model_path",
            1,
            1,
            True,
            True,
            False,
        ),  # gemma3 doesn't support spda
        ("tiny_gemma3_model_path", 1, 1, True, False, True),
        ("tiny_gemma3_model_path", 1, 1, False, True, True),
        ("tiny_gemma3_model_path", 1, 1, True, True, True),
        # CP doesn't support gemma3 due to spda input has attent_mask != None.
    ],
    indirect=True,
)
def test_dtensor_worker_training(training_setup):
    def verify_loss_tensor(loss_tensor):
        assert not torch.isnan(loss_tensor).any(), "Loss should not be NaN"
        assert not torch.isinf(loss_tensor).any(), "Loss should not be Inf"
        return loss_tensor

    policy, data, loss_fn = training_setup

    # Verify resources were created properly
    assert policy is not None, "Training policy was not created properly"
    assert data is not None, "Test data was not created properly"
    assert loss_fn is not None, "Loss function was not created properly"

    # Call prepare_for_training if available
    print("\nPreparing for training...")
    policy.prepare_for_training()

    losses = []
    for steps in range(2):
        results = policy.train(data, loss_fn)

        # Verify results
        assert "loss" in results, "Training results should contain 'loss'"
        loss_tensor = results["loss"]
        verify_loss_tensor(loss_tensor)
        losses.append(loss_tensor[-1].item())

        print(f"Training loss: {results['loss']}")

    policy.finish_training()

    # Verify loss changed between iterations (model parameters were updated)
    assert losses[0] > losses[-1], "Loss should decrease over training iterations"


@pytest.fixture
def logprob_setup(request, two_gpu_virtual_cluster):
    """Setup and teardown specifically for training tests."""
    (
        model_fixture_name,
        tp,
        cp,
        sequence_parallel,
        cpu_offload,
        activation_checkpointing,
    ) = request.param

    # Get the actual model path from the requested fixture
    model_name = request.getfixturevalue(model_fixture_name)
    policy = None
    data = None

    try:
        config = create_test_config(
            model_name, tp, cp, sequence_parallel, cpu_offload, activation_checkpointing
        )
        tokenizer = get_tokenizer(config["tokenizer"])
        print(
            f"Creating logprob Policy with tp={tp}, cpu_offload={cpu_offload}, sequence_parallel={sequence_parallel}, activation_checkpointing={activation_checkpointing}..."
        )
        policy = Policy(
            cluster=two_gpu_virtual_cluster,
            config=config,
            tokenizer=tokenizer,
            init_reference_model=False,
        )

        # Create a test batch
        print("Creating test batch...")
        # set random seed
        torch.manual_seed(66)

        # Create test input_ids and attention_mask
        input_ids = torch.randint(
            0, 32000, (8, 128)
        ).cuda()  # 8 sequences, each of length 128
        attention_mask = torch.ones(8, 128).cuda()

        # Calculate input_lengths (all sequences are full length in this test)
        input_lengths = attention_mask.sum(dim=1).to(torch.int32).cuda()

        data = BatchedDataDict(
            {
                "input_ids": input_ids,
                "input_lengths": input_lengths,
                "attention_mask": attention_mask,  # Keep for compatibility with loss functions
            }
        )

        with torch.no_grad():
            # run the log prob of regular hf model here
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="cuda", torch_dtype=torch.float32
            )
            hf_model.eval()
            outputs = hf_model(**data)

        log_probs = torch.nn.functional.log_softmax(
            outputs.logits.to(torch.float32), dim=-1
        )
        next_tokens = input_ids[:, 1:]
        log_probs = log_probs[:, :-1]
        token_logprobs = log_probs.gather(
            dim=-1, index=next_tokens.unsqueeze(-1)
        ).squeeze(-1)
        token_logprobs = torch.cat(
            [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
        ).cpu()

        data = data.to("cpu")

        # Provide the resources to the test
        yield policy, data, token_logprobs

    except Exception as e:
        print(f"Error during training setup: {e}")
        pytest.skip(f"Training setup failed: {e}")
    finally:
        # Clean up after the test
        print("Cleaning up resources for test")
        policy.shutdown()


@pytest.mark.hf_gated
@pytest.mark.timeout(360)
@pytest.mark.parametrize(
    "logprob_setup",
    [
        # TP=2, CP=1
        ("tiny_qwen2_model_path", 2, 1, False, True, False),
        ("tiny_qwen2_model_path", 2, 1, False, False, False),
        ("tiny_llama_model_path", 2, 1, False, False, False),
        ("tiny_llama_model_path", 2, 1, False, True, False),
        ("tiny_llama_model_path", 2, 1, False, True, True),
        ("tiny_qwen3_model_path", 2, 1, False, True, False),
        ("tiny_qwen3_model_path", 2, 1, False, False, False),
        ("tiny_gemma3_model_path", 2, 1, False, True, False),
        ("tiny_gemma3_model_path", 2, 1, False, False, False),
        # TP=1, CP=2
        ("tiny_qwen2_model_path", 1, 2, False, True, False),
        ("tiny_qwen2_model_path", 1, 2, False, False, False),
        ("tiny_llama_model_path", 1, 2, False, False, False),
        ("tiny_llama_model_path", 1, 2, False, True, False),
        ("tiny_llama_model_path", 1, 2, False, True, True),
        ("tiny_qwen3_model_path", 1, 2, False, True, False),
        ("tiny_qwen3_model_path", 1, 2, False, False, False),
    ],
    indirect=True,
)
def test_dtensor_worker_logprob_tp2_or_cp2_matches_unsharded(logprob_setup):
    policy, data, logprobs = logprob_setup

    # Verify resources were created properly assert policy is not None, "Policy was not created properly"
    assert data is not None, "Test data was not created properly"

    # Generate logprobs
    print("\nGenerating logprobs...")
    policy.prepare_for_lp_inference()
    policy_logprobs = policy.get_logprobs(data)["logprobs"]

    print("## MAX DIFF ###", torch.max(torch.abs(policy_logprobs - logprobs)))
    assert torch.allclose(policy_logprobs, logprobs), (
        f"max diff {torch.max(torch.abs(policy_logprobs - logprobs))}"
    )


@pytest.mark.hf_gated
def test_dtensor_tp_and_tied_model_with_custom_parallel_plan(
    two_gpu_virtual_cluster, tiny_llama_tied_model_path
):
    """Test that DTensor with a tp > 1 and a tied model with a custom parallel plan works."""
    from torch.distributed.tensor.parallel import ColwiseParallel
    from torch.distributed.tensor.placement_types import Replicate

    custom_parallel_plan = {
        "lm_head": ColwiseParallel(output_layouts=Replicate()),
        "model.embed_tokens": ColwiseParallel(output_layouts=Replicate()),
    }
    config = create_test_config(
        model_name=tiny_llama_tied_model_path,
        tp=2,
        cp=1,
        sequence_parallel=False,
        cpu_offload=False,
        activation_checkpointing=False,
        custom_parallel_plan=custom_parallel_plan,
    )
    tokenizer = get_tokenizer(config["tokenizer"])

    policy = Policy(
        tokenizer=tokenizer,
        config=config,
        init_optimizer=False,
        init_reference_model=False,
        cluster=two_gpu_virtual_cluster,
    )

    # Verify that the model is parallelized as expected
    state_dict = ray.get(policy.worker_group.workers[0].return_state_dict.remote())
    total_shape = state_dict["lm_head.weight"].shape
    sharded_shape = state_dict["lm_head.weight"].to_local().shape
    assert total_shape[0] == sharded_shape[0], (
        "lm_head.weight should have the same number of rows"
    )
    assert total_shape[1] == sharded_shape[1] * 2, (
        "lm_head.weight should be sharded across 2 GPUs"
    )

    # Clean up
    policy.shutdown()


@pytest.mark.hf_gated
@pytest.mark.timeout(180)
def test_dtensor_loss_independent_of_microbatch_size_two_gpus(
    two_gpu_virtual_cluster, tiny_llama_model_path
):
    """Tests that changing microbatch size while keeping global batch size constant does not affect loss values in DTensor."""
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
            ),  # give different examples different numbers of valid tokens
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

    # Test with mbs=1, 2 microbatches per GPU
    config = create_test_config(tiny_llama_model_path)
    tokenizer = get_tokenizer(config["tokenizer"])

    print("Creating training Policy with mbs=1...")
    policy_mbs1 = Policy(
        cluster=two_gpu_virtual_cluster,
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

    policy_mbs1.prepare_for_training()
    mbs1_nll_results = policy_mbs1.train(data, nll_loss_fn)
    mbs1_nll_loss = mbs1_nll_results["loss"]

    mbs1_pg_results = policy_mbs1.train(data, pg_loss_fn)
    mbs1_pg_loss = mbs1_pg_results["loss"]

    policy_mbs1.worker_group.shutdown()

    # Test with mbs=2, 1 microbatch per GPU
    config = create_test_config(tiny_llama_model_path)
    config["train_micro_batch_size"] = 2
    config["generation"] = configure_generation_config(config["generation"], tokenizer)

    print("Creating training Policy with mbs=2...")
    policy_mbs2 = Policy(
        cluster=two_gpu_virtual_cluster,
        config=config,
        init_reference_model=False,
        tokenizer=tokenizer,
    )

    # Test NLLLoss and ClippedPGLossFn with mbs=2
    policy_mbs2.prepare_for_training()
    mbs2_nll_results = policy_mbs2.train(data, nll_loss_fn)
    mbs2_nll_loss = mbs2_nll_results["loss"]

    mbs2_pg_results = policy_mbs2.train(data, pg_loss_fn)
    mbs2_pg_loss = mbs2_pg_results["loss"]

    # Verify both loss functions are independent of microbatch size
    torch.testing.assert_close(mbs1_nll_loss, mbs2_nll_loss, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(mbs1_pg_loss, mbs2_pg_loss, rtol=1e-5, atol=1e-5)

    policy_mbs2.worker_group.shutdown()
