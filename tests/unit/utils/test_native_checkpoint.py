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
from tempfile import TemporaryDirectory

import pytest
import torch
from transformers import AutoModelForCausalLM

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.native_checkpoint import (
    ModelState,
    OptimizerState,
    convert_dcp_to_hf,
    load_checkpoint,
    save_checkpoint,
)
from tests.unit.test_utils import SimpleLoss

# Define basic test config
simple_policy_config = {
    "model_name": "Qwen/Qwen3-0.6B",  # "hf-internal-testing/tiny-random-Gemma3ForCausalLM",
    "tokenizer": {
        "name": "Qwen/Qwen3-0.6B",
    },
    "train_global_batch_size": 4,
    "train_micro_batch_size": 1,
    "logprob_batch_size": 1,
    "max_total_sequence_length": 1024,
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
        "enabled": True,
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
    "max_grad_norm": 1.0,
    "generation": {
        "backend": "vllm",
        "temperature": 1.0,
        "colocated": {"enabled": True},
    },
}


@pytest.fixture
def mock_experiment():
    model = torch.nn.ModuleList(
        [
            torch.nn.Linear(4, 4),
            torch.nn.LayerNorm(4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1),
        ]
    ).to("cuda")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    return model, optimizer, scheduler


@pytest.fixture(scope="function")
def cluster(num_gpus):
    """Create a virtual cluster for testing."""
    cluster_name = f"test-cluster-{num_gpus}gpu"
    print(f"Creating virtual cluster '{cluster_name}' for {num_gpus} GPUs...")
    # Create a cluster with num_gpus GPU
    virtual_cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[num_gpus],  # 1 node with num_gpus GPU bundle
        use_gpus=True,
        max_colocated_worker_groups=1,
        num_gpus_per_node=num_gpus,  # Use available GPUs
        name=cluster_name,
    )
    yield virtual_cluster  # Yield only the cluster object
    virtual_cluster.shutdown()


@pytest.fixture(scope="function")
def tokenizer():
    """Initialize tokenizer for the test model."""
    tokenizer = get_tokenizer(simple_policy_config["tokenizer"])
    return tokenizer


@pytest.fixture(scope="function")
def policy(cluster, tokenizer):
    """Initialize the policy."""
    policy = Policy(
        cluster=cluster,
        tokenizer=tokenizer,
        config=simple_policy_config,
        init_optimizer=True,
        init_reference_model=False,
    )
    yield policy
    policy.worker_group.shutdown()


@pytest.fixture(scope="module", autouse=True)
def skip_tied_weight_check_for_all():
    """Automatically skip tied weight check for all tests in this module."""
    os.environ["NRL_SKIP_TIED_WEIGHT_CHECK"] = "1"

    yield

    # Restore the original value
    os.environ.pop("NRL_SKIP_TIED_WEIGHT_CHECK", None)


def get_dummy_state_dict(state_dict, dummy_dict={}):
    """Recursively get the dummy state dict
    by replacing tensors with random ones of the same shape.
    """
    for k in state_dict.keys():
        if isinstance(state_dict[k], dict):
            dummy_dict[k] = get_dummy_state_dict(state_dict[k], {})
        elif isinstance(state_dict[k], torch.Tensor):
            dummy_dict[k] = torch.randn(state_dict[k].shape)
        else:
            dummy_dict[k] = state_dict[k]
    return dummy_dict


def check_dict_equality(dict1, dict2):
    """Recursively check equality of two dictionaries"""
    for k in dict1.keys():
        if isinstance(dict1[k], dict):
            check_dict_equality(dict1[k], dict2[k])
        elif isinstance(dict1[k], torch.Tensor):
            assert torch.allclose(dict1[k], dict2[k])
        else:
            assert dict1[k] == dict2[k]


def assert_recursive_dict_different(dict1, dict2):
    """Recursively assert that two dictionaries are different"""
    try:
        check_dict_equality(dict1, dict2)
    except AssertionError:
        return
    raise AssertionError("Dictionaries are equal")


def test_model_state(mock_experiment):
    test_model, _, _ = mock_experiment
    model_state = ModelState(test_model)
    state_dict = model_state.state_dict()

    ## relu has no parameters
    expected_keys = {
        "0.bias",
        "0.weight",
        "1.bias",
        "1.weight",
        "3.bias",
        "3.weight",
    }
    assert set(state_dict.keys()) == expected_keys

    dummy_model_state_dict = get_dummy_state_dict(state_dict, {})

    ## update the model's state dict and verify that the model's parameters are updated
    model_state.load_state_dict(dummy_model_state_dict)
    new_model_state_dict = model_state.state_dict()
    check_dict_equality(new_model_state_dict, dummy_model_state_dict)


def test_optimizer_state(mock_experiment):
    test_model, optimizer, scheduler = mock_experiment

    optim_state = OptimizerState(test_model, optimizer, scheduler)
    state_dict = optim_state.state_dict()

    assert set(state_dict.keys()) == {"optim", "sched"}

    ## relu has no parameters
    expected_keys = {
        "0.bias",
        "0.weight",
        "1.bias",
        "1.weight",
        "3.bias",
        "3.weight",
    }

    assert set(state_dict["optim"]["state"].keys()) == expected_keys

    dummy_state_dict = get_dummy_state_dict(state_dict, {})

    optim_state.load_state_dict(dummy_state_dict)
    new_state_dict = optim_state.state_dict()
    check_dict_equality(new_state_dict, dummy_state_dict)


def test_save_and_load_model_only(mock_experiment):
    test_model, _, _ = mock_experiment

    with TemporaryDirectory() as tmp_dir:
        save_checkpoint(test_model, os.path.join(tmp_dir, "test_model_only"))
        assert os.path.exists(os.path.join(tmp_dir, "test_model_only"))
        assert not os.path.exists(os.path.join(tmp_dir, "test_model_only-hf"))
        assert set(os.listdir(os.path.join(tmp_dir, "test_model_only"))) == {
            ".metadata",
            "__0_0.distcp",
        }


def test_save_and_load_model_and_optimizer(mock_experiment):
    test_model, optimizer, scheduler = mock_experiment
    for _ in range(5):
        scheduler.step()

    with TemporaryDirectory() as tmp_dir:
        save_checkpoint(
            test_model,
            os.path.join(tmp_dir, "model_and_optimizer/model"),
            optimizer,
            scheduler,
            optimizer_path=os.path.join(tmp_dir, "model_and_optimizer/optimizer"),
        )

        assert set(os.listdir(os.path.join(tmp_dir, "model_and_optimizer/model"))) == {
            ".metadata",
            "__0_0.distcp",
        }
        assert set(
            os.listdir(os.path.join(tmp_dir, "model_and_optimizer/optimizer"))
        ) == {
            ".metadata",
            "__0_0.distcp",
        }

        ## modify the model, optimizer, and scheduler and verify that loading the checkpoint overrides the values
        new_linear = torch.nn.Linear(4, 4)
        new_linear.weight = torch.nn.Parameter(torch.ones([4, 4]).to("cuda"))
        new_linear.bias = torch.nn.Parameter(torch.ones(4).to("cuda"))
        new_model = torch.nn.ModuleList(
            [
                new_linear,
                torch.nn.LayerNorm(4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 1),
            ]
        ).to("cuda")

        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = torch.optim.lr_scheduler.StepLR(
            new_optimizer, step_size=4, gamma=0.2
        )
        load_checkpoint(
            new_model,
            os.path.join(tmp_dir, "model_and_optimizer/model"),
            new_optimizer,
            new_scheduler,
            optimizer_path=os.path.join(tmp_dir, "model_and_optimizer/optimizer"),
        )

    assert scheduler.state_dict() == new_scheduler.state_dict()
    check_dict_equality(new_model.state_dict(), test_model.state_dict())
    check_dict_equality(new_optimizer.state_dict(), optimizer.state_dict())


@pytest.mark.parametrize("num_gpus", [1, 2], ids=["1gpu", "2gpu"])
def test_convert_dcp_to_hf(policy, num_gpus):
    ## warm up with a forward pass
    ## this is needed before saving a checkpoint because FSDP does some lazy initialization
    input_ids = torch.randint(0, 16000, (4, 128))  # 4 sequences, each of length 128
    attention_mask = torch.ones(4, 128)
    input_lengths = attention_mask.sum(dim=1).to(torch.int32)
    dummy_fwd_dict = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "attention_mask": attention_mask,
            "labels": torch.randint(0, 16000, (4, 128)),
            "sample_mask": torch.ones(input_ids.shape[0]),
        }
    )
    policy.train(dummy_fwd_dict, SimpleLoss())

    with TemporaryDirectory() as tmp_dir:
        policy.save_checkpoint(
            os.path.join(tmp_dir, "test_hf_and_dcp"),
        )

        # Dynamically create the expected set of distcp files based on num_gpus
        expected_distcp_files = {f"__{rank}_0.distcp" for rank in range(num_gpus)}
        expected_files = expected_distcp_files.union({".metadata"})

        ## make sure we save both HF and DCP checkpoints
        assert (
            set(os.listdir(os.path.join(tmp_dir, "test_hf_and_dcp"))) == expected_files
        )

        offline_converted_model_path = convert_dcp_to_hf(
            os.path.join(tmp_dir, "test_hf_and_dcp"),
            os.path.join(tmp_dir, "test_hf_and_dcp-hf-offline"),
            simple_policy_config["model_name"],
            # TODO: After the following PR gets merged:
            # https://github.com/NVIDIA-NeMo/RL/pull/148/files
            # tokenizer should be copied from policy/tokenizer/* instead of relying on the model name
            # We can expose a arg at the top level --tokenizer_path to plumb that through.
            # This is more stable than relying on the current NeMo-RL get_tokenizer() which can
            # change release to release.
            simple_policy_config["model_name"],
        )

        offline_converted_model = AutoModelForCausalLM.from_pretrained(
            offline_converted_model_path
        )

        original_model = AutoModelForCausalLM.from_pretrained(
            simple_policy_config["model_name"]
        )

    # Ensure the offline checkpoint is different from the original
    assert_recursive_dict_different(
        offline_converted_model.state_dict(), original_model.state_dict()
    )
