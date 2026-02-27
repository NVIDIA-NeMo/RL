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
#!/usr/bin/env python3

"""
Functional test for converter roundtrip functionality.

This test:
1. Starts with a HuggingFace Qwen/Qwen2-0.5B checkpoint
2. Converts the model to torch DCP format
3. Converts the model to Megatron format (using community import)
4. Converts both the DCP and Megatron checkpoints back to HF format
5. Asserts that the converted DCP and Megatron checkpoints are identical and match the original HF checkpoint
"""

import copy
import os
import tempfile
import time
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.megatron.community_import import (
    export_model_from_megatron,
    import_model_from_hf_name,
)
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.native_checkpoint import convert_dcp_to_hf


def create_test_config() -> Dict[str, Any]:
    """Create a test configuration for SFT training."""
    return {
        "sft": {
            "max_num_epochs": 1,  ## unused, no training is actually done
            "max_num_steps": 2,
            "val_period": 2,
            "val_batches": 1,
            "val_global_batch_size": 4,
            "val_micro_batch_size": 2,
            "val_at_start": False,
            "val_at_end": False,
            "seed": 42,
        },
        "checkpointing": {
            "enabled": True,
            "checkpoint_dir": "/tmp/test_converter_checkpoints",
            "metric_name": "val_loss",
            "higher_is_better": False,
            "keep_top_k": 1,
            "save_period": 2,
        },
        "policy": {
            "model_name": "Qwen/Qwen2-0.5B",
            "tokenizer": {"name": "Qwen/Qwen2-0.5B"},
            "train_global_batch_size": 4,
            "train_micro_batch_size": 2,
            "max_total_sequence_length": 128,
            "precision": "bfloat16",
            "offload_optimizer_for_logprob": False,
            "dtensor_cfg": {
                "_v2": False,
                "enabled": True,
                "cpu_offload": False,
                "sequence_parallel": False,
                "activation_checkpointing": False,
                "tensor_parallel_size": 1,
                "context_parallel_size": 1,
                "custom_parallel_plan": None,
            },
            "dynamic_batching": {"enabled": False},
            "sequence_packing": {"enabled": False},
            "make_sequence_length_divisible_by": 1,
            "max_grad_norm": 1.0,
            "optimizer": {
                "name": "torch.optim.AdamW",
                "kwargs": {
                    "lr": 5.0e-6,
                    "weight_decay": 0.1,
                    "betas": [0.9, 0.98],
                    "eps": 1e-5,
                    "foreach": False,
                    "fused": False,
                },
            },
            "megatron_cfg": {
                "enabled": False,  # We'll use DCP for this test
            },
        },
        "data": {
            "max_input_seq_length": 128,
            "dataset_name": "squad",
            "add_bos": True,
            "add_eos": True,
            "add_generation_prompt": False,
        },
        "logger": {
            "log_dir": "/tmp/test_converter_logs",
            "wandb_enabled": False,
            "tensorboard_enabled": False,
            "monitor_gpus": False,
        },
        "cluster": {
            "gpus_per_node": 1,
            "num_nodes": 1,
        },
    }


def load_model_and_tokenizer(model_name: str):
    """Load the original HF model and tokenizer."""
    print(f"Loading original model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_model_state_dict(model):
    """Get the state dict of a model, ensuring all tensors are on CPU."""
    state_dict = model.state_dict()
    cpu_state_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cpu_state_dict[key] = value.detach().cpu()
        else:
            cpu_state_dict[key] = value
    return cpu_state_dict


def assert_state_dicts_equal(
    state_dict1: Dict[str, Any], state_dict2: Dict[str, Any], name1: str, name2: str
):
    """Assert that two state dictionaries are equal."""
    print(f"Comparing {name1} vs {name2}")

    # Check that keys match
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    if keys1 != keys2:
        missing_in_2 = keys1 - keys2
        missing_in_1 = keys2 - keys1
        raise AssertionError(
            f"State dict keys don't match between {name1} and {name2}.\n"
            f"Keys in {name1} but not in {name2}: {missing_in_2}\n"
            f"Keys in {name2} but not in {name1}: {missing_in_1}"
        )

    # Check that values match
    for key in keys1:
        val1 = state_dict1[key]
        val2 = state_dict2[key]

        if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
            if not torch.allclose(val1, val2, rtol=1e-5, atol=1e-5):
                max_diff = torch.max(torch.abs(val1 - val2)).item()
                raise AssertionError(
                    f"Tensors for key '{key}' don't match between {name1} and {name2}. "
                    f"Max difference: {max_diff}"
                )
        elif val1 != val2:
            raise AssertionError(
                f"Non-tensor values for key '{key}' don't match between {name1} and {name2}. "
                f"{name1}: {val1}, {name2}: {val2}"
            )

    print(f"✓ {name1} and {name2} are identical")


def check_file_exists(path: str) -> bool:
    """Check if a file exists."""
    for _ in range(10):
        if os.path.exists(path):
            return True
        time.sleep(0.5)
    return False


def create_dcp_checkpoint(
    model_name: str, config: Dict[str, Any], temp_dir: str
) -> str:
    """Create a DCP checkpoint without training."""
    print("Creating DCP checkpoint...")

    # Create cluster
    cluster = RayVirtualCluster(
        name="test-converter-cluster",
        bundle_ct_per_node_list=[1],
        use_gpus=True,
        num_gpus_per_node=1,
        max_colocated_worker_groups=1,
    )

    # Get tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])

    # Create policy
    policy = Policy(
        cluster=cluster,
        config=config["policy"],
        tokenizer=tokenizer,
        init_reference_model=False,
    )

    # Save checkpoint without any training
    use_v2 = config["policy"]["dtensor_cfg"]["_v2"]
    dcp_checkpoint_path = os.path.join(
        temp_dir, "dcp_checkpoint" + ("_v2" if use_v2 else "_v1")
    )
    policy.save_checkpoint(
        dcp_checkpoint_path, checkpointing_cfg=config["checkpointing"]
    )

    if not check_file_exists(dcp_checkpoint_path):
        raise FileNotFoundError(
            f"DCP checkpoint creation failed at {dcp_checkpoint_path}"
        )

    print(f"✓ DCP checkpoint saved to: {dcp_checkpoint_path}")
    return dcp_checkpoint_path


def create_megatron_checkpoint(model_name: str, temp_dir: str) -> str:
    """Create a Megatron checkpoint using community import."""
    print("Creating Megatron checkpoint...")

    try:
        from megatron.bridge.training.model_load_save import (
            temporary_distributed_context,
        )
    except ImportError:
        raise ImportError("megatron.bridge.training is not available.")

    with temporary_distributed_context(backend="gloo"):
        megatron_checkpoint_path = os.path.join(temp_dir, "megatron_checkpoint")
        import_model_from_hf_name(model_name, megatron_checkpoint_path)

    if not check_file_exists(megatron_checkpoint_path):
        raise FileNotFoundError(
            f"Megatron checkpoint creation failed at {megatron_checkpoint_path}"
        )

    print(f"✓ Megatron checkpoint saved to: {megatron_checkpoint_path}")
    return os.path.join(megatron_checkpoint_path, "iter_0000000")


def convert_dcp_to_hf_checkpoint(dcp_path: str, model_name: str, temp_dir: str) -> str:
    """Convert DCP checkpoint to HF format."""
    print("Converting DCP to HF format...")

    use_v2 = dcp_path.endswith("_v2")
    hf_path = os.path.join(temp_dir, "dcp_to_hf" + ("_v2" if use_v2 else "_v1"))
    convert_dcp_to_hf(
        dcp_ckpt_path=dcp_path,
        hf_ckpt_path=hf_path,
        model_name_or_path=model_name,
        tokenizer_name_or_path=model_name,
        overwrite=True,
    )

    print(f"✓ DCP to HF conversion saved to: {hf_path}")
    return hf_path


def convert_megatron_to_hf_checkpoint(
    megatron_path: str, model_name: str, temp_dir: str
) -> str:
    """Convert Megatron checkpoint to HF format."""
    print("Converting Megatron to HF format...")

    hf_path = os.path.join(temp_dir, "megatron_to_hf")

    # Get tokenizer for the export
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer_path = os.path.join(temp_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)

    export_model_from_megatron(
        hf_model_name=model_name,
        input_path=megatron_path,
        output_path=hf_path,
        hf_tokenizer_path=tokenizer_path,
        overwrite=True,
    )

    print(f"✓ Megatron to HF conversion saved to: {hf_path}")
    return hf_path


def main():
    """Main test function."""
    print("=" * 80)
    print("Starting Converter Roundtrip Functional Test")
    print("=" * 80)

    # TODO(@ashors): test more models
    model_name = "Qwen/Qwen2-0.5B"

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        # Step 1: Load original HF model
        print("\n" + "=" * 60)
        print("STEP 1: Loading original HuggingFace model")
        print("=" * 60)
        original_model, original_tokenizer = load_model_and_tokenizer(model_name)
        original_state_dict = get_model_state_dict(original_model)

        # Step 2: Create DCP checkpoint
        print("\n" + "=" * 60)
        print("STEP 2: Creating Dtensor V1 DCP checkpoint")
        print("=" * 60)
        config_v1 = create_test_config()
        dcp_checkpoint_path_v1 = create_dcp_checkpoint(model_name, config_v1, temp_dir)

        # Step 3: Create Dtensor V2 DCP checkpoint
        print("\n" + "=" * 60)
        print("STEP 3: Creating Dtensor V2 DCP checkpoint")
        print("=" * 60)
        config_v2 = copy.deepcopy(config_v1)
        config_v2["policy"]["dtensor_cfg"]["_v2"] = True
        config_v2["checkpointing"]["model_save_format"] = "torch_save"
        dcp_checkpoint_path_v2 = create_dcp_checkpoint(model_name, config_v2, temp_dir)

        # Step 4: Create Megatron checkpoint
        print("\n" + "=" * 60)
        print("STEP 4: Creating Megatron checkpoint")
        print("=" * 60)
        megatron_checkpoint_path = create_megatron_checkpoint(model_name, temp_dir)

        # Step 5: Convert Dtensor V1 DCP to HF
        print("\n" + "=" * 60)
        print("STEP 5: Converting Dtensor V1 DCP to HF format")
        print("=" * 60)
        dcp_to_hf_path_v1 = convert_dcp_to_hf_checkpoint(
            dcp_checkpoint_path_v1, model_name, temp_dir
        )

        # Step 6: Convert Dtensor V2 DCP to HF
        print("\n" + "=" * 60)
        print("STEP 6: Converting Dtensor V2 DCP to HF format")
        print("=" * 60)
        dcp_to_hf_path_v2 = convert_dcp_to_hf_checkpoint(
            dcp_checkpoint_path_v2, model_name, temp_dir
        )

        # Step 7: Convert Megatron to HF
        print("\n" + "=" * 60)
        print("STEP 7: Converting Megatron to HF format")
        print("=" * 60)
        megatron_to_hf_path = convert_megatron_to_hf_checkpoint(
            megatron_checkpoint_path, model_name, temp_dir
        )

        # Step 8: Load converted models and compare
        print("\n" + "=" * 60)
        print("STEP 8: Loading converted models and comparing")
        print("=" * 60)

        # Load Dtensor V1 DCP-converted model
        dcp_converted_model_v1 = AutoModelForCausalLM.from_pretrained(
            dcp_to_hf_path_v1, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        dcp_converted_state_dict_v1 = get_model_state_dict(dcp_converted_model_v1)

        # Load Dtensor V2 DCP-converted model
        dcp_converted_model_v2 = AutoModelForCausalLM.from_pretrained(
            dcp_to_hf_path_v2, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        dcp_converted_state_dict_v2 = get_model_state_dict(dcp_converted_model_v2)

        # Load Megatron-converted model
        megatron_converted_model = AutoModelForCausalLM.from_pretrained(
            megatron_to_hf_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        megatron_converted_state_dict = get_model_state_dict(megatron_converted_model)

        # Step 9: Assertions
        print("\n" + "=" * 60)
        print("STEP 9: Running assertions")
        print("=" * 60)

        # Compare Dtensor V1 DCP-converted vs Original HF model
        print("Comparing Dtensor V1 DCP-converted HF model with Original HF model...")
        assert_state_dicts_equal(
            dcp_converted_state_dict_v1,
            original_state_dict,
            "Dtensor V1 DCP-converted HF model",
            "Original HF model",
        )

        # Compare Dtensor V2 DCP-converted vs Original HF model
        print("Comparing Dtensor V2 DCP-converted HF model with Original HF model...")
        assert_state_dicts_equal(
            dcp_converted_state_dict_v2,
            original_state_dict,
            "Dtensor V2 DCP-converted HF model",
            "Original HF model",
        )

        # Compare Megatron-converted vs Original HF model
        print("Comparing Megatron-converted HF model with Original HF model...")
        assert_state_dicts_equal(
            megatron_converted_state_dict,
            original_state_dict,
            "Megatron-converted HF model",
            "Original HF model",
        )

        print(
            "✓ Dtensor V1 and Dtensor V2 DCP and Megatron roundtrip checkpoints are identical!"
        )

        # Verify that both converted models have the expected structure
        expected_keys = set(original_state_dict.keys())
        dcp_keys_v1 = set(dcp_converted_state_dict_v1.keys())
        dcp_keys_v2 = set(dcp_converted_state_dict_v2.keys())
        megatron_keys = set(megatron_converted_state_dict.keys())

        assert dcp_keys_v1 == expected_keys, (
            f"Dtensor V1 DCP converted model missing keys: {expected_keys - dcp_keys_v1}"
        )
        assert dcp_keys_v2 == expected_keys, (
            f"Dtensor V2 DCP converted model missing keys: {expected_keys - dcp_keys_v2}"
        )
        assert megatron_keys == expected_keys, (
            f"Megatron converted model missing keys: {expected_keys - megatron_keys}"
        )

        print("✓ All converted models have the expected structure")

        # Test that we can do a forward pass with both converted models
        print("Testing forward passes...")
        test_input = torch.randint(0, 1000, (1, 10))

        with torch.no_grad():
            dcp_output_v1 = dcp_converted_model_v1(test_input)
            dcp_output_v2 = dcp_converted_model_v2(test_input)
            megatron_output = megatron_converted_model(test_input)

        print(
            "✓ Dtensor V1 and Dtensor V2 DCP and Megatron converted models can perform forward passes"
        )

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)


if __name__ == "__main__":
    main()
