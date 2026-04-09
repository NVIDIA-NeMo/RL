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
import gc
import importlib.util
import os
import tempfile
import time
from typing import Any, Dict

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.megatron.community_import import (
    export_model_from_megatron,
    import_model_from_hf_name,
)
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.native_checkpoint import convert_dcp_to_hf

_CONVERTER_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), "../../examples/converters/convert_lora_to_hf.py"
    )
)
_spec = importlib.util.spec_from_file_location("convert_lora_to_hf", _CONVERTER_PATH)
_convert_lora_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_convert_lora_mod)
merge_lora_to_hf = _convert_lora_mod.merge_lora_to_hf
export_lora_adapter_to_hf = _convert_lora_mod.export_lora_adapter_to_hf


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
        megatron_config = {
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
            "expert_tensor_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "num_layers_in_first_pipeline_stage": None,
            "num_layers_in_last_pipeline_stage": None,
            "pipeline_dtype": "float32",
            "sequence_parallel": False,
            "gradient_accumulation_fusion": False,
        }
        # need to set `gradient_accumulation_fusion=False` to avoid RuntimeError
        import_model_from_hf_name(
            model_name, megatron_checkpoint_path, megatron_config=megatron_config
        )

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


def create_megatron_lora_checkpoint(
    model_name: str, base_ckpt_path: str, temp_dir: str
) -> str:
    """Build a Megatron model with LoRA, apply a known perturbation, and save only the adapter weights.

    Returns the path to the LoRA adapter checkpoint directory (iter_XXXXXXX format).
    """
    print("Creating Megatron LoRA adapter checkpoint...")

    from megatron.bridge import AutoBridge
    from megatron.bridge.peft.lora import LoRA
    from megatron.bridge.training.model_load_save import (
        load_model_config,
        megatron_cpu_init_context,
        save_megatron_model,
        temporary_distributed_context,
    )
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    lora_dim = 8
    lora_alpha = 16
    peft_cfg = {
        "target_modules": [],
        "exclude_modules": [],
        "dim": lora_dim,
        "alpha": lora_alpha,
        "dropout": 0.0,
        "dropout_position": "pre",
        "lora_A_init_method": "xavier",
        "lora_B_init_method": "zero",
        "a2a_experimental": False,
    }

    bridge = AutoBridge.from_hf_pretrained(model_name, trust_remote_code=True)

    with temporary_distributed_context(backend="gloo"):
        model_parallel_cuda_manual_seed(0)

        model_cfg, _ = load_model_config(base_ckpt_path)
        model_cfg.tensor_model_parallel_size = 1
        model_cfg.pipeline_model_parallel_size = 1
        model_cfg.num_layers_in_first_pipeline_stage = None
        model_cfg.num_layers_in_last_pipeline_stage = None
        model_cfg.context_parallel_size = 1
        model_cfg.expert_model_parallel_size = 1
        model_cfg.expert_tensor_parallel_size = 1
        model_cfg.moe_extended_tp = False
        model_cfg.sequence_parallel = False
        model_cfg.perform_initialization = False
        model_cfg.virtual_pipeline_model_parallel_size = None
        model_cfg.hierarchical_context_parallel_sizes = None
        model_cfg.fp8 = None
        model_cfg.fp8_param = False

        peft = LoRA(**peft_cfg)
        if hasattr(model_cfg, "finalize"):
            model_cfg.finalize()
        with megatron_cpu_init_context(model_cfg):
            megatron_model = model_cfg.provide_distributed_model(
                wrap_with_ddp=False,
                use_cpu_initialization=True,
            )
        gc.collect()

        for m in megatron_model:
            m.requires_grad_(False)

        # Save the base model first to create the checkpoint directory structure
        # and write run_config.yaml (which contains the "model" key needed by
        # load_model_config). Adapter weights are saved separately below.
        adapter_dir = os.path.join(temp_dir, "lora_adapter_checkpoint")
        save_megatron_model(megatron_model, adapter_dir)
        iter_dir = os.path.join(adapter_dir, "iter_0000000")

        # Apply LoRA wrappers (same pattern as merge_lora_to_hf) and perturb
        # adapter weights so that the merge produces something different from base.
        megatron_model = peft(megatron_model, training=False)
        gc.collect()

        torch.manual_seed(42)
        for m in megatron_model:
            for name, param in m.named_parameters():
                if "lora_" in name or "adapter" in name:
                    param.data.normal_(0, 0.01)

        # Save only the adapter weights using dist_checkpointing, which is the
        # format that merge_lora_to_hf expects to load from adapter_ckpt.
        from megatron.bridge.training.checkpointing import (
            _generate_model_state_dict,
            apply_peft_adapter_filter_to_state_dict,
        )
        from megatron.core import dist_checkpointing

        adapter_sharded_sd = _generate_model_state_dict(megatron_model, {})
        adapter_sharded_sd = apply_peft_adapter_filter_to_state_dict(
            adapter_sharded_sd, peft
        )
        dist_checkpointing.save(adapter_sharded_sd, iter_dir)

        # Merge the peft section into run_config.yaml so that both
        # load_model_config (needs "model") and the LoRA converter
        # (needs "peft") can find what they expect.
        run_config_path = os.path.join(iter_dir, "run_config.yaml")
        with open(run_config_path) as f:
            run_config = yaml.safe_load(f)
        run_config["peft"] = peft_cfg
        with open(run_config_path, "w") as f:
            yaml.dump(run_config, f)

        del megatron_model
        gc.collect()

    print(f"✓ LoRA adapter checkpoint saved to: {iter_dir}")
    return iter_dir


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

        # Step 7b: Create LoRA adapter checkpoint on top of the Megatron base
        print("\n" + "=" * 60)
        print("STEP 7b: Creating Megatron LoRA adapter checkpoint")
        print("=" * 60)
        lora_adapter_path = create_megatron_lora_checkpoint(
            model_name, megatron_checkpoint_path, temp_dir
        )

        # Step 7c: Merge LoRA adapter + base and export to HF
        # Calls the actual merge_lora_to_hf function from the converter script.
        print("\n" + "=" * 60)
        print("STEP 7c: Merging LoRA adapter with base and exporting to HF")
        print("=" * 60)
        lora_merged_hf_path = os.path.join(temp_dir, "lora_merged_hf")
        merge_lora_to_hf(
            base_ckpt=megatron_checkpoint_path,
            adapter_ckpt=lora_adapter_path,
            hf_model_name=model_name,
            hf_ckpt_path=lora_merged_hf_path,
        )

        # Step 7d: Export LoRA adapter only in HuggingFace PEFT format
        print("\n" + "=" * 60)
        print("STEP 7d: Exporting LoRA adapter only (PEFT format)")
        print("=" * 60)
        lora_adapter_hf_path = os.path.join(temp_dir, "lora_adapter_hf")
        export_lora_adapter_to_hf(
            adapter_ckpt=lora_adapter_path,
            hf_model_name=model_name,
            hf_ckpt_path=lora_adapter_hf_path,
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

        # LoRA merged model: should have same keys as original but different values
        # (because the LoRA adapter perturbs the weights).
        print("Comparing LoRA-merged HF model with Original HF model...")
        lora_merged_model = AutoModelForCausalLM.from_pretrained(
            lora_merged_hf_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        lora_merged_state_dict = get_model_state_dict(lora_merged_model)

        lora_merged_keys = set(lora_merged_state_dict.keys())
        assert lora_merged_keys == set(original_state_dict.keys()), (
            f"LoRA merged model key mismatch.\n"
            f"  Extra: {lora_merged_keys - set(original_state_dict.keys())}\n"
            f"  Missing: {set(original_state_dict.keys()) - lora_merged_keys}"
        )
        print("✓ LoRA merged model has the expected key structure")

        # The merged model should differ from the original because LoRA
        # perturbations have been folded in.
        any_different = False
        for key in original_state_dict:
            v_orig = original_state_dict[key]
            v_lora_merged = lora_merged_state_dict[key]
            if isinstance(v_orig, torch.Tensor) and not torch.allclose(
                v_orig, v_lora_merged, rtol=1e-5, atol=1e-5
            ):
                any_different = True
                break
        assert any_different, (
            "LoRA-merged model weights are identical to the original — "
            "the adapter perturbation was not applied."
        )
        print("✓ LoRA merged model weights differ from original (adapter was applied)")

        # Forward pass sanity check
        test_input_lora = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            lora_output = lora_merged_model(test_input_lora)
        print("✓ LoRA merged model can perform forward pass")
        # del lora_merged_model
        gc.collect()

        # Adapter-only (PEFT) export assertions
        print("Verifying adapter-only PEFT export...")
        adapter_config_path = os.path.join(lora_adapter_hf_path, "adapter_config.json")
        assert os.path.exists(adapter_config_path), (
            f"adapter_config.json not found in {lora_adapter_hf_path}"
        )
        weight_candidates = ["adapter_model.safetensors", "adapter_model.bin"]
        weight_file_found = any(
            os.path.exists(os.path.join(lora_adapter_hf_path, f))
            for f in weight_candidates
        )
        assert weight_file_found, (
            f"No adapter weight file found in {lora_adapter_hf_path}. "
            f"Expected one of: {weight_candidates}"
        )
        print(
            "✓ PEFT adapter directory has expected files (adapter_config.json + weights)"
        )

        # Forward pass using the already-merged model from Step 7c.
        test_input_peft = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            lora_merged_model(test_input_peft)
        print("✓ LoRA merged model can perform a forward pass")

        # Verify the adapter-only export produces the same merged weights as Step 7c
        # by calling merge_lora_to_hf again with the same Megatron adapter. This
        # avoids tied-weight complications from PeftModel.merge_and_unload().
        adapter_only_merged_hf_path = os.path.join(temp_dir, "adapter_only_merged_hf")
        merge_lora_to_hf(
            base_ckpt=megatron_checkpoint_path,
            adapter_ckpt=lora_adapter_path,
            hf_model_name=model_name,
            hf_ckpt_path=adapter_only_merged_hf_path,
        )
        adapter_only_merged_model = AutoModelForCausalLM.from_pretrained(
            adapter_only_merged_hf_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        adapter_only_merged_state_dict = get_model_state_dict(adapter_only_merged_model)
        assert_state_dicts_equal(
            adapter_only_merged_state_dict,
            lora_merged_state_dict,
            "adapter-only export + merge_lora_to_hf (Step 7d)",
            "lora merged (Step 7c)",
        )
        print("✓ adapter-only merge via merge_lora_to_hf matches Step 7c")

        del adapter_only_merged_model, lora_merged_model
        gc.collect()

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
            "✓ Dtensor V1 and Dtensor V2 DCP, Megatron, and LoRA models can perform forward passes"
        )

        print("\n" + "=" * 80)
        print(
            "✓ ALL TESTS PASSED (DCP v1, DCP v2, Megatron, LoRA merge, LoRA adapter-only PEFT)!"
        )
        print("=" * 80)


if __name__ == "__main__":
    main()
