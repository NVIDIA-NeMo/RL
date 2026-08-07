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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nemo_rl.models.megatron.community_import import export_model_from_megatron
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

from tests.functional.converter_test_utils import (
    assert_state_dicts_equal,
    create_dcp_checkpoint,
    create_megatron_checkpoint,
    create_megatron_lora_checkpoint,
    create_test_config,
    get_model_state_dict,
    load_model_and_tokenizer,
)


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
            base_ckpt=megatron_checkpoint_path,
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
        del lora_merged_model
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

        del adapter_only_merged_model
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
            "✓ Dtensor V1 and Dtensor V2 DCP, Megatron, and LoRA merged models can perform forward passes"
        )

        print("\n" + "=" * 80)
        print(
            "✓ ALL TESTS PASSED (DCP v1, DCP v2, Megatron, LoRA merge, LoRA adapter-only PEFT)!"
        )
        print("=" * 80)


if __name__ == "__main__":
    main()
