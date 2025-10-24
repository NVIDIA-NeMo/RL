"""
Standalone script to test vLLM generation with Megatron training workflow.
Based on test_vllm_generation_with_megatron_training from test_vllm_generation.py

Usage:
    # default grpo_math_1B.yaml
    NRL_FORCE_REBUILD_VENVS=true uv run examples/run_standalone_refit.py

    # grpo_math_1B_megatron_fp8.yaml
    NRL_FORCE_REBUILD_VENVS=true uv run examples/run_standalone_refit.py --config examples/configs/grpo_math_1B_megatron_fp8.yaml
"""

import argparse
import gc
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch
from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import refit_policy_generation, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides

# Import setup_data from run_grpo_math.py
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'examples'))
from run_grpo_math import setup_data
import wandb

DEBUG = False
SAMPLE_SIZE = 2

# Conditionally register resolver only if not already registered
if not OmegaConf.has_resolver("mul"):
    OmegaConf.register_new_resolver("mul", lambda a, b: a * b)

# ====================
# Helper Functions
# ====================
# (setup_data is imported from run_grpo_math.py)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Standalone script to test vLLM generation with Megatron training workflow"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (optional, will use defaults if not provided)"
    )

    args, overrides = parser.parse_known_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_math_1B.yaml"
        )

    config = load_config(args.config)

    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    # config = OmegaConf.to_container(config, resolve=True)
    print(f"{config=}")
    return config


def run_generation_and_training_workflow(config):
    """Main workflow: vLLM generation -> Megatron training -> vLLM generation again.
    
    Uses the setup() function from grpo.py for consistent initialization.
    """
    
    try:
        # Step 1: Setup tokenizer (needed for dataset and configure_generation_config)
        model_name = config.policy.model_name
        print(f"Using model: {model_name}")
        print("Creating tokenizer...")
        tokenizer = get_tokenizer({"name": model_name})
        
        # Step 2: Configure generation config (required by setup())
        print("Configuring generation...")
        config.policy.generation = configure_generation_config(
            OmegaConf.to_container(config.policy.generation, resolve=True), 
            tokenizer
        )
        
        # Step 3: Load dataset using setup_data from run_grpo_math.py
        print("\n▶ Setting up data...")
        master_config_dict = OmegaConf.to_container(config, resolve=True)
        dataset, val_dataset, task_to_env, val_task_to_env = setup_data(
            tokenizer,
            master_config_dict["data"],
            master_config_dict["env"],
            master_config_dict["grpo"]["seed"]
        )
        
        # Step 4: Use setup() function from grpo.py (just like run_grpo_math.py)
        print("\n▶ Calling setup() from grpo.py...")
        master_config = master_config_dict  # Already converted above
        
        (
            megatron_policy,      # policy
            vllm_policy,          # policy_generation
            (train_cluster, inference_cluster),  # clusters
            dataloader,
            val_dataloader,
            loss_fn,
            logger,
            checkpointer,
            grpo_save_state,
            master_config,
        ) = setup(master_config, tokenizer, dataset, val_dataset)

        print("  ✓ Setup complete!")
        cluster = train_cluster  # For cleanup
        
        # # Step 5: Do the initial refit (transfer Megatron weights to vLLM)
        print("\n▶ Refitting vLLM policy with Megatron weights...")
        colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]
        refit_policy_generation(megatron_policy, vllm_policy, colocated_inference)
        print("  ✓ Refit complete!")

        megatron_policy.offload_after_refit()
        # Step 6: Get test batch from dataloader (already created by setup())
        print(f"\nPreparing test batch from dataloader...")
        batch = next(iter(dataloader))
        # Convert message_log to flat format for generation
        batched_flat, input_lengths = batched_message_log_to_flat_message(
            batch["message_log"],
            pad_value_dict={"token_ids": tokenizer.pad_token_id},
        )
        input_ids = batched_flat["token_ids"]
        
        test_input_data = BatchedDataDict({
            "input_ids": input_ids,
            "input_lengths": input_lengths,
        })
        
        vllm_policy.prepare_for_generation()
        generation_results = vllm_policy.generate(test_input_data, greedy=True)
        vllm_policy.finish_generation()
        
        # Validate generation outputs
        assert "output_ids" in generation_results
        assert "logprobs" in generation_results
        
        if DEBUG:
            generated_texts = tokenizer.batch_decode(
                generation_results["output_ids"], skip_special_tokens=True
            )
            
            # Also decode input prompts for comparison
            input_texts = tokenizer.batch_decode(
                input_ids, skip_special_tokens=True
            )
            
            print(f"\nvLLM Generation Results:")
            for i, (prompt, response) in enumerate(zip(input_texts, generated_texts)):
                print(f"\n  Sample [{i}]:")
                print(f"    Prompt: {prompt[:80]}...")
                print(f"    Full Response: {response[:150]}...")

        def pad_to_multiple(tensor, multiple, pad_value):
            original_shape = tensor.shape
            if tensor.size(1) % multiple != 0:
                padding_size = (tensor.size(1) // multiple + 1) * multiple - tensor.size(1)
                tensor = torch.cat([tensor, torch.full((tensor.size(0), padding_size, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device)], dim=1)
                print(f"Padded tensor shape: froom {original_shape} to {tensor.shape}")
            else:
                print(f"Tensor shape: {tensor.shape} is already divisible by {multiple}")
            return tensor

        fprop_logprob_data = BatchedDataDict({
            "input_ids": pad_to_multiple(generation_results["output_ids"], master_config["policy"]["make_sequence_length_divisible_by"], tokenizer.pad_token_id),
            "input_lengths": generation_results["unpadded_sequence_lengths"],
        })
        
        megatron_policy.prepare_for_lp_inference()
        fprop_results = megatron_policy.get_logprobs(fprop_logprob_data)
        megatron_policy.offload_after_refit()
        fprop_logprob_data["generation_logprobs"] = pad_to_multiple(generation_results["logprobs"], master_config["policy"]["make_sequence_length_divisible_by"], 0.0)
        
        
        # Calculate logprob difference
        padding_mask = torch.zeros_like(fprop_logprob_data["generation_logprobs"], dtype=torch.bool)
        for i, (input_len, total_valid_len) in enumerate(
            zip(test_input_data.get("input_lengths"), 
                generation_results["unpadded_sequence_lengths"])
        ):
            padding_mask[i, input_len:total_valid_len] = True
        
        abs_diff = torch.abs(fprop_logprob_data["generation_logprobs"] - fprop_results["logprobs"])
        masked_abs_diff = abs_diff.masked_select(padding_mask)
        avg_prob_mult_error = (
            torch.mean(torch.exp(masked_abs_diff))
            if masked_abs_diff.numel() > 0
            else torch.tensor(0.0)
        )

        # Get vllm_precision from master_config
        print("\n" + "="*60)
        print("✓ Successfully completed vLLM generation + Megatron logprob verification!")
        print("="*60)
        print("\nDataset Statistics:")
        print(f"  • Total samples in dataset: {len(dataset)}")
        print(f"  • Samples processed: {len(batch['message_log'])}")
        print(f"  • Average input length: {input_lengths.float().mean():.1f} tokens")
        print(f"  • Average output length: {generation_results['generation_lengths'].float().mean():.1f} tokens")
        print(f"  • Average logprob error: {avg_prob_mult_error:.6f}")

        threshold = 1.05
        if avg_prob_mult_error > threshold:
            raise ValueError(f"  ✗ Warning: Logprobs{avg_prob_mult_error:.6f} exceed threshold ({threshold:.6f})")
        else:
            print(f"✓ Logprobs match within threshold ({threshold})")
        
    finally:
        pass

# ====================
# Main Entry Point
# ====================

if __name__ == "__main__":
    config = parse_args()
    
    # Initialize Ray FIRST (just like run_grpo_math.py line 229)
    print("Initializing Ray...")
    init_ray()
    print("  ✓ Ray initialized\n")
    
    # Convert OmegaConf to dict for easier access
    config_dict = OmegaConf.to_container(config, resolve=True) if isinstance(config, OmegaConf) else config
    
    # Extract key parameters
    model_name = config_dict.policy.model_name
    num_gpus_per_node = config_dict.cluster.gpus_per_node
    
    # Extract megatron config
    megatron_cfg = config_dict.policy.megatron_cfg
    tp = megatron_cfg.tensor_model_parallel_size
    pp = megatron_cfg.pipeline_model_parallel_size
    
    # Extract vLLM config if present
    generation_cfg = config_dict.policy.generation
    vllm_cfg = generation_cfg.vllm_cfg
    vllm_precision = vllm_cfg.precision
    async_engine = vllm_cfg.async_engine
    
    print("="*60)
    print("Tiny Experiment: vLLM + Megatron Training Workflow")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"GPUs per node: {num_gpus_per_node}")
    print(f"Tensor parallel: {tp}")
    print(f"Pipeline parallel: {pp}")
    print(f"Async engine: {async_engine}")
    print(f"vLLM precision: {vllm_precision}")
    print("="*60 + "\n")
    
    run_generation_and_training_workflow(config)