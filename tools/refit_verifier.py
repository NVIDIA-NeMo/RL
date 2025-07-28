#!/usr/bin/env python3
"""
Refitted Policy Comparison Script

This script compares logprobs between a Megatron policy and a vLLM policy
after performing model weight refitting. It demonstrates the workflow for
getting consistent logprobs across different inference backends.

Usage:
    uv run --extra mcore python3 tools/refit_verifier.py --model_name /path/to/model


Example Output:

--- Comparing Logprobs ---

Input prompt: The following are multiple choice questions (with answers) about world religions.

When was the first Buddhist temple constructed in Japan?
A. 325 CE
B. 119 CE
C. 451 CE
D. 596 CE
Answer:
Input tokens: tensor([200000,    954,   2182,    583,   6146,   9031,   5808,    330,   5992,
      8860,     21,   1509,   3817,  99867,   1574,   7022,    812,    290,
      1660, 120819,  55594,  24043,    310,  11197,   1044,     45,     26,
       220,  23325,  13607,    198,     46,     26,    220,  12860,  13607,
       198,     47,     26,    220,  34518,  13607,    198,     48,     26,
       220,  43145,  13607,    198,   4984,     38])

Comparing 10 generated tokens (from position 51 to 60):
vLLM generated logprobs: tensor([-7.0227, -7.1559, -6.4603, -6.7419, -6.3026, -6.8391, -6.3128, -6.6454,
    -7.1514, -6.8304])
Megatron generated logprobs: tensor([-7.0225, -7.1873, -6.4600, -6.7418, -6.3027, -6.8704, -6.2502, -6.6453,
    -7.1518, -6.8304])
Absolute difference: tensor([2.0981e-04, 3.1348e-02, 2.6035e-04, 1.6689e-04, 1.4973e-04, 3.1272e-02,
    6.2590e-02, 1.7643e-04, 3.2902e-04, 4.1485e-05])
Mean absolute difference: 0.012654399499297142
Max absolute difference: 0.06259012222290039

--- Token-by-Token Comparison (Generated Tokens Only) ---
Token           Token ID   Position   vLLM         Megatron     Diff
---------------------------------------------------------------------------
tok_51          pos_51     51         -7.022674    -7.022464    0.000210
tok_52          pos_52     52         -7.155923    -7.187271    0.031348
tok_53          pos_53     53         -6.460307    -6.460047    0.000260
tok_54          pos_54     54         -6.741926    -6.741759    0.000167
tok_55          pos_55     55         -6.302569    -6.302719    0.000150
tok_56          pos_56     56         -6.839099    -6.870371    0.031272
tok_57          pos_57     57         -6.312774    -6.250184    0.062590
tok_58          pos_58     58         -6.645445    -6.645269    0.000176
tok_59          pos_59     59         -7.151441    -7.151770    0.000329
tok_60          pos_60     60         -6.830355    -6.830397    0.000041
"""

import argparse
import copy
import ray
import torch
from transformers import AutoTokenizer

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.algorithms.grpo import refit_policy_generation


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare Megatron and vLLM policy logprobs after refitting")
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="/root/checkpoints/llama4-scout-custom-init",
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--tp_size", 
        type=int, 
        default=1,
        help="Tensor parallelism size (TP) for Megatron"
    )
    parser.add_argument(
        "--ep_size", 
        type=int, 
        default=1,
        help="Expert parallelism size (EP) for Megatron"
    )
    parser.add_argument(
        "--pp_size", 
        type=int, 
        default=1,
        help="Pipeline parallelism size (PP) for Megatron"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=10,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--max_sequence_length", 
        type=int, 
        default=256,
        help="Maximum total sequence length"
    )
    parser.add_argument(
        "--refit_buffer_size_gb", 
        type=int, 
        default=4,
        help="Refit buffer size in GB"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="Here is a short introduction to me:",
        help="Input prompt for generation"
    )
    
    return parser.parse_args()


def setup_configs(args, tokenizer):
    """
    Setup configuration dictionaries for Megatron and vLLM.
    
    Args:
        args: Parsed command line arguments
        tokenizer: HuggingFace tokenizer
        
    Returns:
        tuple: (megatron_config, vllm_config)
    """
    # Megatron Configuration
    megatron_config = {
        "model_name": args.model_name,
        "training_backend": "megatron",
        "train_global_batch_size": 1,
        "train_micro_batch_size": 1,
        "generation_batch_size": 2,
        "learning_rate": 0.0001,
        "logprob_batch_size": 1,
        "generation": {
            "max_total_sequence_length": args.max_sequence_length,
            "max_new_tokens": args.max_sequence_length,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
        },
        "precision": "bfloat16",
        "pipeline_dtype": "bfloat16",
        "parallel_output": True,
        "max_total_sequence_length": args.max_sequence_length,
        "fsdp_offload_enabled": False,
        "max_grad_norm": 1.0,
        "refit_buffer_size_gb": args.refit_buffer_size_gb,
        "make_sequence_length_divisible_by": args.tp_size,
        "optimizer": {
            "type": "adam",
            "kwargs": {
                "lr": 0.0001,
                "weight_decay": 0.0,
                "eps": 1e-8,
            },
        },
        "dtensor_cfg": {
            "enabled": False,
        },
        "dynamic_batching": {
            "enabled": False,
            "train_mb_tokens": 256,
            "logprob_mb_tokens": 256,
            "sequence_length_round": 64,
        },
        "sequence_packing": {
            "enabled": False,
        },
        "megatron_cfg": {
            "enabled": True,
            "empty_unused_memory_level": 1,
            "tensor_model_parallel_size": args.tp_size,
            "sequence_parallel": False,
            "expert_tensor_parallel_size": args.tp_size,
            "expert_model_parallel_size": args.ep_size,
            "pipeline_model_parallel_size": args.pp_size,
            "context_parallel_size": 1,
            "num_layers_in_first_pipeline_stage": None,
            "num_layers_in_last_pipeline_stage": None,
            "activation_checkpointing": False,
            "moe_router_dtype": "fp64",
            "moe_router_load_balancing_type": "none",
            "moe_router_bias_update_rate": 0.,
            "pipeline_dtype": "bfloat16",
            "freeze_moe_router": False,
            "apply_rope_fusion": False,
            "optimizer": {
                "optimizer": "adam",
                "lr": 5.0e-6,
                "min_lr": 5.0e-7,
                "weight_decay": 0.01,
                "bf16": False,
                "fp16": False,
                "params_dtype": "float32",
                # Adam optimizer settings
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_eps": 1e-8,
                # SGD optimizer settings
                "sgd_momentum": 0.9,
                # Distributed optimizer settings
                "use_distributed_optimizer": True,
                "use_precision_aware_optimizer": True,
                "clip_grad": 1.0,
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
                "overlap_grad_reduce": False,
                "overlap_param_gather": False,
                "average_in_collective": False,
                "use_custom_fsdp": False,
                "data_parallel_sharding_strategy": "optim_grads_params"
            }
        },
    }

    # vLLM Configuration
    vllm_tp = args.tp_size * args.ep_size * args.pp_size
    vllm_config = {
        "backend": "vllm",
        "model_name": args.model_name,
        "tokenizer": {
            "name": args.model_name,
        },
        "dtype": "bfloat16",
        "max_new_tokens": args.max_new_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "vllm_cfg": {
            "tensor_parallel_size": vllm_tp,
            "pipeline_parallel_size": 1,
            "gpu_memory_utilization": 0.6,
            "max_model_len": args.max_sequence_length,
            "precision": "bfloat16",
            "async_engine": False,
            "skip_tokenizer_init": False,
            "load_format": "dummy",
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
    
    # Configure vLLM with tokenizer
    vllm_config = configure_generation_config(vllm_config, tokenizer)
    
    return megatron_config, vllm_config


def setup_clusters_and_policies(args, megatron_config, vllm_config, tokenizer):
    """
    Setup Ray clusters and initialize policies.
    
    Args:
        args: Parsed command line arguments
        megatron_config: Megatron configuration dictionary
        vllm_config: vLLM configuration dictionary
        tokenizer: HuggingFace tokenizer
        
    Returns:
        tuple: (megatron_cluster, policy, vllm_inference_policy)
    """
    gpus_per_node = args.tp_size * args.ep_size * args.pp_size
    print(f"Setting up Megatron Cluster with TP={gpus_per_node}")
    megatron_cluster = RayVirtualCluster(
        name="megatron_cluster",
        bundle_ct_per_node_list=[gpus_per_node],
        use_gpus=True,
        num_gpus_per_node=gpus_per_node,
        max_colocated_worker_groups=2,
    )

    print("Instantiating Policy with Megatron backend...")
    policy = Policy(
        cluster=megatron_cluster, 
        config=megatron_config, 
        tokenizer=tokenizer, 
        init_reference_model=False, 
        init_optimizer=False
    )

    # Create vLLM inference configuration with limited generation
    vllm_inference_config = vllm_config.copy()
    vllm_inference_config["max_new_tokens"] = args.max_new_tokens
    vllm_inference_config = configure_generation_config(vllm_inference_config, tokenizer)

    # Create vLLM policy for inference-only logprobs
    vllm_inference_policy = VllmGeneration(cluster=megatron_cluster, config=vllm_inference_config)

    return megatron_cluster, policy, vllm_inference_policy


def prepare_input_data(prompt, tokenizer):
    """
    Tokenize the input prompt and prepare generation data.
    
    Args:
        prompt: Input text prompt
        tokenizer: HuggingFace tokenizer
        
    Returns:
        BatchedDataDict: Prepared input data
    """
    print("Preparing input data...")
    
    # Tokenize the prompt
    tokenized = tokenizer(
        [prompt],
        padding=True,
        truncation=True,
        return_tensors="pt",
        padding_side="right",
    )

    # Calculate input lengths from attention mask
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    input_lengths = attention_mask.sum(dim=1).to(torch.int32)

    generation_data = BatchedDataDict({
        "input_ids": input_ids,
        "input_lengths": input_lengths,
    })
    
    return generation_data


def run_model_refitting(policy, vllm_inference_policy, refit_buffer_size_gb):
    """
    Perform model weight refitting between Megatron and vLLM policies.
    
    Args:
        policy: Megatron policy
        vllm_inference_policy: vLLM inference policy
        refit_buffer_size_gb: Buffer size for refitting in GB
    """
    print("\n--- Performing Model Refitting ---")
    
    # Prepare weights for inter-process communication
    sd_info = policy.prepare_weights_for_ipc()
    print(f"Weight preparation info: {sd_info}")
    
    # Perform the refitting between policies
    refit_policy_generation(policy, vllm_inference_policy, refit_buffer_size_gb)
    print("Model refitting completed")


def generate_and_compare_logprobs(policy, vllm_inference_policy, generation_data):
    """
    Generate outputs and compare logprobs between vLLM and Megatron policies.
    
    Args:
        policy: Megatron policy
        vllm_inference_policy: vLLM inference policy
        generation_data: Input data for generation
        
    Returns:
        tuple: (vllm_logprobs_data, megatron_generation_data)
    """
    # Generate with vLLM for logprobs
    print("\n--- Getting vLLM Policy Logprobs ---")
    vllm_logprobs_data = vllm_inference_policy.generate(generation_data, greedy=True)
    print(f"vLLM Logprobs shape: {vllm_logprobs_data['logprobs'].shape}")
    print(f"vLLM Logprobs sample: {vllm_logprobs_data['logprobs'][0, -10:]}")

    # Generate with Megatron policy
    print("\n--- Getting Megatron Generation ---")
    policy.prepare_for_generation()
    
    # Prepare input data for Megatron using vLLM outputs
    megatron_input_data = copy.deepcopy(generation_data)
    print("=" * 100)
    print(megatron_input_data)
    print(vllm_logprobs_data)
    megatron_input_data["input_ids"] = vllm_logprobs_data["output_ids"]
    megatron_input_data["input_lengths"] = vllm_logprobs_data["unpadded_sequence_lengths"]

    # Get logprobs from Megatron
    policy.prepare_for_lp_inference()
    megatron_generation_data = policy.get_logprobs(megatron_input_data)
    print(f"Megatron Generation shape: {megatron_generation_data['logprobs'].shape}")
    print(f"Megatron Generation sample: {megatron_generation_data['logprobs'][0, -10:]}")

    return vllm_logprobs_data, megatron_generation_data


def analyze_logprob_differences(vllm_logprobs_data, megatron_generation_data, generation_data, tokenizer, prompt):
    """
    Analyze and display differences between vLLM and Megatron logprobs.
    
    Args:
        vllm_logprobs_data: vLLM generation results
        megatron_generation_data: Megatron generation results
        generation_data: Original input data
        tokenizer: HuggingFace tokenizer
        prompt: Original input prompt
    """
    print("\n--- Comparing Logprobs ---")
    print(f"Input prompt: {prompt}")
    print(f"Input tokens: {generation_data['input_ids'][0, :generation_data['input_lengths'][0]]}")

    # Extract generation parameters
    input_length = generation_data["input_lengths"][0].item()
    total_length = vllm_logprobs_data["logprobs"].shape[1]
    generated_length = vllm_logprobs_data["generation_lengths"][0].item()

    if generated_length > 0:
        print(f"\nComparing {generated_length} generated tokens (from position {input_length} to {total_length-1}):")
        
        # Extract generated logprobs
        vllm_gen_logprobs = vllm_logprobs_data["logprobs"][0, input_length:total_length]
        megatron_gen_logprobs = megatron_generation_data["logprobs"][0, input_length:total_length]
        
        print(f"vLLM generated logprobs: {vllm_gen_logprobs}")
        print(f"Megatron generated logprobs: {megatron_gen_logprobs}")

        # Calculate and display differences
        abs_diff = torch.abs(vllm_gen_logprobs - megatron_gen_logprobs)
        print(f"Absolute difference: {abs_diff}")
        print(f"Mean absolute difference: {torch.mean(abs_diff)}")
        print(f"Max absolute difference: {torch.max(abs_diff)}")

        # Detailed token-by-token comparison
        _detailed_token_comparison(
            vllm_gen_logprobs, megatron_gen_logprobs, vllm_logprobs_data,
            input_length, total_length, tokenizer
        )
    else:
        print(f"No generated tokens to compare (input_length: {input_length}, total_length: {total_length})")


def _detailed_token_comparison(vllm_logprobs, megatron_logprobs, vllm_logprobs_data, input_length, total_length, tokenizer):
    """
    Display detailed token-by-token comparison of logprobs.
    
    Args:
        vllm_logprobs: vLLM logprobs for generated tokens
        megatron_logprobs: Megatron logprobs for generated tokens
        vllm_logprobs_data: Vllm generation data
        input_length: Length of input sequence
        total_length: Total sequence length
        tokenizer: HuggingFace tokenizer
    """
    print("\n--- Token-by-Token Comparison (Generated Tokens Only) ---")
    
    if total_length > input_length:
        # Get generated tokens if available
        print("####", vllm_logprobs_data)
        if "output_ids" in vllm_logprobs_data:
            generated_tokens = vllm_logprobs_data["output_ids"][0, input_length:total_length]
        else:
            generated_tokens = torch.arange(input_length, total_length)

        # Display header
        print(f"{'Token':<15} {'Token ID':<10} {'Position':<10} {'vLLM':<12} {'Megatron':<12} {'Diff':<12}")
        print("-" * 75)
        
        # Display each token comparison
        for i, pos in enumerate(range(input_length, total_length)):
            if "output_ids" in vllm_logprobs_data:
                token_id = generated_tokens[i].item()
                token_text = tokenizer.decode([token_id])
            else:
                token_id = f"pos_{pos}"
                token_text = f"tok_{pos}"
                
            vllm_lp = vllm_logprobs[i].item()
            megatron_lp = megatron_logprobs[i].item()
            diff = abs(vllm_lp - megatron_lp)
            
            print(f"{token_text:<15} {token_id:<10} {pos:<10} {vllm_lp:<12.6f} {megatron_lp:<12.6f} {diff:<12.6f}")
    else:
        print("No generated tokens to compare in detail.")


def cleanup_resources(vllm_inference_policy):
    """
    Clean up resources and shutdown policies.
    
    Args:
        vllm_inference_policy: vLLM policy to shutdown
    """
    print("\n--- Cleaning up ---")
    vllm_inference_policy.shutdown()
    print("Cleanup completed successfully!")


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_args()
    
    # Initialize Ray
    ray.init()
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Setup configurations
    megatron_config, vllm_config = setup_configs(args, tokenizer)
    
    # Setup clusters and policies
    megatron_cluster, policy, vllm_inference_policy = setup_clusters_and_policies(
        args, megatron_config, vllm_config, tokenizer
    )
    
    # Prepare input data
    generation_data = prepare_input_data(args.prompt, tokenizer)

    # prepare refit info
    # state_dict_info = policy.prepare_refit_info()
    # vllm_inference_policy.prepare_refit_info(state_dict_info)

    # Perform model refitting
    run_model_refitting(policy, vllm_inference_policy, args.refit_buffer_size_gb)
    
    # Generate and compare logprobs
    vllm_logprobs_data, megatron_generation_data = generate_and_compare_logprobs(
        policy, vllm_inference_policy, generation_data
    )
    
    # Analyze differences
    analyze_logprob_differences(
        vllm_logprobs_data, megatron_generation_data, generation_data, tokenizer, args.prompt
    )
    
    # Cleanup
    cleanup_resources(vllm_inference_policy)
    
    print("Script completed successfully!")


if __name__ == "__main__":
    main()