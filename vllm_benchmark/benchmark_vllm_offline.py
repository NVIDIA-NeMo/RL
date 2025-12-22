"""
Pure vLLM Offline Throughput Benchmark (Duncan-style)

This script measures "ideal" vLLM offline inference performance.
It uses the same dataset and prompt structure as GRPO training for fair comparison.

Key features:
- Uses vLLM directly (no NeMo-RL wrapper)
- Runs inside a vLLM container with direct vLLM installation
- Supports custom vLLM builds for testing modifications
- GRPO-style batching: num_prompts × num_generations_per_prompt
- Supports Data Parallelism (DP) for multi-node/multi-GPU setups

Parallelism:
- TP (Tensor Parallel): Split model across GPUs within a single vLLM instance
- PP (Pipeline Parallel): Split model layers across GPUs
- DP (Data Parallel): Run multiple vLLM instances in parallel, each handling a subset of prompts

Usage:
    # Inside the vLLM container (via benchmark_vllm_offline.sbatch)
    python benchmark_vllm_offline.py \
        --model /path/to/model \
        --tp 4 --pp 1 --dp 4 \
        --num-prompts 64 --num-generations 32

    # To modify vLLM, edit setup.sh to install from your fork/branch
"""

import os
import time
import argparse
import json

# vLLM is available in this environment (container + venv)
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Pure vLLM Offline Throughput Benchmark")
    
    # Model arguments
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    
    # Parallelism arguments
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--dp", type=int, default=1, help="Data parallel size (number of vLLM instances)")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel size (for MoE models)")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes (for distributed backend selection)")
    parser.add_argument("--gpus-per-node", type=int, default=4, help="GPUs per node (GB200=4, H100=8)")
    parser.add_argument("--gpu-model", type=str, default="unknown", help="GPU model name (e.g., GB200, H100)")
    parser.add_argument("--gpu-utilization", type=float, default=0.9, help="GPU memory utilization")
    
    # GRPO-style batch configuration
    parser.add_argument("--num-prompts", type=int, default=64, help="Number of unique prompts")
    parser.add_argument("--num-generations", type=int, default=32, help="Generations per prompt")
    
    # Generation arguments
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max model length")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    
    # Data arguments - can use file or random
    parser.add_argument("--prompts-file", type=str, default=None, 
                        help="JSON file with prompts (list of token_ids lists)")
    parser.add_argument("--random-input-len", type=int, default=150,
                        help="Random input length if no prompts file")
    
    # Output
    parser.add_argument("--output-file", type=str, default=None, help="Output results to JSON file")
    
    return parser.parse_args()


def load_prompts_from_file(prompts_file, num_prompts, num_generations):
    """Load prompts from a JSON file prepared by NeMo-RL."""
    with open(prompts_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Take only num_prompts
    unique_prompts = data[:num_prompts]
    
    # Replicate for GRPO-style batching
    all_prompts = []
    for prompt in unique_prompts:
        for _ in range(num_generations):
            all_prompts.append({"prompt_token_ids": prompt["prompt_token_ids"]})
    
    return all_prompts, len(unique_prompts)


def generate_random_prompts(num_prompts, num_generations, input_len, vocab_size=32000):
    """Generate random prompts for benchmarking."""
    import random
    
    unique_prompts = []
    for _ in range(num_prompts):
        token_ids = [random.randint(100, vocab_size - 1) for _ in range(input_len)]
        unique_prompts.append({"prompt_token_ids": token_ids})
    
    # Replicate for GRPO-style batching
    all_prompts = []
    for prompt in unique_prompts:
        for _ in range(num_generations):
            all_prompts.append({"prompt_token_ids": prompt["prompt_token_ids"]})
    
    return all_prompts, len(unique_prompts)


def run_single_dp_instance(args, all_prompts, dp_rank, dp_size):
    """Run a single DP instance (for DP=1 case or when called from multiprocessing)."""
    
    # For DP > 1, we need to shard prompts across instances
    if dp_size > 1:
        # Each DP rank handles a subset of prompts
        prompts_per_rank = len(all_prompts) // dp_size
        start_idx = dp_rank * prompts_per_rank
        end_idx = start_idx + prompts_per_rank if dp_rank < dp_size - 1 else len(all_prompts)
        my_prompts = all_prompts[start_idx:end_idx]
    else:
        my_prompts = all_prompts
    
    # Initialize vLLM
    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tp,
        "pipeline_parallel_size": args.pp,
        "gpu_memory_utilization": args.gpu_utilization,
        "max_model_len": args.max_model_len,
        "trust_remote_code": args.trust_remote_code,
        "dtype": "bfloat16",
        "disable_log_stats": True,
    }
    
    # Use Ray for multi-GPU within a single instance
    if args.tp > 1 or args.pp > 1:
        llm_kwargs["distributed_executor_backend"] = "ray"
    
    llm = LLM(**llm_kwargs)
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )
    
    # Run generation
    start_time = time.time()
    outputs = llm.generate(my_prompts, sampling_params)
    end_time = time.time()
    
    # Calculate metrics for this DP rank
    total_generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    total_input_tokens = sum(len(prompt["prompt_token_ids"]) for prompt in my_prompts)
    
    return {
        "dp_rank": dp_rank,
        "num_prompts": len(my_prompts),
        "total_input_tokens": total_input_tokens,
        "total_generated_tokens": total_generated_tokens,
        "time_sec": end_time - start_time,
    }


def main():
    args = parse_args()
    
    total_samples = args.num_prompts * args.num_generations
    gpus_per_instance = args.tp * args.pp
    total_gpus = gpus_per_instance * args.dp
    
    # Check if MoE model (EP > 1)
    is_moe = args.ep > 1
    
    print(f"\n{'=' * 70}")
    print("Pure vLLM Offline Throughput Benchmark (Duncan-style)")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    if is_moe:
        print(f"Parallelism: TP={args.tp}, PP={args.pp}, DP={args.dp}, EP={args.ep} (MoE)")
    else:
        print(f"Parallelism: TP={args.tp}, PP={args.pp}, DP={args.dp}")
    print(f"GPUs per vLLM instance: {gpus_per_instance}")
    print(f"Total GPUs: {total_gpus}")
    print(f"GRPO-style batch: {args.num_prompts} prompts × {args.num_generations} generations = {total_samples} total")
    print(f"Max tokens: {args.max_tokens}")
    print(f"{'=' * 70}\n")
    
    # Load or generate prompts
    if args.prompts_file and os.path.exists(args.prompts_file):
        print(f"▶ Loading prompts from {args.prompts_file}...")
        all_prompts, num_unique = load_prompts_from_file(
            args.prompts_file, args.num_prompts, args.num_generations
        )
    else:
        print(f"▶ Generating random prompts (input_len={args.random_input_len})...")
        all_prompts, num_unique = generate_random_prompts(
            args.num_prompts, args.num_generations, args.random_input_len
        )
    
    print(f"  ✓ Prepared {len(all_prompts)} requests ({num_unique} unique prompts)")
    
    if args.dp > 1:
        # DP > 1: Use vLLM's data_parallel_size with Ray backend
        print(f"\n▶ Initializing vLLM (TP={args.tp}, PP={args.pp}, DP={args.dp})...")
        print(f"   Total GPUs = TP*PP*DP = {args.tp}*{args.pp}*{args.dp} = {total_gpus}")
        
        llm_kwargs = {
            "model": args.model,
            "tensor_parallel_size": args.tp,
            "pipeline_parallel_size": args.pp,
            "data_parallel_size": args.dp,  # Enable actual data parallelism!
            "gpu_memory_utilization": args.gpu_utilization,
            "max_model_len": args.max_model_len,
            "trust_remote_code": args.trust_remote_code,
            "dtype": "bfloat16",
            "disable_log_stats": True,
            "distributed_executor_backend": "ray",  # Ray required for DP
        }
        
        # Enable Expert Parallelism for MoE models
        if args.ep > 1:
            llm_kwargs["enable_expert_parallel"] = True
            print(f"   Expert Parallelism enabled (EP={args.ep})")
        
        llm = LLM(**llm_kwargs)
        print(f"  ✓ vLLM initialized with DP={args.dp}")
    else:
        # Single instance (DP=1)
        if is_moe:
            print(f"\n▶ Initializing vLLM (TP={args.tp}, PP={args.pp}, DP=1, EP={args.ep})...")
        else:
            print(f"\n▶ Initializing vLLM (TP={args.tp}, PP={args.pp}, DP=1)...")
        
        llm_kwargs = {
            "model": args.model,
            "tensor_parallel_size": args.tp,
            "pipeline_parallel_size": args.pp,
            "gpu_memory_utilization": args.gpu_utilization,
            "max_model_len": args.max_model_len,
            "trust_remote_code": args.trust_remote_code,
            "dtype": "bfloat16",
            "disable_log_stats": True,
        }
        
        # Enable Expert Parallelism for MoE models
        if args.ep > 1:
            llm_kwargs["enable_expert_parallel"] = True
            print(f"   Expert Parallelism enabled (EP={args.ep})")
        
        # Use Ray only for multi-node (like Duncan's script)
        # Single node: vLLM uses multiprocessing internally
        if args.num_nodes > 1:
            llm_kwargs["distributed_executor_backend"] = "ray"
            print(f"   Using Ray backend (multi-node: {args.num_nodes} nodes)")
        else:
            print(f"   Using default backend (single-node)")
        
        llm = LLM(**llm_kwargs)
        print("  ✓ vLLM initialized")
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )
    
    # Run benchmark
    print(f"\n▶ Starting generation ({len(all_prompts)} requests)...")
    
    start_time = time.time()
    outputs = llm.generate(all_prompts, sampling_params)
    end_time = time.time()
    
    total_time = end_time - start_time
    
    # Calculate metrics
    total_generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    total_input_tokens = sum(len(prompt["prompt_token_ids"]) for prompt in all_prompts)
    
    results = {
        "model": args.model,
        "gpu_model": args.gpu_model,
        "num_nodes": args.num_nodes,
        "gpus_per_node": args.gpus_per_node,
        "tp_size": args.tp,
        "pp_size": args.pp,
        "dp_size": args.dp,
        "ep_size": args.ep,
        "is_moe": is_moe,
        "gpus_per_instance": gpus_per_instance,
        "total_gpus": total_gpus,
        "num_unique_prompts": num_unique,
        "num_generations_per_prompt": args.num_generations,
        "total_requests": len(all_prompts),
        "total_input_tokens": total_input_tokens,
        "total_generated_tokens": total_generated_tokens,
        "total_time_sec": total_time,
        "generation_throughput_tokens_per_sec": total_generated_tokens / total_time,
        "request_throughput_per_sec": len(all_prompts) / total_time,
        "tokens_per_sec_per_gpu": (total_generated_tokens / total_time) / total_gpus,
    }
    
    # Print results
    print(f"\n{'=' * 70}")
    print("Benchmark Results")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"Parallelism:")
    print(f"  • TP Size: {args.tp}")
    print(f"  • PP Size: {args.pp}")
    print(f"  • DP Size: {args.dp}")
    print(f"  • GPUs per instance: {gpus_per_instance}")
    print(f"  • Total GPUs: {total_gpus}")
    print(f"")
    print(f"GRPO-style Configuration:")
    print(f"  • Unique prompts: {num_unique}")
    print(f"  • Generations per prompt: {args.num_generations}")
    print(f"  • Total requests: {len(all_prompts)}")
    print(f"")
    print(f"Tokens:")
    print(f"  • Total input tokens: {total_input_tokens:,}")
    print(f"  • Total generated tokens: {total_generated_tokens:,}")
    print(f"  • Avg generated tokens/request: {total_generated_tokens / len(all_prompts):.1f}")
    print(f"")
    print(f"Performance:")
    print(f"  • Total time: {total_time:.2f} s")
    print(f"  • Generation throughput: {total_generated_tokens / total_time:,.0f} tokens/sec")
    print(f"  • Request throughput: {len(all_prompts) / total_time:.2f} requests/sec")
    print(f"  • Tokens/sec/GPU: {(total_generated_tokens / total_time) / total_gpus:,.0f}")
    print(f"{'=' * 70}")
    
    # Save results to file
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()

