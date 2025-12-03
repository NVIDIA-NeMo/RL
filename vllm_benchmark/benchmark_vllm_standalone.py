"""
Standalone vLLM Benchmark for NeMo-RL

This script uses the same multi-node vLLM infrastructure as grpo.py to run
distributed inference benchmarks. It leverages:
- RayVirtualCluster for GPU resource management
- VllmGeneration for distributed vLLM inference with TP/PP/DP support

Usage:
    # Single node (4 GPUs with TP=4)
    uv run benchmark_vllm_standalone.py --model Qwen/Qwen2.5-32B-Instruct --tp 4

    # Multi-node (16 GPUs: 4 nodes x 4 GPUs, TP=4, DP=4)
    # Submit via ray.sub with NUM_NODES=4
    uv run benchmark_vllm_standalone.py --model Qwen/Qwen2.5-32B-Instruct --tp 4 --dp 4 --num-nodes 4 --gpus-per-node 4
"""

import time
import argparse
from collections import defaultdict

import ray
from transformers import AutoTokenizer

from nemo_rl.data.datasets.response_datasets import load_response_dataset
from nemo_rl.data.datasets.processed_dataset import AllTaskProcessedDataset
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.processors import math_hf_data_processor
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.vllm import VllmGeneration, VllmConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone vLLM Benchmark for NeMo-RL")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-32B-Instruct", help="Model name or path")
    parser.add_argument("--trust-remote-code", action="store_true", default=True, help="Trust remote code")
    
    # Cluster arguments
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--gpus-per-node", type=int, default=4, help="GPUs per node")
    
    # Parallelism arguments
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel size (for MoE models)")
    parser.add_argument("--gpu-utilization", type=float, default=0.7, help="GPU memory utilization")
    
    # Data arguments
    parser.add_argument("--dataset-name", type=str, default="OpenMathInstruct-2", help="Dataset name")
    parser.add_argument("--prompt-file", type=str, default="examples/prompts/cot.txt", help="Path to prompt file")
    parser.add_argument("--num-prompts", type=int, default=64, help="Number of prompts to process")
    
    # Generation arguments
    parser.add_argument("--n", type=int, default=32, help="Number of generations per prompt")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max model length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens to generate")
    
    # vLLM specific arguments
    parser.add_argument("--async-engine", action="store_true", default=False, help="Use async vLLM engine")
    parser.add_argument("--precision", type=str, default="bfloat16", choices=["bfloat16", "float16", "fp8"], help="Model precision")
    
    return parser.parse_args()


def build_vllm_config(args) -> VllmConfig:
    """Build VllmConfig from command line arguments.
    
    All keys required by vllm_worker.py are included here.
    See nemo_rl/models/generation/vllm/vllm_worker.py for usage.
    """
    return {
        "backend": "vllm",
        "model_name": args.model,
        "temperature": args.temperature,
        "top_k": -1,  # Disabled
        "top_p": 1.0,
        "max_new_tokens": args.max_tokens,  # Required by GenerationConfig
        "stop_token_ids": None,  # Required by GenerationConfig
        "stop_strings": [],
        "n": args.n,
        "include_stop_str_in_output": False,
        "colocated": {
            "enabled": False,  # Non-colocated for standalone benchmark
            "resources": {
                "num_nodes": args.num_nodes,
                "gpus_per_node": args.gpus_per_node,
            }
        },
        "vllm_cfg": {
            "tensor_parallel_size": args.tp,
            "pipeline_parallel_size": args.pp,
            "expert_parallel_size": args.ep,
            "gpu_memory_utilization": args.gpu_utilization,
            "precision": args.precision,
            "max_model_len": args.max_model_len,
            "async_engine": args.async_engine,
            "skip_tokenizer_init": False,
            "enforce_eager": False,  # Required by vllm_worker.py
            "load_format": "auto",  # Required by vllm_worker.py
            "hf_overrides": {},
            "max_num_seqs": 256,
            "enable_chunked_prefill": True,
            "disable_log_stats": True,
            "trust_remote_code": args.trust_remote_code,
        },
    }


def main():
    args = parse_args()
    
    # Calculate total GPUs and DP size
    total_gpus = args.num_nodes * args.gpus_per_node
    model_parallel_size = args.tp * args.pp
    dp_size = total_gpus // model_parallel_size
    
    print(f"\n{'=' * 60}")
    print("vLLM Multi-Node Benchmark (NeMo-RL Style)")
    print(f"{'=' * 60}")
    print(f"Model: {args.model}")
    print(f"Cluster: {args.num_nodes} nodes x {args.gpus_per_node} GPUs = {total_gpus} total GPUs")
    print(f"Parallelism: TP={args.tp}, PP={args.pp}, EP={args.ep}, DP={dp_size}")
    print(f"{'=' * 60}\n")
    
    # Validate configuration
    assert total_gpus % model_parallel_size == 0, (
        f"Total GPUs ({total_gpus}) must be divisible by model parallel size ({model_parallel_size})"
    )
    
    # Initialize Ray cluster (connects to existing cluster from ray.sub)
    print("▶ Initializing Ray cluster...")
    init_ray()
    print(f"  ✓ Ray available resources: {ray.available_resources()}")
    
    # Create RayVirtualCluster (same as grpo.py)
    print("\n▶ Creating RayVirtualCluster...")
    cluster = RayVirtualCluster(
        name="vllm_benchmark_cluster",
        bundle_ct_per_node_list=[args.gpus_per_node] * args.num_nodes,
        use_gpus=True,
        num_gpus_per_node=args.gpus_per_node,
        max_colocated_worker_groups=1,
    )
    print(f"  ✓ Cluster initialized with {cluster.world_size()} GPUs across {args.num_nodes} nodes")
    
    # Build vLLM config
    vllm_config = build_vllm_config(args)
    
    # Initialize VllmGeneration (same as grpo.py)
    print("\n▶ Initializing VllmGeneration...")
    policy_generation = VllmGeneration(
        cluster=cluster,
        config=vllm_config,
        name_prefix="vllm_benchmark",
    )
    # Ensure workers are ready
    policy_generation.finish_generation()
    print(f"  ✓ VllmGeneration initialized with DP={policy_generation.dp_size}")
    
    # Load tokenizer
    print(f"\n▶ Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    print("  ✓ Tokenizer loaded")
    
    # Load and process dataset
    print("\n▶ Loading dataset...")
    DATA_CONFIG = {
        "dataset_name": args.dataset_name,
        "prompt_file": args.prompt_file,
        "split": "train_1M",
        "max_input_seq_length": args.max_model_len,
    }
    base_dataset = load_response_dataset(DATA_CONFIG, seed=42)
    
    print("▶ Processing dataset...")
    math_task_spec = TaskDataSpec(
        task_name="math",
        prompt_file=DATA_CONFIG["prompt_file"],
        system_prompt_file=None,
    )
    
    task_data_processors = defaultdict(lambda: (math_task_spec, math_hf_data_processor))
    task_data_processors["math"] = (math_task_spec, math_hf_data_processor)
    
    processed_dataset = AllTaskProcessedDataset(
        base_dataset.formatted_ds["train"],
        tokenizer,
        math_task_spec,
        task_data_processors,
        max_seq_length=DATA_CONFIG["max_input_seq_length"],
    )
    
    # Extract prompts
    print("▶ Extracting prompts...")
    num_samples = min(args.num_prompts, len(processed_dataset))
    
    # Build input data in the format expected by VllmGeneration
    input_ids_list = []
    input_lengths_list = []
    
    for i in range(num_samples):
        datum = processed_dataset[i]
        token_ids = []
        for msg in datum["message_log"]:
            token_ids.extend(msg["token_ids"].tolist())
        input_ids_list.append(token_ids)
        input_lengths_list.append(len(token_ids))
    
    # Pad to same length
    import torch
    max_len = max(input_lengths_list)
    padded_input_ids = []
    for ids in input_ids_list:
        padded = ids + [tokenizer.pad_token_id] * (max_len - len(ids))
        padded_input_ids.append(padded)
    
    input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
    input_lengths = torch.tensor(input_lengths_list, dtype=torch.long)
    
    print(f"  ✓ Prepared {num_samples} prompts (max length: {max_len})")
    
    # Create BatchedDataDict for generation
    generation_data = BatchedDataDict({
        "input_ids": input_ids,
        "input_lengths": input_lengths,
    })
    
    # Run generation
    print(f"\n▶ Starting generation with n={args.n} generations per prompt...")
    print(f"  Total generations: {num_samples * args.n}")
    
    policy_generation.prepare_for_generation()
    
    start_time = time.time()
    
    # Generate (VllmGeneration handles DP internally)
    outputs = policy_generation.generate(generation_data, greedy=False)
    
    policy_generation.finish_generation()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate metrics
    generation_lengths = outputs["generation_lengths"]
    
    total_generated_tokens = generation_lengths.sum().item()
    total_requests = num_samples
    total_generations = total_requests * args.n
    
    # Print results
    print(f"\n{'=' * 60}")
    print("Benchmark Results")
    print(f"{'=' * 60}")
    print(f"Model: {args.model}")
    print(f"Cluster: {args.num_nodes} nodes x {args.gpus_per_node} GPUs = {total_gpus} total GPUs")
    print(f"TP Size: {args.tp}")
    print(f"PP Size: {args.pp}")
    print(f"EP Size: {args.ep}")
    print(f"DP Size: {dp_size}")
    print(f"Prompts: {total_requests}")
    print(f"Generations per prompt (n): {args.n}")
    print(f"Total Generations: {total_generations}")
    print(f"Total Time: {total_time:.2f} s")
    print(f"Total Generated Tokens: {total_generated_tokens}")
    print(f"Throughput (Tokens/sec): {total_generated_tokens / total_time:.2f}")
    print(f"Throughput (Generations/sec): {total_generations / total_time:.2f}")
    print(f"Tokens/sec/GPU: {(total_generated_tokens / total_time) / total_gpus:.2f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
