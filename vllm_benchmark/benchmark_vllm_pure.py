"""
Pure vLLM Offline Throughput Benchmark

This script measures "ideal" vLLM inference performance WITHOUT NeMo-RL overhead.
It uses the same dataset and prompt structure as GRPO training for fair comparison.

Key features:
- Uses vLLM directly (no NeMo-RL wrapper)
- Loads the same math dataset used in GRPO training
- Replicates prompts like GRPO: num_prompts × num_generations_per_prompt
- Multi-node support via Ray (same as vLLM's native multi-node)

Usage:
    # Single node (4 GPUs with TP=4)
    python benchmark_vllm_pure.py --model Qwen/Qwen2.5-32B-Instruct --tp 4

    # Multi-node via SLURM (see benchmark_vllm_pure.sh)
"""

import os
import time
import argparse
from collections import defaultdict

# Set vLLM environment before importing
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

import ray
import torch
from transformers import AutoTokenizer

# NOTE: vLLM is only available in the Ray worker environment, not on the driver node.
# Import vLLM inside Ray actors/functions only.

# NeMo-RL imports for dataset loading only
from nemo_rl.data.datasets.response_datasets import load_response_dataset
from nemo_rl.data.datasets.processed_dataset import AllTaskProcessedDataset
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.processors import math_hf_data_processor


def parse_args():
    parser = argparse.ArgumentParser(description="Pure vLLM Offline Throughput Benchmark")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-32B-Instruct", help="Model name or path")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    
    # Parallelism arguments (for vLLM)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--dp", type=int, default=1, help="Data parallel size (runs multiple vLLM instances)")
    parser.add_argument("--gpu-utilization", type=float, default=0.9, help="GPU memory utilization")
    
    # GRPO-style batch configuration
    parser.add_argument("--num-prompts", type=int, default=64, help="Number of unique prompts (like grpo.num_prompts_per_step)")
    parser.add_argument("--num-generations", type=int, default=32, help="Generations per prompt (like grpo.num_generations_per_prompt)")
    
    # Generation arguments
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max model length")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    
    # Data arguments
    parser.add_argument("--dataset-name", type=str, default="OpenMathInstruct-2", help="Dataset name")
    parser.add_argument("--prompt-file", type=str, default="examples/prompts/cot.txt", help="Path to prompt file")
    
    # Multi-node arguments
    parser.add_argument("--ray-address", type=str, default=None, help="Ray cluster address (for multi-node)")
    
    return parser.parse_args()


def init_ray_if_needed(args):
    """Initialize Ray for vLLM workers.
    
    Ray is always needed because vLLM is only available in the Ray worker environment.
    """
    if args.ray_address:
        ray.init(address=args.ray_address, ignore_reinit_error=True)
        print(f"  ✓ Connected to Ray cluster at {args.ray_address}")
    elif "RAY_ADDRESS" in os.environ:
        ray.init(address="auto", ignore_reinit_error=True)
        print(f"  ✓ Connected to Ray cluster via RAY_ADDRESS")
    else:
        # Initialize local Ray cluster
        ray.init(ignore_reinit_error=True)
        print(f"  ✓ Initialized local Ray cluster")
    
    print(f"  Ray resources: {ray.available_resources()}")


def load_and_prepare_prompts(args, tokenizer):
    """Load dataset and prepare prompts in GRPO style."""
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
    
    # Extract unique prompts
    print(f"▶ Extracting {args.num_prompts} unique prompts...")
    num_samples = min(args.num_prompts, len(processed_dataset))
    unique_prompts = []
    
    for i in range(num_samples):
        datum = processed_dataset[i]
        token_ids = []
        for msg in datum["message_log"]:
            token_ids.extend(msg["token_ids"].tolist())
        unique_prompts.append({"prompt_token_ids": token_ids})
    
    # Replicate prompts like GRPO: each prompt repeated num_generations times
    print(f"▶ Replicating prompts: {num_samples} × {args.num_generations} = {num_samples * args.num_generations}")
    all_prompts = []
    for prompt in unique_prompts:
        for _ in range(args.num_generations):
            all_prompts.append(prompt)
    
    return all_prompts, num_samples


def run_single_instance_benchmark(args, all_prompts, num_unique_prompts):
    """Run benchmark with a single vLLM instance (TP/PP only, no DP).
    
    Note: This function must run inside a Ray actor where vLLM is available.
    For single instance, we use DP=1 which goes through the Ray actor path.
    """
    # This path is not used when DP >= 1; all benchmarks go through Ray actors
    raise NotImplementedError(
        "Single instance benchmark should use DP=1 with Ray actors. "
        "vLLM is not available on the driver node."
    )


@ray.remote
class VLLMWorkerDP:
    """Ray actor for Data Parallel vLLM instances.
    
    vLLM is imported inside this actor because it's only available
    in the Ray worker environment (not on the driver node).
    """
    
    def __init__(self, dp_rank, model, tp, pp, gpu_utilization, max_model_len, 
                 trust_remote_code, temperature, max_tokens):
        # Import vLLM inside the actor (only available in worker env)
        from vllm import LLM, SamplingParams
        
        self.dp_rank = dp_rank
        
        # Each DP worker uses TP GPUs
        llm_kwargs = {
            "model": model,
            "tensor_parallel_size": tp,
            "pipeline_parallel_size": pp,
            "gpu_memory_utilization": gpu_utilization,
            "max_model_len": max_model_len,
            "trust_remote_code": trust_remote_code,
            "dtype": "bfloat16",
            "disable_log_stats": True,
        }
        
        if tp > 1 or pp > 1:
            llm_kwargs["distributed_executor_backend"] = "ray"
        
        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
        )
    
    def generate(self, prompts):
        """Generate for a subset of prompts."""
        outputs = self.llm.generate(prompts, self.sampling_params)
        total_generated = sum(len(output.outputs[0].token_ids) for output in outputs)
        total_input = sum(len(p["prompt_token_ids"]) for p in prompts)
        return {
            "total_generated_tokens": total_generated,
            "total_input_tokens": total_input,
            "num_requests": len(prompts),
        }


def run_dp_benchmark(args, all_prompts, num_unique_prompts):
    """Run benchmark with multiple vLLM instances (Data Parallelism)."""
    
    dp_size = args.dp
    model_parallel_size = args.tp * args.pp
    total_gpus = dp_size * model_parallel_size
    
    print(f"\n▶ Initializing {dp_size} vLLM instances (DP={dp_size}, TP={args.tp}, PP={args.pp})...")
    
    # Create DP workers - pass individual args instead of namespace object
    workers = []
    for dp_rank in range(dp_size):
        # Each worker needs model_parallel_size GPUs
        worker = VLLMWorkerDP.options(
            num_gpus=model_parallel_size
        ).remote(
            dp_rank=dp_rank,
            model=args.model,
            tp=args.tp,
            pp=args.pp,
            gpu_utilization=args.gpu_utilization,
            max_model_len=args.max_model_len,
            trust_remote_code=args.trust_remote_code,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        workers.append(worker)
    
    print(f"  ✓ {dp_size} vLLM instances initialized")
    
    # Split prompts across DP workers
    prompts_per_worker = len(all_prompts) // dp_size
    prompt_splits = []
    for i in range(dp_size):
        start_idx = i * prompts_per_worker
        end_idx = start_idx + prompts_per_worker if i < dp_size - 1 else len(all_prompts)
        prompt_splits.append(all_prompts[start_idx:end_idx])
    
    print(f"\n▶ Starting generation ({len(all_prompts)} requests across {dp_size} workers)...")
    
    start_time = time.time()
    
    # Run generation in parallel
    futures = [worker.generate.remote(prompts) for worker, prompts in zip(workers, prompt_splits)]
    results = ray.get(futures)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Aggregate results
    total_generated_tokens = sum(r["total_generated_tokens"] for r in results)
    total_input_tokens = sum(r["total_input_tokens"] for r in results)
    
    return {
        "total_time": total_time,
        "total_generated_tokens": total_generated_tokens,
        "total_input_tokens": total_input_tokens,
        "total_requests": len(all_prompts),
        "num_unique_prompts": num_unique_prompts,
        "total_gpus": total_gpus,
    }


def main():
    args = parse_args()
    
    total_samples = args.num_prompts * args.num_generations
    total_gpus = args.dp * args.tp * args.pp
    
    print(f"\n{'=' * 70}")
    print("Pure vLLM Offline Throughput Benchmark")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"Parallelism: TP={args.tp}, PP={args.pp}, DP={args.dp}")
    print(f"Total GPUs: {total_gpus}")
    print(f"GRPO-style batch: {args.num_prompts} prompts × {args.num_generations} generations = {total_samples} total")
    print(f"Max tokens: {args.max_tokens}")
    print(f"{'=' * 70}\n")
    
    # Initialize Ray (always needed - vLLM only available in Ray worker env)
    print("▶ Initializing Ray...")
    init_ray_if_needed(args)
    
    # Load tokenizer
    print(f"▶ Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    
    # Load and prepare prompts
    all_prompts, num_unique_prompts = load_and_prepare_prompts(args, tokenizer)
    
    # Run benchmark - always use Ray actors since vLLM is only in worker env
    # Even for DP=1, we need to run through Ray actor
    if args.dp < 1:
        args.dp = 1  # Minimum DP is 1
    results = run_dp_benchmark(args, all_prompts, num_unique_prompts)
    
    # Print results
    print(f"\n{'=' * 70}")
    print("Benchmark Results")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"TP Size: {args.tp}")
    print(f"PP Size: {args.pp}")
    print(f"DP Size: {args.dp}")
    print(f"Total GPUs: {results['total_gpus']}")
    print(f"")
    print(f"GRPO-style Configuration:")
    print(f"  • Unique prompts: {results['num_unique_prompts']}")
    print(f"  • Generations per prompt: {args.num_generations}")
    print(f"  • Total requests: {results['total_requests']}")
    print(f"")
    print(f"Tokens:")
    print(f"  • Total input tokens: {results['total_input_tokens']:,}")
    print(f"  • Total generated tokens: {results['total_generated_tokens']:,}")
    print(f"  • Avg generated tokens/request: {results['total_generated_tokens'] / results['total_requests']:.1f}")
    print(f"")
    print(f"Performance:")
    print(f"  • Total time: {results['total_time']:.2f} s")
    print(f"  • Generation throughput: {results['total_generated_tokens'] / results['total_time']:,.0f} tokens/sec")
    print(f"  • Request throughput: {results['total_requests'] / results['total_time']:.2f} requests/sec")
    print(f"  • Tokens/sec/GPU: {(results['total_generated_tokens'] / results['total_time']) / results['total_gpus']:,.0f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

