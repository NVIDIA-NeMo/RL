import os
import time
import argparse
import torch
from typing import List, Dict, Any
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from collections import defaultdict
import ray

from nemo_rl.data.datasets.response_datasets import load_response_dataset
from nemo_rl.data.datasets.processed_dataset import AllTaskProcessedDataset
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.processors import math_hf_data_processor

def parse_args():
    parser = argparse.ArgumentParser(description="Standalone vLLM Benchmark for NeMo-RL")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-32B-Instruct", help="Model name or path")
    parser.add_argument("--trust-remote-code", action="store_true", default=True, help="Trust remote code")
    
    # Parallelism arguments
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--gpu-utilization", type=float, default=0.7, help="GPU memory utilization")
    
    # Data arguments
    parser.add_argument("--dataset-name", type=str, default="OpenMathInstruct-2", help="Dataset name")
    parser.add_argument("--prompt-file", type=str, default="examples/prompts/cot.txt", help="Path to prompt file")
    parser.add_argument("--num-prompts", type=int, default=64, help="Number of prompts to process")
    
    # Generation arguments
    parser.add_argument("--n", type=int, default=32, help="Number of generations per prompt")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max model length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    
    # Distributed arguments
    parser.add_argument("--multi-node", action="store_true", help="Enable multi-node (connect to Ray cluster)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Connect to Ray if multi-node
    if args.multi_node:
        import sys
        
        # Capture relevant environment variables to propagate to workers
        env_vars = {
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "PATH": os.environ.get("PATH", ""),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            # "VLLM_WORKER_MULTIPROC_METHOD": "spawn", # Try ensuring env is right first
        }
        
        ray.init(
            address="auto", 
            ignore_reinit_error=True,
            runtime_env={
                "py_executable": sys.executable,
                "env_vars": env_vars
            }
        )
        print(f"Connected to Ray cluster using python: {sys.executable}")
        print(f"Ray available resources: {ray.available_resources()}")

    # Configuration
    MODEL_NAME = args.model
    DATA_CONFIG = {
        "dataset_name": args.dataset_name,
        "prompt_file": args.prompt_file,
        "split": "train_1M",
        "max_input_seq_length": args.max_model_len,
    }
    
    SAMPLING_PARAMS = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_model_len,
        n=args.n,
    )
    
    # 1. Load Tokenizer
    print(f"Loading tokenizer for {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=args.trust_remote_code)
    except Exception as e:
        print(f"Failed to load tokenizer for {MODEL_NAME}: {e}")
        return

    # 2. Load Dataset
    print("Loading dataset...")
    base_dataset = load_response_dataset(DATA_CONFIG, seed=42)
    
    # 3. Process Dataset
    print("Processing dataset...")
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
    
    # 4. Extract Prompts
    print("Extracting prompts...")
    num_samples = min(args.num_prompts, len(processed_dataset))
    prompts = []
    
    for i in range(num_samples):
        datum = processed_dataset[i]
        token_ids = []
        for msg in datum["message_log"]:
            token_ids.extend(msg["token_ids"].tolist())
        prompts.append({"prompt_token_ids": token_ids})
        
    print(f"Prepared {len(prompts)} prompts.")
    
    # 5. Initialize vLLM
    print(f"Initializing vLLM with model={MODEL_NAME}, tp={args.tp}, pp={args.pp}...")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        gpu_memory_utilization=args.gpu_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        dtype="bfloat16",
        # disable_log_stats=True, # Match NeMo-RL worker config
    )
    
    # 6. Run Generation
    print("Starting generation...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params=SAMPLING_PARAMS)
    end_time = time.time()
    
    # 7. Metrics
    total_time = end_time - start_time
    total_generated_tokens = sum(len(out.token_ids) for req in outputs for out in req.outputs)
    total_requests = len(prompts)
    total_generations = total_requests * SAMPLING_PARAMS.n
    
    print(f"\n{'='*40}")
    print(f"Benchmark Results")
    print(f"{'='*40}")
    print(f"Model: {MODEL_NAME}")
    print(f"TP Size: {args.tp}")
    print(f"PP Size: {args.pp}")
    print(f"Prompts: {total_requests}")
    print(f"Generations per prompt: {SAMPLING_PARAMS.n}")
    print(f"Total Generations: {total_generations}")
    print(f"Total Time: {total_time:.2f} s")
    print(f"Total Generated Tokens: {total_generated_tokens}")
    print(f"Throughput (Tokens/sec): {total_generated_tokens / total_time:.2f}")
    print(f"Throughput (Requests/sec): {total_generations / total_time:.2f}")
    
    total_gpus = args.tp * args.pp
    print(f"Tokens/sec/GPU: {(total_generated_tokens / total_time) / total_gpus:.2f}")

if __name__ == "__main__":
    main()

