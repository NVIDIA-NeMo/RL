"""
Prepare prompts from NeMo-RL dataset for vLLM benchmark.

This script extracts prompts from the same dataset used in GRPO training
and saves them as a JSON file that can be used by benchmark_vllm_offline.py.

IMPORTANT: To ensure the same prompts as GRPO training:
- Use the same seed as in your GRPO config (default: 42)
- The dataset is loaded and split using this seed
- Prompts are extracted in order (no shuffle in this script)

In GRPO training, the DataLoader shuffles prompts each epoch, but for 
benchmarking we want deterministic prompts, so we extract the first N
prompts from the dataset (after seed-based train/test split).

Usage:
    # Default (matches grpo_math_1B.yaml and inherited configs, seed=42)
    python prepare_prompts.py --output prompts.json --num-prompts 64

    # With custom seed (must match your GRPO config)
    python prepare_prompts.py --output prompts.json --num-prompts 64 --seed 1234
"""

import argparse
import json
from collections import defaultdict

from transformers import AutoTokenizer

from nemo_rl.data.datasets.response_datasets import load_response_dataset
from nemo_rl.data.datasets.processed_dataset import AllTaskProcessedDataset
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.processors import math_hf_data_processor


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare prompts for vLLM benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-32B-Instruct", 
                        help="Model name for tokenizer")
    parser.add_argument("--output", type=str, default="prompts.json", 
                        help="Output JSON file")
    parser.add_argument("--num-prompts", type=int, default=64, 
                        help="Number of prompts to extract (like grpo.num_prompts_per_step)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (must match grpo.seed in your config, default: 42)")
    parser.add_argument("--dataset-name", type=str, default="OpenMathInstruct-2", 
                        help="Dataset name")
    parser.add_argument("--prompt-file", type=str, default="examples/prompts/cot.txt", 
                        help="Path to prompt file")
    parser.add_argument("--max-model-len", type=int, default=4096, 
                        help="Max model length")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"▶ Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Seed: {args.seed} (must match grpo.seed in your config)")
    print(f"  Num prompts: {args.num_prompts}")
    print(f"  Dataset: {args.dataset_name}")
    
    print(f"\n▶ Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    print(f"▶ Loading dataset {args.dataset_name} with seed={args.seed}...")
    DATA_CONFIG = {
        "dataset_name": args.dataset_name,
        "prompt_file": args.prompt_file,
        "split": "train_1M",
        "max_input_seq_length": args.max_model_len,
    }
    # Use the same seed as GRPO training for reproducibility
    base_dataset = load_response_dataset(DATA_CONFIG, seed=args.seed)
    
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
    
    print(f"▶ Extracting {args.num_prompts} prompts...")
    num_samples = min(args.num_prompts, len(processed_dataset))
    prompts = []
    
    for i in range(num_samples):
        datum = processed_dataset[i]
        token_ids = []
        for msg in datum["message_log"]:
            token_ids.extend(msg["token_ids"].tolist())
        prompts.append({
            "prompt_token_ids": token_ids,
            "prompt_length": len(token_ids),
        })
    
    # Save to JSON
    print(f"▶ Saving {len(prompts)} prompts to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(prompts, f)
    
    # Print stats
    lengths = [p["prompt_length"] for p in prompts]
    print(f"\n✓ Done!")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Seed used: {args.seed}")
    print(f"  Min length: {min(lengths)}")
    print(f"  Max length: {max(lengths)}")
    print(f"  Avg length: {sum(lengths) / len(lengths):.1f}")
    print(f"  Output: {args.output}")
    print(f"\n⚠️  Note: These are the first {len(prompts)} prompts from the dataset")
    print(f"   (after seed-based train/test split). GRPO training shuffles")
    print(f"   prompts each epoch, but for benchmarking we use deterministic order.")


if __name__ == "__main__":
    main()

