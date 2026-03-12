#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["transformers"]
# ///
"""Filter JSONL datasets to keep only samples whose tokenized input fits
within a given token budget.

Uses the actual model tokenizer + chat template for accurate measurement.

Usage:
    uv run experiments/filter_data_by_length.py \
        --tokenizer /path/to/model \
        --max-tokens 8192 \
        --inputs data/multichallenge_vanilla.jsonl data/inverse_if.jsonl \
        --suffix _8k
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Filter JSONL by tokenized input length")
    parser.add_argument("--tokenizer", type=str, required=True, help="HF model/tokenizer path")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max token budget (default: 8192)")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True, help="Input JSONL files")
    parser.add_argument("--suffix", type=str, default="_8k", help="Suffix for output files (default: _8k)")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    print(f"Loading tokenizer from {args.tokenizer} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    print(f"  vocab_size={tokenizer.vocab_size}, model_max_length={tokenizer.model_max_length}")

    for input_path in args.inputs:
        if not input_path.exists():
            print(f"\nSkipping {input_path} (not found)")
            continue

        output_path = input_path.with_stem(input_path.stem + args.suffix)
        print(f"\n{'='*60}")
        print(f"Processing: {input_path}")
        print(f"Output:     {output_path}")
        print(f"Max tokens: {args.max_tokens}")

        kept_lines = []
        filtered_count = 0
        token_lengths = []

        with open(input_path) as f:
            for line_num, line in enumerate(f, 1):
                row = json.loads(line)
                messages = row.get("responses_create_params", {}).get("input", [])

                if hasattr(tokenizer, "apply_chat_template"):
                    try:
                        token_ids = tokenizer.apply_chat_template(
                            messages, tokenize=True, add_generation_prompt=True
                        )
                        num_tokens = len(token_ids)
                    except Exception:
                        text = "".join(m.get("content", "") for m in messages)
                        num_tokens = len(tokenizer.encode(text))
                else:
                    text = "".join(m.get("content", "") for m in messages)
                    num_tokens = len(tokenizer.encode(text))

                token_lengths.append(num_tokens)

                if num_tokens <= args.max_tokens:
                    kept_lines.append(line)
                else:
                    filtered_count += 1

        with open(output_path, "w") as f:
            for line in kept_lines:
                f.write(line)

        total = len(token_lengths)
        kept = len(kept_lines)
        print(f"  Total:    {total}")
        print(f"  Kept:     {kept} ({kept/total*100:.1f}%)")
        print(f"  Filtered: {filtered_count} ({filtered_count/total*100:.1f}%)")
        if token_lengths:
            token_lengths.sort()
            print(f"  Token lengths: min={token_lengths[0]}, "
                  f"median={token_lengths[len(token_lengths)//2]}, "
                  f"max={token_lengths[-1]}, "
                  f"p95={token_lengths[int(len(token_lengths)*0.95)]}")


if __name__ == "__main__":
    main()
