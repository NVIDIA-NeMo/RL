#!/usr/bin/env python3
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
"""Model Diagnostic: Check HuggingFace Model Embeddings for Untrained Patterns.

This script loads a HuggingFace model and analyzes the input and output embeddings
to detect patterns that suggest the model may be untrained or improperly initialized.

uv run --extra mcore 3.check_hf_model_embeddings_untrained.py --model nvidia/Nemotron-H-8B-Base-8K
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def format_index_ranges(indices):
    """Format a list of indices into range strings like '0-1,3-6'."""
    if not indices:
        return ""

    ranges = []
    start = end = indices[0]

    for i in range(1, len(indices)):
        if indices[i] == end + 1:
            end = indices[i]
        else:
            ranges.append(str(start) if start == end else f"{start}-{end}")
            start = end = indices[i]

    # Add the last range
    ranges.append(str(start) if start == end else f"{start}-{end}")
    return ",".join(ranges)


def get_token_info(tokenizer, idx):
    """Get token information for a given index."""
    if not tokenizer:
        return "N/A"
    try:
        return repr(tokenizer.decode([idx]))
    except Exception:
        return "N/A"


def print_problematic_embeddings(
    weights, indices, problem_type, metric_values, threshold, tokenizer=None
):
    """Print detailed information about each problematic embedding."""
    if not indices:
        return

    print(f"\n--- Detailed {problem_type} Embeddings ---")
    for idx in indices:
        embedding = weights[idx]
        metric_val = metric_values[idx].item()
        token_info = get_token_info(tokenizer, idx)

        # Get first 2 and last 2 values
        first_two = embedding[:2].tolist()
        last_two = embedding[-2:].tolist()

        print(
            f"Index {idx}: {problem_type} (metric: {metric_val:.2e} > {threshold:.2e})"
        )
        print(f"  Token: {token_info}")
        print(
            f"  Values: [{first_two[0]:.2e}, {first_two[1]:.2e}, ..., {last_two[0]:.2e}, {last_two[1]:.2e}]"
        )


def find_output_embeddings(model):
    """Find the output embeddings layer in various model architectures."""
    if hasattr(model, "get_output_embeddings"):
        return model.get_output_embeddings()
    elif hasattr(model, "lm_head"):
        return model.lm_head
    elif hasattr(model, "embed_out"):
        return model.embed_out
    return None


def check_embedding_layer(
    embeddings,
    layer_name,
    near_zero_threshold,
    identical_threshold,
    tokenizer=None,
    model=None,
):
    """Check an embedding layer for untrained patterns."""
    print(f"\n=== {layer_name} Analysis ===")

    # Check if embeddings are tied (for output embeddings)
    tied_info = ""
    if layer_name == "Output Embeddings" and model and hasattr(model, "config"):
        tied = getattr(model.config, "tie_word_embeddings", False)
        tied_info = f" (Tied: {tied})"
        print(f"Tied word embeddings: {tied}")

    # Get embedding weights
    weights = (
        embeddings.weight.data if hasattr(embeddings, "weight") else embeddings.data
    )

    print(f"Shape: {weights.shape}")
    print(f"Dtype: {weights.dtype}")

    # Check for near-zero embeddings
    near_zero_mask = torch.abs(weights) < near_zero_threshold
    near_zero_rows = near_zero_mask.all(dim=1)
    near_zero_indices = torch.where(near_zero_rows)[0].tolist()

    # Check for identical embeddings using standard deviation
    row_stds = weights.std(dim=1)
    identical_mask = row_stds < identical_threshold
    identical_indices = torch.where(identical_mask)[0].tolist()

    # Print detailed problematic embeddings
    max_abs_values = torch.abs(weights).max(dim=1)[0]
    print_problematic_embeddings(
        weights,
        near_zero_indices,
        "Near-zero",
        max_abs_values,
        near_zero_threshold,
        tokenizer,
    )
    print_problematic_embeddings(
        weights,
        identical_indices,
        "Identical",
        row_stds,
        identical_threshold,
        tokenizer,
    )

    # Return summary data instead of printing
    num_near_zero = len(near_zero_indices)
    num_identical = len(identical_indices)
    total_embeddings = weights.shape[0]

    # Flag potential issues
    issues = []
    if num_near_zero > 0:
        issues.append(f"{num_near_zero} near-zero embeddings")
    if num_identical > 0:
        issues.append(f"{num_identical} identical embeddings")

    return {
        "layer_name": layer_name,
        "tied_info": tied_info,
        "shape": weights.shape,
        "dtype": weights.dtype,
        "num_near_zero": num_near_zero,
        "num_identical": num_identical,
        "total_embeddings": total_embeddings,
        "near_zero_indices": near_zero_indices,
        "identical_indices": identical_indices,
        "near_zero_threshold": near_zero_threshold,
        "identical_threshold": identical_threshold,
        "mean_abs": torch.abs(weights).mean().item(),
        "max_abs": torch.abs(weights).max().item(),
        "min_std": row_stds.min().item(),
        "max_std": row_stds.max().item(),
        "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check HuggingFace model embeddings for untrained patterns"
    )
    parser.add_argument(
        "--model",
        default="nvidia/Nemotron-H-8B-Base-8K",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--near-zero-threshold",
        type=float,
        default=1e-10,
        help="Threshold for detecting near-zero embeddings (default: 1e-10)",
    )
    parser.add_argument(
        "--identical-threshold",
        type=float,
        default=1e-8,
        help="Threshold for detecting identical embeddings via std dev (default: 1e-8)",
    )

    args = parser.parse_args()

    print(f"Loading model: {args.model}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Model loaded successfully")
    print(f"Model type: {type(model).__name__}")
    print(f"Vocabulary size: {len(tokenizer)}")

    # Collect summary data from both embeddings
    summaries = []

    # Check input embeddings
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None:
        input_summary = check_embedding_layer(
            input_embeddings,
            "Input Embeddings",
            args.near_zero_threshold,
            args.identical_threshold,
            tokenizer,
            model,
        )
        summaries.append(input_summary)
    else:
        print("\n⚠️  Could not find input embeddings layer")

    # Check output embeddings
    output_embeddings = find_output_embeddings(model)
    if output_embeddings is not None:
        output_summary = check_embedding_layer(
            output_embeddings,
            "Output Embeddings",
            args.near_zero_threshold,
            args.identical_threshold,
            tokenizer,
            model,
        )
        summaries.append(output_summary)
    else:
        print("\n⚠️  Could not find output embeddings layer")

    # Print summaries together
    print("\n" + "=" * 80)
    print("EMBEDDING SUMMARIES")
    print("=" * 80)

    for summary in summaries:
        print(f"\n--- {summary['layer_name']} Summary{summary['tied_info']} ---")
        print(f"Shape: {summary['shape']}, Dtype: {summary['dtype']}")

        print(
            f"Near-zero embeddings (abs < {summary['near_zero_threshold']:.2e}): {summary['num_near_zero']}/{summary['total_embeddings']} ({100 * summary['num_near_zero'] / summary['total_embeddings']:.1f}%)"
        )
        if summary["near_zero_indices"]:
            print(f"  Indices: {format_index_ranges(summary['near_zero_indices'])}")

        print(
            f"Identical embeddings (std < {summary['identical_threshold']:.2e}): {summary['num_identical']}/{summary['total_embeddings']} ({100 * summary['num_identical'] / summary['total_embeddings']:.1f}%)"
        )
        if summary["identical_indices"]:
            print(f"  Indices: {format_index_ranges(summary['identical_indices'])}")

        print(
            f"Statistics: mean_abs={summary['mean_abs']:.6f}, max_abs={summary['max_abs']:.6f}, std_range=[{summary['min_std']:.6f}, {summary['max_std']:.6f}]"
        )

        if summary["issues"]:
            print(f"⚠️  POTENTIAL ISSUES: {', '.join(summary['issues'])}")
        else:
            print("✅ No obvious untrained patterns detected")

    print("\n=== Final Summary ===")
    print(f"Model: {args.model}")
    print("Analysis complete.")


if __name__ == "__main__":
    main()
