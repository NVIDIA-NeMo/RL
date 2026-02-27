#!/usr/bin/env python3
"""Consolidate NeMo-RL sharded safetensors checkpoint into standard HuggingFace format.

This script reads the FSDP-sharded safetensors files saved by NeMo-RL's
AutomodelCheckpointManager and consolidates them into a single HuggingFace-
compatible directory that can be loaded by vLLM, transformers, etc.

Usage:
    python consolidate_checkpoint.py \
        --input /path/to/step_50/policy/weights \
        --output /path/to/hf_checkpoint \
        --model-name meta-llama/Llama-3.2-1B
"""

import argparse
import json
import logging
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(REPO_ROOT)
# nemo_automodel is a workspace member, add it to the path explicitly
sys.path.append(os.path.join(REPO_ROOT, "3rdparty", "Automodel-workspace", "Automodel"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Consolidate NeMo-RL sharded checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to policy/weights directory (contains model/ subdirectory with sharded safetensors)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for HuggingFace-format checkpoint",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Original HuggingFace model name (e.g., meta-llama/Llama-3.2-1B) for tokenizer",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_model_dir = os.path.join(args.input, "model")
    hf_metadata_dir = os.path.join(input_model_dir, ".hf_metadata")

    if not os.path.isdir(input_model_dir):
        logger.error(f"Input model directory not found: {input_model_dir}")
        sys.exit(1)

    if not os.path.isdir(hf_metadata_dir):
        logger.error(f"HF metadata directory not found: {hf_metadata_dir}")
        sys.exit(1)

    if os.path.exists(args.output) and not args.overwrite:
        logger.error(
            f"Output directory already exists: {args.output}. Use --overwrite to replace."
        )
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Step 1: Read the fqn-to-file-index mapping
    mapping_path = os.path.join(hf_metadata_dir, "fqn_to_file_index_mapping.json")
    with open(mapping_path, "r") as f:
        fqn_to_index_mapping = json.load(f)
    logger.info(
        f"Loaded fqn_to_file_index_mapping with {len(fqn_to_index_mapping)} parameters"
    )

    # Step 2: Consolidate sharded safetensors into standard HuggingFace format
    from nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors import (
        consolidate_safetensors_files,
    )

    logger.info(f"Consolidating shards from {input_model_dir} -> {args.output}")
    consolidate_safetensors_files(
        input_dir=input_model_dir,
        output_dir=args.output,
        fqn_to_index_mapping=fqn_to_index_mapping,
        num_threads=4,
    )
    logger.info("Consolidation complete")

    # Step 3: Copy config.json and generation_config.json from .hf_metadata
    for config_file in ["config.json", "generation_config.json"]:
        src = os.path.join(hf_metadata_dir, config_file)
        dst = os.path.join(args.output, config_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            logger.info(f"Copied {config_file}")

    # Step 4: Save tokenizer from the original HuggingFace model
    logger.info(f"Saving tokenizer from {args.model_name}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)
    logger.info(f"Tokenizer saved")

    # Verify output
    output_files = os.listdir(args.output)
    logger.info(f"Output directory contents: {sorted(output_files)}")
    logger.info(f"Done! HuggingFace checkpoint saved to: {args.output}")


if __name__ == "__main__":
    main()
