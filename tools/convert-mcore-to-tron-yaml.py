#!/usr/bin/env python3
"""
Convert Megatron checkpoint arguments to NeMo-RL YAML configuration.

This script loads a Megatron checkpoint and extracts the model configuration
arguments, then outputs them in a YAML format suitable for NeMo-RL training.

Usage:
    uv run --extra mcore tools/convert-mcore-to-tron-yaml.py <checkpoint_path> [--output <output_file>] [--base-config <config_name>]
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import nemo_rl  # import just to get megatron.training added to path

from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import load_args_from_checkpoint

from nemo.tron.config import CheckpointConfig, ConfigContainer, LoggerConfig, TokenizerConfig
from megatron.core.optimizer import OptimizerConfig


# Mapping of config names to dataclass import paths
CONFIG_MAPPING = {
    'nemotron-h': 'nemo.collections.llm.gpt.model.ssm.NemotronHConfig8B'
}


def import_config_class(config_path: str):
    """
    Import a config class from its module path.
    
    Args:
        config_path: Full module path to the config class
        
    Returns:
        The config class
    """
    try:
        module_path, class_name = config_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import config class {config_path}: {e}")


def load_checkpoint_args(checkpoint_path: str) -> tuple:
    """
    Load arguments from a Megatron checkpoint.
    
    Args:
        checkpoint_path: Path to the Megatron checkpoint directory
        
    Returns:
        Tuple of (main_args, checkpoint_args)
    """
    # Set up minimal sys.argv for Megatron argument parsing
    original_argv = sys.argv
    sys.argv = [
        'does_not_matter.py',
        '--load', checkpoint_path,
    ]
    
    try:
        # Parse arguments
        margs = parse_args()
        
        # Load checkpoint arguments
        margs, checkpoint_args = load_args_from_checkpoint(margs)
        
        # Pretty print margs
        print("=== Megatron Arguments ===")
        for attr in dir(margs):
            if not attr.startswith('_'):
                value = getattr(margs, attr)
                if not callable(value):
                    print(f"{attr}: {value}")
        print("==========================")
        
        return margs, checkpoint_args
        
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


def convert_args_to_config_dataclass(margs, checkpoint_args, base_config_name: str):
    """
    Convert Megatron arguments to a NeMo config dataclass.
    
    Args:
        margs: Main Megatron arguments
        checkpoint_args: Checkpoint-specific arguments
        base_config_name: Name of the base config to use
        
    Returns:
        Instantiated config dataclass with values from margs
    """
    if base_config_name not in CONFIG_MAPPING:
        raise ValueError(f"Unknown base config: {base_config_name}. Available configs: {list(CONFIG_MAPPING.keys())}")
    
    # Import the config class
    model_config_class = import_config_class(CONFIG_MAPPING[base_config_name])
    
    model_config = model_config_class()

    cfg = ConfigContainer(
        model_config=model_config,
        train_config=None,
        optimizer_config=OptimizerConfig(use_distributed_optimizer=False),
        ddp_config=None,
        scheduler_config=None,
        dataset_config=None,
        logger_config=LoggerConfig(),
        tokenizer_config=TokenizerConfig(),
        checkpoint_config=CheckpointConfig(
            async_save=False, save='does-not-matter', save_optim=False, ckpt_format="torch_dist"
        ),
        dist_config=None,
    )

    # Loop over all non-None configs and assign values into config if target config has that key
    for k, v in vars(margs).items():
        found = [] # Pointer to last found configs to print warnings
        for sub_cfg_field_name, sub_cfg in vars(cfg).items():
            if sub_cfg is None or sub_cfg_field_name.startswith('_'):
                continue
            elif not hasattr(sub_cfg, k):
                continue
            elif found:
                print(f"[WARNING] Found multiple potential mappings for config fields in the ConfigContainer for '{k}' in {type(sub_cfg)} and these configs {[type(c) for c in found]}")
            found.append(sub_cfg)
            setattr(sub_cfg, k, v)

    return cfg 


def main():
    parser = argparse.ArgumentParser(
        description="Convert Megatron checkpoint arguments to NeMo-RL YAML configuration"
    )
    parser.add_argument(
        "checkpoint_path",
        help="Path to the Megatron checkpoint directory"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--base-config", "-b",
        choices=list(CONFIG_MAPPING.keys()),
        default="nemotron-h",
        help="Base config class to use for conversion"
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint path
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    elif args.output and os.path.basename(args.output) != 'run_config.yaml':
        raise ValueError("Output file must be named be 'run_config.yaml' since that's what nemo-tron expects.")
    
    try:
        # Load checkpoint arguments
        print(f"Loading checkpoint from: {checkpoint_path}")
        margs, checkpoint_args = load_checkpoint_args(str(checkpoint_path))
        
        # Convert to config dataclass
        print(f"Converting to config dataclass: {args.base_config}")
        cfg = convert_args_to_config_dataclass(margs, checkpoint_args, args.base_config)
        
        print(f"Config instance: {cfg}")
        print(f"Config type: {type(cfg)}")
        
        # Use NeMo's API to write config to YAML
        if args.output:
            cfg.to_yaml(args.output)
            print(f"Configuration written to: {args.output}")
        else:
            print("=== Generated YAML Configuration ===")
            cfg.to_yaml()
            
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()

