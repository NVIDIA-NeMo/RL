#!/usr/bin/env python3
"""
Helper script to export model configuration as shell variables.
Used by shell scripts to read from model_configs.yaml.

Usage:
    # Get config for specific model and cluster
    eval $(python get_model_config.py llama8b h100)
    
    # Or source it
    source <(python get_model_config.py qwen32b gb200)
"""

import sys
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def get_config_path() -> Path:
    return Path(__file__).parent / "model_configs.yaml"


def load_configs():
    config_path = get_config_path()
    if HAS_YAML and config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def export_config(preset: str, cluster_type: str, variant: str = None):
    """Print shell variable exports for the given configuration."""
    configs = load_configs()
    
    if preset not in configs:
        print(f"echo 'Error: Unknown preset {preset}'", file=sys.stderr)
        sys.exit(1)
    
    model_cfg = configs[preset]
    
    # Determine variant
    variant_key = variant if variant else cluster_type
    if variant_key not in model_cfg:
        variant_key = cluster_type
    if variant_key not in model_cfg:
        print(f"echo 'Error: No config for {preset} on {cluster_type}'", file=sys.stderr)
        sys.exit(1)
    
    cluster_cfg = model_cfg[variant_key]
    gen_cfg = cluster_cfg.get("generation", {})
    train_cfg = cluster_cfg.get("training", {})
    
    # Print exports
    print(f"export MODEL_NAME='{model_cfg['model_name']}'")
    print(f"export CONFIG_FILE='{model_cfg['config_file']}'")
    print(f"export NUM_GPUS={cluster_cfg['num_gpus']}")
    print(f"export MAX_SEQLEN={cluster_cfg['max_seqlen']}")
    print(f"export ROLLOUT_GBS={cluster_cfg['rollout_gbs']}")
    print(f"export TRAIN_GBS={cluster_cfg['train_gbs']}")
    print(f"export NUM_PROMPTS={cluster_cfg['num_prompts']}")
    print(f"export NUM_GENERATIONS={cluster_cfg['num_generations']}")
    
    # Generation parallelism
    print(f"export G_TP={gen_cfg.get('tp', 1)}")
    print(f"export G_PP={gen_cfg.get('pp', 1)}")
    print(f"export G_EP={gen_cfg.get('ep', 1)}")
    
    # Training parallelism
    print(f"export T_TP={train_cfg.get('tp', 1)}")
    print(f"export T_CP={train_cfg.get('cp', 1)}")
    print(f"export T_EP={train_cfg.get('ep', 1)}")
    print(f"export T_PP={train_cfg.get('pp', 1)}")
    
    # Derived values
    t_tp = train_cfg.get('tp', 1)
    print(f"export T_SP={'True' if t_tp > 1 else 'False'}")


def list_presets():
    """List available presets."""
    configs = load_configs()
    for preset in configs.keys():
        if preset != "defaults":
            print(preset)


def main():
    if len(sys.argv) < 2:
        print("Usage: python get_model_config.py <preset> <cluster_type> [variant]")
        print("       python get_model_config.py --list")
        sys.exit(1)
    
    if sys.argv[1] == "--list":
        list_presets()
        return
    
    preset = sys.argv[1]
    cluster_type = sys.argv[2] if len(sys.argv) > 2 else "h100"
    variant = sys.argv[3] if len(sys.argv) > 3 else None
    
    export_config(preset, cluster_type, variant)


if __name__ == "__main__":
    main()

