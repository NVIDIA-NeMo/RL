#!/usr/bin/env python3
"""
Flexible GRPO job launcher with customizable parallelism settings.
Auto-detects cluster type (H100: 8 GPUs/node, GB200: 4 GPUs/node).
Reads model configurations from model_configs.yaml.

Usage:
    # Launch with preset configurations
    python launch_grpo.py --preset qwen32b
    python launch_grpo.py --preset llama8b
    python launch_grpo.py --preset llama70b
    
    # Force specific cluster type
    python launch_grpo.py --preset qwen32b --cluster h100
    python launch_grpo.py --preset qwen32b --cluster gb200
    
    # Use specific variant (e.g., high sequence length)
    python launch_grpo.py --preset llama70b --variant h100_highseq
    
    # List available presets
    python launch_grpo.py --list-presets
    
    # Dry run (show command without executing)
    python launch_grpo.py --preset qwen32b --dry-run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Try to import yaml, fall back to basic parsing if not available
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("[WARNING] PyYAML not installed. Using fallback configuration.")

# ============================================
# Cluster Configuration
# ============================================

CLUSTER_CONFIGS = {
    "h100": {
        "gpus_per_node": 8,
        "container": "/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/RL/nemo_rl_v0.4.sqsh",
        "wandb_project_suffix": "h100",
    },
    "gb200": {
        "gpus_per_node": 4,
        "container": "/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/nemo_rl.sqsh",
        "wandb_project_suffix": "gb200",
    },
}


def detect_cluster_type(partition: str = "batch") -> str:
    """Detect cluster type from SLURM GRES configuration."""
    try:
        result = subprocess.run(
            ["sinfo", "-p", partition, "-h", "-o", "%G"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            import re
            match = re.search(r'gpu(?::[^:]+)?:(\d+)', output)
            if match:
                gpus = int(match.group(1))
                if gpus == 8:
                    return "h100"
                elif gpus == 4:
                    return "gb200"
    except Exception as e:
        print(f"[WARNING] Could not auto-detect cluster type: {e}")
    
    return "h100"


def get_cluster_config(cluster_type: Optional[str] = None, partition: str = "batch") -> dict:
    """Get cluster configuration, auto-detecting if not specified."""
    if cluster_type is None:
        cluster_type = detect_cluster_type(partition)
    
    config = CLUSTER_CONFIGS.get(cluster_type.lower(), CLUSTER_CONFIGS["h100"]).copy()
    config["cluster_type"] = cluster_type.lower()
    return config


# ============================================
# Model Configuration Loading
# ============================================

def get_config_path() -> Path:
    """Get path to model_configs.yaml."""
    script_dir = Path(__file__).parent
    return script_dir / "model_configs.yaml"


def load_model_configs() -> Dict[str, Any]:
    """Load model configurations from YAML file."""
    config_path = get_config_path()
    
    if HAS_YAML and config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Fallback hardcoded configs
        return get_fallback_configs()


def get_fallback_configs() -> Dict[str, Any]:
    """Fallback configurations if YAML is not available."""
    return {
        "llama8b": {
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "config_file": "examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-2n8g.yaml",
            "h100": {
                "num_gpus": 16, "max_seqlen": 4096, "rollout_gbs": 2048, "train_gbs": 512,
                "num_prompts": 64, "num_generations": 32,
                "generation": {"tp": 1, "pp": 1},
                "training": {"tp": 1, "cp": 1, "ep": 1, "pp": 2}
            },
            "gb200": {
                "num_gpus": 8, "max_seqlen": 4096, "rollout_gbs": 2048, "train_gbs": 512,
                "num_prompts": 64, "num_generations": 32,
                "generation": {"tp": 1, "pp": 1},
                "training": {"tp": 1, "cp": 1, "ep": 1, "pp": 1}
            }
        },
        "llama70b": {
            "model_name": "meta-llama/Llama-3.1-70B-Instruct",
            "config_file": "examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-2n8g.yaml",
            "h100": {
                "num_gpus": 32, "max_seqlen": 4096, "rollout_gbs": 2048, "train_gbs": 512,
                "num_prompts": 64, "num_generations": 32,
                "generation": {"tp": 4, "pp": 1},
                "training": {"tp": 4, "cp": 1, "ep": 1, "pp": 8}
            },
            "gb200": {
                "num_gpus": 16, "max_seqlen": 4096, "rollout_gbs": 2048, "train_gbs": 512,
                "num_prompts": 64, "num_generations": 32,
                "generation": {"tp": 2, "pp": 1},
                "training": {"tp": 4, "cp": 1, "ep": 1, "pp": 2}
            }
        },
        "qwen32b": {
            "model_name": "Qwen/Qwen3-32B",
            "config_file": "examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n8g.yaml",
            "h100": {
                "num_gpus": 32, "max_seqlen": 4096, "rollout_gbs": 2048, "train_gbs": 512,
                "num_prompts": 64, "num_generations": 32,
                "generation": {"tp": 4, "pp": 1},
                "training": {"tp": 4, "cp": 1, "ep": 1, "pp": 4}
            },
            "gb200": {
                "num_gpus": 32, "max_seqlen": 4096, "rollout_gbs": 2048, "train_gbs": 512,
                "num_prompts": 64, "num_generations": 32,
                "generation": {"tp": 4, "pp": 1},
                "training": {"tp": 4, "cp": 1, "ep": 1, "pp": 4}
            }
        },
        "qwen30b": {
            "model_name": "Qwen/Qwen3-30B-A3B",
            "config_file": "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g.yaml",
            "h100": {
                "num_gpus": 32, "max_seqlen": 4096, "rollout_gbs": 2048, "train_gbs": 512,
                "num_prompts": 64, "num_generations": 32,
                "generation": {"tp": 2, "pp": 1},
                "training": {"tp": 1, "cp": 1, "ep": 8, "pp": 1}
            },
            "gb200": {
                "num_gpus": 16, "max_seqlen": 4096, "rollout_gbs": 2048, "train_gbs": 512,
                "num_prompts": 64, "num_generations": 32,
                "generation": {"tp": 2, "pp": 1},
                "training": {"tp": 1, "cp": 1, "ep": 8, "pp": 1}
            }
        },
        "qwen235b": {
            "model_name": "Qwen/Qwen3-235B-A22B",
            "config_file": "examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n8g.yaml",
            "h100": {
                "num_gpus": 128, "max_seqlen": 8192, "rollout_gbs": 512, "train_gbs": 512,
                "num_prompts": 16, "num_generations": 32,
                "generation": {"tp": 16, "pp": 1},
                "training": {"tp": 2, "cp": 2, "ep": 16, "pp": 8}
            },
            "gb200": {
                "num_gpus": 128, "max_seqlen": 8192, "rollout_gbs": 512, "train_gbs": 512,
                "num_prompts": 16, "num_generations": 32,
                "generation": {"tp": 16, "pp": 1},
                "training": {"tp": 2, "cp": 2, "ep": 16, "pp": 8}
            }
        },
        "deepseek_v3": {
            "model_name": "deepseek-ai/DeepSeek-V3",
            "config_file": "examples/configs/recipes/llm/performance/grpo-deepseek-v3.yaml",
            "h100": {
                "num_gpus": 256, "max_seqlen": 1536, "rollout_gbs": 2048, "train_gbs": 2048,
                "num_prompts": 64, "num_generations": 32,
                "generation": {"tp": 32, "pp": 1},
                "training": {"tp": 1, "cp": 1, "ep": 16, "pp": 16}
            },
            "gb200": {
                "num_gpus": 256, "max_seqlen": 1536, "rollout_gbs": 2048, "train_gbs": 2048,
                "num_prompts": 64, "num_generations": 32,
                "generation": {"tp": 32, "pp": 1},
                "training": {"tp": 1, "cp": 1, "ep": 16, "pp": 16}
            }
        }
    }


def get_model_config(preset: str, cluster_type: str, variant: Optional[str] = None) -> Dict[str, Any]:
    """Get configuration for a specific model and cluster type."""
    configs = load_model_configs()
    
    if preset not in configs:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(configs.keys())}")
    
    model_cfg = configs[preset]
    
    # Determine which variant to use
    if variant:
        variant_key = variant
    else:
        variant_key = cluster_type
    
    if variant_key not in model_cfg:
        # Try cluster_type as fallback
        if cluster_type not in model_cfg:
            raise ValueError(f"No config for {preset} on {cluster_type}. Available: {[k for k in model_cfg.keys() if k not in ['model_name', 'config_file']]}")
        variant_key = cluster_type
    
    cluster_cfg = model_cfg[variant_key]
    
    return {
        "model_name": model_cfg["model_name"],
        "config_file": model_cfg["config_file"],
        **cluster_cfg
    }


def get_available_presets() -> list:
    """Get list of available presets."""
    configs = load_model_configs()
    return [k for k in configs.keys() if k != "defaults"]


def get_preset_variants(preset: str) -> list:
    """Get available variants for a preset."""
    configs = load_model_configs()
    if preset not in configs:
        return []
    model_cfg = configs[preset]
    return [k for k in model_cfg.keys() if k not in ["model_name", "config_file"]]


# ============================================
# Job Building and Launching
# ============================================

def calculate_dp(total_gpus: int, tp: int, pp: int, ep: int = 1) -> int:
    """Calculate Data Parallelism degree."""
    model_parallel = tp * pp * ep
    return max(1, total_gpus // model_parallel)


def build_command(model_cfg: Dict[str, Any], 
                  cluster_config: dict,
                  wandb_project: str = "sync-grpo-benchmark",
                  max_steps: int = 20,
                  time_limit: str = "04:00:00",
                  account: str = "coreai_dlalgo_nemorl",
                  enable_vllm_metrics: bool = False) -> str:
    """Build the sbatch command for a given configuration."""
    
    gpus_per_node = cluster_config["gpus_per_node"]
    container = cluster_config["container"]
    cluster_type = cluster_config["cluster_type"]
    
    num_gpus = model_cfg["num_gpus"]
    num_nodes = num_gpus // gpus_per_node
    
    # Get parallelism settings
    gen_cfg = model_cfg["generation"]
    train_cfg = model_cfg["training"]
    
    g_tp = gen_cfg.get("tp", 1)
    g_pp = gen_cfg.get("pp", 1)
    
    t_tp = train_cfg.get("tp", 1)
    t_cp = train_cfg.get("cp", 1)
    t_ep = train_cfg.get("ep", 1)
    t_pp = train_cfg.get("pp", 1)
    
    # Sequence parallel (enabled if TP > 1)
    t_sp = "True" if t_tp > 1 else "False"
    
    # Segment for sbatch
    segment = min(16, num_nodes) if num_nodes >= 16 else num_nodes
    
    # WandB settings - format: sync-grpo-{cluster}-benchmark
    if wandb_project == "sync-grpo-benchmark":
        full_wandb_project = f"sync-grpo-{cluster_type}-benchmark"
    else:
        full_wandb_project = wandb_project
    model_short = model_cfg["model_name"].split("/")[-1].replace("-", "_").replace(".", "_")
    wandb_name = f"{model_short}_N{num_nodes}xG{gpus_per_node}_Ttp{t_tp}pp{t_pp}ep{t_ep}cp{t_cp}_Gtp{g_tp}pp{g_pp}"
    
    # Job name
    job_name = f"{model_short.lower()[:20]}-N{num_nodes}xG{gpus_per_node}-T.tp{t_tp}.pp{t_pp}-G.tp{g_tp}.pp{g_pp}"
    
    # Build command
    command = f"""NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo_math.py \\
--config {model_cfg['config_file']} \\
cluster.num_nodes={num_nodes} \\
cluster.gpus_per_node={gpus_per_node} \\
policy.model_name={model_cfg['model_name']} \\
policy.max_total_sequence_length={model_cfg['max_seqlen']} \\
policy.generation.vllm_cfg.tensor_parallel_size={g_tp} \\
policy.generation.vllm_cfg.pipeline_parallel_size={g_pp} \\
policy.megatron_cfg.tensor_model_parallel_size={t_tp} \\
policy.megatron_cfg.expert_model_parallel_size={t_ep} \\
policy.megatron_cfg.pipeline_model_parallel_size={t_pp} \\
policy.megatron_cfg.context_parallel_size={t_cp} \\
policy.megatron_cfg.sequence_parallel={t_sp} \\
grpo.async_grpo.enabled=false \\
grpo.val_period=1000 \\
checkpointing.enabled=false \\
grpo.num_prompts_per_step={model_cfg['num_prompts']} \\
grpo.num_generations_per_prompt={model_cfg['num_generations']} \\
policy.sequence_packing.enabled=True \\
policy.train_global_batch_size={model_cfg['train_gbs']} \\
grpo.max_num_steps={max_steps} \\
logger.wandb_enabled=True \\
logger.wandb.project='{full_wandb_project}' \\
logger.wandb.name='{wandb_name}'"""

    # Add vLLM metrics logging if enabled
    if enable_vllm_metrics:
        command += """ \\
policy.generation.vllm_cfg.async_engine=true \\
policy.generation.vllm_cfg.enable_vllm_metrics_logger=true \\
policy.generation.vllm_cfg.vllm_metrics_logger_interval=0.5"""
    
    sbatch_cmd = f"""COMMAND="{command}" \\
CONTAINER={container} \\
HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home \\
HF_DATASETS_CACHE=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/cache \\
WANDB_API_KEY=$WANDB_API_KEY \\
MOUNTS="/lustre:/lustre" \\
sbatch \\
    --nodes={num_nodes} \\
    --account={account} \\
    --job-name={job_name} \\
    --partition=batch_long \\
    --time={time_limit} \\
    --gres=gpu:{gpus_per_node} \\
    --segment {segment} \\
    ray.sub"""
    
    return sbatch_cmd


def print_config_summary(model_cfg: Dict[str, Any], cluster_config: dict, preset: str):
    """Print configuration summary."""
    gpus_per_node = cluster_config["gpus_per_node"]
    cluster_type = cluster_config["cluster_type"].upper()
    
    num_gpus = model_cfg["num_gpus"]
    num_nodes = num_gpus // gpus_per_node
    
    gen_cfg = model_cfg["generation"]
    train_cfg = model_cfg["training"]
    
    g_tp, g_pp = gen_cfg.get("tp", 1), gen_cfg.get("pp", 1)
    t_tp, t_cp = train_cfg.get("tp", 1), train_cfg.get("cp", 1)
    t_ep, t_pp = train_cfg.get("ep", 1), train_cfg.get("pp", 1)
    
    t_dp = calculate_dp(num_gpus, t_tp, t_pp, t_ep)
    g_dp = calculate_dp(num_gpus, g_tp, g_pp)
    rollout_gbs = model_cfg['num_prompts'] * model_cfg['num_generations']
    
    model_short = model_cfg["model_name"].split("/")[-1]
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  {preset.upper()}: {model_short:<47}  ║
╠══════════════════════════════════════════════════════════════════╣
║  Cluster: {cluster_type:<8} | GPUs/Node: {gpus_per_node:<2}                              ║
║  Model: {model_cfg['model_name']:<54} ║
║  Nodes: {num_nodes:<3}  |  Total GPUs: {num_gpus:<4}                                 ║
╠══════════════════════════════════════════════════════════════════╣
║  Batch Settings:                                                 ║
║    Rollout GBS: {rollout_gbs:<5} ({model_cfg['num_prompts']} prompts × {model_cfg['num_generations']} gens)               ║
║    Training GBS: {model_cfg['train_gbs']:<5}                                           ║
║    Max SeqLen: {model_cfg['max_seqlen']:<6}                                            ║
╠══════════════════════════════════════════════════════════════════╣
║  Training Parallelism:                                           ║
║    TP={t_tp:<2} | CP={t_cp:<2} | EP={t_ep:<2} | PP={t_pp:<2} | DP={t_dp:<2}                       ║
║  Generation Parallelism:                                         ║
║    TP={g_tp:<2} | PP={g_pp:<2} | DP={g_dp:<2}                                         ║
╚══════════════════════════════════════════════════════════════════╝
""")


def list_presets(cluster_config: dict):
    """List all available presets."""
    configs = load_model_configs()
    gpus_per_node = cluster_config["gpus_per_node"]
    cluster_type = cluster_config["cluster_type"]
    
    print(f"\n📋 Available Presets (Cluster: {cluster_type.upper()}, {gpus_per_node} GPUs/node):\n")
    print(f"{'Preset':<15} {'Model':<35} {'GPUs':>5} {'Nodes':>5} {'R-GBS':>6} {'T-GBS':>6} {'SeqLen':>7} {'Train (TP,CP,EP,PP)':>20} {'Gen (TP,PP)':>12}")
    print("-" * 140)
    
    for preset in get_available_presets():
        try:
            model_cfg = get_model_config(preset, cluster_type)
            gen_cfg = model_cfg["generation"]
            train_cfg = model_cfg["training"]
            
            num_gpus = model_cfg["num_gpus"]
            num_nodes = num_gpus // gpus_per_node
            rollout_gbs = model_cfg['num_prompts'] * model_cfg['num_generations']
            
            train_str = f"{train_cfg.get('tp',1)},{train_cfg.get('cp',1)},{train_cfg.get('ep',1)},{train_cfg.get('pp',1)}"
            gen_str = f"{gen_cfg.get('tp',1)},{gen_cfg.get('pp',1)}"
            
            model_short = model_cfg['model_name'].split('/')[-1][:32]
            print(f"{preset:<15} {model_short:<35} {num_gpus:>5} {num_nodes:>5} {rollout_gbs:>6} {model_cfg['train_gbs']:>6} {model_cfg['max_seqlen']:>7} {train_str:>20} {gen_str:>12}")
            
            # Show variants
            variants = get_preset_variants(preset)
            other_variants = [v for v in variants if v != cluster_type]
            if other_variants:
                print(f"  └── variants: {', '.join(other_variants)}")
        except Exception as e:
            print(f"{preset:<15} [Error loading config: {e}]")
    
    print(f"\n💡 Usage: python launch_grpo.py --preset <preset_name>")
    print(f"   Variant: python launch_grpo.py --preset llama70b --variant h100_highseq")
    print(f"   Force cluster: python launch_grpo.py --preset qwen32b --cluster h100\n")


def launch_job(preset: str, cluster_config: dict, variant: Optional[str] = None, 
               dry_run: bool = False, **kwargs):
    """Launch a job with the given configuration."""
    cluster_type = cluster_config["cluster_type"]
    model_cfg = get_model_config(preset, cluster_type, variant)
    
    print_config_summary(model_cfg, cluster_config, preset)
    
    cmd = build_command(model_cfg, cluster_config, **kwargs)
    
    if dry_run:
        print("🔍 Dry run - Command that would be executed:\n")
        print(cmd)
        print()
        return
    
    print("🚀 Launching job...")
    
    script_path = f"/tmp/launch_{preset}.sh"
    with open(script_path, 'w') as f:
        f.write(f"#!/bin/bash\n{cmd}\n")
    
    os.chmod(script_path, 0o755)
    
    try:
        result = subprocess.run(['bash', script_path], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Job submitted successfully!")
            print(f"   {result.stdout.strip()}")
        else:
            print(f"❌ Job submission failed:")
            print(f"   {result.stderr.strip()}")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)


def main():
    parser = argparse.ArgumentParser(
        description="GRPO job launcher (reads from model_configs.yaml)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use a preset (auto-detect cluster)
  python launch_grpo.py --preset qwen32b
  
  # Force specific cluster type
  python launch_grpo.py --preset qwen32b --cluster h100
  
  # Use a specific variant
  python launch_grpo.py --preset llama70b --variant h100_highseq
  
  # Dry run
  python launch_grpo.py --preset llama70b --dry-run
  
  # Launch all presets
  python launch_grpo.py --all
"""
    )
    
    # Cluster selection
    parser.add_argument("--cluster", "-c", choices=["h100", "gb200"],
                        help="Force cluster type (default: auto-detect)")
    parser.add_argument("--partition", default="batch",
                        help="SLURM partition for auto-detection")
    
    # Preset selection
    parser.add_argument("--preset", "-p", help="Preset name from model_configs.yaml")
    parser.add_argument("--variant", "-v", help="Specific variant (e.g., h100_highseq)")
    parser.add_argument("--list-presets", "-l", action="store_true",
                        help="List available presets")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Launch all preset configurations")
    
    # Job settings
    parser.add_argument("--wandb-project", default="sync-grpo-benchmark",
                        help="WandB project name (cluster type appended)")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum training steps")
    parser.add_argument("--time", default="04:00:00", help="Job time limit")
    parser.add_argument("--account", default="coreai_dlalgo_nemorl", help="SLURM account")
    
    # Execution options
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Show command without executing")
    parser.add_argument("--enable-vllm-metrics", action="store_true",
                        help="Enable vLLM metrics logging (requires async_engine=true)")
    
    args = parser.parse_args()
    
    # Get cluster configuration
    cluster_config = get_cluster_config(args.cluster, args.partition)
    print(f"[INFO] Cluster: {cluster_config['cluster_type'].upper()} ({cluster_config['gpus_per_node']} GPUs/node)")
    print(f"[INFO] Container: {cluster_config['container']}")
    
    # List presets
    if args.list_presets:
        list_presets(cluster_config)
        return
    
    # Launch all presets
    if args.all:
        print("🚀 Launching all preset configurations...\n")
        for preset in get_available_presets():
            print(f"\n{'='*70}")
            print(f"  Launching: {preset}")
            print(f"{'='*70}")
            try:
                launch_job(preset, cluster_config, dry_run=args.dry_run,
                          wandb_project=args.wandb_project,
                          max_steps=args.max_steps,
                          time_limit=args.time,
                          account=args.account)
            except Exception as e:
                print(f"❌ Error launching {preset}: {e}")
        return
    
    # Use preset
    if args.preset:
        launch_job(args.preset, cluster_config, variant=args.variant,
                  dry_run=args.dry_run,
                  wandb_project=args.wandb_project,
                  max_steps=args.max_steps,
                  time_limit=args.time,
                  account=args.account,
                  enable_vllm_metrics=args.enable_vllm_metrics)
        return
    
    # No valid options
    parser.print_help()
    print("\n💡 Try: python launch_grpo.py --list-presets")


if __name__ == "__main__":
    main()
