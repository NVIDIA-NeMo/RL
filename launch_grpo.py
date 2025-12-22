#!/usr/bin/env python3
"""
Flexible GRPO job launcher with customizable parallelism settings.
Auto-detects cluster type (H100: 8 GPUs/node, GB200: 4 GPUs/node).
Reads model configurations from model_configs.yaml.

Usage:
    # Launch with preset configurations
    python launch_grpo.py --preset qwen32b
    python launch_grpo.py --preset qwen32b,llama8b,qwen30b  # Launch multiple
    
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
        "container_name": "nemo_rl_v0.4.sqsh",  # Container filename (path set dynamically)
        "wandb_project_suffix": "h100",
        "default_partition": "batch_long",
        "use_gres": True,
    },
    "gb200": {
        "gpus_per_node": 4,
        "container_name": "nemo_rl_nightly.sqsh",  # Container filename (path set dynamically)
        "wandb_project_suffix": "gb200",
        "default_partition": "batch",
        "use_gres": True,  # GB200 also needs GRES
    },
}


def detect_cluster_type(partition: str = "batch") -> str:
    """Detect cluster type from GRES configuration, hostname, or partition names."""
    import re
    import socket
    
    # Method 1: Check hostname for GB200-only clusters (lyris, theia)
    try:
        hostname = socket.gethostname().lower()
        if "lyris" in hostname or "theia" in hostname:
            print(f"[DEBUG] Detected GB200/GB300 cluster from hostname: {hostname}")
            return "gb200"
    except Exception as e:
        print(f"[DEBUG] Could not detect from hostname: {e}")
    
    # Method 2: GRES-based detection (most reliable - check actual GPU count)
    try:
        result = subprocess.run(
            ["sinfo", "-p", partition, "-h", "-o", "%G"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            match = re.search(r'gpu:(\d+)', output)
            if match:
                gpus = int(match.group(1))
                print(f"[DEBUG] Detected {gpus} GPUs per node from GRES: {output}")
                if gpus == 8:
                    return "h100"
                elif gpus == 4:
                    return "gb200"
    except Exception as e:
        print(f"[DEBUG] Could not detect from GRES: {e}")
    
    # Method 3: Check available partitions for gb200/gb300 naming (fallback)
    try:
        result = subprocess.run(
            ["sinfo", "-h", "-o", "%P"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            partitions = result.stdout.strip().lower()
            has_gb200 = "gb200" in partitions or "gb300" in partitions
            has_batch = "batch" in partitions
            
            if has_gb200:
                print(f"[DEBUG] Detected GB200/GB300 cluster from partition names")
                return "gb200"
            elif has_batch:
                print(f"[DEBUG] Detected H100 cluster from partition names (batch only)")
                return "h100"
    except Exception as e:
        print(f"[WARNING] Could not auto-detect cluster type: {e}")
    
    return "h100"


def get_cluster_config(cluster_type: Optional[str] = None, partition: str = "batch") -> dict:
    """Get cluster configuration, auto-detecting if not specified."""
    import socket
    
    if cluster_type is None:
        cluster_type = detect_cluster_type(partition)
    
    config = CLUSTER_CONFIGS.get(cluster_type.lower(), CLUSTER_CONFIGS["h100"]).copy()
    config["cluster_type"] = cluster_type.lower()
    
    # Set container path: current directory + container_name
    container_name = config.get("container_name", "nemo_rl.sqsh")
    config["container"] = str(Path.cwd() / container_name)
    
    # Override settings for GB200 on lyris/theia nodes (no GRES, use gb200 partition, different account)
    if cluster_type.lower() == "gb200":
        try:
            hostname = socket.gethostname().lower()
            if "lyris" in hostname or "theia" in hostname:
                config["default_partition"] = "gb200"
                config["use_gres"] = False  # lyris/theia don't use GRES
                config["default_account"] = "coreai_dlalgo_llm"  # lyris/theia use different account
                print(f"[DEBUG] Using 'gb200' partition (no GRES, account=coreai_dlalgo_llm) for lyris/theia node")
        except Exception:
            pass
    
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
                "num_gpus": 16, "max_seqlen": 4096, "rollout_gbs": 2048, "train_gbs": 512,
                "num_prompts": 64, "num_generations": 32,
                "generation": {"tp": 1, "pp": 1},
                "training": {"tp": 4, "cp": 1, "ep": 1, "pp": 1}
            },
            "gb200_tp2": {
                "num_gpus": 16, "max_seqlen": 4096, "rollout_gbs": 2048, "train_gbs": 512,
                "num_prompts": 64, "num_generations": 32,
                "generation": {"tp": 1, "pp": 1},
                "training": {"tp": 2, "cp": 1, "ep": 1, "pp": 1}
            },
            "gb200_tp1": {
                "num_gpus": 16, "max_seqlen": 4096, "rollout_gbs": 2048, "train_gbs": 512,
                "num_prompts": 64, "num_generations": 32,
                "generation": {"tp": 1, "pp": 1},
                "training": {"tp": 1, "cp": 1, "ep": 1, "pp": 1}
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
            "config_file": "examples/configs/recipes/llm/performance/grpo-deepseek-v3-32n8g.yaml",
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

def calculate_dp(total_gpus: int, tp: int, pp: int, cp: int = 1) -> int:
    """Calculate Data Parallelism degree (Megatron-compatible).
    
    DP = world_size / (TP × PP × CP)
    Note: EP is NOT included as it operates within the DP group.
    """
    model_parallel = tp * pp * cp
    return max(1, total_gpus // model_parallel)


def build_command(model_cfg: Dict[str, Any], 
                  cluster_config: dict,
                  wandb_project: str = "sync-grpo-benchmark",
                  max_steps: int = 20,
                  time_limit: str = "04:00:00",
                  account: Optional[str] = None,
                  partition: Optional[str] = None,
                  enable_vllm_metrics: bool = False,
                  vllm_metrics_interval: float = 0.5) -> str:
    """Build the sbatch command for a given configuration."""
    
    gpus_per_node = cluster_config["gpus_per_node"]
    container = cluster_config["container"]
    cluster_type = cluster_config["cluster_type"]
    use_gres = cluster_config.get("use_gres", True)
    
    # Use cluster-specific default partition if not specified
    if partition is None:
        partition = cluster_config.get("default_partition", "batch_long")
    
    # Use cluster-specific default account if not specified
    if account is None:
        account = cluster_config.get("default_account", "coreai_dlalgo_nemorl")
    
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
    
    # Segment for sbatch (GB200 NVLink topology aware)
    # - num_nodes <= 16: use num_nodes as segment
    # - num_nodes > 16 and divisible by 16: use 16
    # - otherwise: find largest divisor from 18 down to 1
    if num_nodes <= 16:
        segment = num_nodes
    elif num_nodes % 16 == 0:
        segment = 16
    else:
        # Find largest divisor starting from 18
        segment = num_nodes  # fallback
        for seg in range(18, 0, -1):
            if num_nodes % seg == 0:
                segment = seg
                break
    
    # Calculate Data Parallelism (Megatron-compatible: DP = world_size / (TP × PP × CP))
    # Note: EP is NOT included as it operates within the DP group
    t_dp = num_gpus // (t_tp * t_cp * t_pp) if (t_tp * t_cp * t_pp) > 0 else 1
    g_dp = num_gpus // (g_tp * g_pp) if (g_tp * g_pp) > 0 else 1
    
    # WandB settings - format: sync-grpo-{cluster}-benchmark
    if wandb_project == "sync-grpo-benchmark":
        full_wandb_project = f"sync-grpo-{cluster_type}-benchmark"
    else:
        full_wandb_project = wandb_project
    model_short = model_cfg["model_name"].split("/")[-1].replace("-", "_").replace(".", "_")
    wandb_name = f"{model_short}_N{num_nodes}xG{gpus_per_node}_Ttp{t_tp}cp{t_cp}ep{t_ep}pp{t_pp}dp{t_dp}_Gtp{g_tp}pp{g_pp}dp{g_dp}"
    
    # Job name (SLURM squeue display)
    # Format: Model_NxG_T.tp#.cp#.ep#.pp#.dp#_G.tp#.pp#.dp#
    # Extract a clean short model name (e.g., "llama8b", "qwen32b")
    model_name_lower = model_short.lower()
    # Try to extract model size (e.g., "8b", "70b", "32b") and combine with model family
    import re
    size_match = re.search(r'(\d+b)', model_name_lower)
    if "llama" in model_name_lower:
        short_name = f"llama{size_match.group(1)}" if size_match else "llama"
    elif "qwen" in model_name_lower:
        short_name = f"qwen{size_match.group(1)}" if size_match else "qwen"
    elif "deepseek" in model_name_lower:
        short_name = "deepseek"
    else:
        # Fallback: use first 10 chars, strip trailing underscores
        short_name = model_name_lower[:10].rstrip("_")
    
    job_name = f"{short_name}_{num_nodes}x{gpus_per_node}_T.tp{t_tp}.cp{t_cp}.ep{t_ep}.pp{t_pp}.dp{t_dp}_G.tp{g_tp}.pp{g_pp}.dp{g_dp}"
    
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
        command += f""" \\
policy.generation.vllm_cfg.async_engine=true \\
policy.generation.vllm_cfg.enable_vllm_metrics_logger=true \\
policy.generation.vllm_cfg.vllm_metrics_logger_interval={vllm_metrics_interval}"""
    
    # Build GRES option only for clusters that use it (H100 uses GRES, GB200 doesn't)
    gres_line = f"--gres=gpu:{gpus_per_node} \\\n    " if use_gres else ""
    
    # Check required environment variables
    env_warnings = []
    hf_home = os.environ.get("HF_HOME", "")
    hf_cache = os.environ.get("HF_DATASETS_CACHE", "")
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    
    if not hf_home:
        env_warnings.append("⚠️  HF_HOME is not set. Add 'export HF_HOME=/path/to/hf_home' to ~/.bashrc")
    if not hf_cache:
        env_warnings.append("⚠️  HF_DATASETS_CACHE is not set. Add 'export HF_DATASETS_CACHE=/path/to/cache' to ~/.bashrc")
    if not wandb_key:
        env_warnings.append("⚠️  WANDB_API_KEY is not set. Add 'export WANDB_API_KEY=your_key' to ~/.bashrc")
    
    if env_warnings:
        print("\n" + "\n".join(env_warnings) + "\n")
    
    # Build log directory structure: exp_logs/{model}/{parallelism_config}/
    rollout_gbs = model_cfg['num_prompts'] * model_cfg['num_generations']
    log_subdir = f"T.tp{t_tp}.cp{t_cp}.ep{t_ep}.pp{t_pp}_G.tp{g_tp}.pp{g_pp}_R{rollout_gbs}_T{model_cfg['train_gbs']}_S{model_cfg['max_seqlen']}"
    base_log_dir = str(Path.cwd() / "exp_logs" / short_name / log_subdir)
    
    sbatch_cmd = f"""COMMAND="{command}" \\
CONTAINER={container} \\
GPUS_PER_NODE={gpus_per_node} \\
BASE_LOG_DIR={base_log_dir} \\
HF_HOME=$HF_HOME \\
HF_DATASETS_CACHE=$HF_DATASETS_CACHE \\
WANDB_API_KEY=$WANDB_API_KEY \\
MOUNTS="/lustre:/lustre" \\
sbatch \\
    --nodes={num_nodes} \\
    --account={account} \\
    --job-name={job_name} \\
    --partition={partition} \\
    --time={time_limit} \\
    --output={base_log_dir}/%j-logs/slurm-%j.out \\
    {gres_line}--segment {segment} \\
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
    
    t_dp = calculate_dp(num_gpus, t_tp, t_pp, t_cp)  # Megatron-compatible: DP = world_size / (TP × PP × CP)
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
    
    # Check if requested node configuration is available (e.g. partition mismatch)
    # This prevents obscure sbatch errors like "Requested node configuration is not available"
    if cluster_type == "gb200" and "batch_long" in cmd:
         # GB200 usually requires specific partition or constraint if batch_long is H100 only
         # But here we just warn, assuming user knows their cluster
         pass

    print("🚀 Launching job...")
    print("\n📝 Command to be executed:")
    print("-" * 60)
    print(cmd)
    print("-" * 60)
    print()
    
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
    parser.add_argument("--account", default=None, help="SLURM account (default: auto based on cluster)")
    parser.add_argument("--job-partition", default=None, help="SLURM job partition (default: auto based on cluster)")
    
    # Execution options
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Show command without executing")
    parser.add_argument("--enable-vllm-metrics", action="store_true",
                        help="Enable vLLM metrics logging (requires async_engine=true)")
    parser.add_argument("--vllm-metrics-interval", type=float, default=0.5,
                        help="vLLM metrics logging interval in seconds (default: 0.5)")
    
    args = parser.parse_args()
    
    # Infer cluster type from variant name if not explicitly specified
    cluster_type = args.cluster
    if cluster_type is None and args.variant:
        # Parse variant name to infer cluster type (e.g., "gb200_tp1" -> "gb200")
        variant_lower = args.variant.lower()
        if variant_lower.startswith("gb200"):
            cluster_type = "gb200"
            print(f"[INFO] Inferred cluster type 'gb200' from variant '{args.variant}'")
        elif variant_lower.startswith("h100"):
            cluster_type = "h100"
            print(f"[INFO] Inferred cluster type 'h100' from variant '{args.variant}'")
    
    # Get cluster configuration
    cluster_config = get_cluster_config(cluster_type, args.partition)
    print(f"[INFO] Cluster: {cluster_config['cluster_type'].upper()} ({cluster_config['gpus_per_node']} GPUs/node)")
    print(f"[INFO] Container: {cluster_config['container']}")
    
    # List presets
    if args.list_presets:
        list_presets(cluster_config)
        return
    
    # Launch all presets
    if args.all:
        variant_msg = f" with variant '{args.variant}'" if args.variant else ""
        print(f"🚀 Launching all preset configurations{variant_msg}...\n")
        
        launched = 0
        skipped = 0
        for preset in get_available_presets():
            # If variant is specified, check if this preset has that variant
            if args.variant:
                available_variants = get_preset_variants(preset)
                if args.variant not in available_variants:
                    print(f"⏭️  Skipping {preset}: variant '{args.variant}' not available (has: {available_variants})")
                    skipped += 1
                    continue
            
            print(f"\n{'='*70}")
            print(f"  Launching: {preset}" + (f" ({args.variant})" if args.variant else ""))
            print(f"{'='*70}")
            try:
                launch_job(preset, cluster_config, variant=args.variant,
                          dry_run=args.dry_run,
                          wandb_project=args.wandb_project,
                          max_steps=args.max_steps,
                          time_limit=args.time,
                          account=args.account,
                          partition=args.job_partition,
                          enable_vllm_metrics=args.enable_vllm_metrics,
                          vllm_metrics_interval=args.vllm_metrics_interval)
                launched += 1
            except Exception as e:
                print(f"❌ Error launching {preset}: {e}")
        
        print(f"\n✅ Launched: {launched}, Skipped: {skipped}")
        return
    
    # Use preset
    if args.preset:
        presets = [p.strip() for p in args.preset.split(',')]
        
        if len(presets) > 1:
            print(f"🚀 Launching {len(presets)} selected presets: {', '.join(presets)}\\n")
        
        for p in presets:
            if len(presets) > 1:
                print(f"\\n{'='*70}")
                print(f"  Launching: {p}")
                print(f"{'='*70}")

            try:
                launch_job(p, cluster_config, variant=args.variant,
                        dry_run=args.dry_run,
                        wandb_project=args.wandb_project,
                        max_steps=args.max_steps,
                        time_limit=args.time,
                        account=args.account,
                        partition=args.job_partition,
                        enable_vllm_metrics=args.enable_vllm_metrics,
                        vllm_metrics_interval=args.vllm_metrics_interval)
            except Exception as e:
                print(f"❌ Error launching {p}: {e}")
        return
    
    # No valid options
    parser.print_help()
    print("\n💡 Try: python launch_grpo.py --list-presets")


if __name__ == "__main__":
    main()
