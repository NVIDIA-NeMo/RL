#!/usr/bin/env python3
"""
Flexible GRPO job launcher with customizable parallelism settings.

Usage:
    # Launch with preset configurations
    python launch_grpo.py --preset qwen32b
    python launch_grpo.py --preset qwen30b
    python launch_grpo.py --preset llama8b
    python launch_grpo.py --preset llama70b
    
    # Launch with custom settings
    python launch_grpo.py --model Qwen/Qwen3-32B --gpus 16 \
        --train-tp 4 --train-pp 1 --train-ep 1 --train-cp 1 \
        --gen-tp 1 --gen-pp 1
    
    # List available presets
    python launch_grpo.py --list-presets
    
    # Dry run (show command without executing)
    python launch_grpo.py --preset qwen32b --dry-run
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

# ============================================
# Preset Configurations (from user's table)
# ============================================

@dataclass
class ModelConfig:
    """Configuration for a model experiment."""
    name: str                    # Display name
    model_name: str              # HuggingFace model name
    total_gpus: int              # Target total GPUs
    gpus_per_node: int           # GPUs per node (4 for B200/GB200)
    # Training parallelism
    t_tp: int                    # Tensor Parallel
    t_cp: int                    # Context Parallel
    t_ep: int                    # Expert Parallel
    t_pp: int                    # Pipeline Parallel
    # Generation parallelism
    g_tp: int                    # Tensor Parallel
    g_pp: int                    # Pipeline Parallel
    # Config file (base config to use)
    config_file: str
    # Optional overrides
    max_seqlen: int = 4096
    train_gbs: int = 512
    num_prompts: int = 64
    num_generations: int = 32


# B200/GB200: 4 GPUs per node, 192GB memory per GPU
# Settings are derived from H100 benchmarks (half the GPUs)
PRESETS = {
    # ============================================
    # Qwen3-32B: 16 GPUs (4 nodes)
    # H100 32 GPUs → GB200 16 GPUs
    # Training: TP4, CP1, EP1, PP1, DP4 (MP=4, DP=4)
    # Generation: TP1, PP1, DP16 (B200 192GB fits 32B model)
    # Note: B200 192GB allows PP=1 (no pipeline bubble)
    # ============================================
    "qwen32b": ModelConfig(
        name="Qwen3-32B",
        model_name="Qwen/Qwen3-32B",
        total_gpus=16,
        gpus_per_node=4,
        t_tp=4, t_cp=1, t_ep=1, t_pp=1,
        g_tp=1, g_pp=1,
        config_file="examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n8g.yaml",
    ),
    
    # ============================================
    # Qwen3-30B-A3B (MoE): 16 GPUs (4 nodes)
    # H100 32 GPUs → GB200 16 GPUs
    # Training: TP1, CP1, EP8, PP1, DP2 (MP=8, DP=2, EP=8 for MoE)
    # Generation: TP1, PP1, DP16 (MoE 3B active fits easily)
    # ============================================
    "qwen30b": ModelConfig(
        name="Qwen3-30B-A3B",
        model_name="Qwen/Qwen3-30B-A3B",
        total_gpus=16,
        gpus_per_node=4,
        t_tp=1, t_cp=1, t_ep=8, t_pp=1,
        g_tp=1, g_pp=1,
        config_file="examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g.yaml",
    ),
    
    # ============================================
    # Llama-3.1-8B-Instruct: 8 GPUs (2 nodes)
    # H100 16 GPUs → GB200 8 GPUs
    # Training: TP1, CP1, EP1, PP1, DP8 (MP=1, DP=8)
    # Generation: TP1, PP1, DP8
    # ============================================
    "llama8b": ModelConfig(
        name="Llama-3.1-8B",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        total_gpus=8,
        gpus_per_node=4,
        t_tp=1, t_cp=1, t_ep=1, t_pp=1,
        g_tp=1, g_pp=1,
        config_file="examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-2n8g.yaml",
    ),
    
    # ============================================
    # Llama-3.1-70B-Instruct: 16 GPUs (4 nodes)
    # H100 32 GPUs → GB200 16 GPUs
    # Training: TP4, CP1, EP1, PP2, DP2 (MP=8, DP=2)
    # Generation: TP2, PP1, DP8 (70B needs TP2 for KV cache)
    # Note: B200 192GB allows PP=2 (reduced pipeline bubble)
    # ============================================
    "llama70b": ModelConfig(
        name="Llama-3.1-70B",
        model_name="meta-llama/Llama-3.1-70B-Instruct",
        total_gpus=16,
        gpus_per_node=4,
        t_tp=4, t_cp=1, t_ep=1, t_pp=2,
        g_tp=2, g_pp=1,
        config_file="examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-2n8g.yaml",
        max_seqlen=4096,
    ),
    
    # ============================================
    # Llama-3.1-70B Low Rollout GBS: 16 GPUs (4 nodes)
    # H100 32 GPUs → GB200 16 GPUs
    # Rollout GBS: 512 (16 prompts × 32 gens)
    # Training: TP4, CP1, EP1, PP2, DP2 (reduced PP for higher DP)
    # Generation: TP2, PP1, DP8 (B200 192GB allows lower TP)
    # ============================================
    "llama70b-lowgbs": ModelConfig(
        name="Llama-3.1-70B-LowGBS",
        model_name="meta-llama/Llama-3.1-70B-Instruct",
        total_gpus=16,
        gpus_per_node=4,
        t_tp=4, t_cp=1, t_ep=1, t_pp=2,
        g_tp=2, g_pp=1,
        config_file="examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-2n8g.yaml",
        max_seqlen=4096,
        train_gbs=512,
        num_prompts=16,
        num_generations=32,
    ),
    
    # ============================================
    # Llama-3.1-70B High Sequence Length: 16 GPUs (4 nodes)
    # Same as above but with 16384 sequence length
    # Training: TP4, CP1, EP1, PP2, DP2 (reduced PP for higher DP)
    # Generation: TP2, PP1, DP8
    # ============================================
    "llama70b-highseq": ModelConfig(
        name="Llama-3.1-70B-HighSeq",
        model_name="meta-llama/Llama-3.1-70B-Instruct",
        total_gpus=16,
        gpus_per_node=4,
        t_tp=4, t_cp=1, t_ep=1, t_pp=2,
        g_tp=2, g_pp=1,
        config_file="examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-2n8g.yaml",
        max_seqlen=16384,
    ),
}


def calculate_dp(total_gpus: int, tp: int, pp: int, ep: int = 1) -> int:
    """Calculate Data Parallelism degree."""
    model_parallel = tp * pp * ep
    return total_gpus // model_parallel


def build_command(config: ModelConfig, 
                  wandb_project: str = "sync-grpo-gb200-benchmark",
                  max_steps: int = 20,
                  time_limit: str = "04:00:00",
                  account: str = "coreai_dlalgo_nemorl") -> str:
    """Build the sbatch command for a given configuration."""
    
    num_nodes = config.total_gpus // config.gpus_per_node
    
    # Calculate DP
    t_dp = calculate_dp(config.total_gpus, config.t_tp, config.t_pp, config.t_ep)
    g_dp = calculate_dp(config.total_gpus, config.g_tp, config.g_pp)
    
    # Sequence parallel (enabled if TP > 1)
    t_sp = "True" if config.t_tp > 1 else "False"
    
    # Segment for sbatch (16 for large jobs, num_nodes for small)
    segment = 16 if num_nodes >= 16 else num_nodes
    
    # Build WandB name (no special characters for Hydra)
    wandb_name = f"{config.name.replace('-', '_').replace('.', '_')}_N{num_nodes}xG{config.gpus_per_node}_Ttp{config.t_tp}pp{config.t_pp}ep{config.t_ep}cp{config.t_cp}_Gtp{config.g_tp}pp{config.g_pp}"
    
    # Build job name
    job_name = f"{config.name.lower().replace('.', '')}-N{num_nodes}xG{config.gpus_per_node}-T.tp{config.t_tp}.pp{config.t_pp}-G.tp{config.g_tp}.pp{config.g_pp}"
    
    # Build the COMMAND string
    command = f"""NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo_math.py \\
--config {config.config_file} \\
cluster.num_nodes={num_nodes} \\
cluster.gpus_per_node={config.gpus_per_node} \\
policy.model_name={config.model_name} \\
policy.max_total_sequence_length={config.max_seqlen} \\
policy.generation.vllm_cfg.tensor_parallel_size={config.g_tp} \\
policy.generation.vllm_cfg.pipeline_parallel_size={config.g_pp} \\
policy.megatron_cfg.tensor_model_parallel_size={config.t_tp} \\
policy.megatron_cfg.expert_model_parallel_size={config.t_ep} \\
policy.megatron_cfg.pipeline_model_parallel_size={config.t_pp} \\
policy.megatron_cfg.context_parallel_size={config.t_cp} \\
policy.megatron_cfg.sequence_parallel={t_sp} \\
grpo.async_grpo.enabled=false \\
grpo.val_period=1000 \\
checkpointing.enabled=false \\
grpo.num_prompts_per_step={config.num_prompts} \\
grpo.num_generations_per_prompt={config.num_generations} \\
policy.sequence_packing.enabled=True \\
policy.train_global_batch_size={config.train_gbs} \\
grpo.max_num_steps={max_steps} \\
logger.wandb_enabled=True \\
logger.wandb.project='{wandb_project}' \\
logger.wandb.name='{wandb_name}'"""
    
    # Full sbatch command
    sbatch_cmd = f"""COMMAND="{command}" \\
CONTAINER=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/nemo_rl.sqsh \\
HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home \\
HF_DATASETS_CACHE=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/cache \\
WANDB_API_KEY=$WANDB_API_KEY \\
MOUNTS="/lustre:/lustre" \\
sbatch \\
    --nodes={num_nodes} \\
    --account={account} \\
    --job-name={job_name} \\
    --partition=batch \\
    --time={time_limit} \\
    --gres=gpu:4 \\
    --segment {segment} \\
    ray.sub"""
    
    return sbatch_cmd


def print_config_summary(config: ModelConfig):
    """Print a summary of the configuration."""
    num_nodes = config.total_gpus // config.gpus_per_node
    t_dp = calculate_dp(config.total_gpus, config.t_tp, config.t_pp, config.t_ep)
    g_dp = calculate_dp(config.total_gpus, config.g_tp, config.g_pp)
    rollout_gbs = config.num_prompts * config.num_generations
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  {config.name:^60}  ║
╠══════════════════════════════════════════════════════════════════╣
║  Model: {config.model_name:<54} ║
║  Nodes: {num_nodes:<3}  |  GPUs/Node: {config.gpus_per_node:<2}  |  Total GPUs: {config.total_gpus:<4}         ║
╠══════════════════════════════════════════════════════════════════╣
║  Batch Settings:                                                 ║
║    Rollout GBS: {rollout_gbs:<5} ({config.num_prompts} prompts × {config.num_generations} gens)               ║
║    Training GBS: {config.train_gbs:<5}                                           ║
║    Max SeqLen: {config.max_seqlen:<6}                                            ║
╠══════════════════════════════════════════════════════════════════╣
║  Training Parallelism:                                           ║
║    TP={config.t_tp:<2} | CP={config.t_cp:<2} | EP={config.t_ep:<2} | PP={config.t_pp:<2} | DP={t_dp:<2}                       ║
║  Generation Parallelism:                                         ║
║    TP={config.g_tp:<2} | PP={config.g_pp:<2} | DP={g_dp:<2}                                         ║
╚══════════════════════════════════════════════════════════════════╝
""")


def list_presets():
    """List all available presets."""
    print("\n📋 Available Presets:\n")
    print(f"{'Preset':<18} {'Model':<30} {'GPUs':>5} {'R-GBS':>6} {'T-GBS':>6} {'SeqLen':>7} {'Train (TP,CP,EP,PP,DP)':>22} {'Gen (TP,PP,DP)':>14}")
    print("-" * 120)
    
    for name, config in PRESETS.items():
        t_dp = calculate_dp(config.total_gpus, config.t_tp, config.t_pp, config.t_ep)
        g_dp = calculate_dp(config.total_gpus, config.g_tp, config.g_pp)
        train_str = f"{config.t_tp},{config.t_cp},{config.t_ep},{config.t_pp},{t_dp}"
        gen_str = f"{config.g_tp},{config.g_pp},{g_dp}"
        rollout_gbs = config.num_prompts * config.num_generations
        print(f"{name:<18} {config.model_name:<30} {config.total_gpus:>5} {rollout_gbs:>6} {config.train_gbs:>6} {config.max_seqlen:>7} {train_str:>22} {gen_str:>14}")
    
    print("\n💡 Usage: python launch_grpo.py --preset <preset_name>")
    print("   Example: python launch_grpo.py --preset qwen32b --dry-run\n")


def launch_job(config: ModelConfig, dry_run: bool = False, **kwargs):
    """Launch a job with the given configuration."""
    print_config_summary(config)
    
    cmd = build_command(config, **kwargs)
    
    if dry_run:
        print("🔍 Dry run - Command that would be executed:\n")
        print(cmd)
        print()
        return
    
    print("🚀 Launching job...")
    
    # Write to temporary script and execute
    script_content = f"""#!/bin/bash
{cmd}
"""
    
    script_path = f"/tmp/launch_{config.name.lower().replace('-', '_')}.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
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
        os.remove(script_path)


def main():
    parser = argparse.ArgumentParser(
        description="Flexible GRPO job launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use a preset configuration
  python launch_grpo.py --preset qwen32b
  
  # Dry run to see the command
  python launch_grpo.py --preset llama70b --dry-run
  
  # Custom configuration
  python launch_grpo.py --model Qwen/Qwen3-32B --gpus 16 \\
      --train-tp 4 --train-pp 1 --gen-tp 1 --gen-pp 1
  
  # Launch all presets
  python launch_grpo.py --all
"""
    )
    
    # Preset selection
    parser.add_argument("--preset", "-p", choices=list(PRESETS.keys()),
                        help="Use a preset configuration")
    parser.add_argument("--list-presets", "-l", action="store_true",
                        help="List available presets")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Launch all preset configurations")
    
    # Custom configuration
    parser.add_argument("--model", help="HuggingFace model name")
    parser.add_argument("--gpus", type=int, help="Total number of GPUs")
    parser.add_argument("--gpus-per-node", type=int, default=4,
                        help="GPUs per node (default: 4 for B200/GB200)")
    
    # Training parallelism
    parser.add_argument("--train-tp", type=int, default=1, help="Training Tensor Parallel")
    parser.add_argument("--train-cp", type=int, default=1, help="Training Context Parallel")
    parser.add_argument("--train-ep", type=int, default=1, help="Training Expert Parallel")
    parser.add_argument("--train-pp", type=int, default=1, help="Training Pipeline Parallel")
    
    # Generation parallelism
    parser.add_argument("--gen-tp", type=int, default=1, help="Generation Tensor Parallel")
    parser.add_argument("--gen-pp", type=int, default=1, help="Generation Pipeline Parallel")
    
    # Job settings
    parser.add_argument("--wandb-project", default="sync-grpo-gb200-benchmark",
                        help="WandB project name")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum training steps")
    parser.add_argument("--time", default="04:00:00", help="Job time limit")
    parser.add_argument("--account", default="coreai_dlalgo_nemorl", help="SLURM account")
    
    # Execution options
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Show command without executing")
    
    args = parser.parse_args()
    
    # List presets
    if args.list_presets:
        list_presets()
        return
    
    # Launch all presets
    if args.all:
        print("🚀 Launching all preset configurations...\n")
        for name, config in PRESETS.items():
            print(f"\n{'='*70}")
            print(f"  Launching: {name}")
            print(f"{'='*70}")
            launch_job(config, dry_run=args.dry_run,
                      wandb_project=args.wandb_project,
                      max_steps=args.max_steps,
                      time_limit=args.time,
                      account=args.account)
        return
    
    # Use preset
    if args.preset:
        config = PRESETS[args.preset]
        launch_job(config, dry_run=args.dry_run,
                  wandb_project=args.wandb_project,
                  max_steps=args.max_steps,
                  time_limit=args.time,
                  account=args.account)
        return
    
    # Custom configuration
    if args.model and args.gpus:
        # Find a suitable base config
        if "70b" in args.model.lower() or "70B" in args.model:
            base_config = "examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-2n8g.yaml"
        elif "32b" in args.model.lower() or "32B" in args.model:
            base_config = "examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n8g.yaml"
        elif "30b" in args.model.lower() or "30B" in args.model:
            base_config = "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g.yaml"
        else:
            base_config = "examples/configs/recipes/llm/performance/grpo-llama3.1-8b-instruct-2n8g.yaml"
        
        # Extract short name
        short_name = args.model.split('/')[-1] if '/' in args.model else args.model
        
        config = ModelConfig(
            name=short_name,
            model_name=args.model,
            total_gpus=args.gpus,
            gpus_per_node=args.gpus_per_node,
            t_tp=args.train_tp,
            t_cp=args.train_cp,
            t_ep=args.train_ep,
            t_pp=args.train_pp,
            g_tp=args.gen_tp,
            g_pp=args.gen_pp,
            config_file=base_config,
        )
        
        launch_job(config, dry_run=args.dry_run,
                  wandb_project=args.wandb_project,
                  max_steps=args.max_steps,
                  time_limit=args.time,
                  account=args.account)
        return
    
    # No valid options provided
    parser.print_help()
    print("\n💡 Try: python launch_grpo.py --list-presets")


if __name__ == "__main__":
    main()

