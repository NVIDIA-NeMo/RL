#!/usr/bin/env python3
"""
vLLM Throughput Benchmark Runner

Reads model configurations from model_configs.yaml and submits
vLLM throughput benchmarks for specified models and platforms.

Usage:
    # Run all GB200 models
    python run_benchmarks.py --platform gb200

    # Run specific models
    python run_benchmarks.py --platform gb200 --models llama8b qwen32b

    # Dry run (print commands without executing)
    python run_benchmarks.py --platform gb200 --dry-run

    # Override settings
    python run_benchmarks.py --platform gb200 --num-prompts 128 --max-seqlen 8192
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict:
    """Load model_configs.yaml"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_model_config(config: dict, model_name: str, platform: str) -> dict | None:
    """Get model configuration for specified platform"""
    if model_name not in config:
        return None
    
    model_config = config[model_name]
    if platform not in model_config:
        return None
    
    platform_config = model_config[platform].copy()
    platform_config["model_path"] = model_config.get("model_name", "")
    
    return platform_config


def calculate_nodes(num_gpus: int, gpus_per_node: int = 4) -> int:
    """Calculate number of nodes from GPU count"""
    return num_gpus // gpus_per_node


def build_benchmark_command(
    model_path: str,
    num_nodes: int,
    tp_size: int,
    pp_size: int,
    num_prompts: int,
    num_generations: int,
    max_model_len: int,
    input_lens: str,
    output_lens: str,
    ep_size: int = 1,
    script_dir: Path = None,
) -> dict:
    """Build environment variables and command for benchmark"""
    env = {
        "MODEL_PATH": model_path,
        "NUM_NODES": str(num_nodes),
        "TP_SIZE": str(tp_size),
        "PP_SIZE": str(pp_size),
        "THROUGHPUT_NUM_PROMPTS": str(num_prompts),
        "THROUGHPUT_NUM_GENERATIONS": str(num_generations),
        "MAX_MODEL_LEN": str(max_model_len),
        "INPUT_LENS": input_lens,
        "OUTPUT_LENS": output_lens,
    }
    
    if ep_size > 1:
        env["EP_SIZE"] = str(ep_size)
    
    cmd = ["./run_vllm_benchmark.sh", "run-throughput"]
    
    return {"env": env, "cmd": cmd}


def run_benchmark(
    model_name: str,
    config: dict,
    platform: str,
    gpus_per_node: int,
    overrides: dict,
    script_dir: Path,
    dry_run: bool = False,
) -> bool:
    """Run benchmark for a single model"""
    model_config = get_model_config(config, model_name, platform)
    if model_config is None:
        print(f"⚠️  No {platform} config found for {model_name}, skipping...")
        return False
    
    # Extract settings
    num_gpus = model_config.get("num_gpus", 8)
    num_nodes = calculate_nodes(num_gpus, gpus_per_node)
    
    gen_config = model_config.get("generation", {})
    tp_size = gen_config.get("tp", 1)
    pp_size = gen_config.get("pp", 1)
    
    # Get common settings with overrides
    num_prompts = overrides.get("num_prompts", model_config.get("num_prompts", 64))
    num_generations = overrides.get("num_generations", model_config.get("num_generations", 32))
    max_model_len = overrides.get("max_seqlen", model_config.get("max_seqlen", 4096))
    input_lens = overrides.get("input_lens", "64 100 150")
    output_lens = overrides.get("output_lens", "2048 4096")
    
    # Calculate DP
    dp_size = num_gpus // (tp_size * pp_size)
    
    model_path = model_config.get("model_path", "")
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name} ({model_path})")
    print(f"{'='*60}")
    print(f"  Platform: {platform.upper()}")
    print(f"  GPUs: {num_gpus} ({num_nodes} nodes × {gpus_per_node} GPUs)")
    print(f"  Generation Parallelism: TP={tp_size}, PP={pp_size}, DP={dp_size}")
    print(f"  Workload:")
    print(f"    - num_prompts: {num_prompts}")
    print(f"    - num_generations (n): {num_generations}")
    print(f"    - total sequences: {num_prompts * num_generations}")
    print(f"    - max_model_len: {max_model_len}")
    print(f"    - input_lens: {input_lens}")
    print(f"    - output_lens: {output_lens}")
    
    # Build command
    bench_cmd = build_benchmark_command(
        model_path=model_path,
        num_nodes=num_nodes,
        tp_size=tp_size,
        pp_size=pp_size,
        num_prompts=num_prompts,
        num_generations=num_generations,
        max_model_len=max_model_len,
        input_lens=input_lens,
        output_lens=output_lens,
        script_dir=script_dir,
    )
    
    # Print command
    env_str = " \\\n    ".join(f"{k}={v}" for k, v in bench_cmd["env"].items())
    cmd_str = " ".join(bench_cmd["cmd"])
    print(f"\n  Command:")
    print(f"    {env_str} \\")
    print(f"    {cmd_str}")
    
    if dry_run:
        print(f"\n  [DRY RUN] Skipping execution")
        return True
    
    # Execute
    print(f"\n  Submitting job...")
    env = os.environ.copy()
    env.update(bench_cmd["env"])
    
    result = subprocess.run(
        bench_cmd["cmd"],
        cwd=script_dir,
        env=env,
        capture_output=False,
    )
    
    if result.returncode == 0:
        print(f"  ✅ Job submitted successfully")
        return True
    else:
        print(f"  ❌ Job submission failed (exit code: {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="vLLM Throughput Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all GB200 models
  python run_benchmarks.py --platform gb200

  # Run specific models
  python run_benchmarks.py --platform gb200 --models llama8b qwen32b qwen30b

  # Dry run (print commands without executing)
  python run_benchmarks.py --platform gb200 --dry-run

  # Override settings
  python run_benchmarks.py --platform gb200 --num-prompts 128 --max-seqlen 8192

  # List available models
  python run_benchmarks.py --list-models
        """
    )
    
    parser.add_argument(
        "--platform",
        type=str,
        default="gb200",
        choices=["h100", "gb200", "h100_lowgbs", "h100_highseq", "gb200_lowgbs", "gb200_highseq"],
        help="Target platform (default: gb200)"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["llama8b", "qwen32b", "qwen30b"],
        help="Model names to benchmark (default: llama8b qwen32b qwen30b)"
    )
    
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=4,
        help="GPUs per node (default: 4 for GB200, 8 for H100)"
    )
    
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Override num_prompts (default: from config)"
    )
    
    parser.add_argument(
        "--num-generations",
        type=int,
        default=None,
        help="Override num_generations/n (default: from config)"
    )
    
    parser.add_argument(
        "--max-seqlen",
        type=int,
        default=None,
        help="Override max_seqlen (default: from config)"
    )
    
    parser.add_argument(
        "--input-lens",
        type=str,
        default="64 100 150",
        help="Input lengths to test (default: '64 100 150')"
    )
    
    parser.add_argument(
        "--output-lens",
        type=str,
        default="2048 4096",
        help="Output lengths to test (default: '2048 4096')"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model_configs.yaml (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Find script directory and config
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent
    
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = base_dir / "model_configs.yaml"
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    print(f"📄 Loading config: {config_path}")
    config = load_config(config_path)
    
    # Auto-detect GPUs per node based on platform
    if args.gpus_per_node == 4 and args.platform.startswith("h100"):
        args.gpus_per_node = 8
        print(f"   Auto-detected: {args.gpus_per_node} GPUs/node for H100")
    
    # List models and exit
    if args.list_models:
        print("\n📋 Available models in config:")
        for model_name in config:
            if model_name == "defaults":
                continue
            model = config[model_name]
            platforms = [k for k in model.keys() if k not in ["model_name", "config_file"]]
            print(f"  - {model_name}: {', '.join(platforms)}")
        sys.exit(0)
    
    # Build overrides
    overrides = {}
    if args.num_prompts is not None:
        overrides["num_prompts"] = args.num_prompts
    if args.num_generations is not None:
        overrides["num_generations"] = args.num_generations
    if args.max_seqlen is not None:
        overrides["max_seqlen"] = args.max_seqlen
    overrides["input_lens"] = args.input_lens
    overrides["output_lens"] = args.output_lens
    
    print(f"\n🚀 vLLM Throughput Benchmark Runner")
    print(f"   Platform: {args.platform.upper()}")
    print(f"   Models: {', '.join(args.models)}")
    print(f"   GPUs per node: {args.gpus_per_node}")
    if args.dry_run:
        print(f"   Mode: DRY RUN")
    
    # Run benchmarks
    success_count = 0
    for i, model_name in enumerate(args.models, 1):
        print(f"\n[{i}/{len(args.models)}] Processing {model_name}...")
        
        success = run_benchmark(
            model_name=model_name,
            config=config,
            platform=args.platform,
            gpus_per_node=args.gpus_per_node,
            overrides=overrides,
            script_dir=script_dir,
            dry_run=args.dry_run,
        )
        
        if success:
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary: {success_count}/{len(args.models)} benchmarks submitted")
    print(f"{'='*60}")
    
    if not args.dry_run:
        print(f"\nCheck job status: squeue -u $USER")
        print(f"Results will be in: vllm_standalone_perf_exp/throughput/<job_id>-logs/")


if __name__ == "__main__":
    main()

