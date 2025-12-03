#!/usr/bin/env python3
"""
Collect and analyze GRPO benchmark results from WandB and/or local logs.

Usage:
    python collect_grpo_results.py                     # Show all results
    python collect_grpo_results.py --wandb             # Fetch from WandB
    python collect_grpo_results.py --project PROJECT   # Specify WandB project
    python collect_grpo_results.py --output results.csv
"""

import argparse
import json
import os
import re
import glob
from pathlib import Path
from datetime import datetime
from typing import Optional
import subprocess


def get_running_jobs():
    """Get list of currently running SLURM job IDs."""
    try:
        result = subprocess.run(
            ['squeue', '-u', os.environ.get('USER', ''), '-h', '-o', '%i %j'],
            capture_output=True, text=True, timeout=10
        )
        jobs = {}
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    jobs[parts[0]] = parts[1]
        return jobs
    except Exception:
        return {}


def parse_wandb_name(name: str) -> dict:
    """Parse WandB run name to extract configuration."""
    result = {
        'model': 'Unknown',
        'num_nodes': 0,
        'gpus_per_node': 0,
        'total_gpus': 0,
        # Training parallelism
        't_tp': 1,
        't_pp': 1,
        't_ep': 1,
        't_cp': 1,
        't_vpp': 1,
        # Generation parallelism
        'g_tp': 1,
        'g_pp': 1,
        'g_ep': 1,
    }
    
    # Extract model name (first part before -N)
    model_match = re.match(r'^([^-]+(?:-[^N][^-]*)*)', name)
    if model_match:
        result['model'] = model_match.group(1)
    
    # Extract NxG (nodes x gpus_per_node)
    nxg_match = re.search(r'-N(\d+)xG(\d+)', name)
    if nxg_match:
        result['num_nodes'] = int(nxg_match.group(1))
        result['gpus_per_node'] = int(nxg_match.group(2))
        result['total_gpus'] = result['num_nodes'] * result['gpus_per_node']
    
    # Extract Training params: Train(tp4.pp4.ep1.cp1.vpp1)
    train_match = re.search(r'Train\(tp(\d+)\.pp(\d+)\.ep(\d+)\.cp(\d+)\.vpp(\d+)\)', name)
    if train_match:
        result['t_tp'] = int(train_match.group(1))
        result['t_pp'] = int(train_match.group(2))
        result['t_ep'] = int(train_match.group(3))
        result['t_cp'] = int(train_match.group(4))
        result['t_vpp'] = int(train_match.group(5))
    
    # Extract Generation params: Gen(tp4.pp1.ep1)
    gen_match = re.search(r'Gen\(tp(\d+)\.pp(\d+)\.ep(\d+)\)', name)
    if gen_match:
        result['g_tp'] = int(gen_match.group(1))
        result['g_pp'] = int(gen_match.group(2))
        result['g_ep'] = int(gen_match.group(3))
    
    # Calculate DP (Data Parallelism)
    mp = result['t_tp'] * result['t_pp']
    if mp > 0 and result['total_gpus'] > 0:
        result['t_dp'] = result['total_gpus'] // mp
    else:
        result['t_dp'] = 1
    
    # Generation DP
    g_mp = result['g_tp'] * result['g_pp']
    if g_mp > 0 and result['total_gpus'] > 0:
        result['g_dp'] = result['total_gpus'] // g_mp
    else:
        result['g_dp'] = 1
    
    return result


def fetch_wandb_results(project: str, entity: Optional[str] = None) -> list:
    """Fetch results from WandB."""
    try:
        import wandb
    except ImportError:
        print("⚠️  wandb not installed. Install with: pip install wandb")
        return []
    
    api = wandb.Api()
    
    try:
        if entity:
            runs = api.runs(f"{entity}/{project}")
        else:
            runs = api.runs(project)
    except Exception as e:
        print(f"⚠️  Failed to fetch WandB runs: {e}")
        return []
    
    results = []
    for run in runs:
        try:
            # Parse name for config
            config = parse_wandb_name(run.name)
            
            # Get summary metrics
            summary = run.summary._json_dict
            
            result = {
                'run_id': run.id,
                'run_name': run.name,
                'status': run.state,
                'created_at': run.created_at,
                **config,
                # Metrics
                'step_time_sec': summary.get('train/step_time', summary.get('step_time', 0)),
                'tokens_per_sec': summary.get('train/tokens_per_sec', summary.get('tokens_per_sec', 0)),
                'tokens_per_sec_per_gpu': summary.get('train/tokens_per_sec_per_gpu', 0),
                'seq_per_sec': summary.get('train/seq_per_sec', summary.get('seq_per_sec', 0)),
                'gpu_util_mfu': summary.get('train/gpu_util_mfu', summary.get('mfu', 0)),
                'per_gpu_tflops': summary.get('train/per_gpu_tflops', summary.get('tflops', 0)),
                'generation_tokens_per_sec': summary.get('generation/tokens_per_sec', 0),
                'generation_tokens_per_sec_per_gpu': summary.get('generation/tokens_per_sec_per_gpu', 0),
                'train_accuracy': summary.get('train/accuracy', summary.get('accuracy', 0)),
                'val_accuracy': summary.get('val/accuracy', 0),
            }
            
            # Calculate tokens_per_sec_per_gpu if not directly available
            if result['tokens_per_sec_per_gpu'] == 0 and result['tokens_per_sec'] > 0 and result['total_gpus'] > 0:
                result['tokens_per_sec_per_gpu'] = result['tokens_per_sec'] / result['total_gpus']
            
            results.append(result)
        except Exception as e:
            print(f"⚠️  Error processing run {run.name}: {e}")
            continue
    
    return results


def scan_local_logs(log_dir: str = "logs") -> list:
    """Scan local log directories for results."""
    results = []
    log_path = Path(log_dir)
    
    if not log_path.exists():
        print(f"⚠️  Log directory {log_dir} not found")
        return results
    
    # Find all experiment directories
    for exp_dir in sorted(log_path.glob("exp_*")):
        result = {
            'exp_id': exp_dir.name,
            'path': str(exp_dir),
            'status': 'Unknown',
        }
        
        # Try to find metrics from various sources
        # 1. Check for wandb local files
        wandb_dir = exp_dir / "wandb"
        if wandb_dir.exists():
            # Look for run info
            for run_dir in wandb_dir.glob("run-*"):
                config_file = run_dir / "files" / "config.yaml"
                if config_file.exists():
                    # Parse config
                    pass
        
        # 2. Check for tensorboard logs
        tb_dir = exp_dir / "tensorboard"
        if tb_dir.exists():
            result['has_tensorboard'] = True
        
        # 3. Check for any summary files
        summary_file = exp_dir / "summary.json"
        if summary_file.exists():
            try:
                with open(summary_file) as f:
                    summary = json.load(f)
                    result.update(summary)
            except Exception:
                pass
        
        results.append(result)
    
    return results


def print_results_table(results: list, show_all: bool = False):
    """Print results as a formatted table."""
    if not results:
        print("No results found.")
        return
    
    # ANSI color codes
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Filter by status if needed
    if not show_all:
        results = [r for r in results if r.get('status') in ['finished', 'running', 'Completed']]
    
    if not results:
        print("No completed/running results found. Use --all to see all runs.")
        return
    
    # Sort by tokens_per_sec descending
    results = sorted(results, key=lambda x: x.get('tokens_per_sec', 0), reverse=True)
    
    # Print header
    print(f"\n{BOLD}{'='*200}{RESET}")
    print(f"{BOLD}GRPO Benchmark Results{RESET}")
    print(f"{'='*200}")
    
    # Table header
    header = (
        f"{'Model':<20} "
        f"{'N':>3} {'G':>3} {'GPUs':>5} "
        f"{'T.TP':>4} {'T.PP':>4} {'T.EP':>4} {'T.CP':>4} {'T.DP':>4} "
        f"{'G.TP':>4} {'G.PP':>4} {'G.DP':>4} "
        f"{'Step(s)':>8} "
        f"{'Tok/s':>12} "
        f"{'Tok/s/GPU':>12} "
        f"{'Seq/s':>8} "
        f"{'MFU%':>6} "
        f"{'TFLOPs':>8} "
        f"{'Status':<10}"
    )
    print(f"{CYAN}{header}{RESET}")
    print("-" * 200)
    
    for r in results:
        model = r.get('model', 'Unknown')[:18]
        status = r.get('status', 'Unknown')
        
        # Color status
        if status in ['finished', 'Completed']:
            status_str = f"{GREEN}{status:<10}{RESET}"
        elif status == 'running':
            status_str = f"{YELLOW}{status:<10}{RESET}"
        else:
            status_str = f"{RED}{status:<10}{RESET}"
        
        row = (
            f"{model:<20} "
            f"{r.get('num_nodes', 0):>3} "
            f"{r.get('gpus_per_node', 0):>3} "
            f"{r.get('total_gpus', 0):>5} "
            f"{r.get('t_tp', 1):>4} "
            f"{r.get('t_pp', 1):>4} "
            f"{r.get('t_ep', 1):>4} "
            f"{r.get('t_cp', 1):>4} "
            f"{r.get('t_dp', 1):>4} "
            f"{r.get('g_tp', 1):>4} "
            f"{r.get('g_pp', 1):>4} "
            f"{r.get('g_dp', 1):>4} "
            f"{r.get('step_time_sec', 0):>8.2f} "
            f"{r.get('tokens_per_sec', 0):>12,.0f} "
            f"{r.get('tokens_per_sec_per_gpu', 0):>12,.0f} "
            f"{r.get('seq_per_sec', 0):>8.2f} "
            f"{r.get('gpu_util_mfu', 0)*100:>6.1f} "
            f"{r.get('per_gpu_tflops', 0):>8.1f} "
            f"{status_str}"
        )
        print(row)
    
    print("=" * 200)
    print(f"\nTotal runs: {len(results)}")


def save_csv(results: list, output_path: str):
    """Save results to CSV."""
    import csv
    
    if not results:
        print("No results to save.")
        return
    
    # Define columns
    columns = [
        'run_name', 'model', 'num_nodes', 'gpus_per_node', 'total_gpus',
        't_tp', 't_pp', 't_ep', 't_cp', 't_vpp', 't_dp',
        'g_tp', 'g_pp', 'g_ep', 'g_dp',
        'step_time_sec', 'tokens_per_sec', 'tokens_per_sec_per_gpu',
        'seq_per_sec', 'gpu_util_mfu', 'per_gpu_tflops',
        'generation_tokens_per_sec', 'generation_tokens_per_sec_per_gpu',
        'train_accuracy', 'val_accuracy', 'status', 'created_at'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✅ Results saved to {output_path}")


def save_json(results: list, output_path: str):
    """Save results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✅ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect GRPO benchmark results")
    parser.add_argument("--wandb", action="store_true", help="Fetch results from WandB")
    parser.add_argument("--project", default="sync-grpo-gb200-benchmark", help="WandB project name")
    parser.add_argument("--entity", default=None, help="WandB entity (team/user)")
    parser.add_argument("--local", action="store_true", help="Scan local log directories")
    parser.add_argument("--log-dir", default="logs", help="Local log directory")
    parser.add_argument("--all", "-a", action="store_true", help="Show all runs including failed")
    parser.add_argument("--output", "-o", help="Output CSV file")
    parser.add_argument("--json", help="Output JSON file")
    
    args = parser.parse_args()
    
    results = []
    
    # Fetch from WandB
    if args.wandb or not args.local:
        print(f"📊 Fetching results from WandB project: {args.project}")
        wandb_results = fetch_wandb_results(args.project, args.entity)
        results.extend(wandb_results)
        print(f"   Found {len(wandb_results)} runs")
    
    # Scan local logs
    if args.local:
        print(f"📁 Scanning local logs in: {args.log_dir}")
        local_results = scan_local_logs(args.log_dir)
        results.extend(local_results)
        print(f"   Found {len(local_results)} experiments")
    
    # Print table
    print_results_table(results, show_all=args.all)
    
    # Save if requested
    if args.output:
        save_csv(results, args.output)
    
    if args.json:
        save_json(results, args.json)


if __name__ == "__main__":
    main()

