#!/usr/bin/env python3
"""
Collect and analyze benchmark results from multiple runs.

Usage:
    python collect_results.py                    # Show offline results
    python collect_results.py --online           # Show online results
    python collect_results.py --group            # Group by config, show mean±std
    python collect_results.py --all              # Include running/failed jobs
    python collect_results.py 1234567 1234568   # Collect specific job IDs
    python collect_results.py --output results.csv
"""

import argparse
import json
import glob
import os
import subprocess
import re
from pathlib import Path


def get_running_jobs():
    """Get list of currently running SLURM job IDs."""
    try:
        result = subprocess.run(
            ['squeue', '-u', os.environ.get('USER', ''), '-h', '-o', '%i'],
            capture_output=True, text=True, timeout=10
        )
        return set(result.stdout.strip().split('\n')) if result.stdout.strip() else set()
    except Exception:
        return set()


def check_job_status(job_dir, running_jobs):
    """
    Check job status based on files and content.
    Returns: 'Completed', 'Running', 'Failed', or 'Unknown'
    """
    job_id = job_dir.name.replace('-logs', '')
    results_file = job_dir / 'results.json'
    
    # Check if job is still running
    if job_id in running_jobs:
        return 'Running'
    
    # Check if results.json exists and is valid
    if results_file.exists():
        try:
            with open(results_file) as f:
                data = json.load(f)
                if data.get('generation_throughput_tokens_per_sec', 0) > 0:
                    return 'Completed'
        except Exception:
            pass
    
    # Check for error patterns in log files
    error_patterns = [
        r'Error', r'Exception', r'Traceback', r'FAILED', r'OOM',
        r'CUDA out of memory', r'RuntimeError', r'ValueError',
        r'killed', r'CANCELLED', r'timeout'
    ]
    
    for log_file in job_dir.glob('slurm-*.out'):
        try:
            with open(log_file, 'r', errors='ignore') as f:
                content = f.read()
                for pattern in error_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        return 'Failed'
        except Exception:
            pass
    
    for log_file in job_dir.glob('slurm-*.err'):
        try:
            with open(log_file, 'r', errors='ignore') as f:
                content = f.read()
                if content.strip():  # Any content in stderr might indicate issues
                    for pattern in error_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            return 'Failed'
        except Exception:
            pass
    
    # If results.json doesn't exist and job is not running
    if not results_file.exists():
        return 'Failed'
    
    return 'Unknown'


def find_all_job_dirs(job_ids=None, base_dir="."):
    """Find all job directories (including those without results.json)."""
    base_path = Path(base_dir)
    
    if job_ids:
        dirs = []
        for job_id in job_ids:
            job_dir = base_path / f"{job_id}-logs"
            if job_dir.exists():
                dirs.append(job_dir)
    else:
        dirs = [d for d in base_path.glob("*-logs") if d.is_dir()]
    
    return sorted(dirs)


def find_result_files(job_ids=None, base_dir="."):
    """Find all results.json files."""
    base_path = Path(base_dir)
    
    if job_ids:
        # Specific job IDs
        files = []
        for job_id in job_ids:
            pattern = base_path / f"{job_id}-logs" / "results.json"
            if pattern.exists():
                files.append(pattern)
            else:
                print(f"Warning: {pattern} not found")
    else:
        # Find all
        files = list(base_path.glob("*-logs/results.json"))
    
    return sorted(files)


def load_results(files):
    """Load results from JSON files."""
    results = []
    
    for f in files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                # Add file path for reference
                data['_file'] = str(f)
                data['_job_id'] = f.parent.name.replace('-logs', '')
                data['_status'] = 'Completed'
                results.append(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    return results


def load_all_jobs(job_dirs, running_jobs):
    """Load all jobs including running/failed ones."""
    results = []
    
    for job_dir in job_dirs:
        job_id = job_dir.name.replace('-logs', '')
        status = check_job_status(job_dir, running_jobs)
        results_file = job_dir / 'results.json'
        
        if results_file.exists():
            try:
                with open(results_file, 'r') as fp:
                    data = json.load(fp)
                    data['_file'] = str(results_file)
                    data['_job_id'] = job_id
                    data['_status'] = status
                    results.append(data)
                    continue
            except Exception:
                pass
        
        # For running/failed jobs without valid results.json
        # Try to extract info from slurm output
        data = {
            '_file': str(job_dir),
            '_job_id': job_id,
            '_status': status,
            'model': 'unknown',
            'num_nodes': 0,
            'tp_size': 0,
            'pp_size': 0,
            'ep_size': 1,
            'dp_size': 0,
            'total_gpus': 0,
            'generation_throughput_tokens_per_sec': 0,
            'tokens_per_sec_per_gpu': 0,
            'total_time_sec': 0,
            'total_requests': 0,
        }
        
        # Try to parse model info from log
        for log_file in job_dir.glob('slurm-*.out'):
            try:
                with open(log_file, 'r', errors='ignore') as f:
                    content = f.read()
                    # Extract model name
                    m = re.search(r'Model:\s*(\S+)', content)
                    if m:
                        data['model'] = m.group(1)
                    # Extract parallelism
                    m = re.search(r'TP=(\d+)', content)
                    if m:
                        data['tp_size'] = int(m.group(1))
                    m = re.search(r'PP=(\d+)', content)
                    if m:
                        data['pp_size'] = int(m.group(1))
                    m = re.search(r'EP=(\d+)', content)
                    if m:
                        data['ep_size'] = int(m.group(1))
                    m = re.search(r'DP=(\d+)', content)
                    if m:
                        data['dp_size'] = int(m.group(1))
                    m = re.search(r'Nodes:\s*(\d+)', content)
                    if m:
                        data['num_nodes'] = int(m.group(1))
                    m = re.search(r'Total GPUs:\s*(\d+)', content)
                    if m:
                        data['total_gpus'] = int(m.group(1))
                    break
            except Exception:
                pass
        
        results.append(data)
    
    return results


def get_gpu_model(r):
    """Get GPU model, defaulting to GB200 for older results without this field."""
    gpu = r.get('gpu_model', 'GB200')
    if gpu in ('unknown', '', None):
        gpu = 'GB200'
    return gpu

def get_gpus_per_node(r):
    """Get GPUs per node, defaulting to 4 for GB200."""
    gpn = r.get('gpus_per_node')
    if gpn is None or gpn == '?':
        gpn = 4  # Default for GB200
    return gpn

def get_config_key(r):
    """Get a unique key for a configuration."""
    model = r.get('model', 'unknown')
    if '/' in model:
        model = model.split('/')[-1]
    gpu_model = get_gpu_model(r)
    gpus_per_node = get_gpus_per_node(r)
    return (gpu_model, model, r.get('num_nodes', 1), gpus_per_node, r.get('tp_size', 0), r.get('pp_size', 0), r.get('dp_size', 0), r.get('ep_size', 1))


def print_table(results, group_by_config=False, show_status=False):
    """Print results as a formatted table."""
    if not results:
        print("No results found.")
        return
    
    if group_by_config:
        print_grouped_table(results)
    else:
        print_individual_table(results, show_status=show_status)


def get_status_display(status):
    """Get colored status display."""
    status_colors = {
        'Completed': '\033[92mCompleted\033[0m',  # Green
        'Running': '\033[93mRunning\033[0m',      # Yellow
        'Failed': '\033[91mFailed\033[0m',        # Red
        'Unknown': '\033[90mUnknown\033[0m',      # Gray
    }
    return status_colors.get(status, status)


def print_individual_table(results, show_status=False):
    """Print all results individually."""
    # Header
    if show_status:
        print("\n" + "=" * 180)
        print(f"{'Job ID':<10} {'Status':<12} {'GPU':<8} {'Model':<22} {'N':>2} {'G/N':>3} {'TP':>3} {'PP':>3} {'EP':>3} {'DP':>3} {'GPUs':>5} {'Requests':>8} {'Tokens/s':>12} {'Tok/s/GPU':>12} {'Time(s)':>10}")
        print("=" * 180)
    else:
        print("\n" + "=" * 165)
        print(f"{'Job ID':<10} {'GPU':<8} {'Model':<25} {'N':>2} {'G/N':>3} {'TP':>3} {'PP':>3} {'EP':>3} {'DP':>3} {'GPUs':>5} {'Requests':>8} {'Tokens/s':>12} {'Tok/s/GPU':>12} {'Time(s)':>10}")
        print("=" * 165)
    
    # Sort by status first (Completed first), then by throughput
    def sort_key(x):
        status_order = {'Completed': 0, 'Running': 1, 'Failed': 2, 'Unknown': 3}
        return (status_order.get(x.get('_status', 'Unknown'), 3), 
                -x.get('generation_throughput_tokens_per_sec', 0))
    
    results_sorted = sorted(results, key=sort_key)
    
    for r in results_sorted:
        model = r.get('model', 'unknown')
        # Shorten model name
        if '/' in model:
            model = model.split('/')[-1]
        max_len = 20 if show_status else 23
        if len(model) > max_len:
            model = model[:max_len-3] + "..."
        
        job_id = r.get('_job_id', '?')
        gpu_model = get_gpu_model(r)[:7]  # Truncate GPU model name
        status = r.get('_status', 'Completed')
        
        if show_status:
            print(f"{job_id:<10} "
                  f"{get_status_display(status):<21} "  # Extra space for ANSI codes
                  f"{gpu_model:<8} "
                  f"{model:<22} "
                  f"{r.get('num_nodes', 1):>2} "
                  f"{get_gpus_per_node(r):>3} "
                  f"{r.get('tp_size', '?'):>3} "
                  f"{r.get('pp_size', '?'):>3} "
                  f"{r.get('ep_size', 1):>3} "
                  f"{r.get('dp_size', '?'):>3} "
                  f"{r.get('total_gpus', '?'):>5} "
                  f"{r.get('total_requests', '?'):>8} "
                  f"{r.get('generation_throughput_tokens_per_sec', 0):>12,.0f} "
                  f"{r.get('tokens_per_sec_per_gpu', 0):>12,.0f} "
                  f"{r.get('total_time_sec', 0):>10.1f}")
        else:
            print(f"{job_id:<10} "
                  f"{gpu_model:<8} "
                  f"{model:<25} "
                  f"{r.get('num_nodes', 1):>2} "
                  f"{get_gpus_per_node(r):>3} "
                  f"{r.get('tp_size', '?'):>3} "
                  f"{r.get('pp_size', '?'):>3} "
                  f"{r.get('ep_size', 1):>3} "
                  f"{r.get('dp_size', '?'):>3} "
                  f"{r.get('total_gpus', '?'):>5} "
                  f"{r.get('total_requests', '?'):>8} "
                  f"{r.get('generation_throughput_tokens_per_sec', 0):>12,.0f} "
                  f"{r.get('tokens_per_sec_per_gpu', 0):>12,.0f} "
                  f"{r.get('total_time_sec', 0):>10.1f}")
    
    if show_status:
        print("=" * 180)
        # Print summary
        completed = sum(1 for r in results if r.get('_status') == 'Completed')
        running = sum(1 for r in results if r.get('_status') == 'Running')
        failed = sum(1 for r in results if r.get('_status') == 'Failed')
        print(f"\nSummary: {completed} Completed, {running} Running, {failed} Failed")
    else:
        print("=" * 165)


def print_grouped_table(results):
    """Print results grouped by configuration with mean/std."""
    from collections import defaultdict
    import statistics
    
    # Group by config
    groups = defaultdict(list)
    for r in results:
        key = get_config_key(r)
        groups[key].append(r)
    
    # Header
    print("\n" + "=" * 170)
    print(f"{'GPU':<8} {'Model':<22} {'N':>2} {'G/N':>3} {'TP':>3} {'PP':>3} {'EP':>3} {'DP':>3} {'GPUs':>5} {'Runs':>5} {'Tokens/s (mean±std)':>25} {'Tok/s/GPU':>12} {'Time(s)':>12}")
    print("=" * 170)
    
    # Calculate stats and sort by mean throughput
    stats = []
    for (gpu_model, model, nodes, gpus_per_node, tp, pp, dp, ep), group in groups.items():
        throughputs = [r.get('generation_throughput_tokens_per_sec', 0) for r in group]
        per_gpu = [r.get('tokens_per_sec_per_gpu', 0) for r in group]
        times = [r.get('total_time_sec', 0) for r in group]
        gpus = group[0].get('total_gpus', 0)
        
        mean_tp = statistics.mean(throughputs)
        std_tp = statistics.stdev(throughputs) if len(throughputs) > 1 else 0
        mean_per_gpu = statistics.mean(per_gpu)
        mean_time = statistics.mean(times)
        
        stats.append({
            'gpu_model': gpu_model,
            'model': model,
            'nodes': nodes,
            'gpus_per_node': gpus_per_node,
            'tp': tp,
            'pp': pp,
            'dp': dp,
            'ep': ep,
            'gpus': gpus,
            'runs': len(group),
            'mean_tp': mean_tp,
            'std_tp': std_tp,
            'mean_per_gpu': mean_per_gpu,
            'mean_time': mean_time,
            'job_ids': [r.get('_job_id', '?') for r in group]
        })
    
    # Sort by mean throughput
    stats_sorted = sorted(stats, key=lambda x: x['mean_tp'], reverse=True)
    
    for s in stats_sorted:
        model = s['model']
        if len(model) > 20:
            model = model[:17] + "..."
        gpu_model = (s['gpu_model'] or 'GB200')[:7]
        
        if s['std_tp'] > 0:
            tp_str = f"{s['mean_tp']:,.0f} ± {s['std_tp']:,.0f}"
        else:
            tp_str = f"{s['mean_tp']:,.0f}"
        
        print(f"{gpu_model:<8} "
              f"{model:<22} "
              f"{s['nodes']:>2} "
              f"{s['gpus_per_node']:>3} "
              f"{s['tp']:>3} "
              f"{s['pp']:>3} "
              f"{s['ep']:>3} "
              f"{s['dp']:>3} "
              f"{s['gpus']:>5} "
              f"{s['runs']:>5} "
              f"{tp_str:>25} "
              f"{s['mean_per_gpu']:>12,.0f} "
              f"{s['mean_time']:>12.1f}")
    
    print("=" * 170)
    
    # Print job IDs for reference
    print("\nJob IDs per configuration:")
    for s in stats_sorted:
        ep_str = f" EP={s['ep']}" if s['ep'] > 1 else ""
        print(f"  [{s['gpu_model'][:6]}] {s['model'][:18]} N={s['nodes']} G/N={s['gpus_per_node']} TP={s['tp']} PP={s['pp']}{ep_str} DP={s['dp']}: {', '.join(s['job_ids'])}")


def save_csv(results, output_path):
    """Save results to CSV."""
    import csv
    
    if not results:
        print("No results to save.")
        return
    
    # Get all keys
    keys = set()
    for r in results:
        keys.update(r.keys())
    keys = sorted(keys)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect benchmark results")
    parser.add_argument("job_ids", nargs="*", help="Job IDs to collect (default: all)")
    parser.add_argument("--online", action="store_true", help="Collect online benchmark results (default: offline)")
    parser.add_argument("--base-dir", help="Override base directory")
    parser.add_argument("--group", "-g", action="store_true", help="Group results by configuration and show mean±std")
    parser.add_argument("--all", "-a", action="store_true", help="Include running/failed jobs with status column")
    parser.add_argument("--output", "-o", help="Output CSV file")
    parser.add_argument("--json", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Determine base directory
    if args.base_dir:
        base_dir = args.base_dir
    elif args.online:
        base_dir = "vllm_standalone_perf_exp/online"
        print("Collecting ONLINE benchmark results...")
    else:
        base_dir = "vllm_standalone_perf_exp/offline"
        print("Collecting OFFLINE benchmark results...")
    
    # Find and load results
    if args.all:
        running_jobs = get_running_jobs()
        job_dirs = find_all_job_dirs(args.job_ids or None, base_dir)
        print(f"Found {len(job_dirs)} job directories")
        results = load_all_jobs(job_dirs, running_jobs)
        print(f"Loaded {len(results)} jobs (including running/failed)")
    else:
        files = find_result_files(args.job_ids or None, base_dir)
        print(f"Found {len(files)} result files")
        results = load_results(files)
        print(f"Loaded {len(results)} results")
    
    # Print table
    print_table(results, group_by_config=args.group, show_status=args.all)
    
    # Save if requested
    if args.output:
        save_csv(results, args.output)
    
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.json}")


if __name__ == "__main__":
    main()

