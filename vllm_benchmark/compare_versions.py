#!/usr/bin/env python3
"""
vLLM Version Comparison Tool
============================
Compares benchmark results between different vLLM versions.

Usage:
    # Compare latest results for each version
    python compare_versions.py
    
    # Compare specific job IDs
    python compare_versions.py --v0110-job 587395 --nightly-job 587399
    
    # Export to CSV
    python compare_versions.py --output comparison.csv
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import glob


def find_latest_results(base_dir: Path, model_pattern: str = "*") -> dict:
    """Find the latest results.json for each model."""
    results = {}
    
    # Look for throughput results
    throughput_dir = base_dir / "throughput"
    if not throughput_dir.exists():
        return results
    
    for model_dir in throughput_dir.iterdir():
        if not model_dir.is_dir():
            continue
        if model_pattern != "*" and model_pattern not in model_dir.name:
            continue
            
        # Find latest job directory
        job_dirs = sorted(
            [d for d in model_dir.rglob("*-logs") if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        for job_dir in job_dirs:
            results_file = job_dir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file) as f:
                        data = json.load(f)
                    if data:
                        # Use first result to get version info
                        first_result = data[0] if isinstance(data, list) else data
                        version = first_result.get("vllm_version", "unknown")
                        model = first_result.get("model", model_dir.name)
                        
                        key = (model, version)
                        if key not in results:
                            results[key] = {
                                "job_dir": str(job_dir),
                                "data": data,
                                "version": version,
                                "model": model,
                            }
                except Exception as e:
                    print(f"Warning: Could not read {results_file}: {e}")
    
    return results


def load_results_by_job(base_dir: Path, job_id: str) -> dict | None:
    """Load results for a specific job ID."""
    # Search for the job directory
    for job_dir in base_dir.rglob(f"{job_id}-logs"):
        results_file = job_dir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
            if data:
                first_result = data[0] if isinstance(data, list) else data
                return {
                    "job_dir": str(job_dir),
                    "data": data,
                    "version": first_result.get("vllm_version", "unknown"),
                    "model": first_result.get("model", "unknown"),
                }
    return None


def extract_metrics(result_data: list | dict) -> dict:
    """Extract key metrics from results."""
    if isinstance(result_data, dict):
        result_data = [result_data]
    
    metrics = {}
    for item in result_data:
        input_len = item.get("input_len", 0)
        output_len = item.get("actual_output_len", item.get("output_len", 0))
        key = f"I{input_len}_O{output_len}"
        
        metrics[key] = {
            "input_len": input_len,
            "output_len": output_len,
            "throughput_req_s": item.get("throughput_requests_per_sec", 0),
            "throughput_tok_s": item.get("throughput_tokens_per_sec", 0),
            "throughput_tok_s_gpu": item.get("throughput_tokens_per_sec_per_gpu", 0),
            "total_time": item.get("total_time_sec", 0),
            "tp": item.get("tp_size", 1),
            "pp": item.get("pp_size", 1),
        }
    
    return metrics


def format_number(n, decimals=2):
    """Format number with thousands separator."""
    if isinstance(n, (int, float)):
        if n >= 1000:
            return f"{n:,.{decimals}f}"
        return f"{n:.{decimals}f}"
    return str(n)


def print_comparison_table(comparisons: list[dict]):
    """Print a nicely formatted comparison table."""
    
    print("\n" + "=" * 100)
    print("vLLM VERSION COMPARISON RESULTS")
    print("=" * 100)
    
    # Group by model
    by_model = {}
    for comp in comparisons:
        model = comp["model"]
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(comp)
    
    for model, model_comps in by_model.items():
        print(f"\n{'─' * 100}")
        print(f"Model: {model}")
        print(f"{'─' * 100}")
        
        # Header
        print(f"{'Config':<15} │ {'Version':<12} │ {'Throughput (tok/s)':<20} │ {'Per GPU':<15} │ {'Time (s)':<10} │ {'Diff %':<10}")
        print(f"{'─' * 15}─┼─{'─' * 12}─┼─{'─' * 20}─┼─{'─' * 15}─┼─{'─' * 10}─┼─{'─' * 10}")
        
        for comp in model_comps:
            config = f"I{comp['input_len']}_O{comp['output_len']}"
            
            # v0.11.0 row
            v0110_tok = comp.get("v0110_tok_s", 0)
            v0110_gpu = comp.get("v0110_tok_s_gpu", 0)
            v0110_time = comp.get("v0110_time", 0)
            
            print(f"{config:<15} │ {'v0.11.0':<12} │ {format_number(v0110_tok):<20} │ {format_number(v0110_gpu):<15} │ {format_number(v0110_time):<10} │ {'baseline':<10}")
            
            # nightly row
            nightly_tok = comp.get("nightly_tok_s", 0)
            nightly_gpu = comp.get("nightly_tok_s_gpu", 0)
            nightly_time = comp.get("nightly_time", 0)
            
            # Calculate diff
            if v0110_tok > 0:
                diff_pct = ((nightly_tok - v0110_tok) / v0110_tok) * 100
                diff_str = f"{diff_pct:+.1f}%"
                if diff_pct > 5:
                    diff_str = f"🚀 {diff_str}"
                elif diff_pct < -5:
                    diff_str = f"⚠️  {diff_str}"
            else:
                diff_str = "N/A"
            
            print(f"{'':<15} │ {'nightly':<12} │ {format_number(nightly_tok):<20} │ {format_number(nightly_gpu):<15} │ {format_number(nightly_time):<10} │ {diff_str:<10}")
            print(f"{'─' * 15}─┼─{'─' * 12}─┼─{'─' * 20}─┼─{'─' * 15}─┼─{'─' * 10}─┼─{'─' * 10}")
    
    print("\n" + "=" * 100)
    print("Legend: 🚀 = nightly faster (>5%), ⚠️  = nightly slower (>5%)")
    print("=" * 100 + "\n")


def print_summary_table(comparisons: list[dict]):
    """Print a summary comparison table."""
    
    print("\n" + "=" * 80)
    print("SUMMARY: v0.11.0 vs Nightly Performance")
    print("=" * 80)
    
    print(f"\n{'Model':<30} │ {'v0.11.0 (tok/s)':<18} │ {'Nightly (tok/s)':<18} │ {'Diff':<12}")
    print(f"{'─' * 30}─┼─{'─' * 18}─┼─{'─' * 18}─┼─{'─' * 12}")
    
    for comp in comparisons:
        model_short = comp["model"].split("/")[-1][:28]
        v0110 = comp.get("v0110_tok_s", 0)
        nightly = comp.get("nightly_tok_s", 0)
        
        if v0110 > 0:
            diff_pct = ((nightly - v0110) / v0110) * 100
            diff_str = f"{diff_pct:+.1f}%"
            if diff_pct > 5:
                diff_str = f"🚀 {diff_str}"
            elif diff_pct < -5:
                diff_str = f"⚠️  {diff_str}"
        else:
            diff_str = "N/A"
        
        print(f"{model_short:<30} │ {format_number(v0110):<18} │ {format_number(nightly):<18} │ {diff_str:<12}")
    
    print(f"{'─' * 30}─┴─{'─' * 18}─┴─{'─' * 18}─┴─{'─' * 12}")
    print()


def export_to_csv(comparisons: list[dict], output_file: str):
    """Export comparison results to CSV."""
    import csv
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Input_Len", "Output_Len", "TP", "PP",
            "v0.11.0_tok_s", "v0.11.0_tok_s_gpu", "v0.11.0_time",
            "Nightly_tok_s", "Nightly_tok_s_gpu", "Nightly_time",
            "Diff_%"
        ])
        
        for comp in comparisons:
            v0110 = comp.get("v0110_tok_s", 0)
            nightly = comp.get("nightly_tok_s", 0)
            diff_pct = ((nightly - v0110) / v0110 * 100) if v0110 > 0 else 0
            
            writer.writerow([
                comp["model"],
                comp["input_len"],
                comp["output_len"],
                comp.get("tp", 1),
                comp.get("pp", 1),
                comp.get("v0110_tok_s", 0),
                comp.get("v0110_tok_s_gpu", 0),
                comp.get("v0110_time", 0),
                comp.get("nightly_tok_s", 0),
                comp.get("nightly_tok_s_gpu", 0),
                comp.get("nightly_time", 0),
                f"{diff_pct:.2f}"
            ])
    
    print(f"Results exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare vLLM version benchmark results")
    parser.add_argument("--base-dir", type=str, 
                       default=str(Path(__file__).parent / "vllm_standalone_perf_exp"),
                       help="Base directory for results")
    parser.add_argument("--v0110-job", type=str, help="Specific job ID for v0.11.0")
    parser.add_argument("--nightly-job", type=str, help="Specific job ID for nightly")
    parser.add_argument("--output", type=str, help="Export to CSV file")
    parser.add_argument("--model", type=str, default="*", help="Filter by model name")
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        # Try parent directory
        base_dir = Path(__file__).parent.parent / "vllm_standalone_perf_exp"
    
    if not base_dir.exists():
        print(f"Error: Results directory not found: {base_dir}")
        print("Make sure benchmarks have completed and results are available.")
        return
    
    print(f"Scanning results in: {base_dir}")
    
    # Find all results
    all_results = find_latest_results(base_dir, args.model)
    
    if not all_results:
        print("No results found. Make sure benchmarks have completed.")
        return
    
    # Group results by model for comparison
    by_model = {}
    for (model, version), result in all_results.items():
        model_name = model.split("/")[-1]  # Short name
        if model_name not in by_model:
            by_model[model_name] = {}
        
        # Categorize by version
        if "0.11" in version:
            by_model[model_name]["v0110"] = result
        else:
            by_model[model_name]["nightly"] = result
    
    # Build comparison data
    comparisons = []
    
    for model_name, versions in by_model.items():
        v0110_data = versions.get("v0110", {}).get("data", [])
        nightly_data = versions.get("nightly", {}).get("data", [])
        
        v0110_metrics = extract_metrics(v0110_data) if v0110_data else {}
        nightly_metrics = extract_metrics(nightly_data) if nightly_data else {}
        
        # Get all config keys
        all_configs = set(v0110_metrics.keys()) | set(nightly_metrics.keys())
        
        for config in all_configs:
            v0110 = v0110_metrics.get(config, {})
            nightly = nightly_metrics.get(config, {})
            
            comp = {
                "model": model_name,
                "input_len": v0110.get("input_len", nightly.get("input_len", 0)),
                "output_len": v0110.get("output_len", nightly.get("output_len", 0)),
                "tp": v0110.get("tp", nightly.get("tp", 1)),
                "pp": v0110.get("pp", nightly.get("pp", 1)),
                "v0110_tok_s": v0110.get("throughput_tok_s", 0),
                "v0110_tok_s_gpu": v0110.get("throughput_tok_s_gpu", 0),
                "v0110_time": v0110.get("total_time", 0),
                "nightly_tok_s": nightly.get("throughput_tok_s", 0),
                "nightly_tok_s_gpu": nightly.get("throughput_tok_s_gpu", 0),
                "nightly_time": nightly.get("total_time", 0),
            }
            comparisons.append(comp)
    
    if not comparisons:
        print("No matching results found for comparison.")
        print("\nAvailable results:")
        for (model, version), result in all_results.items():
            print(f"  - {model} (v{version}): {result['job_dir']}")
        return
    
    # Print results
    print_summary_table(comparisons)
    print_comparison_table(comparisons)
    
    # Export if requested
    if args.output:
        export_to_csv(comparisons, args.output)


if __name__ == "__main__":
    main()

