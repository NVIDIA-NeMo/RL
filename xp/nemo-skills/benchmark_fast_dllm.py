#!/usr/bin/env python3
"""
Fast-dLLM Benchmarking Script

This script runs multiple evaluation configurations to benchmark different Fast-dLLM
acceleration strategies and compare their performance on GSM8K.

Usage:
    python benchmark_fast_dllm.py
    
    # Custom benchmark
    python benchmark_fast_dllm.py --benchmark math:2
    
    # Quick test mode
    python benchmark_fast_dllm.py --quick-test

Requirements:
    - LLaDA OpenAI server running with Fast-dLLM integration
    - NeMo-Skills evaluation framework
"""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime


def run_evaluation(config_name, eval_args, output_dir):
    """Run a single evaluation configuration."""
    print(f"\n{'=' * 60}")
    print(f"🚀 Running: {config_name}")
    print(f"{'=' * 60}")
    
    # Create subdirectory for this configuration
    config_output_dir = os.path.join(output_dir, f"fast-dllm-{config_name.lower().replace(' ', '-')}")
    os.makedirs(config_output_dir, exist_ok=True)
    
    # Build command
    cmd = ["python", "eval_llada.py", "--output-dir", config_output_dir] + eval_args
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run evaluation
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"✅ {config_name} completed successfully in {duration:.1f}s")
        
        return {
            'name': config_name,
            'success': True,
            'duration': duration,
            'output_dir': config_output_dir,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"❌ {config_name} failed after {duration:.1f}s")
        print(f"Error: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        
        return {
            'name': config_name,
            'success': False,
            'duration': duration,
            'error': str(e),
            'stdout': e.stdout,
            'stderr': e.stderr
        }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Fast-dLLM acceleration strategies")
    parser.add_argument("--benchmark", default="gsm8k:4", help="Benchmark to evaluate")
    parser.add_argument("--output-dir", default="./fast-dllm-benchmark", help="Base output directory")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test mode")
    parser.add_argument("--max-samples", type=int, help="Maximum samples for testing")
    
    args = parser.parse_args()
    
    # Create base output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Common evaluation arguments
    base_args = [
        "--benchmark", args.benchmark,
        "--steps", "128",
        "--block-length", "32",
        "--tokens-to-generate", "128",
        "--temperature", "0.0"
    ]
    
    if args.quick_test:
        base_args.extend(["--quick-test"])
    
    if args.max_samples:
        base_args.extend(["--max-samples", str(args.max_samples)])
    
    # Define benchmark configurations
    configs = [
        {
            "name": "No Cache",
            "description": "Baseline without any caching",
            "args": base_args + ["--no-cache"]
        },
        {
            "name": "Prefix Cache",
            "description": "KV caching with prefix cache only",
            "args": base_args + ["--use-cache", "--no-dual-cache"]
        },
        {
            "name": "Dual Cache",
            "description": "Maximum acceleration with dual cache",
            "args": base_args + ["--use-cache", "--use-dual-cache"]
        },
        {
            "name": "Threshold 0.8",
            "description": "Dual cache with confidence threshold parallel decoding",
            "args": base_args + ["--use-cache", "--use-dual-cache", "--threshold", "0.8"]
        },
        {
            "name": "Factor 2.0",
            "description": "Dual cache with dynamic parallel decoding",
            "args": base_args + ["--use-cache", "--use-dual-cache", "--factor", "2.0"]
        },
        {
            "name": "High Steps",
            "description": "Dual cache with higher quality (256 steps)",
            "args": base_args[:] + ["--use-cache", "--use-dual-cache", "--steps", "256"]
        }
    ]
    
    print("🏁 Fast-dLLM Benchmarking Suite")
    print("=" * 60)
    print(f"Benchmark: {args.benchmark}")
    print(f"Output directory: {base_output_dir}")
    print(f"Total configurations: {len(configs)}")
    print("=" * 60)
    
    # Show configurations
    print("\n📋 Benchmark Configurations:")
    for i, config in enumerate(configs, 1):
        print(f"  {i}. {config['name']}: {config['description']}")
    print()
    
    # Run all configurations
    results = []
    total_start_time = time.time()
    
    for config in configs:
        result = run_evaluation(config['name'], config['args'], base_output_dir)
        results.append(result)
        
        # Brief pause between evaluations
        time.sleep(2)
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"Total configurations: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_duration:.1f}s")
    print()
    
    if successful:
        print("✅ Successful Configurations:")
        print("-" * 40)
        for result in successful:
            print(f"  {result['name']:20} | {result['duration']:6.1f}s | {result['output_dir']}")
        print()
    
    if failed:
        print("❌ Failed Configurations:")
        print("-" * 40)
        for result in failed:
            print(f"  {result['name']:20} | Failed after {result['duration']:6.1f}s")
        print()
    
    # Performance comparison (if we have successful results)
    if len(successful) > 1:
        print("🏎️  Performance Comparison:")
        print("-" * 40)
        # Sort by duration (fastest first)
        sorted_results = sorted(successful, key=lambda x: x['duration'])
        baseline_duration = sorted_results[-1]['duration']  # Slowest as baseline
        
        for result in sorted_results:
            speedup = baseline_duration / result['duration']
            print(f"  {result['name']:20} | {result['duration']:6.1f}s | {speedup:.2f}x speedup")
        print()
    
    print("📁 Results Location:")
    print(f"   Base directory: {base_output_dir}")
    print("   Each configuration has its own subdirectory with detailed metrics")
    print()
    
    print("🔍 Next Steps:")
    print("   1. Check individual metrics.json files for accuracy scores")
    print("   2. Compare performance vs quality trade-offs")
    print("   3. Choose optimal configuration for your use case")
    
    return len(failed) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
