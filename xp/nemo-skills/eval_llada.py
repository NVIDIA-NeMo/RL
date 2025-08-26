#!/usr/bin/env python3
"""
Benchmark evaluation script using NeMo-Skills with LLaDA model via OpenAI API.

This script evaluates a LLaDA model running on localhost:8000 (OpenAI-compatible API)
on various benchmarks using the NeMo-Skills evaluation framework.

IMPORTANT: Parameter Mapping Notes:
- tokens_to_generate → max_tokens (automatically mapped by NeMo-Skills)
- top_k is forced to -1 for OpenAI API compatibility
- LLaDA-specific parameters (steps, block_length, cfg_scale, remasking) are passed 
  via NeMo-Skills extra_body mechanism to the OpenAI API
- Generation algorithm selection (generation_algorithm, threshold, factor) 
  are passed via NeMo-Skills extra_body mechanism for optimized generation

Usage:
    # Default GSM8K evaluation
    python eval_llada.py
    
    # Evaluate on a different benchmark
    python eval_llada.py --benchmark math:2
    
    # Quick test mode
    python eval_llada.py --quick-test
    
    # Custom settings
    python eval_llada.py --benchmark gsm8k:1 --temperature 0.8 --max-samples 100
    
    # Different server/model
    python eval_llada.py --server-address http://my-server:8080/v1 --model my-model
    
    # Custom LLaDA settings
    python eval_llada.py --steps 128 --cfg-scale 1.5 --remasking random
    
    # Different generation algorithms
    python eval_llada.py --generation-algorithm basic           # No caching
    python eval_llada.py --generation-algorithm prefix_cache    # Prefix caching
    python eval_llada.py --generation-algorithm dual_cache      # Dual caching (default)
    
    # Advanced Fast-dLLM settings
    python eval_llada.py --generation-algorithm dual_cache --threshold 0.8 --factor 2.0
"""

import os
import argparse
from nemo_skills.pipeline.cli import eval, wrap_arguments


def create_parser():
    """Create argument parser for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate models on various benchmarks using NeMo-Skills with OpenAI-compatible API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Benchmark and evaluation settings
    parser.add_argument(
        "--benchmark", 
        default="gsm8k:4",
        help="Benchmark to evaluate on (format: benchmark_name:num_samples)"
    )
    parser.add_argument(
        "--output-dir", 
        default=".",
        help="Directory to store evaluation results"
    )
    parser.add_argument(
        "--expname",
        default=None,
        help="Experiment name (defaults to 'llada-{benchmark}-eval')"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of problems to evaluate (for quick testing)"
    )
    
    # Server configuration
    parser.add_argument(
        "--server-address",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible server endpoint"
    )
    parser.add_argument(
        "--model",
        default="llada-8b-instruct", 
        help="Model identifier"
    )
    
    # Inference settings
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--top-k", 
        type=int,
        default=-1,
        help="Top-k sampling parameter (will be forced to -1 for OpenAI API compatibility)"
    )
    parser.add_argument(
        "--tokens-to-generate",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    
    # LLaDA-specific parameters
    parser.add_argument(
        "--steps",
        type=int,
        default=128,
        help="LLaDA diffusion steps (1-512, higher = better quality but slower)"
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=32,
        help="LLaDA block length for semi-autoregressive generation"
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=0.0,
        help="LLaDA classifier-free guidance scale (0.0-3.0)"
    )
    parser.add_argument(
        "--remasking",
        default="low_confidence",
        choices=["low_confidence", "random"],
        help="LLaDA remasking strategy"
    )
    
    # Generation algorithm selection
    parser.add_argument(
        "--generation-algorithm",
        default="dual_cache",
        choices=["basic", "prefix_cache", "dual_cache"],
        help="Generation algorithm to use (basic=no cache, prefix_cache=prefix caching, dual_cache=dual caching)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Confidence threshold for parallel decoding (e.g., 0.8)"
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=None,
        help="Factor for dynamic parallel decoding strategy (e.g., 2.0)"
    )
    
    # Execution settings
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without actual evaluation"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true", 
        help="Run quick test mode (10 problems, single sample, overrides some settings)"
    )
    
    return parser


def main():
    """Run benchmark evaluation using LLaDA model with configurable settings."""
    
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Build configuration from parsed arguments
    config = {
        # Evaluation settings
        "benchmarks": args.benchmark,
        "output_dir": args.output_dir,
        "expname": args.expname,
        
        # Server configuration for external OpenAI API
        "server_type": "openai",  # Use OpenAI-compatible server
        "server_address": args.server_address,
        "model": args.model,
        
        # Additional evaluation arguments
        "cluster": None,  # Use local execution (no SLURM cluster)
        "dry_run": args.dry_run,
        
        # Inference settings
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "tokens_to_generate": args.tokens_to_generate,
        "max_samples": args.max_samples,
        "quick_test": args.quick_test,
        
        # LLaDA-specific settings
        "steps": args.steps,
        "block_length": args.block_length,
        "cfg_scale": args.cfg_scale,
        "remasking": args.remasking,
        
        # Generation algorithm selection
        "generation_algorithm": getattr(args, 'generation_algorithm', 'dual_cache'),
        "threshold": args.threshold,
        "factor": args.factor,
    }
    
    # Set default experiment name if not provided
    if config["expname"] is None:
        benchmark_name = args.benchmark.split(":")[0]  # Extract benchmark name (e.g., "gsm8k" from "gsm8k:4")
        config["expname"] = f"llada-{benchmark_name}-eval"
    
    # Override with environment variables if needed (for backward compatibility)
    config["output_dir"] = os.environ.get("EVAL_OUTPUT_DIR", config["output_dir"])
    config["server_address"] = os.environ.get("LLADA_SERVER_URL", config["server_address"])
    
    # Create output directory if it doesn't exist
    os.makedirs(config["output_dir"], exist_ok=True)
    
    print("=" * 60)
    benchmark_name = config["benchmarks"].split(":")[0].upper()
    print(f"{benchmark_name} Evaluation with LLaDA Model")
    print("=" * 60)
    print(f"Server: {config['server_address']}")
    print(f"Model: {config['model']}")
    print(f"Benchmark: {config['benchmarks']}")
    print(f"Output: {config['output_dir']}")
    print(f"Experiment: {config['expname']}")
    print(f"Temperature: {config['temperature']} | Top-p: {config['top_p']} | Top-k: {config['top_k']} (will be set to -1)")
    print(f"Max tokens: {config['tokens_to_generate']}")
    print(f"LLaDA Steps: {config['steps']} | Block length: {config['block_length']}")
    print(f"CFG scale: {config['cfg_scale']} | Remasking: {config['remasking']}")
    print(f"Generation Algorithm: {config['generation_algorithm']}")
    if config['threshold'] is not None:
        print(f"Fast-dLLM Threshold: {config['threshold']}")
    if config['factor'] is not None:
        print(f"Fast-dLLM Factor: {config['factor']}")
    if config["max_samples"]:
        print(f"Max samples: {config['max_samples']} (for testing)")
    print("=" * 60)
    
    if config["quick_test"] or os.environ.get("EVAL_QUICK_TEST") == "1":
        print("🚀 QUICK TEST MODE enabled")
    else:
        print("⚠️  This evaluation may take some time to complete depending on the benchmark.")
    print("   Use --quick-test for a faster test run, or --max-samples N to limit problems.")
    
    # Run evaluation using NeMo-Skills
    try:
        # Only pass generation-specific arguments through wrap_arguments
        # These will be forwarded to the underlying generation script
        generation_args = [
            f"++inference.temperature={config['temperature']}",
            f"++inference.top_p={config['top_p']}", 
            f"++inference.top_k=-1",  # Must be -1 for OpenAI API compatibility
            f"++inference.tokens_to_generate={config['tokens_to_generate']}",
            # Pass LLaDA-specific parameters via extra_body (NeMo-Skills will include these in the OpenAI API request)
            f"++inference.extra_body.steps={config['steps']}",
            f"++inference.extra_body.block_length={config['block_length']}",
            f"++inference.extra_body.cfg_scale={config['cfg_scale']}",
            f"++inference.extra_body.remasking={config['remasking']}",
            # Pass generation algorithm selection via extra_body
            f"++inference.extra_body.generation_algorithm={config['generation_algorithm']}",
            f"++num_chunks=2"
        ]
        
        # Add optional Fast-dLLM parameters if specified
        if config['threshold'] is not None:
            generation_args.append(f"++inference.extra_body.threshold={config['threshold']}")
        if config['factor'] is not None:
            generation_args.append(f"++inference.extra_body.factor={config['factor']}")
        
        print(f"\n🔧 LLaDA generation parameters (via extra_body):")
        print(f"  steps={config['steps']}")
        print(f"  block_length={config['block_length']}")
        print(f"  cfg_scale={config['cfg_scale']}")
        print(f"  remasking={config['remasking']}")
        print(f"  generation_algorithm={config['generation_algorithm']}")
        if config['threshold'] is not None:
            print(f"  threshold={config['threshold']}")
        if config['factor'] is not None:
            print(f"  factor={config['factor']}")
        print("   (Passed via NeMo-Skills extra_body to OpenAI API)")
        
        # Add max_samples if specified
        if config["max_samples"]:
            generation_args.append(f"++max_samples={config['max_samples']}")
        
        # Quick test mode for development/testing (can be enabled via --quick-test or environment variable)
        if config["quick_test"] or os.environ.get("EVAL_QUICK_TEST") == "1":
            # Override benchmark to single sample if not already set
            if not config["max_samples"]:
                benchmark_name = config["benchmarks"].split(":")[0]
                config["benchmarks"] = f"{benchmark_name}:1"  # Single sample
                generation_args.append("++max_samples=10")  # Only 10 problems
            print(f"\n🚀 QUICK TEST MODE: Running with {config['benchmarks']} and limited samples")
        
        # Call the evaluation function with direct parameters
        result = eval(
            ctx=wrap_arguments(" ".join(generation_args)),
            # Core parameters
            benchmarks=config["benchmarks"],
            output_dir=config["output_dir"],
            expname=config["expname"],
            
            # Server configuration  
            server_type=config["server_type"],
            server_address=config["server_address"],
            model=config["model"],
            
            # Optional parameters
            cluster=config["cluster"],
            dry_run=config["dry_run"],
        )
        
        print("\n" + "=" * 60)
        if config["dry_run"]:
            print("DRY RUN COMPLETED - No actual evaluation was performed")
        else:
            print("EVALUATION COMPLETED SUCCESSFULLY!")
            benchmark_name = config["benchmarks"].split(":")[0]
            print(f"Results saved to: {config['output_dir']}")
            print(f"📊 Final metrics: {config['output_dir']}/eval-results/{benchmark_name}/metrics.json")
            print(f"📄 Detailed outputs: {config['output_dir']}/eval-results/{benchmark_name}/output-rs*.jsonl")
            print("🎯 Check the metrics.json file for accuracy scores and detailed statistics.")
        print("=" * 60)
        
        return result
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your LLaDA server is running on localhost:8000")
        print("2. Test the server with: curl http://localhost:8000/v1/models")
        print("3. Check server logs for any errors")
        print("4. Verify the server is OpenAI-compatible")
        print("5. For quick testing, use: python eval_llada.py --quick-test")
        raise


if __name__ == "__main__":
    main()