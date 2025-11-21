#!/usr/bin/env python3
"""
Benchmark evaluation script using NeMo-Skills with diffusion language models via OpenAI API.

This script evaluates LLaDA or Nemotron models running on localhost:8000 (OpenAI-compatible API)
on various benchmarks using the NeMo-Skills evaluation framework.

SUPPORTED MODELS:
- LLaDA models with Fast-dLLM acceleration (generation algorithms: basic, prefix_cache, dual_cache)
- Nemotron models with native diffusion generation (generation algorithm: nemotron)

IMPORTANT: Parameter Mapping Notes:
- tokens_to_generate → max_tokens (automatically mapped by NeMo-Skills)
- top_k is forced to -1 for OpenAI API compatibility
- Model-specific parameters (steps, block_length, cfg_scale, remasking, threshold) are passed 
  via NeMo-Skills extra_body mechanism to the OpenAI API
- Generation algorithm selection determines which model optimizations are used

Usage:
    # Default GSM8K evaluation (LLaDA)
    python eval_llada.py
    
    # Evaluate with Nemotron model
    python eval_llada.py --generation-algorithm nemotron --model nemotron-4b
    
    # Evaluate on a different benchmark
    python eval_llada.py --benchmark math:2
    
    # Quick test mode
    python eval_llada.py --quick-test
    
    # Custom settings
    python eval_llada.py --benchmark gsm8k:1 --temperature 0.8 --max-samples 100
    
    # Different server/model
    python eval_llada.py --server-address http://my-server:8080/v1 --model my-model
    
    # LLaDA model with custom settings
    python eval_llada.py --generation-algorithm dual_cache --steps 128 --cfg-scale 1.5 --remasking random
    
    # LLaDA generation algorithms (Fast-dLLM acceleration)
    python eval_llada.py --generation-algorithm basic           # No caching
    python eval_llada.py --generation-algorithm prefix_cache    # Prefix caching
    python eval_llada.py --generation-algorithm dual_cache      # Dual caching (default for LLaDA)
    
    # Nemotron native generation
    python eval_llada.py --generation-algorithm nemotron        # Native Nemotron generation
    
    # Advanced settings for specific models
    python eval_llada.py --generation-algorithm dual_cache --threshold 0.8 --factor 2.0  # LLaDA Fast-dLLM
    python eval_llada.py --generation-algorithm nemotron --steps 128 --threshold 0.9      # Nemotron native
    
    # Handle truncated outputs (when model cuts off mid-reasoning)
    python eval_llada.py --keep-thinking                    # Don't remove <think> tags, extract answer from full output
    python eval_llada.py --keep-thinking --tokens-to-generate 1024  # Increase tokens + keep thinking mode
"""

import os
import argparse
from nemo_skills.pipeline.cli import eval, wrap_arguments


def create_parser():
    """Create argument parser for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLaDA/Nemotron diffusion models on various benchmarks using NeMo-Skills with OpenAI-compatible API",
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
    parser.add_argument(
        "--cluster",
        default="local",
        help="If you want to run on a cluster via nemo-skills (defaults to 'local')"
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
        help="Model identifier (e.g., llada-8b-instruct, nemotron-4b)"
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
    
    # Diffusion model parameters
    parser.add_argument(
        "--steps",
        type=int,
        default=256,
        help="Diffusion steps (1-512, higher = better quality but slower) - used by both LLaDA and Nemotron"
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=8,
        help="Block length for semi-autoregressive generation - used by both LLaDA and Nemotron"
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale (0.0-3.0) - primarily used by LLaDA models"
    )
    parser.add_argument(
        "--remasking",
        default="low_confidence",
        choices=["low_confidence", "random"],
        help="Remasking strategy - primarily used by LLaDA models"
    )
    
    # Generation algorithm selection
    parser.add_argument(
        "--generation-algorithm",
        default="dual_cache",
        choices=["basic", "prefix_cache", "dual_cache", "nemotron", "dinfer_blockwise", "dinfer_hierarchy", "dinfer_credit", "dinfer_soft"],
        help="Generation algorithm: LLaDA (basic=no cache, prefix_cache=prefix caching, dual_cache=dual caching) or Nemotron (nemotron=native generation)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Confidence threshold - for LLaDA parallel decoding (e.g., 0.8) or Nemotron generation (e.g., 0.9)"
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=None,
        help="Factor for LLaDA dynamic parallel decoding strategy (e.g., 2.0) - not used by Nemotron"
    )
    
    # Soft Token parameters
    parser.add_argument(
        "--soft-token-ratio",
        type=float,
        default=0.5,
        help="Ratio of soft tokens for dInfer Soft Token generation"
    )
    parser.add_argument(
        "--treat-soft-tokens-as-candidates",
        action="store_true",
        help="Whether to treat soft tokens as candidates for decoding (dInfer Soft Token only)"
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
    parser.add_argument(
        "--keep-thinking",
        action="store_true",
        help="Keep <think> tags in generation (don't remove them). Useful when model outputs are truncated."
    )
    
    return parser


def main():
    """Run benchmark evaluation using LLaDA or Nemotron models with configurable settings."""
    
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Adjust defaults for Nemotron models
    if args.generation_algorithm == "nemotron":
        # Suggest better defaults for Nemotron if user hasn't specified custom values
        if args.model == "llada-8b-instruct":  # Default LLaDA model name
            print("ℹ️  Note: Using Nemotron algorithm with default LLaDA model name.")
            print("   Consider using: --model nemotron-4b or similar for clarity.")
        
        # Nemotron typically works well with these defaults
        if not hasattr(args, '_threshold_set') and args.threshold is None:
            print("ℹ️  Note: Nemotron models typically use threshold=0.9 for good results.")
            print("   Add --threshold 0.9 to optimize Nemotron generation.")
    
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
        "cluster": None if args.cluster == "local" else args.cluster,
        "dry_run": args.dry_run,
        
        # Inference settings
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "tokens_to_generate": args.tokens_to_generate,
        "max_samples": args.max_samples,
        "quick_test": args.quick_test,
        "keep_thinking": args.keep_thinking,
        
        # LLaDA-specific settings
        "steps": args.steps,
        "block_length": args.block_length,
        "cfg_scale": args.cfg_scale,
        "remasking": args.remasking,
        
        # Generation algorithm selection
        "generation_algorithm": getattr(args, 'generation_algorithm', 'dual_cache'),
        "threshold": args.threshold,
        "factor": args.factor,
        "soft_token_ratio": args.soft_token_ratio,
        "treat_soft_tokens_as_candidates": args.treat_soft_tokens_as_candidates,
    }
    
    # Set default experiment name if not provided
    if config["expname"] is None:
        benchmark_name = args.benchmark.split(":")[0]  # Extract benchmark name (e.g., "gsm8k" from "gsm8k:4")
        model_type = "nemotron" if config["generation_algorithm"] == "nemotron" else "llada"
        config["expname"] = f"{model_type}-{benchmark_name}-eval"
    
    # Override with environment variables if needed (for backward compatibility)
    config["output_dir"] = os.environ.get("EVAL_OUTPUT_DIR", config["output_dir"])
    config["server_address"] = os.environ.get("LLADA_SERVER_URL", config["server_address"])
    
    # Create output directory if it doesn't exist
    os.makedirs(config["output_dir"], exist_ok=True)
    
    print("=" * 60)
    benchmark_name = config["benchmarks"].split(":")[0].upper()
    model_type = "Nemotron" if config["generation_algorithm"] == "nemotron" else "LLaDA"
    print(f"{benchmark_name} Evaluation with {model_type} Model")
    print("=" * 60)
    print(f"Server: {config['server_address']}")
    print(f"Model: {config['model']}")
    print(f"Benchmark: {config['benchmarks']}")
    print(f"Output: {config['output_dir']}")
    print(f"Experiment: {config['expname']}")
    print(f"Temperature: {config['temperature']} | Top-p: {config['top_p']} | Top-k: {config['top_k']} (will be set to -1)")
    print(f"Max tokens: {config['tokens_to_generate']}")
    print(f"Generation Algorithm: {config['generation_algorithm']}")
    
    # Show model-specific parameters
    if config["generation_algorithm"] == "nemotron":
        print(f"Nemotron Steps: {config['steps']} | Block length: {config['block_length']}")
        if config['threshold'] is not None:
            print(f"Nemotron Threshold: {config['threshold']}")
        print("CFG scale and remasking: Not used by Nemotron")
        if config['factor'] is not None:
            print("⚠️  Factor parameter not used by Nemotron (LLaDA-specific)")
    else:
        print(f"LLaDA Steps: {config['steps']} | Block length: {config['block_length']}")
        print(f"CFG scale: {config['cfg_scale']} | Remasking: {config['remasking']}")
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
            # Set generation_key to 'predicted_answer' to match what evaluation expects
            #f"++generation_key=predicted_answer",
            f"++num_chunks=2"
        ]
        
        # Add optional Fast-dLLM parameters if specified
        if config['threshold'] is not None:
            generation_args.append(f"++inference.extra_body.threshold={config['threshold']}")
        if config['factor'] is not None:
            generation_args.append(f"++inference.extra_body.factor={config['factor']}")
        
        # Add Soft Token parameters
        if config['generation_algorithm'] == "dinfer_soft":
            generation_args.append(f"++inference.extra_body.soft_token_ratio={config['soft_token_ratio']}")
            generation_args.append(f"++inference.extra_body.treat_soft_tokens_as_candidates={config['treat_soft_tokens_as_candidates']}")
        
        model_type_display = "Nemotron" if config['generation_algorithm'] == "nemotron" else "LLaDA"
        print(f"\n🔧 {model_type_display} generation parameters (via extra_body):")
        print(f"  steps={config['steps']}")
        print(f"  block_length={config['block_length']}")
        
        if config['generation_algorithm'] == "nemotron":
            print(f"  generation_algorithm={config['generation_algorithm']} (Nemotron native)")
            if config['threshold'] is not None:
                print(f"  threshold={config['threshold']} (Nemotron generation threshold)")
            if config['cfg_scale'] != 0.0 or config['remasking'] != "low_confidence":
                print("  ⚠️  Note: cfg_scale and remasking are not used by Nemotron")
            if config['factor'] is not None:
                print("  ⚠️  Note: factor is not used by Nemotron (LLaDA-specific)")
        elif config['generation_algorithm'] == "dinfer_soft":
            print(f"  generation_algorithm={config['generation_algorithm']} (dInfer Soft Token)")
            print(f"  soft_token_ratio={config['soft_token_ratio']}")
            print(f"  treat_soft_tokens_as_candidates={config['treat_soft_tokens_as_candidates']}")
        else:
            print(f"  cfg_scale={config['cfg_scale']}")
            print(f"  remasking={config['remasking']}")
            print(f"  generation_algorithm={config['generation_algorithm']} (LLaDA Fast-dLLM)")
            if config['threshold'] is not None:
                print(f"  threshold={config['threshold']} (Fast-dLLM parallel decoding)")
            if config['factor'] is not None:
                print(f"  factor={config['factor']} (Fast-dLLM dynamic decoding)")
        
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
        
        # Add keep_thinking parameter to generation_args if needed
        if config["keep_thinking"]:
            generation_args.append("++remove_thinking=False")
            print("\n⚠️  Keep-thinking mode enabled: <think> tags will NOT be removed from generations")
            print("   This helps when model outputs are truncated and missing </think> tags")
        
        #print("#### generation_args: ", str(wrap_arguments(" ".join(generation_args))), flush=True)
        #exit(2)
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
        print("1. Make sure your diffusion model server (LLaDA/Nemotron) is running on localhost:8000")
        print("2. Test the server with: curl http://localhost:8000/v1/models")
        print("3. Check server logs for any errors")
        print("4. Verify the server is OpenAI-compatible")
        print("5. Verify generation algorithm matches your model type:")
        print("   - LLaDA models: use --generation-algorithm dual_cache (or basic/prefix_cache)")
        print("   - Nemotron models: use --generation-algorithm nemotron")
        print("6. For quick testing, use: python eval_llada.py --quick-test")
        raise


if __name__ == "__main__":
    main()