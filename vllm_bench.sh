#!/bin/bash
# ============================================================
# vLLM Standalone Benchmark - Quick Access Script
# ============================================================
# This is a convenience wrapper for the vLLM benchmark suite.
#
# Usage:
#   ./vllm_bench.sh                    # Show help
#   ./vllm_bench.sh list               # List model configurations
#   ./vllm_bench.sh sweep [--dry-run]  # Run benchmark sweep
#   ./vllm_bench.sh run MODEL          # Run single model benchmark
#   ./vllm_bench.sh results [--all]    # Show results
#
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_BENCH_DIR="$SCRIPT_DIR/vllm_benchmark"

case "${1:-help}" in
    list|ls)
        "$VLLM_BENCH_DIR/benchmark_sweep.sh" --list
        ;;
    
    sweep)
        shift
        "$VLLM_BENCH_DIR/benchmark_sweep.sh" "$@"
        ;;
    
    run)
        shift
        MODEL="$1" "$VLLM_BENCH_DIR/run_vllm_benchmark.sh" run-throughput
        ;;
    
    run-offline)
        shift
        MODEL="$1" "$VLLM_BENCH_DIR/run_vllm_benchmark.sh" run-random
        ;;
    
    run-online)
        shift
        MODEL="$1" "$VLLM_BENCH_DIR/run_vllm_benchmark.sh" run-online
        ;;
    
    results|res)
        shift
        python3 "$VLLM_BENCH_DIR/collect_results.py" --throughput "$@"
        ;;
    
    results-offline)
        shift
        python3 "$VLLM_BENCH_DIR/collect_results.py" "$@"
        ;;
    
    results-online)
        shift
        python3 "$VLLM_BENCH_DIR/collect_results.py" --online "$@"
        ;;
    
    help|--help|-h|*)
        echo "vLLM Standalone Benchmark Suite"
        echo ""
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  list                  List available model configurations"
        echo "  sweep [--dry-run]     Run benchmark sweep for all models"
        echo "  run MODEL             Run throughput benchmark for a model"
        echo "  run-offline MODEL     Run offline (batch) benchmark"
        echo "  run-online MODEL      Run online (server) benchmark"
        echo ""
        echo "  results [--all]       Show throughput benchmark results"
        echo "  results-offline       Show offline benchmark results"
        echo "  results-online        Show online benchmark results"
        echo ""
        echo "Model presets:"
        echo "  Dense:  llama8b, llama70b, qwen3-32b"
        echo "  MoE:    qwen3-30b, qwen3-235b, deepseek-v3"
        echo ""
        echo "Examples:"
        echo "  $0 list                       # Show all configurations"
        echo "  $0 sweep --dry-run            # Preview benchmark sweep"
        echo "  $0 sweep --model qwen3        # Run only Qwen3 models"
        echo "  $0 run qwen3-32b              # Run single model"
        echo "  $0 results                    # View results"
        echo ""
        echo "For advanced usage:"
        echo "  cd vllm_benchmark && ./run_vllm_benchmark.sh help"
        ;;
esac

