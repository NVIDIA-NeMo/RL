#!/usr/bin/env python3
"""
Server Comparison and Recommendation Script

This script helps you choose between the batch server and streaming server
based on your use case and system specifications.
"""

import argparse
import subprocess
import sys
import psutil
import GPUtil

def get_gpu_info():
    """Get GPU information."""
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return None, 0
        
        total_memory = sum(gpu.memoryTotal for gpu in gpus)
        gpu_count = len(gpus)
        return gpus, total_memory, gpu_count
    except Exception:
        try:
            # Fallback to nvidia-smi
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                memory_values = [int(x.strip()) for x in result.stdout.strip().split('\n')]
                total_memory = sum(memory_values)
                gpu_count = len(memory_values)
                return None, total_memory, gpu_count
        except Exception:
            pass
    return None, 0, 0

def recommend_batch_size(gpu_memory_gb, use_case):
    """Recommend batch size based on available GPU memory and use case."""
    if gpu_memory_gb < 16:
        return 2, "Conservative (limited GPU memory)"
    elif gpu_memory_gb < 32:
        return 4, "Balanced (moderate GPU memory)"
    elif gpu_memory_gb < 64:
        if use_case == "evaluation":
            return 8, "Recommended (good balance)"
        else:
            return 4, "Conservative for interactive use"
    else:
        if use_case == "evaluation":
            return 16, "High throughput (plenty of memory)"
        else:
            return 8, "High performance for interactive use"

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def print_section(text):
    """Print a formatted section."""
    print(f"\n🔹 {text}")
    print("-" * 60)

def main():
    parser = argparse.ArgumentParser(description="LLaDA Server Comparison and Recommendation")
    parser.add_argument("--use-case", choices=["evaluation", "interactive", "demo"], 
                       default="evaluation", help="Your primary use case")
    parser.add_argument("--show-commands", action="store_true", 
                       help="Show recommended commands to start servers")
    
    args = parser.parse_args()
    
    print_header("LLaDA Server Comparison & Recommendation Tool")
    
    # Get system information
    print_section("System Analysis")
    
    # CPU info
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    print(f"CPU Cores: {cpu_count}")
    print(f"System RAM: {memory_gb:.1f} GB")
    
    # GPU info
    gpus, gpu_memory_mb, gpu_count = get_gpu_info()
    if gpu_count > 0:
        gpu_memory_gb = gpu_memory_mb / 1024
        print(f"GPUs: {gpu_count} available")
        print(f"Total GPU Memory: {gpu_memory_gb:.1f} GB")
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name} ({gpu.memoryTotal} MB)")
    else:
        gpu_memory_gb = 0
        print("GPUs: None detected (CPU-only mode)")
    
    # Use case analysis
    print_section("Use Case Analysis")
    print(f"Primary use case: {args.use_case}")
    
    if args.use_case == "evaluation":
        print("• Focus: Maximum throughput for benchmark evaluations")
        print("• Typical workload: Many requests, batch processing preferred")
        print("• Latency tolerance: Medium-high (batching acceptable)")
    elif args.use_case == "interactive":
        print("• Focus: Real-time responses for interactive sessions")
        print("• Typical workload: Single requests, immediate responses")
        print("• Latency tolerance: Low (need fast responses)")
    else:  # demo
        print("• Focus: Demonstration and testing purposes")
        print("• Typical workload: Mixed, occasional requests")
        print("• Latency tolerance: Medium (balanced approach)")
    
    # Server comparison
    print_section("Server Comparison")
    
    print("BATCH SERVER:")
    print("  ✅ 3-5x faster throughput")
    print("  ✅ Automatic request batching")
    print("  ✅ GPU memory efficient for multiple requests")
    print("  ✅ Perfect for evaluations and benchmarks")
    print("  ❌ Higher latency per request")
    print("  ❌ No streaming support")
    print("  💾 Memory usage: batch_size × model_size")
    
    print("\nSTREAMING SERVER:")
    print("  ✅ Low latency per request")
    print("  ✅ Real-time streaming responses")
    print("  ✅ Lower memory usage")
    print("  ✅ Better for interactive use")
    print("  ❌ Sequential processing only")
    print("  ❌ Lower overall throughput")
    print("  💾 Memory usage: 1 × model_size")
    
    # Recommendations
    print_section("Recommendations")
    
    if args.use_case == "evaluation":
        recommended_server = "BATCH"
        reason = "Evaluations benefit greatly from batch processing"
        batch_size, batch_reason = recommend_batch_size(gpu_memory_gb, args.use_case)
        
    elif args.use_case == "interactive":
        if gpu_memory_gb >= 32:
            recommended_server = "BATCH"
            reason = "Plenty of memory allows efficient batching even for interactive use"
            batch_size, batch_reason = recommend_batch_size(gpu_memory_gb, args.use_case)
        else:
            recommended_server = "STREAMING"
            reason = "Limited memory + interactive use favors streaming"
            batch_size, batch_reason = 4, "Fallback recommendation"
    else:  # demo
        recommended_server = "BATCH"
        reason = "Demo use can benefit from batch processing with moderate settings"
        batch_size, batch_reason = recommend_batch_size(gpu_memory_gb, args.use_case)
    
    print(f"🎯 RECOMMENDED SERVER: {recommended_server}")
    print(f"   Reason: {reason}")
    
    if recommended_server == "BATCH":
        print(f"🔧 Recommended batch size: {batch_size}")
        print(f"   Reason: {batch_reason}")
        
        # Memory warning
        estimated_memory = batch_size * 8  # Rough estimate: 8GB per request
        if estimated_memory > gpu_memory_gb * 0.8:
            print(f"⚠️  Warning: Estimated memory usage ({estimated_memory:.1f} GB) might exceed available GPU memory")
            print(f"   Consider reducing batch size to {max(2, int(gpu_memory_gb * 0.6 / 8))}")
    
    # Performance expectations
    print_section("Expected Performance")
    
    if gpu_count == 0:
        print("⚠️  CPU-only mode: Expect significantly slower performance")
        print("   Recommendation: Use smaller models or GPU if possible")
    elif gpu_memory_gb < 16:
        print("⚠️  Limited GPU memory: Performance may be constrained")
        print("   Recommendation: Use smaller batch sizes or consider model optimization")
    elif gpu_memory_gb >= 64:
        print("🚀 High-end GPU setup: Excellent performance expected")
        print("   Can handle large batch sizes and high throughput")
    else:
        print("✅ Good GPU setup: Solid performance expected")
        print("   Can handle moderate to good batch sizes")
    
    # Show commands if requested
    if args.show_commands:
        print_section("Recommended Commands")
        
        if recommended_server == "BATCH":
            print("🚀 BATCH SERVER (Recommended):")
            print(f"./xp/llada_api/scripts/start_llada_batch_server.sh \\")
            print(f"  --local \\")
            print(f"  --model-path GSAI-ML/LLaDA-8B-Instruct \\")
            print(f"  --batch-size {batch_size}")
            
            if args.use_case == "evaluation":
                print(f"\n# For maximum throughput (if memory allows):")
                print(f"./xp/llada_api/scripts/start_llada_batch_server.sh \\")
                print(f"  --local \\")
                print(f"  --model-path GSAI-ML/LLaDA-8B-Instruct \\")
                print(f"  --batch-size {min(16, batch_size * 2)} \\")
                print(f"  --max-wait-time 0.05")
            
        else:
            print("🌊 STREAMING SERVER (Recommended):")
            print(f"./xp/llada_api/scripts/start_llada_batch_server.sh \\")
            print(f"  --local \\")
            print(f"  --streaming \\")
            print(f"  --model-path GSAI-ML/LLaDA-8B-Instruct")
        
        print(f"\n# Alternative server:")
        if recommended_server == "BATCH":
            print("./xp/llada_api/scripts/start_llada_batch_server.sh --local --streaming --model-path GSAI-ML/LLaDA-8B-Instruct")
        else:
            print(f"./xp/llada_api/scripts/start_llada_batch_server.sh --local --model-path GSAI-ML/LLaDA-8B-Instruct --batch-size {batch_size}")
    
    # Testing recommendation
    print_section("Testing & Validation")
    print("1. Start your recommended server")
    print("2. Test basic functionality:")
    print("   curl http://localhost:8000/health")
    if recommended_server == "BATCH":
        print("   curl http://localhost:8000/batch/stats")
        print("3. Run performance test:")
        print("   python xp/llada_api/test_batch_server.py")
    else:
        print("3. Test streaming:")
        print("   python xp/llada_api/examples/llada_api_client.py")
    print("4. Run your actual workload:")
    print("   python xp/nemo-skills/eval_llada.py --quick-test")
    
    print_header("Summary")
    print(f"✅ Recommended: {recommended_server} server")
    if recommended_server == "BATCH":
        print(f"⚙️  Batch size: {batch_size}")
        print("🎯 Expected: 3-5x speedup for evaluation workloads")
    else:
        print("🎯 Expected: Lower latency, streaming responses")
    print("🚀 Ready to get started!")

if __name__ == "__main__":
    main()
