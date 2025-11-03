import asyncio
import time
import torch
import traceback
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import math
import random

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Add the RL directory to the path
rl_dir = Path(__file__).parent / "RL"
sys.path.insert(0, str(rl_dir))

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import GenerationDatumSpec, GenerationOutputSpec
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.generation.dynamo.dynamo_generation import DynamoGeneration

@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    name: str
    batch_size: int
    seq_len: int
    max_new_tokens: int
    num_iterations: int = 5
    router_config: Optional[Dict[str, Any]] = None
    use_april_router: bool = False  # Use APRIL router for over-provisioning
    over_sampling_batch_size: Optional[int] = None
    
@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    config_name: str
    batch_size: int
    total_samples: int
    total_time: float
    avg_latency: float
    throughput: float
    min_latency: float
    max_latency: float
    errors: int = 0
    avg_sample_length: float = 0
    max_sample_length: float = 0


# Global temporary numpy dataset for consistent benchmarking
TEMP_BENCHMARK_DATASET = None


def create_temp_benchmark_dataset(max_samples: int = 256, max_seq_length: int = 4096, seed: int = 42):
    """
    Create a temporary numpy dataset for consistent benchmarking across all configs.
    
    Args:
        max_samples: Maximum number of samples to generate (should cover largest batch size)
        max_seq_length: Maximum sequence length for the dataset
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with numpy arrays for input_ids and input_lengths
    """
    print(f"🔧 Creating temporary benchmark dataset: {max_samples} samples, {max_seq_length} max length")
    np.random.seed(seed)
    
    # Generate input_ids
    input_ids = np.random.randint(1, 1000, size=(max_samples, max_seq_length), dtype=np.int64)
    # Generate varying input lengths for each sample
    base_length = 150  # Default base length
    input_lengths = np.array([max(1, base_length - (i % 3)) for i in range(max_samples)], dtype=np.int64)
    
    # Generate output lengths using log-normal distribution for realistic variable-length outputs
    mu = 7.0
    sigma = 1.1
    random.seed(seed)
    remaining_ctx_list = np.array([
        int(min(max_seq_length, math.exp(random.gauss(mu, sigma))))
        for _ in range(max_samples)
    ], dtype=np.int64)
    
    dataset = {
        "input_ids": input_ids,
        "input_lengths": input_lengths,
        "remaining_ctx_list": remaining_ctx_list,
    }
    
    print(f"✅ Temporary dataset created:")
    print(f"   - input_ids shape: {input_ids.shape}")
    print(f"   - input_lengths shape: {input_lengths.shape}")
    print(f"   - remaining_ctx_list shape: {remaining_ctx_list.shape}, mean: {remaining_ctx_list.mean():.1f}, std: {remaining_ctx_list.std():.1f}")
    return dataset


def create_test_data(batch_size: int, max_seq_length: int, input_seq_len: int, aborted_samples: List[BatchedDataDict[GenerationOutputSpec]] = [], device: str = "cpu", finish_remaining=False, sample_offset: int = 0):
    """Create test input data using the temporary benchmark dataset."""
    global TEMP_BENCHMARK_DATASET
    
    assert max_seq_length is not None, "max_seq_length must be provided"
    assert max_seq_length >= input_seq_len, f"max_seq_length ({max_seq_length}) must be >= input_seq_len ({input_seq_len})"
    assert TEMP_BENCHMARK_DATASET is not None, "Temporary benchmark dataset not initialized. Call create_temp_benchmark_dataset() first."

    if finish_remaining and aborted_samples:
        # Only processing remaining aborted samples - don't create new samples
        remaining_ctx_list = []
        input_ids = None
        input_lengths = None
    else:
        # Use data from the temporary benchmark dataset instead of random generation
        dataset_input_ids = TEMP_BENCHMARK_DATASET["input_ids"]
        dataset_input_lengths = TEMP_BENCHMARK_DATASET["input_lengths"]
        dataset_output_lengths = TEMP_BENCHMARK_DATASET["remaining_ctx_list"]
        
        # Ensure we have enough samples in the dataset
        assert sample_offset + batch_size <= len(dataset_input_ids), \
            f"Not enough samples in dataset. Need {sample_offset + batch_size}, have {len(dataset_input_ids)}"
        
        # Extract the batch from the dataset
        batch_input_ids = dataset_input_ids[sample_offset:sample_offset + batch_size, :input_seq_len]
        batch_input_lengths = dataset_input_lengths[sample_offset:sample_offset + batch_size]
        remaining_ctx_list = dataset_output_lengths[sample_offset:sample_offset + batch_size].tolist()
        
        # Convert numpy arrays to torch tensors
        input_ids = torch.from_numpy(batch_input_ids).to(device)
        input_lengths = torch.from_numpy(batch_input_lengths).to(device)
        
        # Adjust input_lengths if they exceed input_seq_len
        input_lengths = torch.clamp(input_lengths, max=input_seq_len)
        
        # Pad new input_ids to max_seq_length (no truncation needed since max_seq_length >= input_seq_len)
        if input_ids.shape[1] < max_seq_length:
            padding = torch.zeros((input_ids.shape[0], max_seq_length - input_ids.shape[1]), dtype=input_ids.dtype, device=device)
            input_ids = torch.cat([input_ids, padding], dim=1)
    
    # Combine with aborted samples if they exist
    if aborted_samples:
        aborted_samples_list = []
        aborted_lengths_list = []
        
        for sample in aborted_samples:
            sample_ids = sample["output_ids"]
            sample_length = sample["unpadded_sequence_lengths"]
            
            # Pad or truncate to match max_seq_length
            if sample_ids.shape[1] < max_seq_length:
                # Pad with zeros
                padding = torch.zeros((sample_ids.shape[0], max_seq_length - sample_ids.shape[1]), dtype=sample_ids.dtype, device=device)
                sample_ids = torch.cat([sample_ids, padding], dim=1)
            
            aborted_samples_list.append(sample_ids)
            aborted_lengths_list.append(sample_length)
            
            # Collect remaining_ctx for aborted samples
            remaining_ctx_list.append(sample["out_remaining_len"])
        
        aborted_input_ids = torch.cat(aborted_samples_list, dim=0)
        aborted_input_lengths = torch.cat(aborted_lengths_list, dim=0)
        
        if finish_remaining:
            input_ids = aborted_input_ids
            input_lengths = aborted_input_lengths
        else:
            input_ids = torch.cat([input_ids, aborted_input_ids], dim=0)
            input_lengths = torch.cat([input_lengths, aborted_input_lengths], dim=0)
    
    print(f"Final input_ids size: {input_ids.size()}")
    print(f"Final input_lengths size: {input_lengths.size()}")
    
    return BatchedDataDict[GenerationDatumSpec]({
        "input_ids": input_ids,
        "input_lengths": input_lengths,
        "remaining_ctx_list": remaining_ctx_list,
    })


async def run_benchmark(dynamo_gen: DynamoGeneration, bench_config: BenchmarkConfig) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    print(f"\n📊 Running benchmark: {bench_config.name}")
    print(f"   Batch size: {bench_config.batch_size}, Seq len: {bench_config.seq_len}")
    print(f"   Max new tokens: {bench_config.max_new_tokens}, Iterations: {bench_config.num_iterations}")
    if bench_config.use_april_router:
        print(f"   🚀 Using APRIL Router (over-sampling batch size factor: {bench_config.over_sampling_batch_size}/{bench_config.batch_size})")
    
    latencies = []
    errors = 0
    total_samples = 0
    avg_sample_length = 0
    max_sample_length = 0
    aborted_samples = []
    sample_offset = 0
    
    for iteration in range(bench_config.num_iterations):
        try:
            # Create test data for this iteration
            # Use sample_offset=0 to ensure all iterations and configs use the same dataset samples
            if bench_config.use_april_router:
                batch_size = bench_config.over_sampling_batch_size
                if aborted_samples:
                    batch_size = batch_size - len(aborted_samples)
            else:
                batch_size = bench_config.batch_size
            data = create_test_data(
                batch_size, 
                max_seq_length=bench_config.max_new_tokens, 
                input_seq_len=bench_config.seq_len, 
                aborted_samples=aborted_samples, 
                finish_remaining=(iteration == bench_config.num_iterations - 1) and bench_config.use_april_router,
                sample_offset=sample_offset  # Always use the same samples from the temporary dataset
            )

            # Time the generation
            start_time = time.time()
            results = []
            aborted_samples = []

            end_time = None

            if bench_config.use_april_router:
                async for sample_idx, result_batch in dynamo_gen.generate_async(data, greedy=True):
                    results.append((sample_idx, result_batch))
                    if len(results) == bench_config.batch_size:
                        dynamo_gen.stop_generation()
                        
                end_time = time.time()
                iteration_time = end_time - start_time

                instance_stats = {}
                print(f"Sanity check result length: {len(results)}")
                for _, result_batch in results:
                    is_aborted = False

                    if "stop_reason" in result_batch and result_batch["stop_reason"] == "abort":
                        is_aborted = True
                        aborted_samples.append(result_batch)


                    # Worker statistics
                    instance_id = result_batch.get("selected_instance", None)
                    if instance_id is None:
                        print("lol no instance_id")
                        print(f"Result batch: {result_batch}")
                        continue

                    if instance_id not in instance_stats:
                        instance_stats[instance_id] = {
                            "completed_count": 0,
                            "aborted_total_tokens": 0,
                            "aborted_count": 0,
                            "completed_total_tokens": 0,
                            "completed_tokens": 0,
                            "aborted_tokens": 0,
                        }

                    if is_aborted:
                        instance_stats[instance_id]["aborted_count"] += 1
                        instance_stats[instance_id]["aborted_total_tokens"] += result_batch["output_ids"].shape[1]
                        instance_stats[instance_id]["aborted_tokens"] += result_batch["generation_lengths"].item()
                    else:
                        instance_stats[instance_id]["completed_count"] += 1
                        instance_stats[instance_id]["completed_total_tokens"] += result_batch["output_ids"].shape[1]
                        instance_stats[instance_id]["completed_tokens"] += result_batch["generation_lengths"].item()
                        avg_sample_length += result_batch["unpadded_sequence_lengths"].item()
                        max_sample_length = max(max_sample_length, result_batch["unpadded_sequence_lengths"].item())
                    

                print(f"Number of results with stop_reason == 'abort': {len(aborted_samples)}")
                # if instance_stats:
                #     print(f"Instance distribution: {instance_stats}")

                print("\n📊 Instance-level Statistics:")
                for instance_id, stats in instance_stats.items():
                    print(f"Instance {instance_id}:")

                    print("  Completed:")
                    avg_completed = stats['completed_total_tokens'] / stats['completed_count'] if stats['completed_count'] > 0 else 0
                    print(f"    Requests: {stats['completed_count']}, Tokens Generated: {stats['completed_tokens']}, Average tokens per request: {avg_completed}")


                    print("  Aborted:")
                    avg_aborted = stats['aborted_total_tokens'] / stats['aborted_count'] if stats['aborted_count'] > 0 else 0
                    print(f"    Requests: {stats['aborted_count']}, Tokens Generated: {stats['aborted_tokens']}, Average tokens per request: {avg_aborted}")

            else:
                # Fallback to existing path if router isn't available
                async for sample_idx, result_batch in dynamo_gen.generate_async(data, greedy=True):
                    results.append((sample_idx, result_batch))
                end_time = time.time()
                iteration_time = end_time - start_time
            
            if results:
                if not bench_config.use_april_router:
                    avg_sample_length += sum(result_batch["unpadded_sequence_lengths"].item() for _, result_batch in results)
                    max_sample_length = max(max_sample_length, result_batch["unpadded_sequence_lengths"].item())
                latencies.append(iteration_time)
                total_samples += len(results)

                if bench_config.use_april_router:
                    total_samples -= len(aborted_samples)
                print(f"   Iteration {iteration + 1}/{bench_config.num_iterations}: {iteration_time:.3f}s ({len(results)} samples)")

                sample_offset += batch_size

                # print(f"Results: {results}")
            else:
                errors += 1
                print(f"   ⚠️  Iteration {iteration + 1} returned no results")


            # dynamo_gen.finish_generation()
            # # Wait a bit between benchmarks for clean state
            # await asyncio.sleep(2)

            # await dynamo_gen.prepare_for_generation()
            # print(f"✅ Infrastructure reinitialized for next iteration")
                
        except Exception as e:
            errors += 1
            print(f"   ❌ Iteration {iteration + 1} failed: {e}")
    
    # Calculate statistics
    if latencies:
        total_time = sum(latencies)
        avg_latency = total_time / len(latencies)
        throughput = total_samples / total_time if total_time > 0 else 0
        min_latency = min(latencies)
        max_latency = max(latencies)
    else:
        total_time = avg_latency = throughput = min_latency = max_latency = 0
    

    return BenchmarkResult(
        config_name=bench_config.name,
        batch_size=bench_config.batch_size,
        total_samples=total_samples,
        total_time=total_time,
        avg_latency=avg_latency,
        throughput=throughput,
        min_latency=min_latency,
        max_latency=max_latency,
        errors=errors,
        avg_sample_length=avg_sample_length/total_samples,
        max_sample_length=max_sample_length
    )


async def benchmark_router_configs():
    """Benchmark different router configurations."""
    global TEMP_BENCHMARK_DATASET
    
    print("🚀 Dynamo KvPushRouter Benchmarking Suite")
    print("=" * 70)

    batch_size = 64
    num_iterations = 5
    max_seq_length = 16384
    
    # Create temporary benchmark dataset for consistent testing across all configs
    TEMP_BENCHMARK_DATASET = create_temp_benchmark_dataset(max_samples=batch_size * num_iterations, max_seq_length=max_seq_length, seed=42)
    
    
    benchmark_configs = [
        BenchmarkConfig(
            name="Round Robin Router - Mock Workers",
            batch_size=batch_size,  # num_generations_per_prompt
            seq_len=150,  # Typical math problem length
            max_new_tokens=max_seq_length,  # max_total_sequence_length
            num_iterations=num_iterations,
        ),
        BenchmarkConfig(
            name="Partial Rollout Router - Mock Workers",
            batch_size=batch_size,  # num_generations_per_prompt
            seq_len=150,  # Typical math problem length
            max_new_tokens=max_seq_length,  # max_total_sequence_length
            num_iterations=num_iterations,
            use_april_router=True,
            over_sampling_batch_size=128,
        ),
    ]
    
    # Run benchmarks
    all_results = []
    
    for bench_config in benchmark_configs:
        # Create configuration
        config = VllmConfig({
            # "model_name": "Qwen/Qwen3-0.6B",
            "model_name": "Qwen/Qwen3-0.6B",
            "served_model_name": "Qwen/Qwen3-0.6B",
            "max_model_len": 16384,
            # "max_model_len": 4096,
            "max_new_tokens": bench_config.max_new_tokens,
            "temperature": 0.7,
            "pad_token_id": 0,
            "data_parallel_size": 4,
        })
        
        # Create DynamoGeneration instance
        dynamo_gen = DynamoGeneration(
            config=config,
            namespace="dynamo",
            component_name="mocker",
            # component_name="backend",
            endpoint_name="generate",
            block_size=16,
        )
        
        try:
            # Initialize infrastructure
            print(f"\n🔧 Initializing infrastructure for: {bench_config.name}")
            success = await dynamo_gen.prepare_for_generation()
            
            if not success:
                print(f"❌ Failed to initialize for {bench_config.name}")
                continue
            
            # Run benchmark
            result = await run_benchmark(dynamo_gen, bench_config)
            all_results.append(result)
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            traceback.print_exc()
        
        finally:
            # Cleanup between benchmarks
            try:
                dynamo_gen.finish_generation()
                # Wait a bit between benchmarks for clean state
                await asyncio.sleep(2)
            except Exception as e:
                print(f"⚠️  Cleanup error: {e}")
    
    return all_results


def print_benchmark_results(results: List[BenchmarkResult]):
    """Print formatted benchmark results."""
    print("\n" + "=" * 70)
    print("📊 BENCHMARK RESULTS")
    print("=" * 70)
    
    for result in results:
        print(f"\n{result.config_name}")
        print("-" * 70)
        print(f"  Batch Size:      {result.batch_size}")
        print(f"  Total Samples:   {result.total_samples}")
        print(f"  Total Time:      {result.total_time:.3f}s")
        print(f"  Avg Latency:     {result.avg_latency:.3f}s")
        print(f"  Throughput:      {result.throughput:.2f} samples/sec")
        print(f"  Min Latency:     {result.min_latency:.3f}s")
        print(f"  Max Latency:     {result.max_latency:.3f}s")
        print(f"  Avg Sample Length: {result.avg_sample_length:.2f} tokens")
        print(f"  Max Sample Length: {result.max_sample_length:.2f} tokens")
        if result.errors > 0:
            print(f"  ⚠️  Errors:        {result.errors}")
    
    print("\n" + "=" * 70)
    print("📈 SUMMARY")
    print("=" * 70)
    
    if results:
        # Find best throughput
        best_throughput = max(results, key=lambda r: r.throughput)
        print(f"🏆 Best Throughput:  {best_throughput.config_name}")
        print(f"   {best_throughput.throughput:.2f} samples/sec")
        
        # Find lowest latency
        valid_results = [r for r in results if r.avg_latency > 0]
        if valid_results:
            best_latency = min(valid_results, key=lambda r: r.avg_latency)
            print(f"\n⚡ Lowest Latency:   {best_latency.config_name}")
            print(f"   {best_latency.avg_latency:.3f}s average")
    
    print("=" * 70)


if __name__ == "__main__":
    try:
        # Run benchmarks
        results = asyncio.run(benchmark_router_configs())
        
        # Print results
        print_benchmark_results(results)
        
        # Determine success
        success = all(r.errors == 0 for r in results) if results else False
        
        print("\n" + "=" * 70)
        if success:
            print("🎉 BENCHMARK COMPLETED SUCCESSFULLY!")
            print("   All configurations tested without errors.")
        else:
            print("⚠️  BENCHMARK COMPLETED WITH SOME ERRORS")
            print("   Check individual results above for details.")
        print("=" * 70)
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)