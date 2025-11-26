# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import asyncio
import logging
import socket
import sys
import time
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np

from dynamo._core import compute_block_hash_for_seq_py
from nemo_rl.models.generation.dynamo.standalone_router import RouterAPI, RouterRequest, RouterResponse
from nemo_rl.models.generation.dynamo.workers import VllmWorkers
from vllm.inputs.data import TokensPrompt
from vllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


def is_port_available(port: int, host: str = "0.0.0.0") -> bool:
    """Check if a port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


async def wait_for_port_available(port: int, host: str = "0.0.0.0", timeout: float = 30.0) -> None:
    """Wait for a port to become available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_available(port, host):
            logger.info(f"Port {port} is now available")
            return
        logger.debug(f"Port {port} still in use, waiting...")
        await asyncio.sleep(1.0)
    raise TimeoutError(f"Port {port} did not become available within {timeout}s")


@dataclass
class RouterBenchmarkConfig:
    name: str
    batch_size: int
    seq_len: int
    router_url: str
    num_iterations: int = 5
    concurrency: int = 32
    request_timeout: float = 1.0
    generation_max_tokens: int = 128
    generation_temperature: float = 0.0

    @property
    def requests_per_iteration(self) -> int:
        return self.batch_size

    @property
    def total_requests(self) -> int:
        return self.requests_per_iteration * self.num_iterations


@dataclass
class RouterBenchmarkResult:
    config_name: str
    total_attempted: int
    total_success: int
    total_errors: int
    total_time: float
    throughput: float
    router_latency: "LatencyStats"
    end_to_end_latency: "LatencyStats"
    generation_latency: Optional["LatencyStats"]
    worker_distribution: Dict[int, int]


@dataclass
class LatencyStats:
    avg: float = 0.0
    min: float = 0.0
    max: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def create_temp_benchmark_dataset(max_samples: int, max_seq_length: int, seed: int) -> Dict[str, np.ndarray]:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")

    logger.info(
        "Creating synthetic dataset with repetitive samples: %s samples, max sequence length %s",
        max_samples,
        max_seq_length,
    )

    # Create a small number of unique patterns that will be repeated
    num_unique_patterns = min(10, max_samples)  # Use 10 unique patterns or fewer
    rng = np.random.default_rng(seed)
    
    # Generate unique patterns
    unique_input_ids = rng.integers(1, 1000, size=(num_unique_patterns, max_seq_length), dtype=np.int64)
    
    # Repeat patterns to fill max_samples
    input_ids = np.tile(unique_input_ids, (max_samples // num_unique_patterns + 1, 1))[:max_samples]
    
    # Create repetitive lengths that cycle through a small set of values
    base_length = min(max_seq_length, 150)
    input_lengths = np.array(
        [max(1, min(max_seq_length, base_length - (i % 3))) for i in range(max_samples)],
        dtype=np.int64,
    )

    logger.info(f"Created {max_samples} samples from {num_unique_patterns} unique patterns")

    return {
        "input_ids": input_ids,
        "input_lengths": input_lengths,
    }


def get_batch_tokens(dataset: Dict[str, np.ndarray], start_idx: int, batch_size: int, seq_len: int) -> List[List[int]]:
    input_ids = dataset["input_ids"]
    input_lengths = dataset["input_lengths"]

    end_idx = start_idx + batch_size
    if end_idx > len(input_ids):
        raise ValueError(
            "Synthetic dataset exhausted. Increase --dataset-samples or adjust benchmark settings."
        )

    tokens_batch: List[List[int]] = []
    for idx in range(start_idx, end_idx):
        length = int(min(seq_len, input_lengths[idx]))
        length = max(length, 1)
        tokens = input_ids[idx, :length].tolist()
        tokens_batch.append(tokens)

    return tokens_batch


def compute_latency_stats(latencies: List[float]) -> LatencyStats:
    if not latencies:
        return LatencyStats()

    lat_array = np.array(latencies)
    return LatencyStats(
        avg=float(lat_array.mean()),
        min=float(lat_array.min()),
        max=float(lat_array.max()),
        p50=float(np.percentile(lat_array, 50)),
        p95=float(np.percentile(lat_array, 95)),
        p99=float(np.percentile(lat_array, 99)),
    )


async def execute_batch(
    client: httpx.AsyncClient,
    router_url: str,
    tokens_batch: List[List[int]],
    block_size: int,
    concurrency: int,
    request_timeout: float,
    generation_workers: VllmWorkers,
    sampling_params: SamplingParams,
) -> Tuple[List[float], List[float], List[float], int, Counter]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    router_latencies: List[float] = []
    total_latencies: List[float] = []
    generation_latencies: List[float] = []
    worker_counts: Counter[int] = Counter()
    failed_requests = 0

    async def send_request(idx: int, prompt_token_ids: List[int]) -> None:
        nonlocal failed_requests
        async with semaphore:
            request_start = time.perf_counter()
            try:
                router_request = RouterRequest(
                    local_hashes=compute_block_hash_for_seq_py(prompt_token_ids, block_size),
                    num_tokens=len(prompt_token_ids),
                )
                response = await client.post(
                    router_url,
                    json=router_request.model_dump(),
                    timeout=request_timeout,
                )
                response.raise_for_status()
                router_data = RouterResponse.model_validate(response.json())
                worker_id_int = router_data.worker_id
            except Exception as exc:  # noqa: BLE001
                failed_requests += 1
                logger.debug("Router request %s failed: %s", idx, exc)
                return

            router_elapsed = time.perf_counter() - request_start

            try:
                tokens_prompt = TokensPrompt(
                    prompt_token_ids=prompt_token_ids
                )
                result_generator = generation_workers.direct(
                    tokens_prompt,
                    worker_id_int,
                    sampling_params,
                )
                async for _ in result_generator:
                    pass
            except Exception as exc:  # noqa: BLE001
                failed_requests += 1
                logger.debug("Generation for request %s failed: %s", idx, exc)
                return
            total_elapsed = time.perf_counter() - request_start
            generation_latencies.append(total_elapsed - router_elapsed)

            router_latencies.append(router_elapsed)
            total_latencies.append(total_elapsed)
            worker_counts[worker_id_int] += 1

    await asyncio.gather(
        *(send_request(idx, tokens) for idx, tokens in enumerate(tokens_batch))
    )

    return router_latencies, total_latencies, generation_latencies, failed_requests, worker_counts


async def run_router_benchmark(
    client: httpx.AsyncClient,
    router_url: str,
    config: RouterBenchmarkConfig,
    dataset: Dict[str, np.ndarray],
    block_size: int,
    generation_workers: VllmWorkers,
) -> RouterBenchmarkResult:
    print(f"\nBenchmark: {config.name}")
    print(f"  Target batch size:      {config.batch_size}")
    print(f"  Requests per iteration: {config.requests_per_iteration}")
    print(f"  Sequence length:        {config.seq_len}")
    print(f"  Concurrency:            {config.concurrency}")
    print(f"  Request timeout:        {config.request_timeout:.2f}s")
    print(f"  Generation max tokens:  {config.generation_max_tokens}")
    print(f"  Generation temperature: {config.generation_temperature:.2f}")

    sampling_params = SamplingParams(
        temperature=config.generation_temperature,
        max_tokens=config.generation_max_tokens,
        top_p=1.0,
    )

    total_attempted = 0
    total_success = 0
    total_errors = 0
    total_time = 0.0
    all_router_latencies: List[float] = []
    all_total_latencies: List[float] = []
    all_generation_latencies: List[float] = []
    aggregated_workers: Counter[int] = Counter()

    sample_offset = 0

    for iteration in range(config.num_iterations):
        requests_this_iteration = config.requests_per_iteration
        tokens_batch = get_batch_tokens(
            dataset,
            sample_offset,
            requests_this_iteration,
            config.seq_len,
        )
        # sample_offset += requests_this_iteration
        total_attempted += requests_this_iteration

        iteration_start = time.perf_counter()
        (
            router_latencies,
            total_latencies,
            generation_latencies,
            failed_requests,
            worker_counts,
        ) = await execute_batch(
            client,
            router_url,
            tokens_batch,
            block_size,
            config.concurrency,
            config.request_timeout,
            generation_workers,
            sampling_params,
        )
        iteration_time = time.perf_counter() - iteration_start

        successes = len(total_latencies)
        total_success += successes
        total_errors += failed_requests
        total_time += iteration_time
        all_router_latencies.extend(router_latencies)
        all_total_latencies.extend(total_latencies)
        if generation_latencies:
            all_generation_latencies.extend(generation_latencies)
        aggregated_workers.update(worker_counts)

        if successes + failed_requests != requests_this_iteration:
            logger.debug(
                "Iteration counts mismatch for %s: successes=%s failed=%s expected=%s",
                config.name,
                successes,
                failed_requests,
                requests_this_iteration,
            )

        print(
            f"    Iteration {iteration + 1}/{config.num_iterations}: "
            f"{iteration_time:.3f}s, successes={successes}, errors={failed_requests}"
        )

    if aggregated_workers:
        distribution = ", ".join(
            f"{worker_id}:{count}" for worker_id, count in sorted(aggregated_workers.items())
        )
        print(f"    Worker assignment totals: {distribution}")

    router_latency_stats = compute_latency_stats(all_router_latencies)
    end_to_end_latency_stats = compute_latency_stats(all_total_latencies)
    generation_latency_stats = (
        compute_latency_stats(all_generation_latencies)
    )

    throughput = (total_success / total_time) if total_time > 0 else 0.0

    return RouterBenchmarkResult(
        config_name=config.name,
        total_attempted=total_attempted,
        total_success=total_success,
        total_errors=total_errors,
        total_time=total_time,
        throughput=throughput,
        router_latency=router_latency_stats,
        end_to_end_latency=end_to_end_latency_stats,
        generation_latency=generation_latency_stats,
        worker_distribution=dict(sorted(aggregated_workers.items())),
    )


async def execute_benchmarks(
    configs: List[RouterBenchmarkConfig],
    dataset: Dict[str, np.ndarray],
    block_size: int,
    num_workers: int,
    base_kv_events_port: int,
    base_metrics_port: int,
    router_port: int,
    router_startup_wait: float,
    model: str,
) -> List[RouterBenchmarkResult]:
    if not configs:
        raise ValueError("No benchmark configurations provided")

    results: List[RouterBenchmarkResult] = []
    
    for idx, config in enumerate(configs):
        logger.info(f"\nStarting fresh router and workers for: {config.name}")
        
        # Wait for port to be available (important for subsequent configs)
        if idx > 0:
            logger.info(f"Waiting for port {router_port} to become available...")
            await wait_for_port_available(router_port, timeout=30.0)
        
        router_task: Optional[asyncio.Task] = None
        generation_workers: Optional[VllmWorkers] = None
        router_api: Optional[RouterAPI] = None
        
        try:
            # Start router
            logger.info("Starting RouterAPI on port %s (workers=%s)", router_port, num_workers)
            router_api = RouterAPI(
                block_size=block_size,
                num_workers=num_workers,
                base_kv_events_port=base_kv_events_port,
                base_metrics_port=base_metrics_port,
                port=router_port,
            )
            router_task = asyncio.create_task(router_api.start())
            await asyncio.sleep(max(0.0, router_startup_wait))
            if router_task.done():
                await router_task
            
            # Start workers
            logger.info("Initializing VllmWorkers")
            generation_workers = VllmWorkers(
                model=model,
                block_size=block_size,
                base_kv_events_port=base_kv_events_port,
                base_metrics_port=base_metrics_port,
                num_workers=num_workers,
            )
            
            # Run benchmark with fresh instances
            max_concurrency = config.concurrency
            max_connections = max(1, max_concurrency * 2)
            limits = httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            )
            
            async with httpx.AsyncClient(limits=limits) as client:
                result = await run_router_benchmark(
                    client,
                    config.router_url,
                    config,
                    dataset,
                    block_size,
                    generation_workers,
                )
                results.append(result)
        
        finally:
            # Cleanup workers first
            if generation_workers is not None:
                logger.info("Shutting down workers...")
                del generation_workers
                await asyncio.sleep(1.0)  # Give workers time to cleanup
            
            # Cleanup router
            if router_task is not None:
                logger.info("Waiting for router task to exit...")
                if router_api is not None:
                    try:
                        await router_api.stop()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Error while stopping router: %s", exc)
                with suppress(asyncio.CancelledError, Exception):
                    await router_task
                await asyncio.sleep(1.0)  # Allow event loop to cleanup
    
    return results


def print_benchmark_results(results: List[RouterBenchmarkResult]) -> None:
    print("\n" + "=" * 70)
    print("ROUTER BENCHMARK RESULTS")
    print("=" * 70)

    for result in results:
        print(f"\n{result.config_name}")
        print("-" * 70)
        print(f"  Requests attempted:   {result.total_attempted}")
        print(f"  Successful requests:  {result.total_success}")
        print(f"  Errors:               {result.total_errors}")
        print(f"  Total time:           {result.total_time:.3f}s")
        print(f"  Throughput:           {result.throughput:.2f} req/s")
        router_stats = result.router_latency
        end_to_end_stats = result.end_to_end_latency
        print("  Router latency (ms):")
        print(
            f"    avg={router_stats.avg * 1000:.2f}, "
            f"p95={router_stats.p95 * 1000:.2f}, "
            f"p99={router_stats.p99 * 1000:.2f}"
        )
        print(
            f"  End-to-end latency (ms): avg={end_to_end_stats.avg * 1000:.2f}, "
            f"p95={end_to_end_stats.p95 * 1000:.2f}, p99={end_to_end_stats.p99 * 1000:.2f}"
        )
        if result.generation_latency is not None:
            gen_stats = result.generation_latency
            print(
                f"  Generation latency (ms): avg={gen_stats.avg * 1000:.2f}, "
                f"p95={gen_stats.p95 * 1000:.2f}, p99={gen_stats.p99 * 1000:.2f}"
            )
        if result.worker_distribution:
            print("  Worker distribution:")
            for worker_id, count in result.worker_distribution.items():
                print(f"    Worker {worker_id}: {count}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results:
        best_throughput = max(results, key=lambda r: r.throughput)
        print(
            f"  Best throughput: {best_throughput.config_name} "
            f"({best_throughput.throughput:.2f} req/s)"
        )

        latency_candidates = [r for r in results if r.end_to_end_latency.avg > 0]
        if latency_candidates:
            best_latency = min(latency_candidates, key=lambda r: r.end_to_end_latency.avg)
            print(
                f"  Lowest avg latency: {best_latency.config_name} "
                f"({best_latency.end_to_end_latency.avg * 1000:.2f} ms)"
            )

    print("=" * 70)


def build_benchmark_configs(args: argparse.Namespace) -> List[RouterBenchmarkConfig]:
    configs = [
        RouterBenchmarkConfig(
            name="Router KV Baseline",
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            router_url=f"http://{args.router_host}:{args.router_port}/find_best_worker",
            num_iterations=args.num_iterations,
            concurrency=max(1, args.concurrency),
            request_timeout=args.request_timeout,
            generation_max_tokens=args.generation_max_tokens,
            generation_temperature=args.generation_temperature,
        ),
        RouterBenchmarkConfig(
            name="Router Round Robin Baseline",
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            router_url=f"http://{args.router_host}:{args.router_port}/find_worker_round_robin",
            num_iterations=args.num_iterations,
            concurrency=max(1, args.concurrency),
            request_timeout=args.request_timeout,
            generation_max_tokens=args.generation_max_tokens,
            generation_temperature=args.generation_temperature,
        )
    ]

    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the router assignment latency and throughput."
    )
    parser.add_argument("--router-host", type=str, default="localhost")
    parser.add_argument("--router-port", type=int, default=7000)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=150)
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--request-timeout", type=float, default=1.0)
    parser.add_argument("--router-startup-wait", type=float, default=1.0)
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        # default="Qwen/Qwen3-0.6B",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--base-kv-events-port", type=int, default=5557)
    parser.add_argument("--base-metrics-port", type=int, default=5657)
    parser.add_argument("--generation-max-tokens", type=int, default=128)
    parser.add_argument("--generation-temperature", type=float, default=0.0)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--dataset-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> int:
    configs = build_benchmark_configs(args)

    samples_needed = max(config.total_requests for config in configs)
    if args.dataset_samples is not None and args.dataset_samples < samples_needed:
        raise ValueError(
            "Provided --dataset-samples is smaller than the number of samples required by the benchmark."
        )

    max_samples = args.dataset_samples or samples_needed
    dataset = create_temp_benchmark_dataset(max_samples, args.max_seq_length, args.seed)

    print("Router Benchmarking Suite")
    print("=" * 70)
    print(f"Block size:      {args.block_size}")
    print(f"Number of benchmark configs: {len(configs)}")
    print("=" * 70)

    results = await execute_benchmarks(
        configs,
        dataset,
        args.block_size,
        args.num_workers,
        args.base_kv_events_port,
        args.base_metrics_port,
        args.router_port,
        args.router_startup_wait,
        args.model,
    )
    print_benchmark_results(results)

    success = all(result.total_errors == 0 for result in results)
    return 0 if success else 1


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    try:
        exit_code = asyncio.run(async_main(args))
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        exit_code = 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Benchmark failed: %s", exc)
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
