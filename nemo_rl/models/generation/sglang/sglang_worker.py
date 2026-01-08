# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import multiprocessing
import os
import time
from typing import Any, Optional

import aiohttp
import ray
import requests
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import _get_free_port_local, _get_node_ip_local
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.sglang.config import SGLangConfig
from nemo_rl.models.generation.sglang.utils import AsyncLoopThread
from nemo_rl.utils.nsys import wrap_with_nvtx_name

logger = logging.getLogger(__name__)


def _require_sglang():
    """Import `sglang` lazily so test collection works without the optional extra."""
    try:
        from sglang.srt.entrypoints.http_server import launch_server
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.utils import kill_process_tree
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Optional dependency `sglang` is required for the SGLang generation backend.\n"
            "Install it via the project extra (e.g. `uv run --extra sglang ...`) to use "
            "`SGLangGenerationWorker`."
        ) from e

    return launch_server, ServerArgs, kill_process_tree


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("sglang_generation_worker")}
)  # pragma: no cover
class SGLangGenerationWorker:
    def __repr__(self) -> str:
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        return f"{self.__class__.__name__}"

    @staticmethod
    def configure_worker(
        num_gpus: int | float, bundle_indices: Optional[tuple[int, list[int]]] = None
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
        """Provides complete worker configuration for SGLang server.

        This method configures the worker based on bundle_indices which tells us
        how many GPUs this server should use.

        Args:
            num_gpus: Original GPU allocation for this worker based on the placement group
            bundle_indices: Tuple of (node_idx, local_bundle_indices) for this server

        Returns:
            tuple with complete worker configuration:
              - 'resources': Resource allocation (e.g., num_gpus)
              - 'env_vars': Environment variables for this worker
              - 'init_kwargs': Parameters to pass to __init__ of the worker
        """
        # Initialize configuration
        resources: dict[str, Any] = {"num_gpus": num_gpus}
        init_kwargs: dict[str, Any] = {}
        env_vars: dict[str, str] = {}

        local_bundle_indices = None
        if bundle_indices is not None:
            node_idx = bundle_indices[0]
            local_bundle_indices = bundle_indices[1]
            init_kwargs["bundle_indices"] = local_bundle_indices

            # Calculate a unique seed from node_idx and bundle_indices
            if len(local_bundle_indices) == 1:
                seed = node_idx * 1024 + local_bundle_indices[0]
            else:
                bundle_id = local_bundle_indices[0] // len(local_bundle_indices)
                seed = node_idx * 1024 + bundle_id

            init_kwargs["seed"] = seed

        # Check if this worker is part of a parallel group (multiple GPUs per server).
        # A worker with local rank =0 owns the server(local_bundle_indices is not None )
        # otherwise it is a placeholder for Ray's resource management (local_bundle_indices is None).
        is_part_of_parallel_workers = (
            local_bundle_indices is not None and len(local_bundle_indices) > 1
        ) or local_bundle_indices is None

        if is_part_of_parallel_workers:
            # For parallel workers, we manage GPU assignment via base_gpu_id
            # All workers see the same global CUDA_VISIBLE_DEVICES, but use different
            # logical GPU ranges via base_gpu_id
            resources["num_gpus"] = 0
            env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
            init_kwargs["fraction_of_gpus"] = num_gpus
        else:
            env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"

        return resources, env_vars, init_kwargs

    def __init__(
        self,
        config: SGLangConfig,
        bundle_indices: Optional[list[int]] = None,
        fraction_of_gpus: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Initialize a SGLang worker for distributed inference.

        Args:
            config: Configuration dictionary for the policy
            bundle_indices: List of local bundle indices for this server.
                          The length of this list determines tp_size (number of GPUs per server).
                          Only needed for the first worker in each server group (model owner).
            fraction_of_gpus: Fraction of GPUs to use for this worker
            seed: Random seed for initialization, if None, then defaults to the config's seed
        """
        self.cfg = config
        self.is_model_owner = bundle_indices is not None
        self.global_rank = int(os.environ.get("RANK", "0"))
        self.sglang_cfg = config["sglang_cfg"]

        # Create a dedicated event loop thread for async operations
        # there will be issues if we use the event loop in the main thread
        self.async_loop_thread = AsyncLoopThread()

        # temp: Maximum concurrent requests per server
        # we may remove this limit in the future
        self.max_concurrent_requests = config.get("max_concurrent_requests", 999999)

        # Only the primary worker (local_rank=0) in each server group starts the SGLang server
        # Secondary workers (local_rank!=0) just returns
        if not self.is_model_owner:
            return

        # `sglang` is an optional dependency; import only when we actually start a server.
        _, ServerArgs, _ = _require_sglang()

        # Determine tp_size from bundle_indices length
        tp_size = len(bundle_indices)

        base_gpu_id = bundle_indices[0] if bundle_indices else 0

        # Get the global CUDA_VISIBLE_DEVICES (all engines see the same global value)
        global_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)

        logger.info(
            f"[SGLang Server] Rank {self.global_rank}: "
            f"base_gpu_id={base_gpu_id}, tp_size={tp_size}, "
            f"bundle_indices={bundle_indices}, global_cvd={global_cvd}"
        )

        # Get current node IP and a free port for the server
        node_ip = _get_node_ip_local()
        free_port = _get_free_port_local()

        # Build SGLang server arguments
        kwargs = {
            "model_path": self.sglang_cfg["model_path"],
            "trust_remote_code": True,
            "random_seed": seed
            if seed is not None
            else self.sglang_cfg.get("random_seed", 1),
            # Memory settings
            "enable_memory_saver": self.sglang_cfg["enable_memory_saver"],
            "gpu_id_step": 1,
            "base_gpu_id": base_gpu_id,
            # Parallel settings
            "tp_size": tp_size,
            "dp_size": self.sglang_cfg["dp_size"],
            "pp_size": self.sglang_cfg["pp_size"],
            "ep_size": self.sglang_cfg["ep_size"],
            # Always skip warmup to prevent warmup timeout
            "skip_server_warmup": self.sglang_cfg.get("skip_server_warmup", True),
            # Server network settings - listen on all interfaces, use the free port we found
            "host": "0.0.0.0",
            "port": free_port,
            "torchao_config": "",
        }

        for key in [
            "dtype",
            "kv_cache_dtype",
            "context_length",
            "max_running_requests",
            "chunked_prefill_size",
            "max_prefill_tokens",
            "schedule_policy",
            "schedule_conservativeness",
            "cpu_offload_gb",
            "log_level",
            "mem_fraction_static",
            "allow_auto_truncate",
        ]:
            if key in self.sglang_cfg:
                kwargs[key] = self.sglang_cfg[key]

        server_args = ServerArgs(**kwargs)
        # Save server_args and base_url for use in generate() and _make_request()
        self.server_args = server_args
        self.base_url = f"http://{node_ip}:{free_port}"

        logger.info(
            f"[SGLang Worker] Rank {self.global_rank} Starting on {self.base_url}, CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', None)}, base_gpu_id: {base_gpu_id}"
        )

        self.session = None
        self.connector = None

        self.server_process = self._launch_server_process(server_args)

    def get_base_url(self) -> str:
        """Get the base URL of this SGLang server."""
        return self.base_url

    def invalidate_kv_cache(self) -> bool:
        """Invalidate KV cache before weight updates (Megatron-style).

        This flushes the cache before weight updates to clear stale cache.
        Uses retry logic to handle cases where there are pending requests.

        Returns:
            bool: True if flush was successful, False otherwise
        """
        if not self.is_model_owner:
            return True

        url = f"{self.base_url}/flush_cache"
        max_attempts = 60
        connection_retry_limit = 5

        # flush_cache will not return status_code 200 when there are pending requests
        for attempt in range(max_attempts):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    if attempt > 0:
                        logger.info(
                            f"[SGLang Worker] Rank {self.global_rank} Cache flushed successfully "
                            f"(attempt {attempt + 1})"
                        )
                    return True
            except requests.exceptions.ConnectionError:
                # Server might not be ready yet - only retry for first few attempts
                if attempt >= connection_retry_limit:
                    logger.warning(
                        f"[SGLang Worker] Rank {self.global_rank} Connection failed after "
                        f"{connection_retry_limit} attempts"
                    )
                    return False
            except Exception as e:
                # For other errors, log and retry (except on last attempt)
                if attempt == max_attempts - 1:
                    logger.error(
                        f"[SGLang Worker] Rank {self.global_rank} Failed to flush cache after "
                        f"{max_attempts} attempts: {e}"
                    )
                    return False

            time.sleep(1)

        # All attempts exhausted without success
        logger.error(
            f"[SGLang Worker] Rank {self.global_rank} Timeout: Cache flush failed after "
            f"{max_attempts} attempts. Server may have pending requests."
        )
        return False

    def get_gpu_uuids(self) -> list[str]:
        """Get list of GPU UUIDs used by this SGLang server.

        Returns:
            List of GPU UUIDs (e.g., ["GPU-xxxxx", "GPU-yyyyy"])
        """
        from nemo_rl.utils.nvml import get_device_uuid

        # Get all GPU UUIDs used by this server
        # SGLang server uses GPUs starting from base_gpu_id with tp_size GPUs
        gpu_uuids = []
        for i in range(self.server_args.tp_size):
            gpu_id = self.server_args.base_gpu_id + i
            uuid = get_device_uuid(gpu_id)
            gpu_uuids.append(uuid)

        return gpu_uuids

    def _merge_stop_strings(self, batch_stop_strings):
        """Merge stop strings from config and batch.

        Args:
            batch_stop_strings: List of stop strings from batch (one per sample)

        Returns:
            List of merged stop strings (one per sample)
        """
        stop_set: set[str] = set()

        # Add stop strings from config
        if self.cfg.get("stop_strings"):
            stop_set.update(self.cfg["stop_strings"])

        # Merge stop strings from batch
        merged_stop_strings = []
        for sample_ss in batch_stop_strings:
            sample_stop_set = stop_set.copy()
            if sample_ss:
                if isinstance(sample_ss, str):
                    sample_stop_set.add(sample_ss)
                elif isinstance(sample_ss, list):
                    sample_stop_set.update(sample_ss)

            merged_stop_strings.append(
                list(sample_stop_set) if sample_stop_set else None
            )

        return merged_stop_strings

    def _build_sampling_params(
        self,
        *,
        greedy: bool,
        stop_strings,
        max_new_tokens: Optional[int] = None,
        input_len: Optional[int] = None,
        context_length: Optional[int] = None,
        sample_index: Optional[int] = None,
    ) -> dict[str, Any]:
        """Build sampling parameters dictionary for SGLang API.

        Args:
            greedy: Whether to use greedy decoding (temperature=0.0)
            stop_strings: Merged stop strings (not used here, handled per sample)
            max_new_tokens: Override max_new_tokens from config if provided
            input_len: Input length for this sample (used for context_length adjustment)
            context_length: Maximum context length (if provided, adjusts max_new_tokens)
            sample_index: Sample index (used for warning messages, 0-indexed)

        Returns:
            Dictionary of sampling parameters compatible with SGLang API
        """
        top_k_cfg = self.cfg.get("top_k")
        top_k_val = 1 if greedy else (top_k_cfg if top_k_cfg is not None else -1)
        temperature = 0.0 if greedy else self.cfg["temperature"]

        base_max_tokens = (
            max_new_tokens if max_new_tokens is not None else self.cfg["max_new_tokens"]
        )

        # TODO: check if this is needed
        final_max_tokens = base_max_tokens
        if context_length is not None and input_len is not None:
            max_allowed_new_tokens = max(0, context_length - input_len - 1)
            if base_max_tokens > max_allowed_new_tokens:
                final_max_tokens = max_allowed_new_tokens
                if sample_index == 0:
                    logger.warning(
                        f"[SGLang Worker] Rank {self.global_rank} Warning: "
                        f"Sample {sample_index} input length ({input_len}) + max_new_tokens ({base_max_tokens}) "
                        f"would exceed context_length ({context_length}). "
                        f"Reducing max_new_tokens to {final_max_tokens} for this sample."
                    )

        # Build sampling params dict
        sampling_params = {
            "temperature": temperature,
            "top_p": self.cfg.get("top_p", 1.0),
            "max_new_tokens": final_max_tokens,
        }

        if top_k_val != -1:
            sampling_params["top_k"] = top_k_val

        stop_token_ids = self.cfg.get("stop_token_ids")
        if stop_token_ids is not None:
            sampling_params["stop_token_ids"] = stop_token_ids

        return sampling_params

    async def _ensure_session(self):
        if self.session is None:
            # Create connector with connection pool limit
            self.connector = aiohttp.TCPConnector(limit=512, limit_per_host=512)
            # Create session with timeout
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
            self.session = aiohttp.ClientSession(
                connector=self.connector, timeout=timeout
            )
        return self.session

    async def _generate_single_sample(
        self,
        input_ids: list[int],
        sampling_params: dict[str, Any],
        stop_string: Optional[str] = None,
    ) -> tuple[list[int], list[float]]:
        """Generate a single sample using SGLang API (async function).

        Args:
            input_ids: List of input token IDs (without padding)
            sampling_params: Dictionary of sampling parameters (temperature, top_p, max_new_tokens, etc.)
            stop_string: Optional stop string for this sample

        Returns:
            Tuple of (generated_tokens, logprobs):
                - generated_tokens: List of generated token IDs
                - logprobs: List of log probabilities for generated tokens
        """
        # Prepare payload for SGLang API
        # Note: stop should be in sampling_params, not in payload top level
        # TODO: double check this
        if stop_string is not None:
            # stop can be a string or list of strings
            sampling_params = sampling_params.copy()  # Don't modify the original
            sampling_params["stop"] = stop_string

        payload = {
            "sampling_params": sampling_params,
            "return_logprob": True,
            "input_ids": input_ids,
        }

        url = f"{self.base_url}/generate"
        headers = {
            "Content-Type": "application/json; charset=utf-8",
        }

        session = await self._ensure_session()

        try:
            async with session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()
                result = await response.json()
        except Exception as e:
            logger.error(
                f"[SGLang Worker] Rank {self.global_rank} Request failed for input_len={len(input_ids)}: {e}"
            )
            raise

        # Extract generated tokens and logprobs
        meta_info = result.get("meta_info", {})
        output_token_logprobs = meta_info.get("output_token_logprobs", [])

        if output_token_logprobs:
            new_tokens = [item[1] for item in output_token_logprobs]
            new_logprobs = [item[0] for item in output_token_logprobs]
        else:
            # Fallback: empty if token logprobs not available
            new_tokens = []
            new_logprobs = []

        return new_tokens, new_logprobs

    async def _generate_async(self, tasks):
        """Execute generation tasks with concurrency control.

        TEMP: Uses a semaphore to limit the number of concurrent requests per server, preventing server overload.
        A router based solution is preffered in the future.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def wrap(idx, coro):
            async with semaphore:
                try:
                    result = await coro
                    return idx, result
                except Exception as e:
                    raise

        wrapped = [wrap(i, t) for i, t in enumerate(tasks)]
        results = [None] * len(tasks)
        count = 0

        for fut in asyncio.as_completed(wrapped):
            idx, value = await fut
            results[idx] = value
            count += 1
            if count % 50 == 0 or count == len(tasks):
                logger.debug(
                    f"[SGLang Worker] Rank {self.global_rank} Completed {count}/{len(tasks)} tasks"
                )

        return results

    def _launch_server_process(self, server_args: Any) -> multiprocessing.Process:
        """Launch the SGLang server process and wait for it to be ready."""
        # Ensure `sglang` is importable when we actually start a server.
        launch_server, _, kill_process_tree = _require_sglang()
        p = multiprocessing.Process(target=launch_server, args=(server_args,))
        p.start()

        # Wait for server to be ready by checking health endpoint
        # Use the base_url we stored earlier
        headers = {
            "Content-Type": "application/json; charset=utf-8",
        }

        max_wait_time = 300  # 5 minutes timeout
        start_time = time.time()
        with requests.Session() as session:
            while True:
                if time.time() - start_time > max_wait_time:
                    kill_process_tree(p.pid)
                    raise TimeoutError(
                        f"[SGLang Server] Rank {self.global_rank} Server failed to start within {max_wait_time}s"
                    )
                try:
                    response = session.get(
                        f"{self.base_url}/health_generate", headers=headers, timeout=10
                    )
                    if response.status_code == 200:
                        logger.info(
                            f"[SGLang Server] Rank {self.global_rank} Server is ready at {self.base_url}"
                        )
                        break
                except requests.RequestException:
                    pass

                if not p.is_alive():
                    raise RuntimeError(
                        f"[SGLang Server] Rank {self.global_rank} Server process terminated unexpectedly."
                    )

                time.sleep(2)
        return p

    @wrap_with_nvtx_name("sglang_genertion_worker/generate")
    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using SGLang generation.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths tensors
            greedy: Whether to use greedy decoding instead of sampling

        Returns:
            BatchedDataDict conforming to GenerationOutputSpec:
                - output_ids: input + generated token IDs with proper padding
                - logprobs: Log probabilities for tokens
                - generation_lengths: Lengths of each response
                - unpadded_sequence_lengths: Lengths of each input + generated sequence
        """
        # Handle empty input case
        if len(data["input_ids"]) == 0:
            return BatchedDataDict[GenerationOutputSpec](
                {
                    "output_ids": torch.zeros((0, 0), dtype=torch.long),
                    "logprobs": torch.zeros((0, 0), dtype=torch.float),
                    "generation_lengths": torch.zeros(0, dtype=torch.long),
                    "unpadded_sequence_lengths": torch.zeros(0, dtype=torch.long),
                }
            )

        input_ids = data["input_ids"]
        input_lengths = data["input_lengths"]
        batch_stop_strings = data.get("stop_strings", [None] * len(input_lengths))
        stop_strings = self._merge_stop_strings(batch_stop_strings)
        batch_size = len(input_lengths)
        pad_token_id = self.cfg["_pad_token_id"]

        # Verify inputs have correct padding
        verify_right_padding(data, pad_value=pad_token_id)

        # Original input length with padding
        padded_input_length = input_ids.size(1)

        logger.debug(
            f"[SGLang Worker] Rank {self.global_rank} batch_size: {batch_size}, padded_input_length: {padded_input_length}"
        )

        if batch_size == 0:
            raise ValueError("Empty batch received")

        context_length = self.sglang_cfg.get("context_length", None)

        # Create async tasks for all samples
        tasks = []
        for i in range(batch_size):
            input_len = input_lengths[i].item()

            # Truncate input if it exceeds context_length
            if context_length is not None and input_len >= context_length:
                input_len = context_length - 1

            valid_input_ids = input_ids[i, :input_len].tolist()

            # Build sampling params for this sample (with context_length adjustment)
            sample_sampling_params = self._build_sampling_params(
                greedy=greedy,
                stop_strings=stop_strings,
                max_new_tokens=None,
                input_len=input_len,
                context_length=context_length,
                sample_index=i,
            )

            tasks.append(
                self._generate_single_sample(
                    input_ids=valid_input_ids,
                    sampling_params=sample_sampling_params,
                    stop_string=stop_strings[i],
                )
            )

        # Execute all requests concurrently using the dedicated event loop thread
        try:
            all_results = self.async_loop_thread.run(self._generate_async(tasks))
        except Exception as e:
            raise

        total_generated_tokens = sum(len(tokens) for tokens, _ in all_results)
        avg_generation_length = (
            total_generated_tokens / batch_size if batch_size > 0 else 0
        )

        # Process results
        output_ids_list = []
        logprobs_list = []
        generation_lengths_list = []
        unpadded_sequence_lengths_list = []
        max_length = 0

        # First pass: calculate max_length
        for i, (new_tokens, new_logprobs) in enumerate(all_results):
            input_len = input_lengths[i].item()
            generation_length = len(new_tokens)
            unpadded_length = input_len + generation_length
            max_length = max(max_length, unpadded_length)

        total_length = max(max_length, padded_input_length)

        for i, (new_tokens, new_logprobs) in enumerate(all_results):
            input_len = input_lengths[i].item()
            generation_length = len(new_tokens)
            unpadded_length = input_len + generation_length

            full_output = torch.full(
                (total_length,), pad_token_id, dtype=input_ids.dtype
            )
            full_output[:input_len] = input_ids[i][:input_len]

            # Add generated tokens after the original input
            if new_tokens:
                full_output[input_len : input_len + len(new_tokens)] = torch.tensor(
                    new_tokens, dtype=input_ids.dtype
                )

            # Construct logprobs: zeros for input tokens, actual logprobs for generated tokens
            full_logprobs = torch.zeros(total_length, dtype=torch.float32)
            if new_logprobs:
                for idx, logprob in enumerate(new_logprobs):
                    position = input_len + idx
                    full_logprobs[position] = logprob

            output_ids_list.append(full_output)
            logprobs_list.append(full_logprobs)
            generation_lengths_list.append(generation_length)
            unpadded_sequence_lengths_list.append(unpadded_length)

        # Stack into tensors
        output_ids = torch.stack(output_ids_list)
        logprobs = torch.stack(logprobs_list)
        generation_lengths = torch.tensor(generation_lengths_list, dtype=torch.long)
        unpadded_sequence_lengths = torch.tensor(
            unpadded_sequence_lengths_list, dtype=torch.long
        )
        logger.debug(
            f"[SGLang Worker] Rank {self.global_rank} Generated {total_generated_tokens} tokens across {batch_size} samples (avg: {avg_generation_length:.1f} tokens/sample)"
        )
        return BatchedDataDict[GenerationOutputSpec](
            {
                "output_ids": output_ids,
                "generation_lengths": generation_lengths,
                "unpadded_sequence_lengths": unpadded_sequence_lengths,
                "logprobs": logprobs,
            }
        )

    def sleep(self):
        # TODO
        pass

    def wake_up(self, **kwargs):
        # TODO
        pass

    def shutdown(self) -> bool:
        """Shutdown the SGLang server process and cleanup async resources.

        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        if not self.is_model_owner:
            if hasattr(self, "async_loop_thread"):
                try:
                    self.async_loop_thread.shutdown()
                    logger.info(
                        f"[SGLang Worker] Rank {self.global_rank} Async loop thread shut down."
                    )
                except Exception as e:
                    logger.error(
                        f"[SGLang Worker] Rank {self.global_rank} Error shutting down async loop thread: {e}"
                    )
            return True

        try:
            # Only model owners started a server process; they require sglang for shutdown.
            _, _, kill_process_tree = _require_sglang()
            if hasattr(self, "session") and self.session is not None:
                try:

                    async def close_session():
                        await self.session.close()
                        if self.connector is not None:
                            await self.connector.close()

                    self.async_loop_thread.run(close_session())
                    logger.info(
                        f"[SGLang Worker] Rank {self.global_rank} aiohttp session closed."
                    )
                except Exception as e:
                    logger.error(
                        f"[SGLang Worker] Rank {self.global_rank} Error closing aiohttp session: {e}"
                    )

            # Shutdown async loop thread after session cleanup
            if hasattr(self, "async_loop_thread"):
                try:
                    self.async_loop_thread.shutdown()
                    logger.info(
                        f"[SGLang Worker] Rank {self.global_rank} Async loop thread shut down."
                    )
                except Exception as e:
                    logger.error(
                        f"[SGLang Worker] Rank {self.global_rank} Error shutting down async loop thread: {e}"
                    )

            if not hasattr(self, "server_process") or self.server_process is None:
                return True

            logger.info(
                f"[SGLang Worker] Rank {self.global_rank} Shutting down server at {self.base_url}..."
            )

            if self.server_process.is_alive():
                kill_process_tree(self.server_process.pid)

            # Wait for the process to terminate
            self.server_process.join(timeout=5.0)

            if self.server_process.is_alive():
                return False
            return True

        except Exception as e:
            logger.error(
                f"[SGLang Worker] Rank {self.global_rank} Error during shutdown: {e}"
            )
            return False

    def _make_request(self, endpoint: str, payload: Optional[dict] = None):
        """Make a POST request to the specified endpoint with the given payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)

        Returns:
            The JSON response from the server
        """
        # Use the stored base_url instead of constructing from server_args
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "Content-Type": "application/json; charset=utf-8",
        }
        response = requests.post(url, json=payload or {}, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json()
