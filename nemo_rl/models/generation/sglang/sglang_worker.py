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

import copy
import gc
import os
import sys
from typing import Any, Optional, cast
import requests
import asyncio
import aiohttp

import time
import ray
import torch
import multiprocessing

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import _get_node_ip_local, _get_free_port_local
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.sglang.config import SGLangConfig
from nemo_rl.models.huggingface.common import ModelFlag
from nemo_rl.utils.nsys import wrap_with_nvtx_name

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree


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
            seed: Random seed for initialization
        """
        self.cfg = config
        self.is_model_owner = bundle_indices is not None
        
        # This is the global worker rank across all workers
        self.global_rank = int(os.environ.get("RANK", "0"))
        
        # Only the primary worker (local_rank=0) in each server group starts the SGLang server
        # Secondary workers (local_rank!=0) just returns
        if not self.is_model_owner:
            return

        # Determine tp_size from bundle_indices length
        tp_size = len(bundle_indices)
        
        base_gpu_id = bundle_indices[0] if bundle_indices else 0
        
        # Get the global CUDA_VISIBLE_DEVICES (all engines see the same global value)
        global_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        
        
        print(
            f"[SGLang Server] Rank {self.global_rank}: "
            f"base_gpu_id={base_gpu_id}, tp_size={tp_size}, "
            f"bundle_indices={bundle_indices}, global_cvd={global_cvd}"
        )

        # Get current node IP and a free port for the server
        node_ip = _get_node_ip_local()
        free_port = _get_free_port_local()
        
        # Build SGLang server arguments
        kwargs = {
            "model_path": self.cfg.get("model_path", ""),
            "trust_remote_code": True,
            "random_seed": seed if seed is not None else self.cfg.get("random_seed", 1),
            # Memory settings
            "enable_memory_saver": self.cfg.get("enable_memory_saver", False),
            "gpu_id_step": 1,
            "base_gpu_id": base_gpu_id,
            # Parallel settings
            "tp_size": tp_size,
            "dp_size": self.cfg.get("dp_size", 1),
            "pp_size": self.cfg.get("pp_size", 1),
            "ep_size": self.cfg.get("ep_size", 1),
            # Always skip warmup to prevent warmup timeout
            "skip_server_warmup": True,
            # Server network settings - listen on all interfaces, use the free port we found
            "host": "0.0.0.0",
            "port": free_port,
            "torchao_config": "",
        }
        
        # Add other config fields if they exist
        for key in [
            "dtype", "kv_cache_dtype", "context_length", "max_running_requests",
            "chunked_prefill_size", "max_prefill_tokens", "schedule_policy",
            "schedule_conservativeness", "cpu_offload_gb", "log_level",
        ]:
            if key in self.cfg:
                kwargs[key] = self.cfg[key]

        server_args = ServerArgs(**kwargs)
        # Save server_args and base_url for use in generate() and _make_request()
        self.server_args = server_args
        self.base_url = f"http://{node_ip}:{free_port}"
        
        print(f"[SGLang Worker] Rank {self.global_rank} Starting on {self.base_url}, CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', None)}, base_gpu_id: {base_gpu_id}")
        
        self.server_process = self._launch_server_process(server_args)


    def _merge_stop_strings(self, batch_stop_strings):
        pass

    def _build_sampling_params(
        self,
        *,
        greedy: bool,
        stop_strings,
        max_new_tokens: Optional[int] = None,
    ):
        pass

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
        if stop_string is not None:
            # stop can be a string or list of strings
            sampling_params = sampling_params.copy()  # Don't modify the original
            sampling_params["stop"] = stop_string
        
        payload = {
            "sampling_params": sampling_params,
            "return_logprob": True,
            "input_ids": input_ids,
        }
        
        # Use aiohttp for async request
        url = f"{self.base_url}/generate"
        headers = {
            "Content-Type": "application/json; charset=utf-8",
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()
                result = await response.json()
        
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

    async def _generate_async(self, tasks: list) -> list:
        """Execute all async generation tasks concurrently.
        
        Args:
            tasks: List of async coroutines for generating samples
            
        Returns:
            List of (tokens, logprobs) tuples for all samples
        """
        return await asyncio.gather(*tasks)

    def _launch_server_process(self, server_args: ServerArgs) -> multiprocessing.Process:
        """Launch the SGLang server process and wait for it to be ready."""
        p = multiprocessing.Process(target=launch_server, args=(server_args,))
        p.start()

        # Wait for server to be ready by checking health endpoint
        # Use the base_url we stored earlier
        headers = {
            "Content-Type": "application/json; charset=utf-8",
        }

        with requests.Session() as session:
            while True:
                try:
                    response = session.get(f"{self.base_url}/health_generate", headers=headers)
                    if response.status_code == 200:
                        print(f"[SGLang Server] Rank {self.global_rank} Server is ready at {self.base_url}")
                        break
                except requests.RequestException:
                    pass

                if not p.is_alive():
                    raise Exception(f"[SGLang Server] Rank {self.global_rank} Server process terminated unexpectedly.")

                time.sleep(2)
        # response = session.get(f"{self.base_url}/get_model_info", headers=headers)
        # print(f"[SGLang Worker] Rank {self.global_rank} model_info: {response.json()}")
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
        stop_strings = data.get("stop_strings", [None] * len(input_lengths))
        batch_size = len(input_lengths)
        pad_token_id = self.cfg.get("_pad_token_id", 0)
        
        # Verify inputs have correct padding
        verify_right_padding(data, pad_value=pad_token_id)
        
        # Original input length with padding
        padded_input_length = input_ids.size(1)
        
        print(f"[SGLang Worker] Rank {self.global_rank} batch_size: {batch_size}, padded_input_length: {padded_input_length}")
        
        # Get generation parameters from config
        max_new_tokens = self.cfg.get("max_new_tokens", 512)
        temperature = 0.0 if greedy else self.cfg.get("temperature", 1.0)
        top_p = self.cfg.get("top_p", 1.0)
        top_k = self.cfg.get("top_k", None)
        
        sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        }
        if top_k is not None:
            sampling_params["top_k"] = top_k
        
        if batch_size == 0:
            raise ValueError("Empty batch received")
        
        # Create async tasks for all samples
        tasks = []
        for i in range(batch_size):
            input_len = input_lengths[i].item()
            valid_input_ids = input_ids[i, :input_len].tolist()
            
            tasks.append(
                self._generate_single_sample(
                    input_ids=valid_input_ids,
                    sampling_params=sampling_params,
                    stop_string=stop_strings[i],
                )
            )
        
        # Execute all requests concurrently
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(
                self._generate_async(tasks),
                loop
            )
            all_results = future.result()
        except RuntimeError:
            all_results = asyncio.run(self._generate_async(tasks))
        
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
        
        total_length = padded_input_length + max_length
        
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
                full_output[input_len : input_len + len(new_tokens)] = (
                    torch.tensor(new_tokens, dtype=input_ids.dtype)
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
        unpadded_sequence_lengths = torch.tensor(unpadded_sequence_lengths_list, dtype=torch.long)
        
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
        """Shutdown the SGLang server process.
        
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        if not self.is_model_owner:
            return True
        
        if not hasattr(self, "server_process") or self.server_process is None:
            return True
        
        try:
            print(
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
            print(
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
        response = requests.post(url, json=payload or {}, headers=headers)
        response.raise_for_status()
        return response.json()