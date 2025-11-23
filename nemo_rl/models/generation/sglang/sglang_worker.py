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

import time
import ray
import torch
import multiprocessing

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.sglang.config import SGLangConfig
from nemo_rl.models.huggingface.common import ModelFlag
from nemo_rl.utils.nsys import wrap_with_nvtx_name

try:
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import kill_process_tree
except ImportError:
    # SGLang may not be installed, but we still want the code to be importable
    launch_server = None
    ServerArgs = None
    kill_process_tree = None




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

        # For SGLang, Ray manages GPU assignment via CUDA_VISIBLE_DEVICES
        # We set num_gpus to 0 and let Ray handle it
        if local_bundle_indices is not None and len(local_bundle_indices) > 1:
            resources["num_gpus"] = 0
            env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
            init_kwargs["fraction_of_gpus"] = num_gpus

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
        
        if not self.is_model_owner:
            return

        # Determine tp_size from bundle_indices length
        # Ray sets CUDA_VISIBLE_DEVICES so each server sees logical GPUs 0, 1, 2, ..., tp_size-1
        tp_size = len(bundle_indices) if bundle_indices else 1
        
        # Build SGLang server arguments
        # Ray automatically sets CUDA_VISIBLE_DEVICES, so base_gpu_id should be 0
        # and gpu_id_step should be 1
        kwargs = {
            "model_path": self.cfg.get("model_path", ""),
            "trust_remote_code": True,
            "random_seed": seed if seed is not None else self.cfg.get("random_seed", 1),
            # Memory settings
            "enable_memory_saver": self.cfg.get("enable_memory_saver", False),
            # GPU settings - Ray handles CUDA_VISIBLE_DEVICES, so we use logical GPU 0
            "gpu_id_step": 1,
            "base_gpu_id": 0,  # Always 0 because Ray sets CUDA_VISIBLE_DEVICES
            # Parallel settings
            "tp_size": tp_size,
            "dp_size": self.cfg.get("dp_size", 1),
            "pp_size": self.cfg.get("pp_size", 1),
            "ep_size": self.cfg.get("ep_size", 1),
            # Always skip warmup to prevent warmup timeout
            "skip_server_warmup": True,
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

    def _launch_server_process(self, server_args: ServerArgs) -> multiprocessing.Process:
        """Launch the SGLang server process and wait for it to be ready."""
        p = multiprocessing.Process(target=launch_server, args=(server_args,))
        p.start()

        if server_args.node_rank != 0:
            return

        base_url = server_args.url()

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {server_args.api_key}",
        }

        with requests.Session() as session:
            while True:
                try:
                    response = session.get(f"{base_url}/health_generate", headers=headers)
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    pass

                if not p.is_alive():
                    raise Exception("Server process terminated unexpectedly.")

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
        pass

    def sleep(self):
        pass

    def wake_up(self, **kwargs):
        pass

    def shutdown(self) -> bool:
        pass

    def _make_request(self, endpoint: str, payload: Optional[dict] = None):
        """Make a POST request to the specified endpoint with the given payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)

        Returns:
            The JSON response from the server
        """
        if self.node_rank != 0:
            return

        url = f"http://{self.server_args.host}:{self.server_args.port}/{endpoint}"
        response = requests.post(url, json=payload or {})
        response.raise_for_status()
        return response.json()