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
import copy
import gc
import os
import sys
import uuid
from collections import defaultdict
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    NotRequired,
    Optional,
    Tuple,
    TypedDict,
    Union,
    cast,
)

import numpy as np
import ray
import torch
from ray.util.placement_group import PlacementGroup

from nemo_rl.distributed.batched_data_dict import BatchedDataDict, SlicedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.virtual_cluster import (
    RayVirtualCluster,
)
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.distributed.worker_groups import (
    RayWorkerBuilder,
    RayWorkerGroup,
)
from nemo_rl.models.generation.interfaces import (
    GenerationConfig,
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.huggingface.common import ModelFlag


class VllmSpecificArgs(TypedDict):
    tensor_parallel_size: int
    pipeline_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    # Additional arguments for vLLM inserted by nemo rl based on the context of when vllm is used
    skip_tokenizer_init: bool
    async_engine: bool
    load_format: NotRequired[str]
    precision: NotRequired[str]


class VllmConfig(GenerationConfig):
    vllm_cfg: VllmSpecificArgs
    vllm_kwargs: NotRequired[dict[str, Any]]


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_generation_worker")}
)
class VllmGenerationWorker:
    def __repr__(self) -> str:
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        return f"{self.__class__.__name__}"

    @staticmethod
    def configure_worker(
        num_gpus: int | float, bundle_indices: Optional[tuple[int, list[int]]] = None
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
        """Provides complete worker configuration for vLLM tensor and pipeline parallelism.

        This method configures the worker based on its role in tensor and pipeline parallelism,
        which is determined directly from the bundle_indices parameter.

        Args:
            num_gpus: Original GPU allocation for this worker based on the placement group
            bundle_indices: Tuple of (node_idx, local_bundle_indices) for parallelism (if applicable)

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

            """
            compute a unique seed from the node_idx and bundle_indices:
            node_idx = 0, bundle_indices = [0, 1, 2, 3] -> seed = 0*1024 + 0
            node_idx = 0, bundle_indices = [4, 5, 6, 7] -> seed = 0*1024 + 1
            node_idx = 1, bundle_indices = [0, 1, 2, 3] -> seed = 1*1024 + 0
            node_idx = 1, bundle_indices = [4, 5, 6, 7] -> seed = 1*1024 + 1
            """
            # For single worker groups, use a simpler seed calculation
            if len(local_bundle_indices) == 1:
                seed = node_idx * 1024 + local_bundle_indices[0]
            else:
                # For parallel groups, use the original calculation
                bundle_id = local_bundle_indices[0] // len(local_bundle_indices)
                seed = node_idx * 1024 + bundle_id

            init_kwargs["seed"] = seed

        # Check if this worker is part of a parallel group (TP or TP+PP).
        # A worker is part of a parallel group if it's a secondary member (local_bundle_indices is None)
        # or if it's a primary member of a group with multiple workers.
        is_part_of_parallel_workers = (
            local_bundle_indices is not None and len(local_bundle_indices) > 1
        ) or local_bundle_indices is None

        if is_part_of_parallel_workers:
            # Ray + vllm likes to manage GPU assignment internally for parallel groups
            resources["num_gpus"] = 0
            env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
            init_kwargs["fraction_of_gpus"] = num_gpus

        env_vars["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        # Skip vllm P2P check and rely on driver to report peer to peer capability.
        env_vars["VLLM_SKIP_P2P_CHECK"] = "1"

        return resources, env_vars, init_kwargs

    def __init__(
        self,
        config: VllmConfig,
        bundle_indices: Optional[list[int]] = None,
        fraction_of_gpus: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Initialize a vLLM worker for distributed inference.

        Args:
            config: Configuration dictionary for the policy
            bundle_indices: List of local bundle indices within a node for parallelism.
                          Only needed for the first worker in each tied worker group.
            fraction_of_gpus: Fraction of GPUs to use for this worker
            seed: Random seed for initialization
        """
        self.cfg = config

        self.model_name = self.cfg["model_name"]
        self.tensor_parallel_size = self.cfg["vllm_cfg"]["tensor_parallel_size"]
        self.pipeline_parallel_size = self.cfg["vllm_cfg"]["pipeline_parallel_size"]
        self.gpu_memory_utilization = self.cfg["vllm_cfg"]["gpu_memory_utilization"]
        self.fraction_of_gpus = fraction_of_gpus
        self.is_model_owner = bundle_indices is not None

        # Store the Python executable being used by this worker
        self.py_executable = sys.executable

        # Skip model loading if we're not the model owner
        if not self.is_model_owner:
            self.llm = None
            self.tokenizer = None
            self.rank = 0
            self.world_size = 1
            return

        # In Ray+vLLM setup, each worker process considers itself rank 0
        # vLLM handles the parallelism internally through Ray
        self.rank = 0
        self.world_size = 1

        # Monkey patch for vLLM to ensure RAY_ADDRESS is set in Ray actors.
        try:
            import vllm.utils
            from vllm.logger import init_logger
            from vllm.utils import cuda_is_initialized, is_in_ray_actor

            logger = init_logger("vllm_patch")

            def _patched_maybe_force_spawn():
                """Patched version of vllm.utils._maybe_force_spawn.

                This patch changes an `elif is_in_ray_actor()` to an `if` statement.
                This ensures that `os.environ["RAY_ADDRESS"]` is set when running
                within a Ray actor, even if CUDA has already been initialized.
                This is crucial for vLLM workers to connect back to the Ray cluster.
                """
                if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn":
                    return

                reason = None
                if cuda_is_initialized():
                    reason = "CUDA is initialized"

                if is_in_ray_actor():
                    # even if we choose to spawn, we need to pass the ray address
                    # to the subprocess so that it knows how to connect to the ray cluster.
                    # env vars are inherited by subprocesses, even if we use spawn.
                    import ray

                    os.environ["RAY_ADDRESS"] = ray.get_runtime_context().gcs_address
                    if reason is None:
                        reason = "In a Ray actor and can only be spawned"

                if reason is not None:
                    logger.warning(
                        "We must use the `spawn` multiprocessing start method. "
                        "Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. "
                        "See https://docs.vllm.ai/en/latest/getting_started/"
                        "troubleshooting.html#python-multiprocessing "
                        "for more information. Reason: %s",
                        reason,
                    )
                    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

            vllm.utils._maybe_force_spawn = _patched_maybe_force_spawn
            logger.info("Successfully patched vllm.utils._maybe_force_spawn.")

            def _patch_vllm_init_workers_ray():
                # Patch the vLLM ray_distributed_executor.py file to pass custom runtime_env in _init_workers_ray call.
                # This allows passing custom py_executable to worker initialization.

                try:
                    import vllm.executor.ray_distributed_executor as ray_executor_module

                    file_to_patch = ray_executor_module.__file__

                    with open(file_to_patch, "r") as f:
                        content = f.read()

                    old_line = "self._init_workers_ray(placement_group)"
                    new_line = f'self._init_workers_ray(placement_group, runtime_env={{"py_executable": "{self.py_executable}"}})'

                    if new_line in content:
                        return

                    if old_line not in content:
                        return

                    patched_content = content.replace(old_line, new_line)

                    # Write back the patched content
                    with open(file_to_patch, "w") as f:
                        f.write(patched_content)

                except (ImportError, FileNotFoundError, PermissionError):
                    # Allow failures gracefully
                    pass

            _patch_vllm_init_workers_ray()

        except (ImportError, AttributeError):
            # vllm not installed or has a different structure, skipping patch.
            pass

        try:
            import vllm

            self.SamplingParams = vllm.SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Please check that the py_executable in the runtime_env of VllmGenerationWorker "
                "covers the vllm dependency. You may have to update nemo_rl/distributed/ray_actor_environment_registry.py. "
                "If you are working interactively, you can install by running  `uv sync --extra vllm` anywhere in the repo."
            )
        vllm_kwargs: dict[str, Any] = copy.deepcopy(self.cfg.get("vllm_kwargs", {}))

        # Calculate total parallel size (TP * PP)
        model_parallel_size = self.tensor_parallel_size * self.pipeline_parallel_size

        # Special handling for parallel case (either TP or PP or both)
        if model_parallel_size > 1:
            # Configure vLLM for tensor/pipeline parallelism within Ray
            # Reset CUDA_VISIBLE_DEVICES to allow vLLM to manage GPU assignment
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(
                self.fraction_of_gpus / model_parallel_size
            )

            # Set bundle indices for parallel workers
            bundle_indices_str = ",".join(map(str, bundle_indices))
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = bundle_indices_str
            print(
                f"VLLM_RAY_BUNDLE_INDICES environment variable set to: {os.environ.get('VLLM_RAY_BUNDLE_INDICES')}"
            )

            # Use Ray for distributed execution in parallel mode
            vllm_kwargs["distributed_executor_backend"] = "ray"
        else:
            # For non-parallel mode, explicitly set executor to None to avoid Ray issues
            vllm_kwargs["distributed_executor_backend"] = None

        os.environ["VLLM_USE_V1"] = os.environ.get("NRL_VLLM_USE_V1", "1")
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        load_format = self.cfg["vllm_cfg"]["load_format"]
        if ModelFlag.VLLM_LOAD_FORMAT_AUTO.matches(self.model_name):
            load_format = "auto"

        llm_kwargs = dict(
            model=self.model_name,
            load_format=load_format,
            skip_tokenizer_init=self.cfg["vllm_cfg"]["skip_tokenizer_init"],
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enable_prefix_caching=torch.cuda.get_device_capability()[0] >= 8,
            dtype=self.cfg["vllm_cfg"]["precision"],
            seed=seed,
            # Don't use cuda-graph by default as it leads to convergence issues (see https://github.com/NVIDIA/NeMo-RL/issues/186)
            enforce_eager=True,
            max_model_len=self.cfg["vllm_cfg"]["max_model_len"],
            trust_remote_code=True,
            worker_extension_cls="nemo_rl.models.generation.vllm_backend.VllmInternalWorkerExtension",
            enable_sleep_mode=True,
            disable_log_stats=True,
            **vllm_kwargs,
        )

        if self.cfg["vllm_cfg"]["async_engine"]:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.v1.engine.async_llm import AsyncLLM

            self.llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**llm_kwargs))
        else:
            self.llm = vllm.LLM(**llm_kwargs)

    def init_collective(self, data: int, ip: str, port: int, world_size: int) -> None:
        self.llm.collective_rpc(
            "init_collective",
            args=(
                data,
                ip,
                port,
                world_size,
            ),
        )

    async def init_collective_async(
        self, data: int, ip: str, port: int, world_size: int
    ) -> None:
        await self.llm.collective_rpc(
            "init_collective",
            args=(
                data,
                ip,
                port,
                world_size,
            ),
        )

    def llm(self):
        return self.llm

    def is_alive(self):
        """Check if the worker is alive."""
        return True

    def _merge_stop_strings(self, batch_stop_strings):
        stop_set: set[str] = set()

        if self.cfg.get("stop_strings"):
            stop_set.update(self.cfg["stop_strings"])

        if batch_stop_strings is not None:
            for sample_ss in batch_stop_strings:
                if sample_ss:
                    stop_set.update(sample_ss)

        return list(stop_set) if stop_set else None

    def _build_sampling_params(
        self,
        *,
        greedy: bool,
        stop_strings,
        max_new_tokens: Optional[int] = None,
    ):
        top_k_cfg = self.cfg["top_k"]
        top_k_val = 1 if greedy else (top_k_cfg if top_k_cfg is not None else -1)

        temperature = 0.0 if greedy else self.cfg["temperature"]

        max_tokens = (
            max_new_tokens if max_new_tokens is not None else self.cfg["max_new_tokens"]
        )

        return self.SamplingParams(
            temperature=temperature,
            top_p=self.cfg["top_p"],
            top_k=top_k_val,
            max_tokens=max_tokens,
            logprobs=0,
            stop_token_ids=self.cfg["stop_token_ids"],
            stop=stop_strings,
            include_stop_str_in_output=True,
        )

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using vLLM generation.

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
            # Return empty BatchedDataDict with all required fields
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
        batch_stop_strings: list[list[str]] = data.get("stop_strings", [])
        stop_strings = self._merge_stop_strings(batch_stop_strings)
        sampling_params = self._build_sampling_params(
            greedy=greedy,
            stop_strings=stop_strings,
        )

        # verify inputs have correct padding
        verify_right_padding(data, pad_value=self.cfg["pad_token_id"])

        # Convert inputs to vLLM format
        batch_size = input_ids.shape[0]
        # Original input length with padding
        padded_input_length = input_ids.size(1)

        # Prepare prompts for vLLM (removing padding)
        prompts = []

        for i in range(batch_size):
            # Use input_lengths to get only valid tokens (not padding)
            valid_length = input_lengths[i].item()
            valid_ids = (
                input_ids[i, :valid_length] if valid_length > 0 else input_ids[i, :0]
            )
            token_ids = valid_ids.tolist()

            prompts.append({"prompt_token_ids": token_ids})

        # Generate outputs
        assert self.llm is not None, (
            "Attempting to generate with either an uninitialized vLLM or non-model-owner"
        )
        outputs = self.llm.generate(prompts, sampling_params)

        # Process the outputs - but preserve the original input padding structure
        output_ids_list = []
        logprobs_list = []
        generation_lengths = []
        unpadded_sequence_lengths = []
        max_length = 0
        for output in outputs:
            max_length = max(max_length, len(output.outputs[0].token_ids))

        for i, output in enumerate(outputs):
            # Extract generated tokens
            sequence_length = input_lengths[i]
            generation = output.outputs[0]
            generated_tokens = list(generation.token_ids)

            # Calculate total sequence length (original input length + generated tokens)
            total_length = padded_input_length + max_length

            # Create a new tensor with the right size and fill with padding token
            full_output = torch.full(
                (total_length,), self.cfg["pad_token_id"], dtype=input_ids.dtype
            )

            # Copy original input (with padding) into the beginning
            full_output[:sequence_length] = input_ids[i][:sequence_length]

            # Add generated tokens after the original input
            full_output[sequence_length : sequence_length + len(generated_tokens)] = (
                torch.tensor(generated_tokens)
            )

            output_ids_list.append(full_output)
            full_logprobs = torch.zeros(total_length, dtype=torch.float32)
            if hasattr(generation, "logprobs") and generation.logprobs:
                try:
                    for idx, logprob_dict in enumerate(generation.logprobs):
                        if logprob_dict:
                            position = sequence_length + idx
                            full_logprobs[position] = next(iter(logprob_dict.items()))[
                                1
                            ].logprob
                except Exception:
                    import traceback

                    traceback.print_exc()

            logprobs_list.append(full_logprobs)

            response_length = sequence_length + len(generated_tokens)
            generation_lengths.append(len(generated_tokens))
            unpadded_sequence_lengths.append(response_length)
            assert response_length <= self.llm.llm_engine.model_config.max_model_len, (
                f"response_length={response_length} > max_model_len={self.llm.llm_engine.model_config.max_model_len}, which should not happen. Please check this behavior in isolation by running `uv run --extra vllm tools/model_diagnostics/1.max_model_len_respected.py {self.llm.llm_engine.model_config.model}` and raise this issue with the vllm team."
            )

        # Create return data conforming to GenerationOutputSpec
        output_ids = torch.stack(output_ids_list)
        logprobs = torch.stack(logprobs_list)

        return_data = BatchedDataDict[GenerationOutputSpec](
            {
                "output_ids": output_ids,
                "logprobs": logprobs,
                "generation_lengths": torch.tensor(
                    generation_lengths, dtype=torch.long
                ),
                "unpadded_sequence_lengths": torch.tensor(
                    unpadded_sequence_lengths, dtype=torch.long
                ),
            }
        )

        return return_data

    async def generate_async(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        greedy: bool = False,
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Generate a batch of data using vLLM's AsyncLLMEngine, yielding results as they are ready.

        Args:
            data: BatchedDataDict with input_ids and input_lengths
            greedy: Whether to use greedy decoding instead of sampling

        Yields:
            Tuple of (original_index, BatchedDataDict conforming to GenerationOutputSpec for the single sequence)
        """
        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "generate_async can only be used when async_engine is enabled in vLLM config."
            )

        # Handle empty input case
        if len(data["input_ids"]) == 0:
            return

        verify_right_padding(data, pad_value=self.cfg["pad_token_id"])

        input_ids_batch = data["input_ids"]
        input_lengths_batch = data["input_lengths"]
        batch_size = input_ids_batch.shape[0]

        # Ensure generate_async only receives single samples (batch_size = 1)
        assert batch_size == 1, (
            f"generate_async is restricted to handle only single samples, "
            f"but received batch_size={batch_size}. Please handle batching outside this method."
        )

        batch_specific_stop_strings_list = data.get(
            "stop_strings", [[] for _ in range(batch_size)]
        )

        # Create tasks for each sample in the batch
        async def process_single_sample(sample_idx):
            """Process a single sample and return the result."""
            current_input_actual_length = input_lengths_batch[sample_idx].item()
            prompt_token_ids_list = (
                input_ids_batch[sample_idx, :current_input_actual_length].tolist()
                if current_input_actual_length > 0
                else []
            )
            prompt = {"prompt_token_ids": prompt_token_ids_list}

            per_sample_stop_strings = None
            if batch_specific_stop_strings_list and sample_idx < len(
                batch_specific_stop_strings_list
            ):
                per_sample_stop_strings = batch_specific_stop_strings_list[sample_idx]

            final_stop_strings_for_sample = self._merge_stop_strings(
                [per_sample_stop_strings] if per_sample_stop_strings else None
            )

            remaining_ctx = (
                self.cfg["vllm_cfg"]["max_model_len"] - current_input_actual_length
            )
            allowed_new_tokens = max(0, min(self.cfg["max_new_tokens"], remaining_ctx))

            # Handle case where no tokens can be generated due to length constraints
            if allowed_new_tokens == 0:
                # Access the input data directly from the function parameters
                input_ids_single_row = input_ids_batch[sample_idx]

                # Create output tensors with just the input (no generated tokens)
                output_ids_single_item_batched = input_ids_single_row[
                    :current_input_actual_length
                ].unsqueeze(0)

                logprobs_single_item = torch.zeros(
                    (1, current_input_actual_length),
                    dtype=torch.float32,
                    device=input_ids_single_row.device,
                )

                generation_lengths_tensor = torch.tensor(
                    [0], dtype=torch.long, device=input_ids_single_row.device
                )

                unpadded_sequence_lengths_tensor = torch.tensor(
                    [current_input_actual_length],
                    dtype=torch.long,
                    device=input_ids_single_row.device,
                )

                result_batch = BatchedDataDict[GenerationOutputSpec](
                    {
                        "output_ids": output_ids_single_item_batched,
                        "logprobs": logprobs_single_item,
                        "generation_lengths": generation_lengths_tensor,
                        "unpadded_sequence_lengths": unpadded_sequence_lengths_tensor,
                    }
                )

                return (sample_idx, result_batch)

            sampling_params_for_request = self._build_sampling_params(
                greedy=greedy,
                stop_strings=final_stop_strings_for_sample,
                max_new_tokens=allowed_new_tokens,
            )

            request_id = str(uuid.uuid4())

            # Generate using vLLM async engine
            vllm_request_generator = self.llm.generate(
                prompt=prompt,
                sampling_params=sampling_params_for_request,
                request_id=request_id,
            )

            # Get the final result from the generator
            final_request_output = None
            async for req_output in vllm_request_generator:
                final_request_output = req_output

            if final_request_output is None:
                raise RuntimeError(f"No output received for request {request_id}")

            # Process the output
            generation_details = final_request_output.outputs[0]
            generated_token_ids = list(generation_details.token_ids)
            num_generated_tokens = len(generated_token_ids)

            original_input_ids_single_row = input_ids_batch[sample_idx]
            final_output_tensor_len = current_input_actual_length + num_generated_tokens

            # Create output_ids tensor for this single item
            output_ids_single_item = torch.full(
                (final_output_tensor_len,),
                self.cfg["pad_token_id"],
                dtype=original_input_ids_single_row.dtype,
                device=original_input_ids_single_row.device,
            )
            # Copy original input (up to its actual length)
            output_ids_single_item[:current_input_actual_length] = (
                original_input_ids_single_row[:current_input_actual_length]
            )
            # Add generated tokens after the actual input
            output_ids_single_item[
                current_input_actual_length : current_input_actual_length
                + num_generated_tokens
            ] = torch.tensor(
                generated_token_ids,
                dtype=original_input_ids_single_row.dtype,
                device=original_input_ids_single_row.device,
            )

            # Reshape to (1, seq_len) for BatchedDataDict
            output_ids_single_item_batched = output_ids_single_item.unsqueeze(0)

            # Create logprobs tensor for this single item
            logprobs_single_item = torch.zeros(
                (1, final_output_tensor_len),
                dtype=torch.float32,
                device=original_input_ids_single_row.device,
            )
            if hasattr(generation_details, "logprobs") and generation_details.logprobs:
                for idx, logprob_dict_per_token in enumerate(
                    generation_details.logprobs
                ):
                    if logprob_dict_per_token and idx < len(generated_token_ids):
                        token_id_at_idx = generated_token_ids[idx]
                        if token_id_at_idx in logprob_dict_per_token:
                            logprob_value = logprob_dict_per_token[
                                token_id_at_idx
                            ].logprob
                            position_in_output_tensor = (
                                current_input_actual_length + idx
                            )
                            if position_in_output_tensor < final_output_tensor_len:
                                logprobs_single_item[0, position_in_output_tensor] = (
                                    logprob_value
                                )

            # Generation lengths
            generation_lengths_tensor = torch.tensor(
                [num_generated_tokens],
                dtype=torch.long,
                device=original_input_ids_single_row.device,
            )

            # Unpadded sequence lengths (actual_input + actual_generated)
            unpadded_total_length = current_input_actual_length + num_generated_tokens
            unpadded_sequence_lengths_tensor = torch.tensor(
                [unpadded_total_length],
                dtype=torch.long,
                device=original_input_ids_single_row.device,
            )

            result_batch = BatchedDataDict[GenerationOutputSpec](
                {
                    "output_ids": output_ids_single_item_batched,
                    "logprobs": logprobs_single_item,
                    "generation_lengths": generation_lengths_tensor,
                    "unpadded_sequence_lengths": unpadded_sequence_lengths_tensor,
                }
            )

            return (sample_idx, result_batch)

        # Create tasks for all samples and yield results as they complete
        sample_tasks = [
            asyncio.create_task(process_single_sample(i)) for i in range(batch_size)
        ]

        # Yield results as they become available
        for completed_task in asyncio.as_completed(sample_tasks):
            try:
                result = await completed_task
                yield result
            except Exception as e:
                # Cancel remaining tasks
                for task in sample_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*sample_tasks, return_exceptions=True)
                raise e

    def generate_text(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate text responses using vLLM generation.

        Args:
            data: BatchedDataDict containing prompts with text strings
            greedy: Whether to use greedy decoding instead of sampling

        Returns:
            BatchedDataDict containing:
                - texts: List of generated text responses
        """
        # Extract stop_strings if provided, else use default from config
        batch_stop_strings: list[list[str] | None] = data.get(
            "stop_strings", [self.cfg.get("stop_strings")] * len(data["prompts"])
        )

        # This function requires all generations have the same stop strings, so we collect all here
        stop_strings: set[str] = set()
        for sample_stop_strings in batch_stop_strings:
            if sample_stop_strings:
                stop_strings.update(sample_stop_strings)

        # Add default stop strings from config
        if self.cfg.get("stop_strings", None):
            stop_strings.update(self.cfg["stop_strings"])

        stop_strings: list[str] | None = (
            list(stop_strings) if len(stop_strings) > 0 else None
        )

        # Read generation parameters from config
        top_k = self.cfg["top_k"] if self.cfg["top_k"] is not None else -1
        sampling_params = self.SamplingParams(
            temperature=self.cfg["temperature"] if not greedy else 0,
            top_p=self.cfg["top_p"],
            top_k=top_k if not greedy else 1,
            max_tokens=self.cfg["max_new_tokens"],
            stop_token_ids=self.cfg["stop_token_ids"],
            stop=stop_strings,
            include_stop_str_in_output=True,  # returning stop strings like hf
        )

        # Generate outputs
        assert self.llm is not None, (
            "Attempting to generate with either an uninitialized vLLM or non-model-owner"
        )
        outputs = self.llm.generate(data["prompts"], sampling_params)
        texts = [output.outputs[0].text for output in outputs]

        # Convert to BatchedDataDict
        return_data: BatchedDataDict[GenerationOutputSpec] = BatchedDataDict(
            {"texts": texts}
        )
        return return_data

    def shutdown(self) -> bool:
        """Clean up vLLM resources."""
        try:
            if self.llm is not None:
                is_async_engine = self.cfg.get("vllm_cfg", {}).get(
                    "async_engine", False
                )

                if is_async_engine:
                    try:
                        self.llm.shutdown()
                    except Exception as e_stop:
                        print(f"Error calling shutdown_background_loop: {e_stop}")
                # Explicitly delete the engine. This may trigger its __del__ method.
                del self.llm

            self.llm = None
            self.tokenizer = None

            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()

            return True
        except Exception as e:
            print(f"Error during vLLM shutdown: {e}")
            return False

    def report_device_id(self) -> list[str]:
        """Report device ID from the vLLM worker."""
        assert self.llm is not None, (
            "Attempting to report device id with either an uninitialized vLLM or non-model-owner"
        )

        if self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "report_device_id cannot be used with async_engine=True. Use report_device_id_async instead."
            )

        list_of_worker_results = self.llm.collective_rpc(
            "report_device_id", args=tuple()
        )
        return cast(list[str], list_of_worker_results)

    async def report_device_id_async(self) -> list[str]:
        """Async version of report_device_id."""
        assert self.llm is not None, (
            "Attempting to report device id with either an uninitialized vLLM or non-model-owner"
        )

        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "report_device_id_async can only be used with async_engine=True. Use report_device_id instead."
            )

        result_or_coro = await self.llm.collective_rpc("report_device_id", args=tuple())

        if asyncio.iscoroutine(result_or_coro):
            list_of_worker_results = await result_or_coro
        else:
            list_of_worker_results = result_or_coro

        return cast(list[str], list_of_worker_results)

    def update_weights_from_ipc_handles(self, data: dict[str, Any]) -> bool:
        """Update weights from IPC handles by delegating to the vLLM Worker implementation.

        Args:
            data (dict): Dictionary mapping device UUIDs (str) to parameter IPC handles.

        Returns:
            bool: True if weights were successfully updated, False otherwise.
        """
        try:
            assert self.llm is not None, (
                "Attempting to update weights with either an uninitialized vLLM or non-model-owner"
            )

            if self.cfg["vllm_cfg"]["async_engine"]:
                raise RuntimeError(
                    "update_weights_from_ipc_handles cannot be used with async_engine=True. Use update_weights_from_ipc_handles_async instead."
                )

            result_or_coro = self.llm.collective_rpc(
                "update_weights_from_ipc_handles", args=(data,)
            )
            worker_result = result_or_coro[0]

            if not worker_result:
                print(
                    f"Error: Worker failed to update weights. Result: {worker_result}"
                )
                return False
            return True
        except Exception as e:
            print(f"Exception during collective_rpc for weight update: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def update_weights_from_ipc_handles_async(self, data: dict[str, Any]) -> bool:
        """Async version of update_weights_from_ipc_handles.

        Args:
            data (dict): Dictionary mapping device UUIDs (str) to parameter IPC handles.

        Returns:
            bool: True if weights were successfully updated, False otherwise.
        """
        try:
            assert self.llm is not None, (
                "Attempting to update weights with either an uninitialized vLLM or non-model-owner"
            )

            if not self.cfg["vllm_cfg"]["async_engine"]:
                raise RuntimeError(
                    "update_weights_from_ipc_handles_async can only be used with async_engine=True. Use update_weights_from_ipc_handles instead."
                )

            result_or_coro = await self.llm.collective_rpc(
                "update_weights_from_ipc_handles", args=(data,)
            )

            if asyncio.iscoroutine(result_or_coro):
                worker_results = await result_or_coro
            else:
                worker_results = result_or_coro

            worker_result = worker_results[0]

            if not worker_result:
                print(
                    f"Error: Worker failed to update weights. Result: {worker_result}"
                )
                return False
            return True
        except Exception as e:
            print(f"Exception during collective_rpc for weight update: {e}")
            import traceback

            traceback.print_exc()
            return False

    def update_weights_from_collective(self, data: dict[str, Any]) -> bool:
        """Update the model weights from collective communication."""
        try:
            assert self.llm is not None, (
                "Attempting to update weights with either an uninitialized vLLM or non-model-owner"
            )

            if self.cfg["vllm_cfg"]["async_engine"]:
                raise RuntimeError(
                    "update_weights_from_collective can only be used with async_engine=False. Use update_weights_from_collective_async instead."
                )

            result_or_coro = self.llm.collective_rpc(
                "update_weights_from_collective", args=(data,)
            )
            worker_result = result_or_coro[0]

            if not worker_result:
                print(
                    f"Error: Worker failed to update weights. Result: {worker_result}"
                )
                return False
            return True
        except Exception as e:
            print(f"Exception during collective_rpc for weight update: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def update_weights_from_collective_async(self, data: dict[str, Any]) -> bool:
        """Async version of update_weights_from_collective."""
        try:
            assert self.llm is not None, (
                "Attempting to update weights with either an uninitialized vLLM or non-model-owner"
            )

            if not self.cfg["vllm_cfg"]["async_engine"]:
                raise RuntimeError(
                    "update_weights_from_collective_async can only be used with async_engine=True. Use update_weights_from_collective instead."
                )

            result_or_coro = await self.llm.collective_rpc(
                "update_weights_from_collective", args=(data,)
            )

            if asyncio.iscoroutine(result_or_coro):
                worker_results = await result_or_coro
            else:
                worker_results = result_or_coro

            worker_result = worker_results[0]

            if not worker_result:
                print(
                    f"Error: Worker failed to update weights. Result: {worker_result}"
                )
                return False
            return True
        except Exception as e:
            print(f"Exception during collective_rpc for weight update: {e}")
            import traceback

            traceback.print_exc()
            return False

    def reset_prefix_cache(self):
        """Reset the prefix cache of vLLM engine."""
        assert self.llm is not None, (
            "Attempting to reset prefix cache with either an uninitialized vLLM or non-model-owner"
        )

        if self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "reset_prefix_cache can only be used with async_engine=False. Use reset_prefix_cache_async instead."
            )

        self.llm.llm_engine.reset_prefix_cache()
        gc.collect()
        torch.cuda.empty_cache()

    async def reset_prefix_cache_async(self):
        """Async version of reset_prefix_cache."""
        assert self.llm is not None, (
            "Attempting to reset prefix cache with either an uninitialized vLLM or non-model-owner"
        )

        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "reset_prefix_cache_async can only be used with async_engine=True. Use reset_prefix_cache instead."
            )

        await self.llm.reset_prefix_cache()
        gc.collect()
        torch.cuda.empty_cache()

    def sleep(self):
        """Put the vLLM engine to sleep."""
        assert self.llm is not None, (
            "Attempting to sleep with either an uninitialized vLLM or non-model-owner"
        )

        if self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "sleep cannot be used with async_engine=True. Use sleep_async instead."
            )

        # Reset the prefix cache to ensure that prefix cache is not reused after weights are updated
        self.llm.llm_engine.reset_prefix_cache()
        self.llm.sleep(level=1)

        gc.collect()
        torch.cuda.empty_cache()

    async def sleep_async(self):
        """Async version of sleep."""
        assert self.llm is not None, (
            "Attempting to sleep with either an uninitialized vLLM or non-model-owner"
        )

        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "sleep_async can only be used with async_engine=True. Use sleep instead."
            )

        # Reset the prefix cache to ensure that prefix cache is not reused after weights are updated
        await self.llm.reset_prefix_cache()
        await self.llm.sleep(level=1)

        gc.collect()
        torch.cuda.empty_cache()

    def wake_up(self, **kwargs):
        """Wake up the vLLM engine."""
        assert self.llm is not None, (
            "Attempting to wake up with either an uninitialized vLLM or non-model-owner"
        )

        if self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "wake_up cannot be used with async_engine=True. Use wake_up_async instead."
            )

        tags = kwargs.get("tags")

        wake_up_args = {}
        if tags is not None:
            wake_up_args["tags"] = tags

        self.llm.wake_up(**wake_up_args)

    async def wake_up_async(self, **kwargs):
        """Async version of wake_up."""
        assert self.llm is not None, (
            "Attempting to wake up with either an uninitialized vLLM or non-model-owner"
        )

        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "wake_up_async can only be used with async_engine=True. Use wake_up instead."
            )

        tags = kwargs.get("tags")

        wake_up_args = {}
        if tags is not None:
            wake_up_args["tags"] = tags

        await self.llm.wake_up(**wake_up_args)

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()


class VllmGeneration(GenerationInterface):
    def __init__(
        self,
        cluster: RayVirtualCluster,
        config: VllmConfig,
        name_prefix: str = "vllm_policy",
        workers_per_node: Optional[Union[int, list[int]]] = None,
    ):
        """Initialize a vLLM policy with distributed workers."""
        # Store config
        self.cfg = config
        if self.cfg["vllm_cfg"]["pipeline_parallel_size"] > 1:
            assert self.cfg["vllm_cfg"]["async_engine"], (
                "When pipeline_parallel_size > 1, async_engine must be set to True in the vLLM configuration. "
                "You can enable it by adding `policy.generation.vllm_cfg.async_engine=true` to your command."
            )

        # Ensure all required VllmConfig fields are present
        missing_keys = [
            key for key in VllmConfig.__required_keys__ if key not in self.cfg
        ]
        assert not missing_keys, (
            f"VLLM Configuration Error: Missing required keys in VllmConfig.\n"
            f"Missing keys: {', '.join(missing_keys)}\n"
            f"Provided keys: {', '.join(self.cfg.keys())}\n"
            f"Please update your configuration to include all required VLLM parameters."
        )

        self.sharding_annotations = NamedSharding(
            layout=np.arange(cluster.world_size()).reshape(
                -1,  # DP
                config["vllm_cfg"]["pipeline_parallel_size"],  # PP
                config["vllm_cfg"]["tensor_parallel_size"],  # TP
            ),
            names=["data_parallel", "pipeline_parallel", "tensor_parallel"],
        )
        self.model_parallel_size = self.sharding_annotations.get_axis_size(
            "tensor_parallel"
        ) * self.sharding_annotations.get_axis_size("pipeline_parallel")

        # Determine if we need cross-node model parallelism
        needs_cross_node_parallelism = (
            self.model_parallel_size > cluster.num_gpus_per_node
        )

        # Initialize placement groups with the appropriate mode
        cluster._init_placement_groups(use_unified_pg=needs_cross_node_parallelism)

        # Create worker builder for VllmGenerationWorker
        worker_builder = RayWorkerBuilder(
            "nemo_rl.models.generation.vllm.VllmGenerationWorker", config
        )

        # Check if we need parallelism-aware worker group creation
        if self.model_parallel_size > 1:
            # For parallelism, create node-aware worker groups
            node_bundle_indices = self._get_tied_worker_bundle_indices(cluster)

            self.worker_group = RayWorkerGroup(
                cluster,
                worker_builder,
                name_prefix=name_prefix,
                bundle_indices_list=node_bundle_indices,
                sharding_annotations=self.sharding_annotations,
            )
        else:
            # Use standard worker group creation for non-parallel case
            self.worker_group = RayWorkerGroup(
                cluster,
                worker_builder,
                name_prefix=name_prefix,
                workers_per_node=workers_per_node,
                sharding_annotations=self.sharding_annotations,
            )

        # Number of data parallel groups is the number of tied worker groups
        self.dp_size = self.worker_group.dp_size

        # Used to track the round-robin selection of worker groups for generate_async
        self.current_generate_dp_shard_idx = 0

        # Save the device uuids for the workers
        self.device_uuids = self._report_device_id()

    def _get_tied_worker_bundle_indices(
        self, cluster: RayVirtualCluster
    ) -> List[Tuple[int, List[int]]]:
        """Calculate bundle indices for tensor and pipeline parallel workers.

        Handles both unified placement groups (for cross-node model parallelism) and
        per-node placement groups (for node-local model parallelism).
        """
        # Get the placement groups from the cluster
        placement_groups = cluster.get_placement_groups()

        if not placement_groups:
            raise ValueError("No placement groups available in the cluster")

        # Total parallel sizes
        tp_size = self.sharding_annotations.get_axis_size("tensor_parallel")
        pp_size = self.sharding_annotations.get_axis_size("pipeline_parallel")
        model_parallel_size = tp_size * pp_size

        if len(placement_groups) == 1:
            # Single unified placement group used when we need multiple nodes for model parallelism
            unified_pg = placement_groups[0]

            def get_node_bundles(
                pg: PlacementGroup,
            ) -> Dict[str, List[int]]:
                # Retrieve mapping from node ID to bundle indices from a placement group.
                try:
                    pg_table = ray.util.placement_group_table(pg)
                    bundle_to_node = pg_table["bundles_to_node_id"]
                except Exception as e:
                    raise RuntimeError(
                        "Failed to retrieve bundle/node mapping from placement group"
                    ) from e

                node_bundles: Dict[str, List[int]] = defaultdict(list)
                for bundle_idx, node_id in bundle_to_node.items():
                    node_bundles[node_id].append(bundle_idx)
                for bundles in node_bundles.values():
                    bundles.sort()
                return dict(node_bundles)

            def allocate_worker_groups(
                pg: PlacementGroup, tp_size: int, pp_size: int
            ) -> List[Tuple[int, List[int]]]:
                # Allocate worker groups for TP and PP training, assuming all nodes have identical bundle counts.

                # Retrieve both bundle mapping and per-node bundles
                pg_table = ray.util.placement_group_table(pg)
                bundle_to_node = pg_table["bundles_to_node_id"]
                node_bundles = get_node_bundles(pg)

                if not node_bundles:
                    raise ValueError("Placement group contains no bundles")

                # Ensure all nodes have the same number of bundles
                counts = [len(b) for b in node_bundles.values()]
                assert len(set(counts)) == 1, (
                    "All nodes must have identical bundle counts"
                )

                total = sum(counts)
                model_parallel_size = tp_size * pp_size
                num_groups = total // model_parallel_size
                if num_groups == 0:
                    raise ValueError(
                        "Unable to allocate any worker groups with the available resources."
                    )

                # Create reproducible node indices
                sorted_nodes = sorted(node_bundles)
                node_idx = {nid: idx for idx, nid in enumerate(sorted_nodes)}

                # Flatten bundles in node order
                flat: List[int] = []
                for nid in sorted_nodes:
                    flat.extend(node_bundles[nid])

                # Slice into groups and assign logical index
                groups: List[Tuple[int, List[int]]] = []
                for i in range(num_groups):
                    slice_ = flat[
                        i * model_parallel_size : (i + 1) * model_parallel_size
                    ]
                    first_node = bundle_to_node[slice_[0]]
                    groups.append((node_idx[first_node], slice_))

                return groups

            tied_groups = allocate_worker_groups(unified_pg, tp_size, pp_size)
        else:
            tied_groups = []
            # For per-node PGs, each PG represents a node
            for pg_idx, pg in enumerate(placement_groups):
                if pg.bundle_count == 0:
                    continue

                # Check if this PG has enough bundles for at least one group
                num_groups_in_pg = pg.bundle_count // model_parallel_size

                # Create groups within this PG
                for group_idx in range(num_groups_in_pg):
                    start_idx = group_idx * model_parallel_size
                    end_idx = start_idx + model_parallel_size
                    bundle_indices = list(range(start_idx, end_idx))
                    # Use pg_idx as the node identifier
                    tied_groups.append((pg_idx, bundle_indices))

        if not tied_groups:
            raise ValueError(
                "Unable to allocate any worker groups with the available resources."
            )

        return tied_groups

    def _report_device_id(self) -> list[list[str]]:
        """Report the device ID of vllm workers."""
        # Choose the appropriate method based on async_engine setting
        method_name = (
            "report_device_id_async"
            if self.cfg["vllm_cfg"]["async_engine"]
            else "report_device_id"
        )
        # Use run_all_workers_single_data for methods that don't need data
        futures = self.worker_group.run_all_workers_single_data(
            method_name, run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"]
        )
        # Wait for all futures to complete
        results = ray.get(futures)
        return results

    def init_collective(
        self, ip: str, port: int, world_size: int
    ) -> list[ray.ObjectRef]:
        """Initialize the collective communication."""
        if not self.worker_group or not self.worker_group.workers:
            raise RuntimeError("Worker group is not initialized")

        # Choose the appropriate method based on async_engine setting
        method_name = (
            "init_collective_async"
            if self.cfg["vllm_cfg"]["async_engine"]
            else "init_collective"
        )

        # Prepare rank
        total_workers = len(self.worker_group.workers)
        if self.dp_size == 0:
            raise RuntimeError(
                "Data parallel size is zero, cannot initialize collective."
            )
        workers_per_group = total_workers // self.dp_size
        rank_prefix_list = list(range(0, total_workers, workers_per_group))

        # Send world_size and rank for init collective to all workers
        futures = self.worker_group.run_all_workers_multiple_data(
            method_name,
            data=rank_prefix_list,
            run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
            common_kwargs={"ip": ip, "port": port, "world_size": world_size},
        )

        # this function should co-work with lm_policy, so we should wait for all futures to complete outside
        return futures

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using vLLM."""
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            "input_ids and input_lengths are required in data for vLLM generation"
        )

        # Shard the data across the tied worker groups
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data: list[SlicedDataDict] = data.shard_by_batch_size(
            dp_size, allow_uneven_shards=True
        )
        future_bundle = self.worker_group.run_all_workers_sharded_data(
            "generate",
            sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=None,  # just run on tp rank 0
            output_is_replicated=None,
            common_kwargs={"greedy": greedy},
        )

        # Get results from the workers, respecting tied worker groups (only one result per tied worker group)
        results = self.worker_group.get_all_worker_results(future_bundle)

        # Combine results from all tied worker groups
        combined: BatchedDataDict[GenerationOutputSpec] = BatchedDataDict.from_batches(
            results, pad_value_dict={"output_ids": self.cfg["pad_token_id"]}
        )

        # Verify the output has all required fields
        required_keys = [
            "output_ids",
            "generation_lengths",
            "unpadded_sequence_lengths",
            "logprobs",
        ]
        missing_keys = [key for key in required_keys if key not in combined]
        if missing_keys:
            raise ValueError(
                f"Missing required keys for GenerationOutputSpec: {missing_keys}"
            )

        return combined

    def generate_text(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate text responses using vLLM."""
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )

        # Shard the data across the tied worker groups
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data: list[SlicedDataDict] = data.shard_by_batch_size(
            dp_size, allow_uneven_shards=True
        )
        future_bundle = self.worker_group.run_all_workers_sharded_data(
            "generate_text",
            sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=None,  # just run on tp rank 0
            output_is_replicated=None,
            common_kwargs={"greedy": greedy},
        )

        # Get results from the workers, respecting tied worker groups (only one result per tied worker group)
        results = self.worker_group.get_all_worker_results(future_bundle)

        # Combine results from all tied worker groups
        combined: BatchedDataDict[GenerationOutputSpec] = BatchedDataDict.from_batches(
            results, pad_value_dict={"output_ids": self.cfg["pad_token_id"]}
        )

        # Verify the output has all required fields
        required_keys = ["texts"]
        missing_keys = [key for key in required_keys if key not in combined]
        if missing_keys:
            raise ValueError(
                f"Missing required keys for GenerationOutputSpec: {missing_keys}"
            )

        return combined

    async def generate_async(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Generate responses asynchronously, yielding individual samples as they complete.

        This method provides per-sample streaming across all workers, yielding each
        sample result as soon as it's ready, regardless of which worker processed it.
        """
        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "generate_async can only be used when async_engine is enabled in VllmConfig."
            )

        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            "input_ids and input_lengths are required in data for vLLM generation"
        )

        # Handle empty input case
        if len(data["input_ids"]) == 0:
            return

        # Determine the leader worker for the current data parallel shard
        leader_worker_idx = self.worker_group.get_dp_leader_worker_idx(
            self.current_generate_dp_shard_idx
        )

        # Run the generate_async method on the selected leader worker. This returns an ObjectRefGenerator.
        worker_gen_proxy = self.worker_group.run_single_worker_single_data(
            method_name="generate_async",
            worker_idx=leader_worker_idx,
            data=data,
            greedy=greedy,
        )

        # Increment the round-robin worker group index
        self.current_generate_dp_shard_idx += 1
        self.current_generate_dp_shard_idx %= self.worker_group.dp_size

        # Create a queue to collect sample results from the worker as they complete
        result_queue = asyncio.Queue()
        finished = False

        async def consume_worker_generator(worker_idx, worker_gen):
            """Consume a single worker generator and put sample results in the queue."""
            nonlocal finished
            worker_name = f"Worker-{worker_idx}"
            try:
                async for sample_result_ref in worker_gen:
                    sample_result = await sample_result_ref
                    await result_queue.put(("sample", sample_result))
            except Exception as e:
                # Log the error before putting it in the queue for better debugging
                import traceback

                print(f"Exception in worker {worker_name}")
                traceback.print_exc()
                await result_queue.put(("error", e))
            finally:
                finished = True
                await result_queue.put(("worker_done", None))

        # Start the task to consume the worker generator
        worker_task = asyncio.create_task(
            consume_worker_generator(leader_worker_idx, worker_gen_proxy)
        )

        # Yield sample results as they become available from the worker
        timeout_seconds = float(
            os.environ.get("NRL_VLLM_ASYNC_TIMEOUT_SECONDS", "600")
        )  # Default 10 minutes

        while not finished:
            try:
                msg_type, item = await asyncio.wait_for(
                    result_queue.get(), timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                print(
                    f"Timeout waiting for results after {timeout_seconds}s. Worker has not finished."
                )
                print(
                    f"For longer sequences, increase the timeout by setting: export NRL_VLLM_ASYNC_TIMEOUT_SECONDS={int(timeout_seconds * 2)}"
                )
                # Cancel the task
                if not worker_task.done():
                    worker_task.cancel()
                await asyncio.gather(worker_task, return_exceptions=True)
                raise RuntimeError(
                    f"Timeout waiting for worker results after {timeout_seconds}s. "
                    f"For longer sequences, increase timeout by setting: export NRL_VLLM_ASYNC_TIMEOUT_SECONDS={int(timeout_seconds * 2)}"
                )

            if msg_type == "sample":
                # Yield individual sample result immediately
                yield item
            elif msg_type == "error":
                # Cancel the task and propagate error
                if not worker_task.done():
                    worker_task.cancel()
                await asyncio.gather(worker_task, return_exceptions=True)
                raise item
            elif msg_type == "worker_done":
                # Worker finished, just continue the loop
                pass
            else:
                raise RuntimeError(f"Unexpected message type: {msg_type}")

        # Verify the task is actually done
        assert worker_task.done(), (
            f"Worker task {leader_worker_idx} should be done but isn't"
        )

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Wake workers up for colocated inference."""
        # non-colocated no need to wake up
        if not self.cfg["colocated"]["enabled"]:
            return True

        try:
            # Choose the appropriate method based on async_engine setting
            method_name = (
                "wake_up_async" if self.cfg["vllm_cfg"]["async_engine"] else "wake_up"
            )
            # Use run_all_workers_single_data for methods that don't need data
            futures = self.worker_group.run_all_workers_single_data(
                method_name,
                run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
                **kwargs,
            )
            # Wait for all futures to complete
            results = ray.get(futures)
            return all(result for result in results if result is not None)
        except Exception as e:
            print(f"Error during policy preparation: {e}")
            return False

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Sleep workers and reset prefix cache."""
        try:
            # Choose the appropriate method based on setting
            # non-colocated only needs reset prefix cache, no need to sleep.
            if self.cfg["colocated"]["enabled"]:
                method_name = (
                    "sleep_async" if self.cfg["vllm_cfg"]["async_engine"] else "sleep"
                )
            else:
                method_name = (
                    "reset_prefix_cache_async"
                    if self.cfg["vllm_cfg"]["async_engine"]
                    else "reset_prefix_cache"
                )
            # Use run_all_workers_single_data for methods that don't need data
            futures = self.worker_group.run_all_workers_single_data(
                method_name,
                run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
            )
            # Wait for all futures to complete
            results = ray.get(futures)
            return all(result for result in results if result is not None)
        except Exception as e:
            print(f"Error during policy preparation: {e}")
            return False

    def shutdown(self) -> bool:
        """Shut down all vLLM workers and clean up resources."""
        try:
            # Use the worker group's shutdown method with the worker's cleanup method
            return self.worker_group.shutdown(cleanup_method="shutdown")
        except Exception as e:
            print(f"Error during policy shutdown: {e}")
            return False

    def update_weights(self, ipc_handles: dict[str, Any]) -> bool:
        """Update weights of the policy using IPC handles, considering tensor parallelism.

        For tp > 1, only the leader in each tensor parallel tied worker group will update weights.

        Args:
            ipc_handles (dict): Dictionary mapping device UUIDs (str) to parameter IPC handles.

        Returns:
            bool: True if weights were successfully updated, False otherwise.
        """
        if not self.worker_group or not self.worker_group.workers:
            return False

        # Choose the appropriate method based on async_engine setting
        method_name = (
            "update_weights_from_ipc_handles_async"
            if self.cfg["vllm_cfg"]["async_engine"]
            else "update_weights_from_ipc_handles"
        )

        # Only send the ipc handles required by the current worker
        ipc_handles_list = []
        for worker_device_uuids in self.device_uuids:
            worker_ipc_handles = {
                device_uuid: ipc_handles[device_uuid]
                for device_uuid in worker_device_uuids
            }
            ipc_handles_list.append(worker_ipc_handles)

        try:
            # Directly pass ipc_handles to the method
            futures = self.worker_group.run_all_workers_multiple_data(
                method_name,
                ipc_handles_list,
                run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
            )
            # Wait for all futures to complete
            results = ray.get(futures)
            return all(result for result in results if result is not None)
        except Exception as e:
            print(f"Error during update weights: {e}")
            return False

    def update_weights_from_collective(
        self, info: dict[str, Any]
    ) -> list[ray.ObjectRef]:
        """Update weights of the policy using collective communication."""
        if not self.worker_group or not self.worker_group.workers:
            raise RuntimeError("Worker group is not initialized")

        # Choose the appropriate method based on async_engine setting
        method_name = (
            "update_weights_from_collective_async"
            if self.cfg["vllm_cfg"]["async_engine"]
            else "update_weights_from_collective"
        )

        # Use run_all_workers_single_data to send data to all workers
        futures = self.worker_group.run_all_workers_single_data(
            method_name,
            data=info,
            run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        )

        # this function should co-work with lm_policy, so we should wait for all futures to complete outside
        return futures

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        futures = self.worker_group.run_all_workers_single_data("start_gpu_profiling")
        ray.get(futures)

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        futures = self.worker_group.run_all_workers_single_data("stop_gpu_profiling")
        ray.get(futures)

    def __del__(self) -> None:
        """Shuts down the worker groups when the object is deleted or is garbage collected.

        This is an extra safety net in case the user forgets to call shutdown() and the pointer to
        the object is lost due to leaving a function scope. It's always recommended that the
        user calls shutdown().
        """
        self.shutdown()
