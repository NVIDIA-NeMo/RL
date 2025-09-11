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

import gc
import os
from typing import List, Optional, TypedDict, Union
from transformers import AutoTokenizer

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    PY_EXECUTABLES,
    RayVirtualCluster,
)
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from nemo_rl.models.generation.interfaces import (
    GenerationConfig,
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.custom_logits_processors import  ThinkingBudgetLogitsProcessor
from nemo_rl.models.huggingface.common import ModelFlag


class VllmSpecificArgs(TypedDict):
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    # Additional arguments for vLLM inserted by nemo rl based on the context of when vllm is used
    skip_tokenizer_init: bool


class VllmConfig(GenerationConfig):
    vllm_cfg: VllmSpecificArgs


@ray.remote
class VllmGenerationWorker:
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.VLLM

    def __repr__(self):
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        return f"{self.__class__.__name__}"

    @staticmethod
    def configure_worker(
        num_gpus: int | float,
        bundle_indices: Optional[tuple] = None,
        seed_offset: int = 0,
    ) -> tuple[dict, dict, dict]:
        """Provides complete worker configuration for vLLM tensor parallelism.

        This method configures the worker based on its role in tensor parallelism,
        which is determined directly from the bundle_indices parameter.

        Args:
            num_gpus: Original GPU allocation for this worker based on the placement group
            bundle_indices: Tuple of (node_idx, local_bundle_indices) for tensor parallelism (if applicable)

        Returns:
            tuple with complete worker configuration:
              - 'resources': Resource allocation (e.g., num_gpus)
              - 'env_vars': Environment variables for this worker
              - 'init_kwargs': Parameters to pass to __init__ of the worker
        """
        # Initialize configuration
        resources = {"num_gpus": num_gpus}
        init_kwargs = {}
        env_vars = {}

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
            bundle_id = local_bundle_indices[0] // len(local_bundle_indices)
            seed = node_idx * 1024 + bundle_id + seed_offset
            init_kwargs["seed"] = seed

        is_part_of_tp_workers = (
            local_bundle_indices is not None and len(local_bundle_indices) > 1
        ) or local_bundle_indices is None
        if is_part_of_tp_workers:
            # Ray + vllm likes to manage GPU assignment internally
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
        bundle_indices: Optional[list] = None,
        fraction_of_gpus: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Initialize a vLLM worker for distributed inference.

        Args:
            config: Configuration dictionary for the policy
            bundle_indices: List of local bundle indices within a node for tensor parallelism.
                          Only needed for the first worker in each tied worker group.
        """
        self.cfg = config

        self.model_name = self.cfg["model_name"]
        self.tensor_parallel_size = self.cfg["vllm_cfg"]["tensor_parallel_size"]
        self.gpu_memory_utilization = self.cfg["vllm_cfg"]["gpu_memory_utilization"]
        self.fraction_of_gpus = fraction_of_gpus
        self.is_model_owner = bundle_indices is not None

        # Skip model loading if we're not the model owner
        if not self.is_model_owner:
            self.llm = None
            self.tokenizer = None
            self.rank = 0
            self.world_size = 1
            return

        # In Ray+vLLM setup, each worker process considers itself rank 0
        # vLLM handles the tensor parallelism internally through Ray
        self.rank = 0
        self.world_size = 1

        try:
            import vllm

            self.SamplingParams = vllm.SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Please check that VllmGenerationWorker.DEFAULT_PY_EXECUTABLE covers the vllm dependency. "
                "If you are working interactively, you can install by running  `uv sync --extra vllm` anywhere in the repo."
            )
        vllm_kwargs = self.cfg.get("vllm_kwargs", {}).copy()

        # Special handling for tensor parallel case
        if self.tensor_parallel_size > 1:
            # Configure vLLM for tensor parallelism within Ray

            # Reset CUDA_VISIBLE_DEVICES to allow vLLM to manage GPU assignment
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(
                self.fraction_of_gpus / self.tensor_parallel_size
            )

            # Set bundle indices for tensor parallelism workers
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))

            # Use Ray for distributed execution in TP mode
            vllm_kwargs["distributed_executor_backend"] = "ray"
        else:
            # For non-TP mode, explicitly set executor to None to avoid Ray issues
            vllm_kwargs["distributed_executor_backend"] = None

        load_format = self.cfg["vllm_cfg"]["load_format"]
        if ModelFlag.VLLM_LOAD_FORMAT_AUTO.matches(self.model_name):
            load_format = "auto"

        self.llm = vllm.LLM(
            model=self.model_name,
            enable_prefix_caching=False,
            # Training pipeline will set this to "dummy" and eval will load real weights using 'auto'
            load_format=load_format,
            skip_tokenizer_init=self.cfg["vllm_cfg"]["skip_tokenizer_init"],
            tensor_parallel_size=self.cfg["vllm_cfg"]["tensor_parallel_size"],
            gpu_memory_utilization=self.cfg["vllm_cfg"]["gpu_memory_utilization"],
            dtype=self.cfg["vllm_cfg"]["precision"],
            enable_chunked_prefill=False,
            seed=seed,
            # Don't use cuda-graph by default as it leads to convergence issue (see https://github.com/NVIDIA/NeMo-RL/issues/186)
            enforce_eager=True,
            max_model_len=self.cfg["vllm_cfg"]["max_model_len"],
            trust_remote_code=True,
            worker_extension_cls="nemo_rl.models.generation.vllm_backend.VllmInternalWorkerExtension",
            enable_sleep_mode=True,
            disable_log_stats=True,
            **vllm_kwargs,
        )

    def llm(self):
        return self.llm

    def is_alive(self):
        """Check if the worker is alive."""
        return True

    def _create_logits_processors(self, logits_processors):
        """Create logits processors from configuration."""
        # you can pass an arbitrary list of token ids to to end the thinking section.
        max_tokens = logits_processors.get("max_thinking_tokens", 64)
        think_end_token_ids = logits_processors.get("think_end_token_ids", [1885, 74045, 1062]) # </think>
        thinking_processor = ThinkingBudgetLogitsProcessor(think_max_tokens=max_tokens, think_end_token_ids=think_end_token_ids)
        return [thinking_processor]

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
        # this function requires all generations have the same stop strings, so we collect all here
        batch_stop_strings = data.get("stop_strings", [])
        stop_strings = set()
        for sample_stop_strings in batch_stop_strings:
            if sample_stop_strings:
                stop_strings.update(sample_stop_strings)

        # Add default stop strings from config
        if self.cfg.get("stop_strings", None):
            stop_strings.update(self.cfg["stop_strings"])

        stop_strings = list(stop_strings)

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

        # Read generation parameters from config
        top_k = self.cfg["top_k"] if self.cfg["top_k"] is not None else -1
        
        # Create custom logits processors
        logits_processors = self._create_logits_processors(self.cfg["logits_processors"])
        
        sampling_params = self.SamplingParams(
            temperature=self.cfg["temperature"] if not greedy else 0,
            top_p=self.cfg["top_p"],
            # we use a default of -1 if unset so that 'null'/None is a common disable value
            top_k=top_k if not greedy else 1,
            max_tokens=self.cfg["max_new_tokens"],
            logprobs=0,  # Return logprobs for the generated tokens
            stop_token_ids=self.cfg["stop_token_ids"],
            stop=stop_strings,
            include_stop_str_in_output=True,  # returning stop strings like hf
            logits_processors=logits_processors if logits_processors else None,
        )

        # Generate outputs
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
        batch_stop_strings = data.get(
            "stop_strings", [self.cfg.get("stop_strings")] * len(data["prompts"])
        )

        # This function requires all generations have the same stop strings, so we collect all here
        stop_strings = set()
        for sample_stop_strings in batch_stop_strings:
            if sample_stop_strings:
                stop_strings.update(sample_stop_strings)

        # Add default stop strings from config
        if self.cfg.get("stop_strings", None):
            stop_strings.update(self.cfg["stop_strings"])

        stop_strings = list(stop_strings) if len(stop_strings) > 0 else None

        # Read generation parameters from config
        top_k = self.cfg["top_k"] if self.cfg["top_k"] is not None else -1
        
        # Create custom logits processors
        logits_processors = self._create_logits_processors()
        
        sampling_params = self.SamplingParams(
            temperature=self.cfg["temperature"] if not greedy else 0,
            top_p=self.cfg["top_p"],
            top_k=top_k if not greedy else 1,
            max_tokens=self.cfg["max_new_tokens"],
            stop_token_ids=self.cfg["stop_token_ids"],
            stop=stop_strings,
            include_stop_str_in_output=True,  # returning stop strings like hf
            logits_processors=logits_processors if logits_processors else None,
        )

        # Generate outputs
        outputs = self.llm.generate(data["prompts"], sampling_params)
        texts = [output.outputs[0].text for output in outputs]

        # Convert to BatchedDataDict
        return_data = BatchedDataDict({"texts": texts})
        return return_data

    def shutdown(self):
        """Clean up vLLM resources."""
        try:
            # Clear caches and free memory
            self.llm = None
            self.tokenizer = None

            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()

            return True
        except Exception as e:
            print(f"Error during vLLM shutdown: {e}")
            return False

    def report_device_id(self) -> str:
        return self.llm.collective_rpc("report_device_id", args=tuple())[0]

    def update_weights_from_ipc_handles(self, ipc_handles):
        """Update weights from IPC handles by delegating to the vLLM Worker implementation.

        Args:
            ipc_handles (dict): Dictionary mapping device UUIDs to parameter IPC handles.

        Returns:
            bool: True if weights were successfully updated, False otherwise.
        """
        try:
            # Use collective_rpc to delegate to the UpdatableVllmInternalWorker implementation
            return self.llm.collective_rpc(
                "update_weights_from_ipc_handles", args=(ipc_handles,)
            )[0]
        except Exception as e:
            print(f"Error updating weights: {e}")
            return False

    def sleep(self):
        # Reset the prefix cache to ensure that prefix cache is not reused after weights are updated
        self.llm.llm_engine.reset_prefix_cache()
        self.llm.sleep(level=1)
        gc.collect()
        torch.cuda.empty_cache()

    def wake_up(self, **kwargs):
        # tags like ["weights", "kv_cache"]
        # We can call this function with just tags=["weights"] while doing refit to
        # avoid spiking memory with the kv_cache while the training fwk is awake.
        if "tags" in kwargs:
            self.llm.wake_up(tags=kwargs["tags"])
        else:
            self.llm.wake_up()


class VllmGeneration(GenerationInterface):
    def __init__(
        self,
        cluster: RayVirtualCluster,
        config: VllmConfig,
        name_prefix: str = "vllm_policy",
        workers_per_node: Optional[Union[int, List[int]]] = None,
    ):
        """Initialize a vLLM policy with distributed workers."""
        # Store config
        self.cfg = config
        # Ensure all required VllmConfig fields are present
        missing_keys = [
            key for key in VllmConfig.__annotations__ if key not in self.cfg
        ]
        assert not missing_keys, (
            f"VLLM Configuration Error: Missing required keys in VllmConfig.\n"
            f"Missing keys: {', '.join(missing_keys)}\n"
            f"Provided keys: {', '.join(self.cfg.keys())}\n"
            f"Please update your configuration to include all required VLLM parameters."
        )

        self.tensor_parallel_size = self.cfg["vllm_cfg"]["tensor_parallel_size"]

        # Create worker builder for VllmGenerationWorker
        worker_builder = RayWorkerBuilder(VllmGenerationWorker, config)

        if self.tensor_parallel_size > 1:
            # For tensor parallelism, create node-aware worker groups
            node_bundle_indices = self._get_tied_worker_bundle_indices(cluster)

            self.worker_group = RayWorkerGroup(
                cluster,
                worker_builder,
                name_prefix=name_prefix,
                bundle_indices_list=node_bundle_indices,
            )
        else:
            # Use standard worker group creation for non-TP case
            self.worker_group = RayWorkerGroup(
                cluster,
                worker_builder,
                name_prefix=name_prefix,
                workers_per_node=workers_per_node,
            )

        # Number of data parallel groups is the number of tied worker groups
        self.dp_size = self.worker_group.group_count

    def _get_tied_worker_bundle_indices(self, cluster):
        """Calculate bundle indices for tensor parallel workers."""
        # Get the placement groups (nodes) from the cluster
        placement_groups = cluster.get_placement_groups()

        tied_worker_groups = []

        # For each node (placement group), create tied worker groups of size tensor_parallel_size
        for node_idx, pg in enumerate(placement_groups):
            # How many bundles (GPUs) are on this node
            bundles_on_node = pg.bundle_count
            tied_worker_groups_on_node = bundles_on_node // self.tensor_parallel_size

            if tied_worker_groups_on_node > 0:
                for group_idx in range(tied_worker_groups_on_node):
                    # Local bundle indices for this tied worker group (consecutive GPUs on this node)
                    start_idx = group_idx * self.tensor_parallel_size
                    end_idx = start_idx + self.tensor_parallel_size
                    local_bundle_indices = list(range(start_idx, end_idx))
                    tied_worker_groups.append((node_idx, local_bundle_indices))

        if not tied_worker_groups:
            raise ValueError(
                f"Cannot create any tensor parallel tied worker groups with size {self.tensor_parallel_size}. "
                f"Make sure each node has at least {self.tensor_parallel_size} GPUs."
            )

        return tied_worker_groups

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
        sharded_data = data.shard_by_batch_size(self.dp_size, allow_uneven_shards=True)
        future_bundle = self.worker_group.run_all_workers_multiple_data(
            "generate",
            sharded_data,
            common_kwargs={"greedy": greedy},
            only_on="tied_leader",
        )

        # Get results from the workers, respecting tied worker groups (only one result per tied worker group)
        results = self.worker_group.get_all_worker_results(future_bundle)

        # Combine results from all tied worker groups
        combined = BatchedDataDict.from_batches(
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

        # Get total batch size
        batch_size = len(data["prompts"])

        # Shard the data across the tied worker groups
        sharded_data = data.shard_by_batch_size(self.dp_size, batch_size=batch_size)
        future_bundle = self.worker_group.run_all_workers_multiple_data(
            "generate_text",
            sharded_data,
            common_kwargs={"greedy": greedy},
            only_on="tied_leader",
        )

        # Get results from the workers, respecting tied worker groups (only one result per tied worker group)
        results = self.worker_group.get_all_worker_results(future_bundle)

        # Combine results from all tied worker groups
        combined = BatchedDataDict.from_batches(
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

    def prepare_for_generation(self, *args, **kwargs):
        """Abstract method that must be implemented by subclasses."""
        try:
            # Use run_all_workers_single_data for methods that don't need data
            futures = self.worker_group.run_all_workers_single_data(
                "wake_up", only_on="tied_leader", **kwargs
            )
            # Wait for all futures to complete
            results = ray.get(futures)
            return all(result for result in results if result is not None)
        except Exception as e:
            print(f"Error during policy preparation: {e}")
            return False

    def finish_generation(self, *args, **kwargs):
        """Abstract method that must be implemented by subclasses."""
        try:
            # Use run_all_workers_single_data for methods that don't need data
            futures = self.worker_group.run_all_workers_single_data(
                "sleep", only_on="tied_leader"
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

    def update_weights(self, ipc_handles):
        """Update weights of the policy using IPC handles, considering tensor parallelism.

        For tp > 1, only the leader in each tensor parallel tied worker group will update weights.

        Args:
            ipc_handles (dict): Dictionary mapping device UUIDs to parameter IPC handles.

        Returns:
            bool: True if weights were successfully updated, False otherwise.
        """
        if not self.worker_group or not self.worker_group.workers:
            return False

        try:
            # Directly pass ipc_handles to the method
            futures = self.worker_group.run_all_workers_single_data(
                "update_weights_from_ipc_handles",
                only_on="tied_leader",
                ipc_handles=ipc_handles,
            )
            # Wait for all futures to complete
            results = ray.get(futures)
            return all(result for result in results if result is not None)
        except Exception as e:
            print(f"Error updating weights: {e}")
            return False

    def __del__(self):
        """Shuts down the worker groups when the object is deleted or is garbage collected.

        This is an extra safety net in case the user forgets to call shutdown() and the pointer to
        the object is lost due to leaving a function scope. It's always recommended that the
        user calls shutdown().
        """
        self.shutdown()
