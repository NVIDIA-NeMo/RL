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
import logging
import os
from typing import List, Optional, TypedDict, Union

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
            # Training pipeline will set this to "dummy" and eval will load real weights using 'auto'
            load_format=load_format,
            skip_tokenizer_init=self.cfg["vllm_cfg"]["skip_tokenizer_init"],
            tensor_parallel_size=self.cfg["vllm_cfg"]["tensor_parallel_size"],
            gpu_memory_utilization=self.cfg["vllm_cfg"]["gpu_memory_utilization"],
            # Disable prefix caching for devices with compute capability < 8 (Volta) due to vllm segfault.
            enable_prefix_caching=torch.cuda.get_device_capability()[0] >= 8,
            dtype=self.cfg["vllm_cfg"]["precision"],
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

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False, enable_principle_genrm_logprobs: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using vLLM generation.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths tensors
            greedy: Whether to use greedy decoding instead of sampling
            enable_principle_genrm_logprobs: Whether to extract Yes/No logprobs for principle_genrm training

        Returns:
            BatchedDataDict conforming to GenerationOutputSpec:
                - output_ids: input + generated token IDs with proper padding
                - logprobs: Log probabilities for tokens
                - generation_lengths: Lengths of each response
                - unpadded_sequence_lengths: Lengths of each input + generated sequence
                - principle_genrm_yes_logprobs: (optional) Yes token logprobs for principle_genrm
                - principle_genrm_no_logprobs: (optional) No token logprobs for principle_genrm
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
        )

        # Generate outputs
        outputs = self.llm.generate(prompts, sampling_params)

        # Process the outputs - but preserve the original input padding structure
        output_ids_list = []
        logprobs_list = []
        generation_lengths = []
        unpadded_sequence_lengths = []
        complete_sequences = []  # For principle_genrm logprob calculation
        
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

            # Store complete sequence for principle_genrm if needed
            if enable_principle_genrm_logprobs:
                # Create the complete sequence: input + generated tokens
                input_tokens = input_ids[i, :sequence_length].tolist()
                complete_sequence = input_tokens + generated_tokens
                complete_sequences.append(complete_sequence)

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

        # Add principle_genrm logprobs if enabled - do additional forward pass
        if enable_principle_genrm_logprobs and complete_sequences:
            try:
                yes_logprobs, no_logprobs = self._get_yes_no_logprobs_for_sequences(complete_sequences)
                return_data["principle_genrm_yes_logprobs"] = torch.tensor(yes_logprobs, dtype=torch.float)
                return_data["principle_genrm_no_logprobs"] = torch.tensor(no_logprobs, dtype=torch.float)
            except Exception as e:
                raise e

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

    def _get_yes_no_logprobs_for_sequences(self, sequences: list[list[int]]) -> tuple[list[float], list[float]]:
        """Get Yes/No logprobs for a batch of sequences using an additional forward pass.
        
        This method takes complete sequences (input + generated tokens) and uses sequence[:-1] 
        as the prompt to generate the last token, extracting Yes/No logprobs from the generation.
        
        Args:
            sequences: List of complete token sequences (input + generated tokens)
            
        Returns:
            Tuple of (yes_logprobs, no_logprobs) lists
        """
        if not sequences:
            return [], []
        
        # Get tokenizer from the vLLM engine and compute the *single* token ids for
        # " Yes" and " No".  Using a leading space guarantees that most BPE/GPT
        # tokenizers treat each as a single token (e.g., "ĠYes").
        tokenizer = self.llm.llm_engine.tokenizer.tokenizer

        yes_token_ids = tokenizer.encode(" Yes", add_special_tokens=False)
        if len(yes_token_ids) != 1:
            raise ValueError(
                f"Expected ' Yes' to be a single token, but got {len(yes_token_ids)} tokens: {yes_token_ids}"
                " This is required for principle_genrm logprob calculation."
            )
        yes_token_id = yes_token_ids[0]

        no_token_ids = tokenizer.encode(" No", add_special_tokens=False)
        if len(no_token_ids) != 1:
            raise ValueError(
                f"Expected ' No' to be a single token, but got {len(no_token_ids)} tokens: {no_token_ids}"
                " This is required for principle_genrm logprob calculation."
            )
        no_token_id = no_token_ids[0]
        
        # Prepare prompts using sequence[:-1] to regenerate the last token
        prompts = []
        last_tokens = []
        for seq in sequences:
            if len(seq) <= 2:
                raise ValueError(f"Sequence is too short: {seq}, need at least 3 tokens (input + content_token + eos)")
            else:
                # The sequence ends with [content token(" Yes" or " No"), eos_token]
                # We want to use everything except both the content token AND EOS token as prompt 
                # to regenerate the content token
                prompt_tokens = seq[:-2]  # Remove both content token and EOS token
                prompts.append({"prompt_token_ids": prompt_tokens})
                last_tokens.append(seq[-2])  # Use second-to-last token (the actual content token)
                
        # Create sampling params to generate exactly 1 token with logprobs
        sampling_params = self.SamplingParams(
            # Use temperature=1.0 so the logits are unscaled when computing probabilities.
            # Explicitly disable top-k / top-p filtering as they would renormalise the
            # distribution and shift log-probs.  We still request the top-20 tokens so
            # Yes/No are almost certainly returned even for large vocabularies.
            temperature=1.0,
            top_k=-1,
            top_p=1.0,
            max_tokens=1,  # Generate exactly one token
            logprobs=20,   # Get top-20 logprobs to ensure we capture Yes/No tokens (vLLM max)
            prompt_logprobs=None,  # Don't need prompt logprobs
        )
        
        # Do the additional forward pass
        outputs = self.llm.generate(prompts, sampling_params)
        
        yes_logprobs = []
        no_logprobs = []
        
        for output, expected_last_token in zip(outputs, last_tokens):
            yes_logprob = -float('inf')
            no_logprob = -float('inf')
            
            try:
                generation = output.outputs[0]
                # Check if we have logprobs for the generated token
                if hasattr(generation, 'logprobs') and generation.logprobs and len(generation.logprobs) > 0:
                    # Get logprobs for the first (and only) generated token
                    first_token_logprobs = generation.logprobs[0]
                    
                    if first_token_logprobs:
                        # Extract Yes/No logprobs from the logprob dictionary
                        if yes_token_id in first_token_logprobs:
                            yes_logprob = first_token_logprobs[yes_token_id].logprob
                        if no_token_id in first_token_logprobs:
                            no_logprob = first_token_logprobs[no_token_id].logprob
                        
                        # Debug: Check if we're getting suspiciously perfect scores
                        if yes_logprob == 0.0 or no_logprob == 0.0:
                            logging.warning(f"Suspiciously perfect logprob detected!")
                            logging.warning(f"Generated token: {generation.token_ids[0] if generation.token_ids else None}")
                            logging.warning(f"Yes logprob: {yes_logprob}, No logprob: {no_logprob}")
                            logging.warning(f"Raw Yes entry: {first_token_logprobs.get(yes_token_id, 'NOT_FOUND')}")
                            logging.warning(f"Raw No entry: {first_token_logprobs.get(no_token_id, 'NOT_FOUND')}")
                            # Show top few logprobs for comparison
                            sorted_logprobs = sorted(first_token_logprobs.items(), key=lambda x: x[1].logprob, reverse=True)
                            logging.warning(f"Top 3 logprobs: {[(k, v.logprob) for k, v in sorted_logprobs[:3]]}")
                        
                        # Check for edge cases that might cause issues
                        if yes_logprob == -float('inf') and no_logprob == -float('inf'):
                            logging.warning(f"Neither Yes ({yes_token_id}) nor No ({no_token_id}) found in top-20 logprobs. Available: {list(first_token_logprobs.keys())[:5]}...")
                        
                        if expected_last_token not in [yes_token_id, no_token_id]:
                            logging.warning(f"Expected token {expected_last_token} is not Yes ({yes_token_id}) or No ({no_token_id}). This sequence may not end with a valid judgment.")
                
            except Exception as e:
                logging.error(f"Error extracting Yes/No logprobs from generation: {e}")
            
            yes_logprobs.append(yes_logprob)
            no_logprobs.append(no_logprob)
        
        return yes_logprobs, no_logprobs


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
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False, enable_principle_genrm_logprobs: bool = False
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
            common_kwargs={"greedy": greedy, "enable_principle_genrm_logprobs": enable_principle_genrm_logprobs},
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
