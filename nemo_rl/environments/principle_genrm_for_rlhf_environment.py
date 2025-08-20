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

import logging
import os
import uuid
from typing import Dict, List, Optional, Tuple, TypedDict

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES, RayVirtualCluster
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)


class PrincipleGenrmForRLHFConfig(TypedDict):
    num_workers: int
    model_name: str  # The principle GenRM model to use
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    # Default sampling parameters for the verifier
    temperature: Optional[float]
    max_tokens: Optional[int]
    stop: Optional[List[str]]
    max_concurrency: Optional[int]  # Maximum concurrent step calls
    reasoning_split_word: Optional[str]  # Configurable split word for response processing (default: "</think>")


class PrincipleGenrmForRLHFMetadata(TypedDict):
    principle: str  # The principle to check against
    conversation_history: Optional[List[Dict[str, str]]]  # Full conversation history


@ray.remote
class AsyncPrincipleGenrmWorker:
    """Worker that serves a principle GenRM model using vllm.AsyncLLMEngine for verifying responses.
    
    This worker evaluates whether responses follow given principles by extracting
    Yes/No token logprobs and computing rewards.
    """

    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.VLLM
    # No default template needed - the model's chat template handles formatting

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: Optional[int] = None,
        disable_log_stats: bool = True,
        **engine_kwargs,
    ):
        from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.sampling_params import SamplingParams

        self.SamplingParams = SamplingParams
        
        # Set up HF cache
        hf_hub_cache_path = os.environ.get("HF_HUB_CACHE", HUGGINGFACE_HUB_CACHE)
        if not os.path.isdir(hf_hub_cache_path):
            try:
                os.makedirs(hf_hub_cache_path, exist_ok=True)
                logging.info(f"Created HF cache directory for worker: {hf_hub_cache_path}")
            except OSError as e:
                logging.warning(f"Worker could not create HF cache directory {hf_hub_cache_path}: {e}")

        # Create vLLM engine
        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            disable_log_stats=disable_log_stats,
            download_dir=hf_hub_cache_path,
            **engine_kwargs,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # Get tokenizer for applying chat template
        self.tokenizer = self.engine.engine.tokenizer.tokenizer
        self._setup_yes_no_tokens()
        
        logging.info(f"AsyncPrincipleGenrmWorker initialized with model: {model_name}")

    def _setup_yes_no_tokens(self):
        """Set up the Yes/No token IDs for logprob extraction."""
        # Store multiple possible token IDs for Yes and No
        self.yes_token_ids = []
        self.no_token_ids = []
        
        # Check different possible encodings
        yes_variations = [" Yes", "ĠYes"]
        no_variations = [" No", "ĠNo"]
        
        # Try to encode each variation and store valid single-token encodings
        for yes_var in yes_variations:
            try:
                token_ids = self.tokenizer.encode(yes_var, add_special_tokens=False)
                if len(token_ids) == 1:
                    self.yes_token_ids.append(token_ids[0])
                    logging.info(f"Found Yes token: '{yes_var}' -> {token_ids[0]}")
            except:
                pass
        
        for no_var in no_variations:
            try:
                token_ids = self.tokenizer.encode(no_var, add_special_tokens=False)
                if len(token_ids) == 1:
                    self.no_token_ids.append(token_ids[0])
                    logging.info(f"Found No token: '{no_var}' -> {token_ids[0]}")
            except:
                pass
        
        # Remove duplicates
        self.yes_token_ids = list(set(self.yes_token_ids))
        self.no_token_ids = list(set(self.no_token_ids))
        
        if not self.yes_token_ids:
            raise ValueError("Could not find any valid Yes token encodings")
        if not self.no_token_ids:
            raise ValueError("Could not find any valid No token encodings")
        
        logging.info(f"Yes token IDs: {self.yes_token_ids}")
        logging.info(f"No token IDs: {self.no_token_ids}")

    async def verify_with_principle(
        self,
        request_id: str,
        conversation: List[Dict[str, str]],
        principle: str,
        sampling_params_dict: dict,
    ) -> Tuple[str, float]:
        """Verifies a response against a principle using logprob-based reward calculation.

        Args:
            request_id: Unique ID for this generation request
            conversation: The full conversation history (list of role/content dicts)
            principle: The principle to check against
            sampling_params_dict: Sampling parameters for generation

        Returns:
            Tuple of (request_id, reward) where reward = logprob(" Yes") - logprob(" No")
        """
        # Construct messages with principle role for the chat template
        messages = conversation.copy()
        messages.append({"role": "principle", "content": principle})
        
        # Apply the model's chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate the full response with logprobs in a single pass
        sampling_params = self.SamplingParams(
            temperature=sampling_params_dict.get("temperature", 1.0),
            max_tokens=sampling_params_dict.get("max_tokens", 8192),
            stop=sampling_params_dict.get("stop", None),
            logprobs=20,  # Request top-20 logprobs for each generated token
            prompt_logprobs=None,  # Don't need prompt logprobs
        )
        
        # Generate the response with logprobs
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        if not final_output or not final_output.outputs:
            raise ValueError(f"No output received for request {request_id}")
        
        generation = final_output.outputs[0]
        generated_text = generation.text
        
        # Check if we have a valid judgment
        if "Final Judgement:" not in generated_text:
            logging.warning(f"No 'Final Judgement:' found in output for request {request_id}: {generated_text}")
            return request_id, 0.0
        
        # Extract Yes/No logprobs from the generated tokens
        yes_logprob, no_logprob = self._extract_judgment_logprobs(generation)
        
        # Calculate reward: logprob(" Yes") - logprob(" No")
        reward = yes_logprob - no_logprob
        
        # Reward 0.0 if Yes or No is not in the top-20 logprobs
        if yes_logprob == -float('inf') or no_logprob == -float('inf'):
            reward = 0.0
            logging.warning(
                f"Penalty applied for request {request_id}: yes_logprob={yes_logprob}, no_logprob={no_logprob}"
            )
        
        logging.info(
            f"Request {request_id}: yes_logprob={yes_logprob:.4f}, "
            f"no_logprob={no_logprob:.4f}, reward={reward:.4f}, generated_text={generated_text}"
        )
        
        return request_id, reward

    def _extract_judgment_logprobs(self, generation) -> Tuple[float, float]:
        """Extract Yes/No logprobs from the generated output.
        
        The model should generate text ending with "Final Judgement: Yes" or "Final Judgement: No".
        We look for the Yes/No token which is typically the second-to-last token (before EOS).
        
        Args:
            generation: The generation output from vLLM
            
        Returns:
            Tuple of (yes_logprob, no_logprob)
        """
        yes_logprob = -float('inf')
        no_logprob = -float('inf')
        
        if not hasattr(generation, 'logprobs') or not generation.logprobs:
            logging.warning("No logprobs found in generation")
            return yes_logprob, no_logprob
        
        # The Yes/No token is typically the second-to-last token (last token is often EOS)
        # Try multiple positions in case of variations
        # target_positions = [-2, -1, -3] if len(generation.logprobs) >= 3 else range(len(generation.logprobs))
        target_positions = [-2] # only check the second-to-last token
        for pos in target_positions:
            if abs(pos) > len(generation.logprobs):
                continue
                
            token_logprobs = generation.logprobs[pos]
            if not token_logprobs:
                continue
            
            # Check if this position contains any Yes/No token variations
            found_yes = False
            found_no = False
            
            # Check all Yes token variations
            for yes_token_id in self.yes_token_ids:
                if yes_token_id in token_logprobs:
                    yes_logprob = max(yes_logprob, token_logprobs[yes_token_id].logprob)
                    found_yes = True
            
            # Check all No token variations
            for no_token_id in self.no_token_ids:
                if no_token_id in token_logprobs:
                    no_logprob = max(no_logprob, token_logprobs[no_token_id].logprob)
                    found_no = True
            
            if found_yes or found_no:
                # Debug logging
                actual_token = generation.token_ids[pos] if hasattr(generation, 'token_ids') else None
                actual_token_text = self.tokenizer.decode([actual_token]) if actual_token else "Unknown"
                logging.warning(
                    f"--------------------------------"
                    f"Found Yes/No logprobs at position {pos}: "
                    f"actual_token='{actual_token_text}', "
                    f"yes_logprob={yes_logprob:.4f}, no_logprob={no_logprob:.4f}"
                )
                break
        
        return yes_logprob, no_logprob


@ray.remote
class PrincipleGenrmForRLHFEnvironment(EnvironmentInterface):
    """Environment that uses a principle GenRM model as a verifier for RLHF.
    
    This environment evaluates whether generated responses follow specified principles
    by using a trained principle GenRM model to compute logprob-based rewards.
    """
    
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def __init__(self, cfg: PrincipleGenrmForRLHFConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        
        tensor_parallel_size = cfg.get("tensor_parallel_size", 1)
        
        # Set up placement groups for GPU allocation
        if tensor_parallel_size == 1:
            bundle_ct_per_node_list = [1] * self.num_workers
            self.virtual_cluster = RayVirtualCluster(
                bundle_ct_per_node_list=bundle_ct_per_node_list,
                use_gpus=True,
                name="principle_genrm_rlhf_vc",
            )
            self.virtual_cluster.print_cluster_grid()
            placement_groups = self.virtual_cluster.get_placement_groups()
        else:
            self.virtual_cluster = None
            placement_groups = []
        
        # Pass environment variables to workers
        env_vars_to_pass = {}
        for key in ["HF_HUB_CACHE", "HF_HOME", "TRANSFORMERS_CACHE", "HF_TOKEN"]:
            if key in os.environ:
                env_vars_to_pass[key] = os.environ[key]
        
        worker_options = {
            "runtime_env": {
                "py_executable": AsyncPrincipleGenrmWorker.DEFAULT_PY_EXECUTABLE,
                "env_vars": env_vars_to_pass,
            },
            "num_gpus": tensor_parallel_size,
        }
        
        # Create workers
        self.workers = []
        for i in range(self.num_workers):
            if tensor_parallel_size == 1:
                pg_index = i % len(placement_groups)
                pg = placement_groups[pg_index]
                scheduling_kwargs = dict(
                    scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                        placement_group=pg
                    )
                )
            else:
                scheduling_kwargs = {}
            
            worker = AsyncPrincipleGenrmWorker.options(
                **worker_options,
                **scheduling_kwargs,
            ).remote(
                model_name=cfg["model_name"],
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.85),
                max_model_len=cfg.get("max_model_len"),
            )
            self.workers.append(worker)
        
        logging.info(f"Created {len(self.workers)} AsyncPrincipleGenrmWorker actors.")
        self._request_counter = 0
        self._actor_id_prefix = str(uuid.uuid4())[:8]

    def shutdown(self):
        for worker in self.workers:
            ray.kill(worker)
        if self.virtual_cluster is not None:
            self.virtual_cluster.shutdown()

    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[PrincipleGenrmForRLHFMetadata],
    ) -> EnvironmentReturn:
        """Step function for the principle GenRM verifier environment.

        Args:
            message_log_batch: List of conversations containing the assistant responses to evaluate
            metadata: List of metadata containing principles to check and conversation history

        Returns:
            EnvironmentReturn with rewards based on principle adherence
        """
        # Prepare default sampling parameters
        default_sampling_params = {
            "temperature": self.cfg.get("temperature", 0.0),  # Use greedy for consistent judgments
            "max_tokens": self.cfg.get("max_tokens", 200),  # Enough for reasoning + judgment
            "stop": self.cfg.get("stop", None),
        }
        
        # Create verification tasks
        futures = []
        for i, (message_log, single_metadata) in enumerate(
            zip(message_log_batch, metadata)
        ):
            request_id = f"env_{self._actor_id_prefix}_step_{self._request_counter}_{i}"
            worker_idx = i % self.num_workers
            
            principle = single_metadata["principle"]
            
            # Get conversation history from metadata
            conversation_history = single_metadata.get("conversation_history", None)
            assert conversation_history is not None, "Conversation history is required"
            # Extract the last assistant response
            assistant_response = message_log[-1]["content"]
            
            # Only split if reasoning_split_word is provided
            if self.cfg.get("reasoning_split_word"):
                assistant_response = assistant_response.split(self.cfg["reasoning_split_word"])[-1].lstrip()
            
            # Build the full conversation: history + new assistant response(s)
            full_conversation = conversation_history.copy()
            full_conversation.append({"role": "assistant", "content": assistant_response})
            
            future = self.workers[worker_idx].verify_with_principle.remote(
                request_id,
                full_conversation,
                principle,
                default_sampling_params,
            )
            futures.append(future)
        
        self._request_counter += 1
        
        # Get results
        results_tuples: List[Tuple[str, float]] = ray.get(futures)
        rewards = [reward for _, reward in results_tuples]
        
        # Create observations
        observations = []
        for reward, single_meta in zip(rewards, metadata):
            principle = single_meta["principle"]
            observations.append(
                {
                    "role": "environment",
                    "content": f"Environment: Principle adherence reward = {reward:.4f}\nPrinciple: {principle}",
                }
            )
        
        # Create return tensors
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).cpu()
        terminateds_tensor = torch.ones_like(rewards_tensor).cpu()
        
        next_stop_strings = [None] * len(message_log_batch)
        
        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards_tensor,
            terminateds=terminateds_tensor,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        """Computes metrics for this environment."""
        return batch, {}