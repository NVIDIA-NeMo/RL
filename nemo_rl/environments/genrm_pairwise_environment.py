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
from typing import Dict, List, Optional, Tuple, TypedDict, Any
import itertools

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES, RayVirtualCluster
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.metrics import (
    calculate_pass_rate_per_prompt,
)
from nemo_rl.environments.utils import extract_answer_from_box
from nemo_rl.environments.pairwise_reward_aggregators import create_aggregator


class GenRMPairwiseConfig(TypedDict):
    num_workers: int
    model_name: str  # GenRM model name, e.g., "nvidia/Llama-3_3-Nemotron-Super-49B-GenRM"
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    num_generations_per_prompt: int  # e.g., 8 - number of responses to generate per prompt
    # Default sampling parameters for the GenRM
    temperature: Optional[float]
    max_tokens: Optional[int]
    stop: Optional[List[str]]
    max_concurrency: Optional[int]  # Maximum concurrent step calls for the environment actor
    reasoning_split_word: Optional[str]  # Default: "</think>"
    # Reward aggregation configuration
    aggregator_method: Optional[str]  # e.g., "individual_scores", "weighted_win_loss", "simple_tiebreaker", etc.
    aggregator_config: Optional[Dict[str, Any]]  # Additional config for the aggregator


class GenRMEnvironmentMetadata(TypedDict):
    conversation_history: List[Dict[str, str]]  # The conversation history in user/assistant format
    extract_box: Optional[bool]  # Whether to extract answer from box


@ray.remote
class AsyncGenRMWorker:
    """Worker that serves GenRM using vLLM AsyncEngine for pairwise response comparisons."""

    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.VLLM

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: Optional[int] = None,
        disable_log_stats: bool = True,
        reasoning_split_word: Optional[str] = "</think>",
        **engine_kwargs,
    ):
        # Configure logging for Ray worker
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            force=True
        )
        
        # Imports moved here to be within the Ray actor's context
        from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.sampling_params import SamplingParams
        from vllm.inputs import TokensPrompt
        from transformers import AutoTokenizer

        self.SamplingParams = SamplingParams
        self.TokensPrompt = TokensPrompt
        
        # Setup HF cache path
        hf_home_cache_path = os.environ.get("HF_HOME", HUGGINGFACE_HUB_CACHE)
        if not os.path.isdir(hf_home_cache_path):
            try:
                os.makedirs(hf_home_cache_path, exist_ok=True)
                logging.info(
                    f"Created HF cache directory for GenRM worker: {hf_home_cache_path}"
                )
            except OSError as e:
                logging.warning(
                    f"GenRM worker could not create HF cache directory {hf_home_cache_path}: {e}. "
                    "This might lead to download issues if the default cache is not writable."
                )

        # Load tokenizer for chat template functionality
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=hf_home_cache_path,
            trust_remote_code=True,
        )
        
        # Initialize AsyncEngine with GenRM model
        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            disable_log_stats=disable_log_stats,
            download_dir=hf_home_cache_path,
            ignore_patterns=[
                "*.safetensors.index.json",
                "*.pt",
                "*.bin.index.json",
                "*.gitattributes",
            ],
            trust_remote_code=True,  # GenRM models typically need trust_remote_code
            **engine_kwargs,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.reasoning_split_word = reasoning_split_word
        logging.info(f"AsyncGenRMWorker initialized with GenRM model: {model_name}")

    def _format_genrm_messages(
        self,
        conversation_history: List[Dict[str, str]], 
        response_1: str, 
        response_2: str
    ) -> List[Dict[str, str]]:
        """Format the conversation and responses into GenRM's expected message format."""
        # Build messages list in the format expected by GenRM
        messages = conversation_history.copy()
        
        # Add the responses to be compared
        messages.extend([
            {"role": "response_1", "content": response_1},
            {"role": "response_2", "content": response_2}
        ])
        
        return messages

    async def compare_responses(
        self,
        request_id: str,
        conversation_history: List[Dict[str, str]],
        response_1: str,
        response_2: str,
        sampling_params_dict: dict,
    ) -> Tuple[str, float, float, float]:
        """Compare two responses using GenRM via AsyncEngine.

        Args:
            request_id: Unique ID for this comparison request
            conversation_history: List of conversation messages in user/assistant format
            response_1: First response to compare
            response_2: Second response to compare  
            sampling_params_dict: Parameters for vLLM sampling

        Returns:
            Tuple of (request_id, individual_score_1, individual_score_2, ranking_score)
        """
        try:
            # Format messages for GenRM
            if self.reasoning_split_word and self.reasoning_split_word in response_1:
                response_1 = response_1.split(self.reasoning_split_word)[-1].lstrip()
            if self.reasoning_split_word and self.reasoning_split_word in response_2:
                response_2 = response_2.split(self.reasoning_split_word)[-1].lstrip()
            messages = self._format_genrm_messages(conversation_history, response_1, response_2)
            
            # Apply chat template to get text for debugging
            chat_template_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # logging.info(f"GenRM chat template text for {request_id}:\n{chat_template_text}")
            
            # Apply chat template and tokenize
            token_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors=None,  # Return list of token IDs
            )
            
            # logging.info(f"GenRM tokenized prompt for {request_id}: {len(token_ids)} tokens")
            
            # Create sampling parameters
            sampling_params = self.SamplingParams(**sampling_params_dict)
            
            # Create TokensPrompt object for vLLM
            tokens_prompt = self.TokensPrompt(prompt_token_ids=token_ids)
            
            # Generate using AsyncEngine with TokensPrompt
            results_generator = self.engine.generate(tokens_prompt, sampling_params, request_id)
            
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            
            if final_output and final_output.outputs:
                generated_text = final_output.outputs[0].text.strip()
                
                # Split by reasoning word if provided
                if self.reasoning_split_word and self.reasoning_split_word in generated_text:
                    generated_text = generated_text.split(self.reasoning_split_word)[-1].lstrip()
                
                # Parse the scores from GenRM output
                individual_score_1, individual_score_2, ranking_score = self._parse_genrm_output(generated_text)
                
                logging.info(f"GenRM comparison {request_id}: scores=({individual_score_1}, {individual_score_2}), ranking={ranking_score}, generated_text={generated_text}")
                
                return request_id, individual_score_1, individual_score_2, ranking_score
            else:
                logging.warning(f"No output received from GenRM for request {request_id}")
                return request_id, 3.0, 3.0, 3.5
                
        except Exception as e:
            logging.error(f"Error in GenRM comparison {request_id}: {e}")
            return request_id, 3.0, 3.0, 3.5

    def _parse_genrm_output(self, output: str) -> Tuple[float, float, float]:
        """Parse GenRM output to extract individual and ranking scores."""
        import re
        
        try:
            individual_scores = None
            ranking_score = None
            
            # Extract individual scores section using regex
            individual_pattern = r'\[The Begin of Individual Scores\](.*?)\[The End of Individual Scores\]'
            individual_match = re.search(individual_pattern, output, re.DOTALL)
            if individual_match:
                individual_section_text = individual_match.group(1).strip()
                individual_boxed_content = extract_answer_from_box(individual_section_text)
                if individual_boxed_content and ',' in individual_boxed_content:
                    # Individual scores: "2, 4"
                    parts = individual_boxed_content.split(',')
                    if len(parts) == 2:
                        try:
                            individual_scores = (float(parts[0].strip()), float(parts[1].strip()))
                        except ValueError:
                            logging.warning(f"Could not parse individual scores: {individual_boxed_content}")
            
            # Extract ranking score section using regex
            ranking_pattern = r'\[The Begin of Ranking Score\](.*?)\[The End of Ranking Score\]'
            ranking_match = re.search(ranking_pattern, output, re.DOTALL)
            if ranking_match:
                ranking_section_text = ranking_match.group(1).strip()
                ranking_boxed_content = extract_answer_from_box(ranking_section_text)
                if ranking_boxed_content:
                    try:
                        ranking_score = float(ranking_boxed_content)
                    except ValueError:
                        logging.warning(f"Could not parse ranking score: {ranking_boxed_content}")
            
            # Default values if parsing fails
            if individual_scores is None:
                individual_scores = (3.0, 3.0)  # Neutral scores
            if ranking_score is None:
                ranking_score = 3.5  # Neutral ranking
                
            return individual_scores[0], individual_scores[1], ranking_score
            
        except Exception as e:
            logging.error(f"Error parsing GenRM output: {e}. Output was: {output}")
            return 3.0, 3.0, 3.5


@ray.remote
class GenRMPairwiseEnvironment(EnvironmentInterface):
    """Environment that uses GenRM for pairwise comparison of multiple responses per prompt."""
    
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def __init__(self, cfg: GenRMPairwiseConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.num_generations_per_prompt = cfg["num_generations_per_prompt"]
        
        # Initialize the reward aggregator
        aggregator_method = cfg.get("aggregator_method", "simple_tiebreaker")  # Default to simple_tiebreaker
        aggregator_config = cfg.get("aggregator_config", {})
        self.reward_aggregator = create_aggregator(aggregator_method, **aggregator_config)
        logging.info(f"Initialized GenRM environment with {self.reward_aggregator.name} aggregator")
        
        tensor_parallel_size = cfg.get("tensor_parallel_size", 1)

        # Create RayVirtualCluster for GPU allocation if needed
        if tensor_parallel_size == 1:
            bundle_ct_per_node_list = [tensor_parallel_size] * self.num_workers
            
            self.virtual_cluster = RayVirtualCluster(
                bundle_ct_per_node_list=bundle_ct_per_node_list,
                use_gpus=True,
                name="genrm_pairwise_vc",
            )
            self.virtual_cluster.print_cluster_grid()
            placement_groups = self.virtual_cluster.get_placement_groups()
        else:
            self.virtual_cluster = None
            placement_groups = []

        # Pass down environment variables to workers
        env_vars_to_pass = {}
        for key in [
            "HF_HOME",
            "TRANSFORMERS_CACHE", 
            "WANDB_API_KEY",
            "HUGGINGFACE_HUB_DISABLE_XET",
            "HF_TOKEN",
        ]:
            if key in os.environ:
                env_vars_to_pass[key] = os.environ[key]
        
        env_vars_to_pass.setdefault("HUGGINGFACE_HUB_DISABLE_XET", "1")

        worker_options = {
            "runtime_env": {
                "py_executable": AsyncGenRMWorker.DEFAULT_PY_EXECUTABLE,
                "env_vars": env_vars_to_pass,
            },
            "num_gpus": tensor_parallel_size,
        }

        # Create GenRM workers
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
                
            worker = AsyncGenRMWorker.options(
                **worker_options,
                **scheduling_kwargs,
            ).remote(
                model_name=cfg["model_name"],
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.85),
                max_model_len=cfg.get("max_model_len"),
                reasoning_split_word=cfg.get("reasoning_split_word", "</think>"),
            )
            self.workers.append(worker)
            
        logging.info(f"Created {len(self.workers)} AsyncGenRMWorker actors.")
        self._request_counter = 0
        self._actor_id_prefix = str(uuid.uuid4())[:8]
        self._last_additional_metrics = {}  # Store additional metrics from last step for global_post_process_and_metrics

    def shutdown(self):
        for worker in self.workers:
            ray.kill(worker)
        if self.virtual_cluster is not None:
            self.virtual_cluster.shutdown()

    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[GenRMEnvironmentMetadata],
    ) -> EnvironmentReturn:
        """Step function for GenRM pairwise comparison environment.
        
        Args:
            message_log_batch: List of conversations, where each conversation is a list of messages
            metadata: List of metadata for each conversation
            
        Returns:
            EnvironmentReturn with rewards based on pairwise comparison aggregation
        """
        
        def get_prompt_key(conversation_history: List[Dict[str, str]]) -> str:
            """Extract the conversation history as a grouping key."""
            # Create a key from the conversation history (the prompt context)
            prompt_parts = []
            for msg in conversation_history:
                prompt_parts.append(f"{msg['role']}: {msg['content']}")
            
            return " | ".join(prompt_parts)
        
        # Group responses by prompt (conversation history from metadata)
        prompt_groups = {}
        for i, (conversation, single_metadata) in enumerate(zip(message_log_batch, metadata)):
            prompt_key = get_prompt_key(single_metadata["conversation_history"])
            if prompt_key not in prompt_groups:
                prompt_groups[prompt_key] = {
                    "conversations": [],
                    "metadata": single_metadata,
                    "indices": []
                }
            prompt_groups[prompt_key]["conversations"].append(conversation)
            prompt_groups[prompt_key]["indices"].append(i)
        
        # Prepare default sampling parameters
        default_sampling_params = {
            "temperature": self.cfg.get("temperature", 0.0),
            "max_tokens": self.cfg.get("max_tokens", 32768),
            "stop": self.cfg.get("stop", None),
        }
        
        # Collect all pairwise comparison tasks
        comparison_futures = []
        comparison_metadata = []  # Track which prompt group and response indices each comparison belongs to
        
        for prompt_key, group_data in prompt_groups.items():
            conversations = group_data["conversations"]
            group_metadata = group_data["metadata"]
            conversation_history = group_metadata["conversation_history"]
            
            # Extract responses from conversations (assuming last message is assistant response)
            responses = []
            for conversation in conversations:
                assert len(conversation) >= 1, "Each conversation should have at least one message"
                # Get the last assistant message as the response
                assistant_msgs = [msg for msg in conversation if msg["role"] == "assistant"]
                assert len(assistant_msgs) >= 1, "Each conversation should have at least one assistant message"
                responses.append(assistant_msgs[-1]["content"])
            
            # Check that we have the expected number of generations per prompt
            if len(responses) != self.num_generations_per_prompt:
                raise ValueError(
                    f"Expected {self.num_generations_per_prompt} generations per prompt, "
                    f"but found {len(responses)} responses for prompt key: {prompt_key[:100]}... "
                    f"This may because generations for the same prompt are distributed to multiple dp ranks, can will cause error when calculating rewards."
                )
            
            
            # Generate all pairwise comparisons for this prompt group
            for i, j in itertools.permutations(range(len(responses)), 2):
                request_id = f"genrm_{self._actor_id_prefix}_step_{self._request_counter}_pk{hash(prompt_key)}_r{i}_r{j}"
                worker_idx = len(comparison_futures) % self.num_workers
                
                future = self.workers[worker_idx].compare_responses.remote(
                    request_id,
                    conversation_history,
                    responses[i],
                    responses[j],
                    default_sampling_params,
                )
                comparison_futures.append(future)
                comparison_metadata.append((prompt_key, i, j))
        
        self._request_counter += 1
        
        # Get all comparison results
        comparison_results = ray.get(comparison_futures)
        
        # Aggregate pairwise comparisons into final scores for each response using the configured aggregator
        final_scores = self.reward_aggregator.aggregate_scores(
            comparison_results, comparison_metadata, prompt_groups
        )
        
        # Get additional metrics from the aggregator (e.g., individual scores for ranking-based aggregators)
        self._last_additional_metrics = self.reward_aggregator.get_additional_metrics(
            comparison_results, comparison_metadata, prompt_groups, final_scores
        )
        
        # Create observations and prepare return values in the same order as input
        observations = []
        all_metadata = []
        rewards_list = []
        
        for i, (conversation, single_metadata) in enumerate(zip(message_log_batch, metadata)):
            prompt_key = get_prompt_key(single_metadata["conversation_history"])
            
            # Find which response index this is within its group
            group_indices = prompt_groups[prompt_key]["indices"]
            response_idx_in_group = group_indices.index(i)
            
            # Get the score for this response (default to 0.5 if no comparisons were made)
            if prompt_key in final_scores:
                score = final_scores[prompt_key][response_idx_in_group]
            else:
                score = 0.5  # Neutral score for single responses
            
            observations.append({
                "role": "environment", 
                "content": f"Environment: {self.reward_aggregator.name} Score = {score:.3f}"
            })
            all_metadata.append(single_metadata)
            rewards_list.append(score)
        
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32).cpu()
        terminateds_tensor = torch.ones_like(rewards_tensor).cpu()
        next_stop_strings = [None] * len(rewards_list)
        
        return EnvironmentReturn(
            observations=observations,
            metadata=all_metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards_tensor,
            terminateds=terminateds_tensor,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        """Computes metrics for the GenRM pairwise environment."""
        
        metrics = {}
        
        # Add additional metrics from the aggregator (e.g., individual scores for ranking-based aggregators)
        if self._last_additional_metrics:
            metrics.update(self._last_additional_metrics)

        return batch, metrics
