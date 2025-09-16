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
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict

import ray
import torch

from nemo_rl.environments.utils import extract_answer_from_box
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES, RayVirtualCluster
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.metrics import (
    calculate_pass_rate_per_prompt,
)

class LLMJudgeAsyncConfig(TypedDict):
    num_workers: int
    model_name: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    # Default sampling parameters for the judge
    temperature: Optional[float]
    max_tokens: Optional[int]
    stop: Optional[List[str]] # Note: vLLM's AsyncEngine uses 'stop'
    max_concurrency: Optional[int] # Maximum concurrent step calls for the environment actor
    # Logging configuration
    log_dir: Optional[str] # Directory to save conversation logs
    log_conversations: Optional[bool] # Whether to log conversations
    # Any other vllm.SamplingParams can be added here

class LLMJudgeEnvironmentMetadata(TypedDict): # Reusing from previous
    reference_answer: Optional[str]
    evaluation_criteria: Optional[str]
    judge_prompt_template: Optional[str] # Added for per-sample judge prompts
    extract_box: Optional[bool]
    question: Optional[str] # Added to store the question in metadata

@ray.remote
class AsyncVLLMWorker:
    """
    Worker that serves an LLM using vllm.AsyncLLMEngine for judging responses.
    Each call to judge() handles a single prompt asynchronously.
    """
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.VLLM
    DEFAULT_JUDGE_PROMPT_TEMPLATE = """Please respond to the following prompt:

{prompt}
"""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: Optional[int] = None,
        disable_log_stats: bool = True, # Reduce verbose VLLM logging
        **engine_kwargs, # Allow passing other EngineArgs
    ):
        # Imports moved here to be within the Ray actor's context,
        # ensuring they are resolved correctly in the worker environment.
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.sampling_params import SamplingParams
        from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
        from transformers import AutoTokenizer
        
        self.SamplingParams = SamplingParams
        # Attempt to use HF_HOME from env, otherwise default to huggingface_hub's default cache
        # This ensures the worker tries to use the same cache path as the driver.
        hf_home_cache_path = os.environ.get("HF_HOME", HUGGINGFACE_HUB_CACHE)
        if not os.path.isdir(hf_home_cache_path):
            try:
                os.makedirs(hf_home_cache_path, exist_ok=True)
                logging.info(f"Created HF cache directory for worker: {hf_home_cache_path}")
            except OSError as e:
                logging.warning(f"Worker could not create HF cache directory {hf_home_cache_path}: {e}. "
                                 "This might lead to download issues if the default cache is not writable.")


        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            disable_log_stats=disable_log_stats,
            download_dir=hf_home_cache_path, # Explicitly tell vLLM where to download/look for models
            ignore_patterns=["*.safetensors.index.json", "*.pt", "*.bin.index.json", "*.gitattributes"], # Ignore common problematic files
            trust_remote_code=True, # Required for models with custom code like GenRM
            **engine_kwargs
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # Load tokenizer for chat template functionality
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=hf_home_cache_path,
            trust_remote_code=True,
        )
        
        logging.info(f"AsyncVLLMWorker initialized with model: {model_name}")
    
    async def get_response(self, request_id: str, messages: List[Dict[str, str]], metadata: LLMJudgeEnvironmentMetadata, sampling_params_dict: dict, system_prompt: Optional[str] = None) -> str:
        """Generate response from conversation messages.
        
        Args:
            request_id: Unique request identifier
            messages: List of conversation messages in OpenAI format [{"role": "user", "content": "..."}, ...]
            metadata: Environment metadata
            sampling_params_dict: Sampling parameters for generation
            system_prompt: Optional system prompt to prepend to conversation
            
        Returns:
            Tuple of (request_id, generated_text)
        """
        # Add system prompt if provided
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        # Apply chat template to format the conversation
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        logging.info(f"Formatted Model Prompt: {formatted_prompt}")
        
        sampling_params = self.SamplingParams(**sampling_params_dict)
        results_generator = self.engine.generate(formatted_prompt, sampling_params, request_id)
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        

        if final_output and not final_output.finished:
            logging.info(f"Request {request_id} did not finish within the token limit, Output: {final_output}")

        return request_id, final_output.outputs[0].text.strip()
        


@ray.remote
class AdvTrainingWithReferenceEnvironment(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM # The environment actor itself uses system python

    def __init__(self, cfg: LLMJudgeAsyncConfig):
        self.cfg = cfg
        self.num_target_workers = cfg["target_model"]["num_workers"]
        self.num_reference_workers = cfg["reference_model"]["num_workers"]
        self.num_judge_workers = cfg["judge_model"]["num_workers"]
        
        target_tensor_parallel_size = cfg["target_model"]["tensor_parallel_size"]
        reference_tensor_parallel_size = cfg["reference_model"]["tensor_parallel_size"]
        judge_tensor_parallel_size = cfg["judge_model"]["tensor_parallel_size"]

        # Initialize conversation logging
        self.log_conversations = cfg.get("log_conversations", True)
        self.log_dir = None
        self.step_counter = 0
        
        if self.log_conversations:
            log_dir = cfg.get("log_dir", "./env_logs")
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Use the provided log_dir directly for step logs; do not create a UUID subdirectory
            self.env_log_dir = self.log_dir

            # Initialize step counter from existing step logs if present
            try:
                existing_steps = []
                for step_file in self.env_log_dir.glob("step_*.json"):
                    match = re.search(r"step_(\\d+)\\.json$", step_file.name)
                    if match:
                        existing_steps.append(int(match.group(1)))
                if len(existing_steps) > 0:
                    self.step_counter = max(existing_steps) + 1
            except Exception as e:
                logging.warning(f"Could not infer step counter from existing logs in {self.env_log_dir}: {e}")
            
            logging.info(f"Environment logging enabled. Logs will be saved to: {self.env_log_dir}")

        # Pass down critical environment variables (HF cache, etc.) to workers.
        env_vars_to_pass = {}
        for key in [
            "HF_HOME",
            "TRANSFORMERS_CACHE",
            "WANDB_API_KEY",
            "HUGGINGFACE_HUB_DISABLE_XET",  # Often set to "1" to bypass xet errors.
            "HF_TOKEN",
            "VLLM_USE_V1",  # vLLM v0.10.1+ V1 engine control
        ]:
            if key in os.environ:
                env_vars_to_pass[key] = os.environ[key]

        # Ensure xet is disabled to avoid CAS service issues if not explicitly set.
        env_vars_to_pass.setdefault("HUGGINGFACE_HUB_DISABLE_XET", "1")

        #########################################################
        #                       Target Workers                    
        #########################################################
        
        # Create placement groups for workers that need single-GPU bundles (TP=1)
        single_gpu_workers = 0
        if target_tensor_parallel_size == 1:
            single_gpu_workers += self.num_target_workers
        if reference_tensor_parallel_size == 1:
            single_gpu_workers += self.num_reference_workers
        if judge_tensor_parallel_size == 1:
            single_gpu_workers += self.num_judge_workers
            
        if single_gpu_workers > 0:
            bundle_ct_per_node_list = [1] * single_gpu_workers  # 1 GPU per single-GPU worker

            self.virtual_cluster = RayVirtualCluster(
                bundle_ct_per_node_list=bundle_ct_per_node_list,
                use_gpus=True,
                name="llm_judge_async_vc",
            )
            # self.virtual_cluster.print_cluster_grid()
            placement_groups = self.virtual_cluster.get_placement_groups()
        else:
            # No placement group / virtual cluster -> rely on Ray scheduler.
            self.virtual_cluster = None
            placement_groups = []

        

        print(f"Using py_executable: {AsyncVLLMWorker.DEFAULT_PY_EXECUTABLE}")
        print(f"Using env_vars: {env_vars_to_pass}")
        print(f"Using tensor_parallel_size: {target_tensor_parallel_size}")
        print(f"Using num_workers: {self.num_target_workers}")
        target_worker_options = {
            "runtime_env": {
                "py_executable": AsyncVLLMWorker.DEFAULT_PY_EXECUTABLE,
                "env_vars": env_vars_to_pass,
            },
            "num_gpus": target_tensor_parallel_size,
        }
        
        
        self.target_workers = []
        print(f"Creating {self.num_target_workers} AsyncVLLMWorker actors.")
        for i in range(self.num_target_workers):
            # If tensor_parallel_size == 1, we can safely pin the actor to a
            # single-GPU bundle inside the placement group. Otherwise, each
            # actor needs multiple GPUs and cannot fit into the 1-GPU bundles
            # created by RayVirtualCluster. In that case we rely on Ray's
            # default scheduler (no placement group) to allocate all requested
            # GPUs on the same node.

            if target_tensor_parallel_size == 1:
                pg_index = i % len(placement_groups)
                pg = placement_groups[pg_index]
                scheduling_kwargs = dict(
                    scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                        placement_group=pg
                    )
                )
            else:
                # No placement group – let Ray handle multi-GPU allocation.
                # TODO @yashaswikarnati: improve with custom scheduling strategy
                scheduling_kwargs = {}
            worker = AsyncVLLMWorker.options(
                **target_worker_options,
                **scheduling_kwargs,
            ).remote(
                model_name=cfg["target_model"]["model_name"],
                tensor_parallel_size=cfg["target_model"]["tensor_parallel_size"],
                gpu_memory_utilization=cfg["target_model"].get("gpu_memory_utilization", 0.85),
                max_model_len=cfg["target_model"].get("max_model_len", 8192),
                # Pass any other engine args from cfg if needed
            )
            self.target_workers.append(worker)
        
        logging.info(f"Created {len(self.target_workers)} AsyncVLLMWorker actors.")
        self._request_counter = 0 # For generating unique request IDs per step call
        self._actor_id_prefix = str(uuid.uuid4())[:8] # Unique prefix for this actor instance

        #########################################################
        #                       Reference Workers                    
        #########################################################

        print(f"Using py_executable: {AsyncVLLMWorker.DEFAULT_PY_EXECUTABLE}")
        print(f"Using env_vars: {env_vars_to_pass}")
        print(f"Using tensor_parallel_size: {reference_tensor_parallel_size}")
        print(f"Using num_workers: {self.num_reference_workers}")

        reference_worker_options = {
            "runtime_env": {
                "py_executable": AsyncVLLMWorker.DEFAULT_PY_EXECUTABLE,
                "env_vars": env_vars_to_pass,
            },
            "num_gpus": reference_tensor_parallel_size,
        }

        self.reference_workers = []
        print(f"Creating {self.num_reference_workers} AsyncVLLMWorker actors.")
        for i in range(self.num_reference_workers):
            if reference_tensor_parallel_size == 1:
                # Reference workers use placement groups AFTER target workers (only if target workers also use TP=1)
                target_workers_using_pg = self.num_target_workers if target_tensor_parallel_size == 1 else 0
                pg_index = (target_workers_using_pg + i) % len(placement_groups)
                pg = placement_groups[pg_index]
                scheduling_kwargs = dict(
                    scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                        placement_group=pg
                    )
                )
            else:
                # No placement group – let Ray handle multi-GPU allocation.
                # TODO @yashaswikarnati: improve with custom scheduling strategy
                scheduling_kwargs = {}
            worker = AsyncVLLMWorker.options(
                **reference_worker_options,
                **scheduling_kwargs,
            ).remote(
                model_name=cfg["reference_model"]["model_name"],
                tensor_parallel_size=cfg["reference_model"]["tensor_parallel_size"],
                gpu_memory_utilization=cfg["reference_model"].get("gpu_memory_utilization", 0.85),
                max_model_len=cfg["reference_model"].get("max_model_len", 40000),
            )
            self.reference_workers.append(worker)

        #########################################################
        #                       Judge Workers                    
        #########################################################

        print(f"Using py_executable: {AsyncVLLMWorker.DEFAULT_PY_EXECUTABLE}")
        print(f"Using env_vars: {env_vars_to_pass}")
        print(f"Using tensor_parallel_size: {judge_tensor_parallel_size}")
        print(f"Using num_workers: {self.num_judge_workers}")

        judge_worker_options = {
            "runtime_env": {
                "py_executable": AsyncVLLMWorker.DEFAULT_PY_EXECUTABLE,
                "env_vars": env_vars_to_pass,
            },
            "num_gpus": judge_tensor_parallel_size,
        }

        self.judge_workers = []
        print(f"Creating {self.num_judge_workers} AsyncVLLMWorker actors.")
        for i in range(self.num_judge_workers):
            if judge_tensor_parallel_size == 1:
                # Judge workers use placement groups AFTER target and reference workers
                target_workers_using_pg = self.num_target_workers if target_tensor_parallel_size == 1 else 0
                reference_workers_using_pg = self.num_reference_workers if reference_tensor_parallel_size == 1 else 0
                pg_index = (target_workers_using_pg + reference_workers_using_pg + i) % len(placement_groups)
                pg = placement_groups[pg_index]
                scheduling_kwargs = dict(
                    scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                        placement_group=pg
                    )
                )
            else:
                # No placement group – let Ray handle multi-GPU allocation.
                # TODO @yashaswikarnati: improve with custom scheduling strategy
                scheduling_kwargs = {}
            worker = AsyncVLLMWorker.options(
                **judge_worker_options,
                **scheduling_kwargs,
            ).remote(
                model_name=cfg["judge_model"]["model_name"],
                tensor_parallel_size=cfg["judge_model"]["tensor_parallel_size"],
                gpu_memory_utilization=cfg["judge_model"].get("gpu_memory_utilization", 0.85),
                max_model_len=cfg["judge_model"].get("max_model_len", 40000),
            )
            self.judge_workers.append(worker)
        #########################################################

    def shutdown(self):
        for worker in self.target_workers + self.reference_workers + self.judge_workers:
            ray.kill(worker)
        if self.virtual_cluster is not None:
            self.virtual_cluster.shutdown()

    def _log_step_data(self, step_data: Dict) -> None:
        """Log conversation data for this step to a JSON file."""
        if not self.log_conversations or not hasattr(self, 'env_log_dir'):
            return
            
        try:
            # Create filename with step counter
            filename = f"step_{self.step_counter:06d}.json"
            log_file = self.env_log_dir / filename
            
            # Add metadata
            step_data["step_number"] = self.step_counter
            step_data["timestamp"] = uuid.uuid4().hex
            
            # Write to file
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(step_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logging.warning(f"Failed to log step data: {e}")

    def extract_score(self, judge_response: str) -> float:
        individual_pattern = r'\[The Begin of Individual Scores\](.*?)\[The End of Individual Scores\]'
        individual_match = re.search(individual_pattern, judge_response, re.DOTALL)
        if individual_match:
            individual_section_text = individual_match.group(1).strip()
            individual_boxed_content = extract_answer_from_box(individual_section_text)
            if individual_boxed_content and ',' in individual_boxed_content:
                # Individual scores: "2, 4"
                parts = individual_boxed_content.split(',')
                if len(parts) == 2:
                    try:
                        individual_scores = (float(parts[0].strip()), float(parts[1].strip()))
                        return individual_scores
                    except ValueError:
                        logging.warning(f"Could not parse individual scores: {individual_boxed_content}")
        return 0, 0

    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[LLMJudgeEnvironmentMetadata],
    ) -> EnvironmentReturn:
        """
        message_log_batch: List[List[Dict[str, str]]] - List of conversations, each conversation is a list of messages
        metadata: List[LLMJudgeEnvironmentMetadata] - List of metadata for each conversation

        Returns:
            EnvironmentReturn: A NamedTuple containing the following fields:
                - observations (List[Dict[str, str]]): A list of observations, where
                  each observation is a dictionary representing the judge's feedback
                  (e.g., `{"role": "environment", "content": "Environment: Score = X.XX"}`).
        """
        generated_prompts = []
        for conversation, single_metadata in zip(message_log_batch, metadata):
            assert len(conversation) == 2, "LLMJudgeAsyncEnvironment only supports single turn conversations for now"
            
            generated_prompts.append(
                [{
                    "role": "user",
                    "content": conversation[-1]["content"].split("</prompt>")[0].strip().split("<prompt>")[-1].strip(),
                }]
            )

        #########################################################
        #                       Target Responses                    
        #########################################################

        futures = []
        
        # Prepare target sampling parameters from target model config
        target_sampling_params = {
            "temperature": self.cfg["target_model"].get("temperature", 0.0),
            "max_tokens": self.cfg["target_model"].get("max_tokens", 512),
            "stop": self.cfg["target_model"].get("stop", None),
        }
        system_prompt = self.cfg["target_model"].get("system_prompt", None)

        # For each batch, loop through each conversation and send it to a target worker asynchronously
        for i, (prompt, single_metadata) in enumerate(zip(generated_prompts, metadata)):
            # Generate a unique request ID for vLLM for this specific call to target
            request_id = f"env_{self._actor_id_prefix}_step_{self._request_counter}_{i}"
            
            worker_idx = i % self.num_target_workers # Simple round-robin
            
            current_target_params = target_sampling_params.copy()

            future = self.target_workers[worker_idx].get_response.remote(
                request_id, prompt, single_metadata, current_target_params,
                system_prompt,
            )
            futures.append(future)
        
        self._request_counter += 1 # Increment for the next step call

        target_responses: List[Tuple[str, float]] = ray.get(futures)

        target_responses = [target_response for _, target_response in target_responses]

        print("length of target_responses: ", len(target_responses))
        print("length of generated_prompts: ", len(generated_prompts))


        #########################################################
        #                       Reference Responses                    
        #########################################################

        reference_futures = []
        
        # Prepare reference sampling parameters from reference model config
        reference_sampling_params = {
            "temperature": self.cfg["reference_model"].get("temperature", 0.0),
            "max_tokens": self.cfg["reference_model"].get("max_tokens", 512),
            "stop": self.cfg["reference_model"].get("stop", None),
        }
        reference_system_prompt = self.cfg["reference_model"].get("system_prompt", None)

        # For each batch, loop through each conversation and send it to a reference worker asynchronously
        for i, (prompt, single_metadata) in enumerate(zip(generated_prompts, metadata)):
            # Generate a unique request ID for vLLM for this specific call to reference
            request_id = f"env_{self._actor_id_prefix}_step_{self._request_counter}_ref_{i}"
            
            worker_idx = i % self.num_reference_workers # Simple round-robin
            
            current_reference_params = reference_sampling_params.copy()

            future = self.reference_workers[worker_idx].get_response.remote(
                request_id, prompt, single_metadata, current_reference_params,
                reference_system_prompt,
            )
            reference_futures.append(future)
        
        self._request_counter += 1 # Increment for the next step call

        reference_responses: List[Tuple[str, float]] = ray.get(reference_futures)

        reference_responses = [reference_response for _, reference_response in reference_responses]

        print("length of reference_responses: ", len(reference_responses))
        
        
        
        #########################################################
        #                       Judge Responses                    
        #########################################################
        judge_prompts = []
        for i, target_response in enumerate(target_responses):
            print(f"Generated Prompt: {generated_prompts[i]}")
            print(f"Target Response: {target_response}")    
            print(f"Reference Response: {reference_responses[i]}")

            judge_prompts.append(
                [
                    {
                        "role": "user",
                        "content": generated_prompts[i][0]["content"],
                    },
                    {
                        "role": "response_1",
                        "content": target_response,
                    },
                    {
                        "role": "response_2",
                        "content": reference_responses[i],
                    }
                ]
            )
        
        # Prepare judge sampling parameters from judge model config
        judge_sampling_params = {
            "temperature": self.cfg["judge_model"].get("temperature", 0.0),
            "max_tokens": self.cfg["judge_model"].get("max_tokens", 2048),  # Judges typically need more tokens
            "stop": self.cfg["judge_model"].get("stop", None),
        }
        judge_system_prompt = self.cfg["judge_model"].get("system_prompt", None)
        
        judge_futures = []  # Create a new futures list for judge workers
        for i, prompt in enumerate(judge_prompts):
            request_id = f"env_{self._actor_id_prefix}_step_{self._request_counter}_judge_{i}"
            worker_idx = i % self.num_judge_workers
            
            current_judge_params = judge_sampling_params.copy()
            
            future = self.judge_workers[worker_idx].get_response.remote(request_id, prompt, metadata[i], current_judge_params, judge_system_prompt)
            judge_futures.append(future)
        
        judge_responses: List[Tuple[str, float]] = ray.get(judge_futures)

        judge_responses = [judge_response for _, judge_response in judge_responses]
        
        print("length of judge_responses: ", len(judge_responses))
        print("length of target_responses: ", len(target_responses))
        
        rewards = []
        for i, judge_response in enumerate(judge_responses):
            score1, score2 = self.extract_score(judge_response)
            print(f"Score: {score1}, {score2}")
            rewards.append(score2 - score1)
            
            print(f"Reward for {i}: {rewards[-1]}")
            print(f"Judge Prompt: {judge_prompts[i]}")
            print(f"Judge Response: {judge_response}")
        


        observations = [{"role": "environment", "content": f"Environment: Response = {target_response}"} for target_response in target_responses]
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).cpu()
        terminateds_tensor = torch.ones_like(rewards_tensor).cpu()
        
        next_stop_strings = [None] * len(message_log_batch) 

        # Log conversation data

        if self.log_conversations:
            step_data = {
                "input_conversations": message_log_batch,
                "generated_prompts": generated_prompts,
                "target_responses": target_responses,
                "reference_responses": reference_responses,
                "judge_prompts": judge_prompts,
                "judge_responses": judge_responses,
                "rewards": rewards,
                "metadata": metadata
            }
            self._log_step_data(step_data)
            
        # Increment step counter
        self.step_counter += 1

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata, 
            next_stop_strings=next_stop_strings,
            rewards=rewards_tensor,
            terminateds=terminateds_tensor,
            answers=None,  # No answers for judge environment
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        """Computes metrics for this environment given a global rollout batch.

        Every rank will run this function, so you're free to use distributed
        calculations if you'd prefer for heavy metrics.
        """
        batch["rewards"] = (
            batch["rewards"] * batch["is_end"]
        )  # set a reward of 0 for any incorrectly ended sequences
        if (batch["rewards"] == 1).float().sum() > 0:
            correct_solution_generation_lengths = (
                (batch["generation_lengths"] - batch["prompt_lengths"])[
                    batch["rewards"] == 1
                ]
                .float()
                .mean()
                .item()
            )
        else:
            correct_solution_generation_lengths = 0

        metrics = {
            # "table": table, TODO @sahilj WIP
            "accuracy": batch["rewards"].mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], batch["rewards"]
            ),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
        }

        return batch, metrics
