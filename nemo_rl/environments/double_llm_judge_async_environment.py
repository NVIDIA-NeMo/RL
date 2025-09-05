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
import re
import uuid
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

        # It's critical that download_dir is set for vLLM if HF_HOME is being customized,
        # or if there are any doubts about vLLM picking up the environment variable.
        # Also add ignore_patterns to prevent issues with problematic aux files.
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
    
    async def get_response(self, request_id: str, messages: List[Dict[str, str]], metadata: LLMJudgeEnvironmentMetadata, sampling_params_dict: dict) -> str:
        """Generate response from conversation messages.
        
        Args:
            request_id: Unique request identifier
            messages: List of conversation messages in OpenAI format [{"role": "user", "content": "..."}, ...]
            metadata: Environment metadata
            sampling_params_dict: Sampling parameters for generation
            
        Returns:
            Tuple of (request_id, generated_text)
        """
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
        
    async def judge(
        self,
        request_id: str,
        question: str,
        attacker_response: str,
        metadata: LLMJudgeEnvironmentMetadata,
        sampling_params_dict: dict,
    ) -> Tuple[str, float]:
        """
        Judges a single response using the LLM.

        Args:
            request_id: A unique ID for this generation request.
            question: The question string.
            response_to_judge: The assistant's response string.
            metadata: Metadata containing reference answer and criteria.
            sampling_params_dict: Dictionary to initialize vllm.SamplingParams.

        Returns:
            Tuple of (request_id, score)
        """

        response_to_judge = attacker_response.split("</prompt>")[0].strip().split("<prompt>")[-1].strip()
        # Prioritize metadata's judge_prompt_template, then default
        # Note that if you want to use a custom judge_prompt_template, you may need to change the verdict extraction logic accordingly
        current_judge_prompt = self.DEFAULT_JUDGE_PROMPT_TEMPLATE
        
        prompt = current_judge_prompt.format(
            prompt=attacker_response,
        )
        logging.info(f"Target Model Prompt: {prompt}")

        sampling_params = self.SamplingParams(**sampling_params_dict)
        
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        score = 0.0
        if final_output and not final_output.finished:
            logging.info(f"Request {request_id} did not finish within the token limit, but we will score it anyway. Output: {final_output}")
        elif final_output:
            generated_text = final_output.outputs[0].text.strip()
            generated_text_lower = generated_text.lower()

            has_yes = "yes" in generated_text_lower

            if has_yes:
                score = 1.0
                logging.info(f"Parsed 'yes' for request {request_id}. Score: {score}. Output: '{generated_text}'")
            else:
                score = 0.0
                logging.info(f"No 'yes' found in {request_id}. Score: {score}. Output: '{generated_text}'")
            
        else:
            logging.warning(f"No output received from LLM for request {request_id}.")

        return request_id, score


@ray.remote
class DoubleLLMJudgeAsyncEnvironment(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM # The environment actor itself uses system python

    def __init__(self, cfg: LLMJudgeAsyncConfig):
        self.cfg = cfg
        # Use single workers for batching efficiency
        self.num_target_workers = 1
        self.num_judge_workers = 1
        
        target_tensor_parallel_size = cfg["target_model"]["tensor_parallel_size"]
        judge_tensor_parallel_size = cfg["judge_model"]["tensor_parallel_size"]

        # Pass down critical environment variables (HF cache, etc.) to workers.
        env_vars_to_pass = {}
        for key in [
            "HF_HOME",
            "TRANSFORMERS_CACHE",
            "WANDB_API_KEY",
            "HUGGINGFACE_HUB_DISABLE_XET",  # Often set to "1" to bypass xet errors.
            "HF_TOKEN",
        ]:
            if key in os.environ:
                env_vars_to_pass[key] = os.environ[key]

        # Ensure xet is disabled to avoid CAS service issues if not explicitly set.
        env_vars_to_pass.setdefault("HUGGINGFACE_HUB_DISABLE_XET", "1")

        #########################################################
        #                       Target Workers                    
        #########################################################
        
        # Simplified placement group logic for single workers
        # Only create placement groups if both models use TP=1 (single GPU each)
        if target_tensor_parallel_size == 1 and judge_tensor_parallel_size == 1:
            # 2 workers total, each needs 1 GPU
            bundle_ct_per_node_list = [1, 1]  # 1 GPU per worker

            self.virtual_cluster = RayVirtualCluster(
                bundle_ct_per_node_list=bundle_ct_per_node_list,
                use_gpus=True,
                name="llm_judge_async_vc",
            )
            placement_groups = self.virtual_cluster.get_placement_groups()
        else:
            # Multi-GPU workers -> rely on Ray scheduler
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
        
        
        # Create single target worker for batching
        print(f"Creating 1 target AsyncVLLMWorker actor with TP={target_tensor_parallel_size}")
        
        if target_tensor_parallel_size == 1 and placement_groups:
            # Use first placement group for single GPU
            scheduling_kwargs = dict(
                scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                    placement_group=placement_groups[0]
                )
            )
        else:
            # Multi-GPU or no placement groups -> let Ray handle allocation
            scheduling_kwargs = {}
            
        self.target_worker = AsyncVLLMWorker.options(
            **target_worker_options,
            **scheduling_kwargs,
        ).remote(
            model_name=cfg["target_model"]["model_name"],
            tensor_parallel_size=cfg["target_model"]["tensor_parallel_size"],
            gpu_memory_utilization=cfg["target_model"].get("gpu_memory_utilization", 0.85),
            max_model_len=cfg["target_model"].get("max_model_len"),
        )
        
        logging.info(f"Created target AsyncVLLMWorker actor.")
        self._request_counter = 0 # For generating unique request IDs per step call
        self._actor_id_prefix = str(uuid.uuid4())[:8] # Unique prefix for this actor instance

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

        # Create single judge worker for batching
        print(f"Creating 1 judge AsyncVLLMWorker actor with TP={judge_tensor_parallel_size}")
        
        if judge_tensor_parallel_size == 1 and placement_groups:
            # Use second placement group for single GPU (target uses first)
            pg_index = 1 if len(placement_groups) > 1 else 0
            scheduling_kwargs = dict(
                scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                    placement_group=placement_groups[pg_index]
                )
            )
        else:
            # Multi-GPU or no placement groups -> let Ray handle allocation
            scheduling_kwargs = {}
            
        self.judge_worker = AsyncVLLMWorker.options(
            **judge_worker_options,
            **scheduling_kwargs,
        ).remote(
            model_name=cfg["judge_model"]["model_name"],
            tensor_parallel_size=cfg["judge_model"]["tensor_parallel_size"],
            gpu_memory_utilization=cfg["judge_model"].get("gpu_memory_utilization", 0.85),
            max_model_len=cfg["judge_model"].get("max_model_len"),
        )
        #########################################################

    def shutdown(self):
        ray.kill(self.target_worker)
        ray.kill(self.judge_worker)
        if self.virtual_cluster is not None:
            self.virtual_cluster.shutdown()

    def extract_score(self, judge_response: str) -> float:
        individual_pattern = r'\[The Begin of Individual Scores\](.*?)\[The End of Individual Scores\]'
        individual_match = re.search(individual_pattern, judge_response, re.DOTALL)
        if individual_match:
            individual_section_text = individual_match.group(1).strip()
            individual_boxed_content = extract_answer_from_box(individual_section_text)
            if individual_boxed_content:
                try:
                    individual_score = float(individual_boxed_content)
                except ValueError:
                    individual_score = 5.1
                    logging.warning(f"Could not parse individual scores: {individual_boxed_content}")
            return individual_score
        return 5.0

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

        futures = []
        
        # Prepare default sampling parameters from config
        default_sampling_params = {
            "temperature": self.cfg.get("temperature", 0.0),
            "max_tokens": self.cfg.get("max_tokens", 512),
            "stop": self.cfg.get("stop", None),
        }

        # Send all requests to the single target worker for batching
        for i, (prompt, single_metadata) in enumerate(zip(generated_prompts, metadata)):
            # Generate a unique request ID for vLLM for this specific call
            request_id = f"env_{self._actor_id_prefix}_step_{self._request_counter}_{i}"
            current_sampling_params = default_sampling_params.copy()

            future = self.target_worker.get_response.remote(
                request_id, prompt, single_metadata, current_sampling_params,
            )
            futures.append(future)
        
        self._request_counter += 1 # Increment for the next step call

        target_responses: List[Tuple[str, float]] = ray.get(futures)

        target_responses = [target_response for _, target_response in target_responses]

        print("length of target_responses: ", len(target_responses))
        print("length of generated_prompts: ", len(generated_prompts))

        
        
        
        #########################################################
        #                       Judge Responses                    
        #########################################################
        judge_prompts = []
        for i, target_response in enumerate(target_responses):
            print(f"Generated Prompt: {generated_prompts[i]}")
            print(f"Target Response: {target_response}")
            judge_prompts.append(
                [
                    {
                        "role": "user",
                        "content": generated_prompts[i][0]["content"],
                    },
                    {
                        "role": "response_1",
                        "content": target_response,
                    }
                ]
            )
        
        # Send all judge requests to the single judge worker for batching
        judge_futures = []  # Create a new futures list for judge workers
        
        for i, prompt in enumerate(judge_prompts):
            request_id = f"env_{self._actor_id_prefix}_step_{self._request_counter}_{i}"
            future = self.judge_worker.get_response.remote(request_id, prompt, metadata[i], current_sampling_params)
            judge_futures.append(future)
        
        judge_responses: List[Tuple[str, float]] = ray.get(judge_futures)

        judge_responses = [judge_response for _, judge_response in judge_responses]
        
        print("length of judge_responses: ", len(judge_responses))
        print("length of target_responses: ", len(target_responses))
        
        rewards = []
        for i, judge_response in enumerate(judge_responses):
            score = self.extract_score(judge_response)
            print(f"Score: {score}")
            rewards.append(score)
            
            print(f"Reward for {i}: {rewards[-1]}")
            print(f"Judge Prompt: {judge_prompts[i]}")
            print(f"Judge Response: {judge_response}")
        


        observations = [{"role": "environment", "content": f"Environment: Response = {target_response}"} for target_response in target_responses]
        
        rewards_tensor = -1 * torch.tensor(rewards, dtype=torch.float32).cpu()
        terminateds_tensor = torch.ones_like(rewards_tensor).cpu()
        
        next_stop_strings = [None] * len(message_log_batch) 

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
