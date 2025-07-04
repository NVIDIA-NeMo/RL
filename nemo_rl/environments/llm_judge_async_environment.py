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
from nemo_rl.environments.metrics import (
    calculate_pass_rate_per_prompt,
)
from nemo_rl.environments.utils import extract_answer_from_box


class LLMJudgeAsyncConfig(TypedDict):
    num_workers: int
    model_name: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    # Default sampling parameters for the judge
    temperature: Optional[float]
    max_tokens: Optional[int]
    stop: Optional[List[str]]  # Note: vLLM's AsyncEngine uses 'stop'
    max_concurrency: Optional[
        int
    ]  # Maximum concurrent step calls for the environment actor
    reasoning_split_word: Optional[
        str
    ]  # Configurable split word for response processing (default: "</think>")
    # Any other vllm.SamplingParams can be added here


class LLMJudgeEnvironmentMetadata(TypedDict):  # Reusing from previous
    reference_answer: Optional[str]
    evaluation_criteria: Optional[str]
    judge_prompt_template: Optional[str]  # Added for per-sample judge prompts
    extract_box: Optional[bool]
    question: Optional[str]  # Added to store the question in metadata


@ray.remote
class AsyncVLLMWorker:
    """Worker that serves an LLM using vllm.AsyncLLMEngine for judging responses.

    Each call to judge() handles a single prompt asynchronously.
    """

    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.VLLM
    DEFAULT_JUDGE_PROMPT_TEMPLATE = """You are an expert judge. You will be given a question, a model's prediction to evaluate, a reference answer, and evaluation criteria.

Question:
{question}

Model's Prediction:
{response}

Reference Answer:
{reference}

Now, evaluate if the response is correct based on the following evaluation criteria:
{criteria}

Answer yes or no, then give your reasoning.
"""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: Optional[int] = None,
        disable_log_stats: bool = True,  # Reduce verbose VLLM logging
        reasoning_split_word: Optional[
            str
        ] = None,  # Configurable split word for response processing
        **engine_kwargs,  # Allow passing other EngineArgs
    ):
        # Imports moved here to be within the Ray actor's context,
        # ensuring they are resolved correctly in the worker environment.
        from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.sampling_params import SamplingParams

        self.SamplingParams = SamplingParams
        # Attempt to use HF_HUB_CACHE from env, otherwise default to huggingface_hub's default cache
        # This ensures the worker tries to use the same cache path as the driver.
        hf_hub_cache_path = os.environ.get("HF_HUB_CACHE", HUGGINGFACE_HUB_CACHE)
        if not os.path.isdir(hf_hub_cache_path):
            try:
                os.makedirs(hf_hub_cache_path, exist_ok=True)
                logging.info(
                    f"Created HF cache directory for worker: {hf_hub_cache_path}"
                )
            except OSError as e:
                logging.warning(
                    f"Worker could not create HF cache directory {hf_hub_cache_path}: {e}. "
                    "This might lead to download issues if the default cache is not writable."
                )

        # It's critical that download_dir is set for vLLM if HF_HUB_CACHE is being customized,
        # or if there are any doubts about vLLM picking up the environment variable.
        # Also add ignore_patterns to prevent issues with problematic aux files.
        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            disable_log_stats=disable_log_stats,
            download_dir=hf_hub_cache_path,  # Explicitly tell vLLM where to download/look for models
            ignore_patterns=[
                "*.safetensors.index.json",
                "*.pt",
                "*.bin.index.json",
                "*.gitattributes",
            ],  # Ignore common problematic files
            **engine_kwargs,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.reasoning_split_word = reasoning_split_word  # Store as-is, could be None
        logging.info(f"AsyncVLLMWorker initialized with model: {model_name}")

    async def judge(
        self,
        request_id: str,
        question: str,
        response_to_judge: str,
        metadata: LLMJudgeEnvironmentMetadata,
        sampling_params_dict: dict,
    ) -> Tuple[str, float]:
        """Judges a single response using the LLM.

        Args:
            request_id: A unique ID for this generation request.
            question: The question string.
            response_to_judge: The assistant's response string.
            metadata: Metadata containing reference answer and criteria.
            sampling_params_dict: Dictionary to initialize vllm.SamplingParams.

        Returns:
            Tuple of (request_id, score)
        """
        reference = metadata.get("reference_answer", "")
        criteria = metadata.get("evaluation_criteria", "")
        extract_box = metadata.get("extract_box", False)

        # Only split if reasoning_split_word is provided
        if self.reasoning_split_word:
            response_to_judge = response_to_judge.split(self.reasoning_split_word)[
                -1
            ].lstrip()
        response_to_judge = (
            extract_answer_from_box(response_to_judge)
            if extract_box
            else response_to_judge
        )

        response_to_judge = "None" if response_to_judge is None else response_to_judge
        # Prioritize metadata's judge_prompt_template, then default
        # Note that if you want to use a custom judge_prompt_template, you may need to change the verdict extraction logic accordingly
        current_judge_prompt = (
            metadata.get("judge_prompt_template") or self.DEFAULT_JUDGE_PROMPT_TEMPLATE
        )

        prompt = current_judge_prompt.format(
            question=question,
            response=response_to_judge,
            reference=reference,
            criteria=criteria,
        )
        logging.info(f"Prompt: {prompt}")

        sampling_params = self.SamplingParams(**sampling_params_dict)

        results_generator = self.engine.generate(prompt, sampling_params, request_id)

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        score = 0.0
        if final_output and not final_output.finished:
            logging.info(
                f"Request {request_id} did not finish within the token limit, but we will score it anyway. Output: {final_output}"
            )
        elif final_output:
            generated_text = final_output.outputs[0].text.strip()
            generated_text_lower = generated_text.lower()

            has_yes = "yes" in generated_text_lower

            if has_yes:
                score = 1.0
                logging.info(
                    f"Parsed 'yes' for request {request_id}. Score: {score}. Output: '{generated_text}'"
                )
            else:
                score = 0.0
                logging.info(
                    f"No 'yes' found in {request_id}. Score: {score}. Output: '{generated_text}'"
                )

        else:
            logging.warning(f"No output received from LLM for request {request_id}.")

        return request_id, score


@ray.remote
class LLMJudgeAsyncEnvironment(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = (
        PY_EXECUTABLES.SYSTEM
    )  # The environment actor itself uses system python

    def __init__(self, cfg: LLMJudgeAsyncConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]

        tensor_parallel_size = cfg.get("tensor_parallel_size", 1)

        # Only create a RayVirtualCluster (and thereby reserve GPU bundles)
        # if we actually need single-GPU bundles (tensor_parallel_size == 1).
        if tensor_parallel_size == 1:
            bundle_ct_per_node_list = [
                tensor_parallel_size
            ] * self.num_workers  # == [1] * num_workers

            self.virtual_cluster = RayVirtualCluster(
                bundle_ct_per_node_list=bundle_ct_per_node_list,
                use_gpus=True,
                name="llm_judge_async_vc",
            )
            self.virtual_cluster.print_cluster_grid()
            placement_groups = self.virtual_cluster.get_placement_groups()
        else:
            # No placement group / virtual cluster -> rely on Ray scheduler.
            self.virtual_cluster = None
            placement_groups = []

        # Pass down critical environment variables (HF cache, etc.) to workers.
        env_vars_to_pass = {}
        for key in [
            "HF_HUB_CACHE",
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

        worker_options = {
            "runtime_env": {
                "py_executable": AsyncVLLMWorker.DEFAULT_PY_EXECUTABLE,
                "env_vars": env_vars_to_pass,
            },
            "num_gpus": tensor_parallel_size,
        }

        self.workers = []
        for i in range(self.num_workers):
            # If tensor_parallel_size == 1, we can safely pin the actor to a
            # single-GPU bundle inside the placement group. Otherwise, each
            # actor needs multiple GPUs and cannot fit into the 1-GPU bundles
            # created by RayVirtualCluster. In that case we rely on Ray's
            # default scheduler (no placement group) to allocate all requested
            # GPUs on the same node.

            if tensor_parallel_size == 1:
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
                **worker_options,
                **scheduling_kwargs,
            ).remote(
                model_name=cfg["model_name"],
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.85),
                max_model_len=cfg.get("max_model_len"),
                reasoning_split_word=cfg.get("reasoning_split_word"),
                # Pass any other engine args from cfg if needed
            )
            self.workers.append(worker)

        logging.info(f"Created {len(self.workers)} AsyncVLLMWorker actors.")
        self._request_counter = 0  # For generating unique request IDs per step call
        self._actor_id_prefix = str(uuid.uuid4())[
            :8
        ]  # Unique prefix for this actor instance

    def shutdown(self):
        for worker in self.workers:
            ray.kill(worker)
        if self.virtual_cluster is not None:
            self.virtual_cluster.shutdown()

    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[LLMJudgeEnvironmentMetadata],
    ) -> EnvironmentReturn:
        """Step function for the LLM judge environment.

        Args:
            message_log_batch: List[List[Dict[str, str]]] - List of conversations, each conversation is a list of messages
            metadata: List[LLMJudgeEnvironmentMetadata] - List of metadata for each conversation

        Returns:
            EnvironmentReturn: A NamedTuple containing the following fields:
                - observations (List[Dict[str, str]]): A list of observations, where
                  each observation is a dictionary representing the judge's feedback
                  (e.g., `{"role": "environment", "content": "Environment: Score = X.XX"}`).
        """
        assistant_responses = []
        questions = []
        for conversation, single_metadata in zip(message_log_batch, metadata):
            assert len(conversation) == 2, (
                "LLMJudgeAsyncEnvironment only supports single turn conversations for now"
            )

            # Read question from metadata instead of parsing conversation
            question = single_metadata.get("question")
            assert question is not None, "Question not found in metadata"
            questions.append(question)
            assistant_responses.append(conversation[-1]["content"])

        futures = []

        # Prepare default sampling parameters from config
        default_sampling_params = {
            "temperature": self.cfg.get("temperature", 0.0),
            "max_tokens": self.cfg.get("max_tokens", 512),
            "stop": self.cfg.get("stop", None),
        }

        # For each batch, loop through each conversation and send it to a judge worker asynchronously
        for i, (question_str, response_str, single_metadata) in enumerate(
            zip(questions, assistant_responses, metadata)
        ):
            # Generate a unique request ID for vLLM for this specific call to judge
            request_id = f"env_{self._actor_id_prefix}_step_{self._request_counter}_{i}"

            worker_idx = i % self.num_workers  # Simple round-robin

            current_sampling_params = default_sampling_params.copy()

            future = self.workers[worker_idx].judge.remote(
                request_id,
                question_str,
                response_str,
                single_metadata,
                current_sampling_params,
            )
            futures.append(future)

        self._request_counter += 1  # Increment for the next step call

        results_tuples: List[Tuple[str, float]] = ray.get(futures)

        # Assuming ray.get(futures) preserves the order, which it does.
        scores = [score for _, score in results_tuples]

        observations = []
        for score, single_meta in zip(scores, metadata):
            ref_answer = single_meta.get("reference_answer", "N/A")
            observations.append(
                {
                    "role": "environment",
                    "content": f"Environment: Score = {score:.2f}\nGround Truth: {ref_answer}",
                }
            )

        rewards_tensor = torch.tensor(scores, dtype=torch.float32).cpu()
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
