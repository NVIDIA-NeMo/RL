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
import contextlib
import importlib
import importlib.util
import io
import logging
import sys
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, List, NotRequired, Optional, TypedDict

import ray
import torch
from pydantic import BaseModel, Field, model_validator

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.metrics import (
    calculate_pass_rate_per_prompt,
)
from nemo_rl.environments.rewards import (
    bbox_giou_reward,
    combine_reward_functions,
    exact_answer_alphanumeric_reward,
    format_reward,
    geo3k_reward,
    math_expression_reward,
)
from nemo_rl.environments.utils import chunk_list_to_workers


class CustomRewardFunctionConfig(BaseModel, extra="forbid"):
    """Import location for a user-defined VLM reward function."""

    name: str = Field(min_length=1)
    function: str = Field(min_length=1)
    module: Optional[str] = None
    file: Optional[str] = None

    @model_validator(mode="after")
    def _require_exactly_one_source(self) -> "CustomRewardFunctionConfig":
        if (self.module is None) == (self.file is None):
            raise ValueError("exactly one of 'module' or 'file' must be set")
        return self


class VLMEnvConfig(TypedDict):
    num_workers: int
    stop_strings: NotRequired[Optional[list[str]]]  # Default stop strings for this env
    reward_functions: List[dict[str, Any]]  # list of reward functions and their weights
    custom_reward_functions: NotRequired[list[CustomRewardFunctionConfig]]


RewardFunction = Callable[[str, str], tuple[float, Optional[bool]]]

_BUILTIN_REWARD_FUNCTIONS: dict[str, RewardFunction] = {
    "format": format_reward,
    "exact_alnum": exact_answer_alphanumeric_reward,
    "math_expr": math_expression_reward,
    "bbox_giou": bbox_giou_reward,
    "geo3k": geo3k_reward,
}


def _load_reward_module(config: CustomRewardFunctionConfig) -> ModuleType:
    """Load the module containing a configured custom reward function."""
    if config.module is not None:
        try:
            return importlib.import_module(config.module)
        except Exception as error:
            raise ValueError(
                f"Failed to import module '{config.module}' for custom reward "
                f"function '{config.name}'"
            ) from error

    assert config.file is not None  # Enforced by CustomRewardFunctionConfig.
    file_path = Path(config.file)
    if not file_path.is_file():
        raise ValueError(
            f"Python file '{file_path}' for custom reward function "
            f"'{config.name}' does not exist"
        )

    module_name = f"nemo_rl_custom_reward_{abs(hash(file_path.resolve()))}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ValueError(
            f"Could not create an import spec for custom reward file '{file_path}'"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        sys.modules.pop(module_name, None)
        raise ValueError(
            f"Failed to load Python file '{file_path}' for custom reward "
            f"function '{config.name}'"
        ) from error
    return module


def _load_custom_reward_function(
    raw_config: CustomRewardFunctionConfig | dict[str, Any],
) -> tuple[str, RewardFunction]:
    """Load and validate one custom reward function from user configuration."""
    config = CustomRewardFunctionConfig.model_validate(raw_config)
    module = _load_reward_module(config)

    try:
        reward_function = getattr(module, config.function)
    except AttributeError as error:
        source = config.module if config.module is not None else config.file
        raise ValueError(
            f"Custom reward source '{source}' has no attribute "
            f"'{config.function}' for reward '{config.name}'"
        ) from error

    if not callable(reward_function):
        raise ValueError(
            f"Custom reward '{config.name}' resolves to non-callable attribute "
            f"'{config.function}'"
        )
    return config.name, reward_function


def _get_reward_function_registry(cfg: VLMEnvConfig) -> dict[str, RewardFunction]:
    """Return built-in and configured custom VLM reward functions by name."""
    registry = dict(_BUILTIN_REWARD_FUNCTIONS)
    custom_reward_configs = cfg.get("custom_reward_functions")
    if custom_reward_configs is None:
        return registry

    for raw_config in custom_reward_configs:
        config = CustomRewardFunctionConfig.model_validate(raw_config)
        if config.name in registry:
            raise ValueError(
                f"Reward function name '{config.name}' is already registered; custom "
                "reward names must not replace built-in or previously configured functions"
            )
        name, reward_function = _load_custom_reward_function(config)
        registry[name] = reward_function
    return registry


def _resolve_configured_reward_functions(
    cfg: VLMEnvConfig,
) -> list[tuple[RewardFunction, float]]:
    """Resolve weighted reward functions selected by a VLM environment config."""
    registry = _get_reward_function_registry(cfg)
    reward_functions = []
    for reward_func_cfg in cfg["reward_functions"]:
        reward_func_name: str = reward_func_cfg["name"]
        reward_func_weight: float = reward_func_cfg["weight"]
        reward_func_kwargs: Optional[dict] = reward_func_cfg.get("kwargs", None)

        try:
            reward_func = registry[reward_func_name]
        except KeyError as error:
            available_names = ", ".join(sorted(registry))
            raise ValueError(
                f"Invalid reward function: {reward_func_name}. "
                f"Available reward functions: {available_names}"
            ) from error

        if reward_func_kwargs is not None:
            reward_func = partial(reward_func, **reward_func_kwargs)
        reward_functions.append((reward_func, reward_func_weight))

    if len(reward_functions) == 0:
        raise ValueError("No reward functions provided")
    return reward_functions


@contextlib.contextmanager
def _mute_output():
    devnull_out, devnull_err = io.StringIO(), io.StringIO()
    with (
        contextlib.redirect_stdout(devnull_out),
        contextlib.redirect_stderr(devnull_err),
    ):
        yield


@ray.remote
class VLMVerifyWorker:
    def __init__(self, cfg: VLMEnvConfig) -> None:
        logging.getLogger("vlm_worker").setLevel(logging.CRITICAL)
        reward_functions = _resolve_configured_reward_functions(cfg)
        self.verify_func = combine_reward_functions(reward_functions)

    def verify(
        self, pred_responses: list[str], ground_truths: list[str]
    ) -> list[float]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: list[str]. The predicted responses from the LLM.
            ground_truths: list[str]. The ground truth responses.

        Returns:
            list[float]. The rewards for each predicted response.
        """
        results = []
        for response, ground_truth in zip(pred_responses, ground_truths):
            try:
                with _mute_output():
                    try:
                        ret_score, _ = self.verify_func(ground_truth, response)
                    except Exception as e:
                        ret_score = 0.0
                        print(f"Error in verify_func: {e}")
                results.append(float(ret_score))
            except Exception as e:
                print(f"Error in verify: {e}")
                results.append(0.0)
        return results


class VLMEnvironmentMetadata(TypedDict):
    ground_truth: str


@ray.remote(max_restarts=-1, max_task_retries=-1)
class VLMEnvironment(EnvironmentInterface):
    def __init__(self, cfg: VLMEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.workers = [
            VLMVerifyWorker.options(  # type: ignore # (decorated with @ray.remote)
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote(cfg)
            for _ in range(self.num_workers)
        ]

    def shutdown(self) -> None:
        # shutdown all workers
        for worker in self.workers:
            ray.kill(worker)

    def step(  # type: ignore[override]
        self,
        message_log_batch: list[list[dict[str, str]]],
        metadata: list[VLMEnvironmentMetadata],
        return_extracted_answer: bool = False,
    ) -> EnvironmentReturn:
        """Runs a step in the vlm environment.

        Args:
            message_log: list[list[dict[str, str]]]. A batch of OpenAI-API-like message logs that represent interactions with the VLM.
            metadata: list[VLMEnvironmentMetadata]. The grader will use the 'ground_truth' key to evaluate correctness.

        Returns:
            EnvironmentReturn: A tuple containing:
                - list[dict[str, str]]: Observations/responses batch
                - list[dict]: Updated metadata
                - list[str]: Next stop strings for the next turn
                - Tensor: Rewards tensor
                - Tensor: Done flags tensor
        """
        # Extract the assistant's responses from the message history
        # Each message list should have at least one assistant response
        assistant_response_batch = []
        for conversation in message_log_batch:
            assistant_responses = [
                interaction["content"]
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        ground_truths = [g["ground_truth"] for g in metadata]

        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_ground_truths = chunk_list_to_workers(ground_truths, self.num_workers)

        # # Process each chunk in parallel
        futures = [
            self.workers[i].verify.remote(chunk, ground_truth_chunk)
            for i, (chunk, ground_truth_chunk) in enumerate(
                zip(chunked_assistant_response_batch, chunked_ground_truths)
            )
        ]

        results = ray.get(futures)

        # flatten the results
        results = [item for sublist in results for item in sublist]
        observations = [
            {
                "role": "environment",
                "content": "Environment: correct"
                if result
                else "Environment: incorrect",
            }
            for result in results
        ]

        # create a tensor of rewards and done flags
        rewards = torch.tensor(results).cpu()
        done = torch.ones_like(rewards).cpu()

        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=done,
            answers=None,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
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
