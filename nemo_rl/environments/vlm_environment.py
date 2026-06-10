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
import io
import logging
from functools import partial
from typing import Any, Callable, List, NotRequired, Optional, TypedDict

import ray
import torch

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
    exact_answer_alphanumeric_reward,
    exact_answer_alphanumeric_with_fallback_reward,
    format_reward,
    math_expression_reward,
)
from nemo_rl.environments.utils import chunk_list_to_workers


class VLMEnvConfig(TypedDict):
    num_workers: int
    stop_strings: NotRequired[Optional[list[str]]]  # Default stop strings for this env
    reward_functions: List[dict[str, Any]]  # list of reward functions and their weights


@contextlib.contextmanager
def _mute_output():
    devnull_out, devnull_err = io.StringIO(), io.StringIO()
    with (
        contextlib.redirect_stdout(devnull_out),
        contextlib.redirect_stderr(devnull_err),
    ):
        yield


def _build_named_reward_functions(
    cfg: VLMEnvConfig,
) -> list[tuple[str, Callable[[str, str], tuple[float, Optional[bool]]], float]]:
    """Resolve ``cfg['reward_functions']`` into a list of (name, fn, weight) tuples."""
    resolved: list[
        tuple[str, Callable[[str, str], tuple[float, Optional[bool]]], float]
    ] = []
    for reward_func_cfg in cfg["reward_functions"]:
        reward_func_name: str = reward_func_cfg["name"]
        reward_func_weight: float = reward_func_cfg["weight"]
        reward_func_kwargs: Optional[dict] = reward_func_cfg.get("kwargs", None)
        reward_func: Callable[[str, str], tuple[float, Optional[bool]]]
        if reward_func_name == "format":
            reward_func = format_reward
        elif reward_func_name == "exact_alnum":
            reward_func = exact_answer_alphanumeric_reward
        elif reward_func_name == "exact_alnum_with_fallback":
            reward_func = exact_answer_alphanumeric_with_fallback_reward
        elif reward_func_name == "math_expr":
            reward_func = math_expression_reward
        elif reward_func_name == "bbox_giou":
            reward_func = bbox_giou_reward
        else:
            raise ValueError(f"Invalid reward function: {reward_func_name}")

        if reward_func_kwargs is not None:
            reward_func = partial(reward_func, **reward_func_kwargs)

        resolved.append((reward_func_name, reward_func, reward_func_weight))
    if len(resolved) == 0:
        raise ValueError("No reward functions provided")
    return resolved


@ray.remote
class VLMVerifyWorker:
    def __init__(self, cfg: VLMEnvConfig) -> None:
        logging.getLogger("vlm_worker").setLevel(logging.CRITICAL)
        named = _build_named_reward_functions(cfg)
        self._reward_names: list[str] = [name for name, _, _ in named]
        self._reward_fns: list[Callable[[str, str], tuple[float, Optional[bool]]]] = [
            fn for _, fn, _ in named
        ]
        weights = [w for _, _, w in named]
        # Same renormalization as combine_reward_functions: the combined
        # reward equals sum(weight_i * raw_i) with weights summing to 1.
        weight_arr = [w / sum(weights) for w in weights]
        self._reward_weights: list[float] = weight_arr

    def reward_names(self) -> list[str]:
        """Return the ordered list of configured reward-function names."""
        return list(self._reward_names)

    def verify_with_components(
        self, pred_responses: list[str], ground_truths: list[str]
    ) -> tuple[list[float], list[list[float]]]:
        """Score each (response, ground_truth) and return both totals and components.

        Returns:
            (combined, components) where ``combined[i]`` is the weighted total
            reward for sample ``i`` (matching the historical ``verify`` return)
            and ``components[i]`` is a list of weighted per-function scores in
            the same order as ``reward_names()``. Summing ``components[i]`` ==
            ``combined[i]`` (modulo float error) by construction.
        """
        combined: list[float] = []
        components: list[list[float]] = []
        for response, ground_truth in zip(pred_responses, ground_truths):
            sample_components = [0.0] * len(self._reward_fns)
            try:
                with _mute_output():
                    for idx, (fn, w) in enumerate(
                        zip(self._reward_fns, self._reward_weights)
                    ):
                        try:
                            raw, _ = fn(ground_truth, response)
                        except Exception as e:
                            raw = 0.0
                            print(f"Error in reward fn {self._reward_names[idx]}: {e}")
                        sample_components[idx] = float(raw) * float(w)
            except Exception as e:
                print(f"Error in verify_with_components: {e}")
            combined.append(float(sum(sample_components)))
            components.append(sample_components)
        return combined, components

    def verify(
        self, pred_responses: list[str], ground_truths: list[str]
    ) -> list[float]:
        """Backward-compat scalar reward (sum of weighted components)."""
        combined, _ = self.verify_with_components(pred_responses, ground_truths)
        return combined


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
        # Names of the configured reward functions, in the order
        # `step()` returns them as columns of `EnvironmentReturn.rewards`.
        # Used by the validation loop in `nemo_rl/algorithms/grpo.py::validate`
        # to label per-component reward metrics.
        self._reward_component_names: list[str] = [
            entry["name"] for entry in cfg["reward_functions"]
        ]

    def reward_component_names(self) -> list[str]:
        """Public Ray-callable accessor for the per-component reward names.

        Returns the same ordering used by the K-column rewards tensor that
        ``step()`` emits, so callers can map ``rewards[:, i]`` back to the
        configured reward function name (e.g. ``"format"``, ``"exact_alnum"``).
        """
        return list(self._reward_component_names)

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

        # Use verify_with_components so per-reward-function scores survive
        # back to the rollout layer; the rollout's existing multi-reward
        # plumbing turns the (N, K) tensor into per-component ``reward<i+1>``
        # batch columns, and validation reads those for per-component logging.
        futures = [
            self.workers[i].verify_with_components.remote(chunk, ground_truth_chunk)
            for i, (chunk, ground_truth_chunk) in enumerate(
                zip(chunked_assistant_response_batch, chunked_ground_truths)
            )
        ]

        chunk_results = ray.get(futures)

        combined: list[float] = []
        components: list[list[float]] = []
        for chunk_combined, chunk_components in chunk_results:
            combined.extend(chunk_combined)
            components.extend(chunk_components)

        observations = [
            {
                "role": "environment",
                "content": "Environment: correct"
                if score
                else "Environment: incorrect",
            }
            for score in combined
        ]

        # Build a (N, K) rewards tensor of weighted components. Summing along
        # dim=1 reproduces the historical scalar `total_reward` GRPO uses for
        # advantage computation.
        if len(components) > 0 and len(components[0]) > 0:
            rewards = torch.tensor(components, dtype=torch.float32).cpu()
        else:
            # K=0 (no reward fns configured) is rejected at worker init, but
            # keep the fallback for type stability if `combined` ends up empty.
            rewards = torch.tensor(combined, dtype=torch.float32).cpu()
        done = torch.ones(rewards.shape[0], dtype=rewards.dtype).cpu()

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
