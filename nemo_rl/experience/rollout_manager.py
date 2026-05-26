# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
import json
import warnings
from typing import Any, Optional

import torch
from transformers import PreTrainedTokenizerBase
from wandb import Table

from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.interfaces import Completion, PromptGroupRecord
from nemo_rl.experience.rollouts import (
    _calculate_single_metric,
    _tensorize_by_key,
    run_sample_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import GenerationConfig, GenerationInterface
from nemo_rl.utils.timer import Timer

TokenizerType = PreTrainedTokenizerBase


class AsyncRolloutManager:
    """Manages per-prompt multi-turn rollouts, producing a PromptGroupRecord per call.

    Each run_rollout takes one prompt and returns num_generations_per_prompt completions
    generated concurrently via asyncio.gather.
    """

    def __init__(
        self,
        policy_generation: GenerationInterface,
        tokenizer: TokenizerType,
        task_to_env: dict[str, EnvironmentInterface],
        max_seq_len: int,
        num_generations_per_prompt: int,
        max_rollout_turns: int = 999999,
    ) -> None:
        self._policy_generation = policy_generation
        self._tokenizer = tokenizer
        self._task_to_env = task_to_env
        self._max_seq_len = max_seq_len
        self._num_generations_per_prompt = num_generations_per_prompt
        self._max_rollout_turns = max_rollout_turns

    async def run_rollout(self, input_sample: DatumSpec) -> PromptGroupRecord:
        """Run num_generations_per_prompt rollouts for one prompt.

        Args:
            input_sample: A single prompt (one DatumSpec entry).

        Returns:
            PromptGroupRecord with num_generations_per_prompt completions.
        """
        timer = Timer()
        timer_prefix = "timing/rollout"
        timer.start(f"{timer_prefix}/total")

        with timer.time(f"{timer_prefix}/run_rollouts"):
            completions = list(
                await asyncio.gather(
                    *[
                        self._generate_single_completion(input_sample, gen_idx)
                        for gen_idx in range(self._num_generations_per_prompt)
                    ]
                )
            )

        with timer.time(f"{timer_prefix}/compute_metrics"):
            rollout_metrics = self._compute_rollout_metrics(completions)

        timer.stop(f"{timer_prefix}/total")
        rollout_metrics.update(timer.get_timing_metrics("sum"))

        return PromptGroupRecord(
            prompt_idx=input_sample["idx"],
            prompt=copy.deepcopy(input_sample["message_log"]),
            extra_env_info=input_sample["extra_env_info"],
            metadata={"task_name": input_sample["task_name"]},
            completions=completions,
            rollout_metrics=rollout_metrics,
        )

    async def _generate_single_completion(
        self, input_sample: DatumSpec, gen_idx: int
    ) -> Completion:
        """Run one rollout for a single generation index and return a Completion."""
        sample_state = {
            "message_log": copy.deepcopy(input_sample["message_log"]),
            "extra_env_info": copy.deepcopy(input_sample["extra_env_info"]),
            "task_name": input_sample["task_name"],
            "stop_strings": input_sample.get("stop_strings", None),
            "idx": gen_idx,
        }
        final_state, sample_metrics = await run_sample_multi_turn_rollout(
            sample_idx=gen_idx,
            initial_sample_state=sample_state,
            policy_generation=self._policy_generation,
            tokenizer=self._tokenizer,
            task_to_env=self._task_to_env,
            max_seq_len=self._max_seq_len,
            max_rollout_turns=self._max_rollout_turns,
        )

        env_extras: dict[str, Any] = dict(final_state["extra_env_info"])
        for k, v in final_state.items():
            if isinstance(k, str) and k.startswith("reward") and k[6:].isdigit():
                env_extras[k] = (
                    float(v.item()) if isinstance(v, torch.Tensor) else float(v)
                )

        return Completion(
            message_log=final_state["message_log"],
            env_extras=env_extras,
            truncated=sample_metrics["truncated"],
            reward=float(final_state["total_reward"].item()),
        )

    def _compute_rollout_metrics(
        self, completions: list[Completion]
    ) -> dict[str, Any]:
        """Aggregate per-sample metrics across all completions."""
        n = len(completions)
        rollout_metrics: dict[str, Any] = {
            **_calculate_single_metric(
                [
                    sum(1 for m in c.message_log if m["role"] == "user")
                    for c in completions
                ],
                n,
                "turns_per_sample",
            ),
            **_calculate_single_metric(
                [sum(len(m["token_ids"]) for m in c.message_log) for c in completions],
                n,
                "total_tokens_per_sample",
            ),
            **_calculate_single_metric(
                [
                    sum(
                        len(m["token_ids"])
                        for m in c.message_log
                        if m["role"] == "assistant"
                    )
                    for c in completions
                ],
                n,
                "gen_tokens_per_sample",
            ),
            **_calculate_single_metric(
                [c.reward for c in completions],
                n,
                "total_reward",
            ),
            "natural_termination_rate": sum(not c.truncated for c in completions) / n,
            "truncation_rate": sum(c.truncated for c in completions) / n,
        }

        # Necessary for downstream nemo rl logging/printing.
        rollout_metrics["mean_gen_tokens_per_sample"] = rollout_metrics[
            "gen_tokens_per_sample/mean"
        ]
        return rollout_metrics


class AsyncNemoGymRolloutManager:
    """Manages per-prompt NeMo-Gym rollouts, producing a PromptGroupRecord per call.

    Each run_rollout takes one prompt and returns num_generations_per_prompt completions
    batched through a single NeMo-Gym run_rollouts call.
    """

    def __init__(
        self,
        tokenizer: TokenizerType,
        task_to_env: dict[str, EnvironmentInterface],
        generation_config: GenerationConfig,
        num_generations_per_prompt: int,
        max_seq_len: Optional[int] = None,
        max_rollout_turns: Optional[int] = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._task_to_env = task_to_env
        self._generation_config = generation_config
        self._num_generations_per_prompt = num_generations_per_prompt
        self._max_seq_len = max_seq_len
        self._max_rollout_turns = max_rollout_turns
        self._engine_max_model_len = generation_config["vllm_cfg"]["max_model_len"]  # type: ignore

        self._validate_init_params()

    async def run_rollout(self, input_sample: DatumSpec) -> PromptGroupRecord:
        """Run num_generations_per_prompt rollouts for one prompt.

        Args:
            input_sample: A single prompt (one DatumSpec entry).

        Returns:
            PromptGroupRecord with num_generations_per_prompt completions.
        """
        timer = Timer()
        timer_prefix = "timing/rollout"
        timer.start(f"{timer_prefix}/total")

        rollout_inputs = self._build_inputs(input_sample)
        completions, rollout_metrics = await self._run_rollouts(
            rollout_inputs, timer, timer_prefix
        )

        timer.stop(f"{timer_prefix}/total")
        rollout_metrics.update(timer.get_timing_metrics("sum"))

        return PromptGroupRecord(
            prompt_idx=input_sample["idx"],
            prompt=input_sample["message_log"],
            extra_env_info=input_sample["extra_env_info"],
            metadata={"task_name": "nemo_gym"},
            completions=completions,
            rollout_metrics=rollout_metrics,
        )

    def _validate_init_params(self) -> None:
        """Validate initialization parameters."""
        # Validate generation config.
        for key in ["stop_strings", "stop_token_ids", "top_k"]:
            assert not self._generation_config[key], (  # type: ignore
                f"{key} is not supported in the generation config in NeMo-Gym path!"
            )

        # Validate max_seq_len.
        if (
            self._max_seq_len is not None
            and self._max_seq_len > self._engine_max_model_len
        ):
            warnings.warn(
                f"policy max_total_sequence_length ({self._max_seq_len}) is greater than the "
                f"generation engine's max_model_len ({self._engine_max_model_len}). The engine "
                "will truncate sequences to its own limit, so the policy cap will not be "
                "honored. Lower max_total_sequence_length or raise the engine's max_model_len."
            )

        # Validate max_rollout_turns.
        assert self._max_rollout_turns is None, (
            "`max_rollout_turns` is not supported in NeMo-Gym path!"
        )

        # Validate num_generations_per_prompt.
        assert self._num_generations_per_prompt >= 1, (
            "`num_generations_per_prompt` must be >= 1!"
        )

    def _build_inputs(self, input_sample: DatumSpec) -> list[dict]:
        """Build N row dicts from input_sample, applying generation config params."""
        # Build a template row from the input_sample's extra_env_info, applying generation params.
        template_row: dict = copy.deepcopy(input_sample["extra_env_info"])  # type: ignore

        # We do not translate max_seq_len into row-level max_tokens here because that would
        # change semantics from "total sequence length" to "max new tokens".
        responses_create_params = template_row["responses_create_params"]
        responses_create_params["temperature"] = self._generation_config["temperature"]
        responses_create_params["top_p"] = self._generation_config["top_p"]

        # Configure max_output_tokens to respect the max_new_tokens setting.
        # Will clamp max_output_tokens in vllm_worker_async.py so that input + output <= max_seq_len
        existing = responses_create_params.get("max_output_tokens")
        responses_create_params["max_output_tokens"] = (
            min(existing, self._generation_config["max_new_tokens"])
            if existing is not None
            else self._generation_config["max_new_tokens"]
        )

        # Build N rows with distinct rowidxs so run_rollouts can sort them correctly.
        rows = []
        for i in range(self._num_generations_per_prompt):
            row = copy.deepcopy(template_row)
            row["_rowidx"] = i
            rows.append(row)
        return rows

    async def _run_rollouts(
        self, inputs: list[dict], timer: Timer, timer_prefix: str
    ) -> tuple[list[Completion], dict[str, Any]]:
        """Dispatch rows to NeMo-Gym and return completions + metrics."""
        nemo_gym_env = self._task_to_env["nemo_gym"]

        # Run generation.
        with timer.time(f"{timer_prefix}/run_rollouts"):
            results, env_timing_metrics = await nemo_gym_env.run_rollouts.remote(
                inputs, self._tokenizer, timer_prefix
            )
            # Convert results to completions.
            completions = [self._result_to_completion(r) for r in results]

        # Compute rollout metrics.
        with timer.time(f"{timer_prefix}/compute_metrics"):
            rollout_metrics = self._compute_rollout_metrics(
                completions, inputs[0]["agent_ref"]["name"]
            )

        rollout_metrics.update(env_timing_metrics)

        return completions, rollout_metrics

    def _result_to_completion(self, result: dict) -> Completion:
        """Convert one run_rollouts result dict into a Completion."""
        # Tensorize token fields.
        _tensorize_by_key(result["input_message_log"], "token_ids")
        _tensorize_by_key(result["message_log"], "token_ids")
        _tensorize_by_key(
            [m for m in result["message_log"] if m["role"] == "assistant"],
            "generation_logprobs",
        )

        # Calculate truncation.
        truncated = (
            sum(len(m["token_ids"]) for m in result["message_log"])
            == self._engine_max_model_len
        )

        return Completion(
            message_log=result["message_log"],
            env_extras=result["full_result"],
            truncated=truncated,
            reward=float(result["full_result"]["reward"]),
        )

    def _compute_rollout_metrics(
        self,
        completions: list[Completion],
        agent_name: str,
    ) -> dict[str, Any]:
        """Aggregate per-sample and per-agent metrics."""
        n = len(completions)

        # Aggregate metrics across all samples
        rollout_metrics: dict[str, Any] = {
            **_calculate_single_metric(
                [
                    sum(1 for m in c.message_log if m["role"] == "user")
                    for c in completions
                ],
                n,
                "turns_per_sample",
            ),
            **_calculate_single_metric(
                [sum(len(m["token_ids"]) for m in c.message_log) for c in completions],
                n,
                "total_tokens_per_sample",
            ),
            **_calculate_single_metric(
                [
                    sum(
                        len(m["token_ids"])
                        for m in c.message_log
                        if m["role"] == "assistant"
                    )
                    for c in completions
                ],
                n,
                "gen_tokens_per_sample",
            ),
            **_calculate_single_metric(
                [c.reward for c in completions],
                n,
                "total_reward",
            ),
            "natural_termination_rate": sum(not c.truncated for c in completions) / n,
            "truncation_rate": sum(c.truncated for c in completions) / n,
        }

        # Agent-level metrics.
        agent_extras = [c.env_extras for c in completions]
        for key in agent_extras[0].keys():
            values = [
                float(r[key])
                for r in agent_extras
                if isinstance(r.get(key), (bool, int, float))
            ]
            if values:
                rollout_metrics.update(
                    _calculate_single_metric(values, n, f"{agent_name}/{key}")
                )
        rollout_metrics[f"{agent_name}/full_result"] = Table(
            data=[[json.dumps(r, separators=(",", ":"))] for r in agent_extras],
            columns=["Full result"],
        )

        # Necessary for downstream nemo rl logging/printing.
        rollout_metrics["mean_gen_tokens_per_sample"] = rollout_metrics[
            "gen_tokens_per_sample/mean"
        ]
        return rollout_metrics
