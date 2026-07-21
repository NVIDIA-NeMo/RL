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
import inspect
import json
import math
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers import PreTrainedTokenizerBase
from wandb import Table

from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.interfaces import Completion, PromptGroupRecord
from nemo_rl.experience.metric_utils import calculate_single_metric, pct
from nemo_rl.experience.rollout_checkpoint import (
    ROLLOUT_CHECKPOINT_SCHEMA_VERSION,
    CompletedSiblingRecord,
    IncompatibleCheckpointError,
    PersistAck,
    RolloutWorkItem,
    StorageUnavailableError,
)
from nemo_rl.experience.rollouts import _tensorize_by_key, calculate_rewards
from nemo_rl.models.generation.interfaces import (
    GenerationConfig,
    GenerationDatumSpec,
    GenerationInterface,
)
from nemo_rl.utils.timer import Timer

TokenizerType = PreTrainedTokenizerBase
SiblingCompleteCallback = Callable[[int, Completion], Coroutine[Any, Any, None]]


@dataclass(frozen=True)
class RolloutCheckpointWritePolicy:
    """Bounded retry policy for completed-sibling persistence."""

    max_pending_writes: int
    write_timeout_s: float
    max_retries: int
    retry_backoff_s: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_pending_writes, int)
            or isinstance(self.max_pending_writes, bool)
            or self.max_pending_writes < 1
        ):
            raise ValueError("max_pending_writes must be an integer >= 1")
        if (
            not isinstance(self.write_timeout_s, (int, float))
            or isinstance(self.write_timeout_s, bool)
            or not math.isfinite(self.write_timeout_s)
            or self.write_timeout_s <= 0
        ):
            raise ValueError("write_timeout_s must be a finite number > 0")
        if (
            not isinstance(self.max_retries, int)
            or isinstance(self.max_retries, bool)
            or self.max_retries < 0
        ):
            raise ValueError("max_retries must be an integer >= 0")
        if (
            not isinstance(self.retry_backoff_s, (int, float))
            or isinstance(self.retry_backoff_s, bool)
            or not math.isfinite(self.retry_backoff_s)
            or self.retry_backoff_s < 0
        ):
            raise ValueError("retry_backoff_s must be a finite number >= 0")


@dataclass(frozen=True)
class RolloutCheckpointIOPolicy(RolloutCheckpointWritePolicy):
    """Bounded retry policy for checkpoint-writer reads and writes."""

    load_timeout_s: float
    max_load_retries: int
    load_retry_backoff_s: float

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not isinstance(self.load_timeout_s, (int, float))
            or isinstance(self.load_timeout_s, bool)
            or not math.isfinite(self.load_timeout_s)
            or self.load_timeout_s <= 0
        ):
            raise ValueError("load_timeout_s must be a finite number > 0")
        if (
            not isinstance(self.max_load_retries, int)
            or isinstance(self.max_load_retries, bool)
            or self.max_load_retries < 0
        ):
            raise ValueError("max_load_retries must be an integer >= 0")
        if (
            not isinstance(self.load_retry_backoff_s, (int, float))
            or isinstance(self.load_retry_backoff_s, bool)
            or not math.isfinite(self.load_retry_backoff_s)
            or self.load_retry_backoff_s < 0
        ):
            raise ValueError("load_retry_backoff_s must be a finite number >= 0")


class AsyncRolloutImpl:
    """Manages per-prompt multi-turn rollouts, producing a PromptGroupRecord per call.

    Each run_rollout takes one prompt and returns the requested number of completions
    generated concurrently via asyncio.gather.
    """

    def __init__(
        self,
        tokenizer: TokenizerType,
        max_seq_len: int,
        policy_generation: GenerationInterface,
        max_rollout_turns: int = 999999,
        **kwargs: Any,
    ) -> None:
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len
        self._max_rollout_turns = max_rollout_turns
        self._policy_generation = policy_generation

    async def run_rollout(
        self,
        input_sample: DatumSpec,
        *,
        env_handles: dict[str, EnvironmentInterface],
        num_generations_per_prompt: int,
    ) -> PromptGroupRecord:
        """Run the requested number of rollouts for one prompt.

        Args:
            input_sample: A single prompt (one DatumSpec entry).
            env_handles: Environments used to score and advance the rollouts.
            num_generations_per_prompt: Number of completions to generate.

        Returns:
            PromptGroupRecord with the requested number of completions.
        """
        timer = Timer()
        timer_prefix = "timing/rollout"
        timer.start(f"{timer_prefix}/total")

        with timer.time(f"{timer_prefix}/run_rollouts"):
            results = list(
                await asyncio.gather(
                    *[
                        self._run_single_rollout(
                            input_sample,
                            traj_idx,
                            env_handles=env_handles,
                        )
                        for traj_idx in range(num_generations_per_prompt)
                    ]
                )
            )
            completions = [c for c, _ in results]
            all_sample_metrics = [m for _, m in results]

        with timer.time(f"{timer_prefix}/aggregate_metrics"):
            rollout_metrics = self._aggregate_rollout_metrics(
                completions, all_sample_metrics
            )

        timer.stop(f"{timer_prefix}/total")
        rollout_metrics.update(timer.get_timing_metrics("sum"))

        return PromptGroupRecord(
            prompt_idx=input_sample["idx"],
            prompt=input_sample["message_log"],
            extra_env_info=input_sample["extra_env_info"],
            metadata={"task_name": input_sample["task_name"]},
            completions=completions,
            rollout_metrics=rollout_metrics,
        )

    async def _run_single_rollout(
        self,
        input_sample: DatumSpec,
        traj_idx: int,
        *,
        env_handles: dict[str, EnvironmentInterface],
    ) -> tuple[Completion, dict]:
        """Run one multi-turn rollout for a single generation index."""
        current_message_log = copy.deepcopy(input_sample["message_log"])
        current_extra_env_info = copy.deepcopy(input_sample["extra_env_info"])
        current_stop_strings = input_sample.get("stop_strings", None)
        task_name = input_sample["task_name"]

        total_reward = 0.0
        turn_count = 0
        # token statistics
        total_token_count = 0
        assistant_token_count = 0
        env_token_count = 0
        # truncated statistics
        terminated = False
        truncated = False
        max_turns_reached = False

        # Track per-turn metrics
        turn_gen_tokens = []
        turn_input_tokens = []
        turn_total_tokens = []
        # Track per-turn per-worker token accounting if available
        per_worker_token_counts = {}  # worker_idx -> token_count

        for _ in range(self._max_rollout_turns):
            if terminated or truncated:
                break

            turn_count += 1

            # Generate response for this sample using async generation
            try:
                (
                    assistant_message,
                    input_lengths,
                    gen_metrics,
                ) = await self._generate_response(
                    current_message_log,
                    current_stop_strings,
                )
                current_message_log.append(assistant_message)

                # Check if response was truncated (hit max_tokens without stop token)
                response_truncated = gen_metrics.pop("_response_truncated", None)
                if response_truncated is not None and response_truncated[0]:
                    truncated = True

                # Update token counts
                gen_token_count = len(assistant_message["token_ids"])
                assistant_token_count += gen_token_count
                total_token_count += gen_token_count
                turn_gen_tokens.append(gen_token_count)
                turn_input_tokens.append(int(input_lengths))
                turn_total_tokens.append(int(input_lengths) + gen_token_count)
                # Per-worker load accounting
                if "gen_leader_worker_idx" in gen_metrics:
                    worker_idx = int(gen_metrics["gen_leader_worker_idx"])
                    per_worker_token_counts[worker_idx] = (
                        per_worker_token_counts.get(worker_idx, 0) + gen_token_count
                    )

            except Exception as e:
                print(
                    f"Error generating response for prompt_idx {input_sample['idx']}, traj_idx {traj_idx}: {e}"
                )
                break

            # Create single-sample batch for environment interaction
            sample_batch = BatchedDataDict[DatumSpec](
                {
                    "message_log": [current_message_log],
                    "extra_env_info": [current_extra_env_info],
                    "task_name": [task_name],
                }
            )
            # Get environment feedback.
            # calculate_rewards uses blocking ray.get internally. Running it
            # directly on the asyncio event loop (which this coroutine runs on)
            # blocks every other in-flight rollout coroutine for the entire env
            # step. In this case, need to wrap with asyncio.to_thread to make
            # this function yieldable.
            env_output = await asyncio.to_thread(
                calculate_rewards, sample_batch, env_handles
            )

            # Update reward and termination statistics
            # Multi-reward isn't supported in RolloutManager now, see
            # https://github.com/NVIDIA-NeMo/RL/issues/2625 for more details.
            assert isinstance(env_output.rewards, torch.Tensor)
            total_reward += float(env_output.rewards[0].item())
            terminated = env_output.terminateds[0].item()
            env_obs_content = env_output.observations[0]["content"]
            tokenized_obs = self._tokenizer(
                env_obs_content, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]

            # Check for sequence length overflow
            if (
                input_lengths + gen_token_count + len(tokenized_obs)
                >= self._max_seq_len
            ):
                # Truncate environment observation
                max_env_tokens = self._max_seq_len - input_lengths - gen_token_count
                if max_env_tokens > 0:
                    tokenized_obs = tokenized_obs[:max_env_tokens]
                else:
                    tokenized_obs = torch.empty(0, dtype=tokenized_obs.dtype)
                truncated = True

            current_message_log.append(
                {
                    "role": env_output.observations[0]["role"],
                    "content": env_obs_content,
                    "token_ids": tokenized_obs,
                }
            )

            # Update token counts
            env_token_count += len(tokenized_obs)
            total_token_count += len(tokenized_obs)

            # Update sample state for next turn
            if not terminated and not truncated:
                if env_output.next_stop_strings[0] is not None:
                    current_stop_strings = env_output.next_stop_strings[0]
                if env_output.metadata[0] is not None:
                    current_extra_env_info = env_output.metadata[0]

        else:
            # Reached max turns without termination or truncation.
            max_turns_reached = True

        completion = Completion(
            message_log=current_message_log,
            env_extras=current_extra_env_info,
            truncated=truncated,
            reward=total_reward,
        )
        sample_metrics = {
            "turn_count": turn_count,
            "total_tokens": total_token_count,
            "assistant_tokens": assistant_token_count,
            "env_tokens": env_token_count,
            "terminated": terminated,
            "max_turns_reached": max_turns_reached,
            "turn_gen_tokens": turn_gen_tokens,
            "turn_input_tokens": turn_input_tokens,
            "turn_total_tokens": turn_total_tokens,
            "per_worker_token_counts": per_worker_token_counts,
        }
        return completion, sample_metrics

    async def _generate_response(
        self,
        message_log: list[dict],
        stop_strings: list[str] | None,
    ) -> tuple[dict, torch.Tensor, dict[str, Any]]:
        """Generate a single-turn response for one sample.

        Returns:
            Tuple of (assistant_message, input_lengths, gen_metrics)
        """
        # Prepare generation input
        input_ids = torch.cat([m["token_ids"] for m in message_log]).unsqueeze(0)
        input_lengths = torch.tensor([input_ids.shape[1]], dtype=torch.int32)
        generation_input_data = BatchedDataDict[GenerationDatumSpec](
            {
                "input_ids": input_ids,
                "input_lengths": input_lengths,
                "stop_strings": [stop_strings],
            }
        )

        # Generate response
        # TODO: update generate_async to return a single item directly
        output = None
        async for _idx, output in self._policy_generation.generate_async(
            generation_input_data
        ):
            pass

        # Build assistant message
        input_len = int(input_lengths[0].item())
        total_len = int(output["unpadded_sequence_lengths"][0].item())
        output_ids = output["output_ids"]
        generated_ids = output_ids[0, input_len:total_len]

        assistant_message: dict = {
            "role": "assistant",
            "content": self._tokenizer.decode(generated_ids, skip_special_tokens=True),
            "token_ids": generated_ids,
        }
        if "logprobs" in output:
            assistant_message["generation_logprobs"] = output["logprobs"][
                0, input_len:total_len
            ]

        # Calculate generation metrics
        gen_metrics: dict[str, Any] = {}
        if "gen_leader_worker_idx" in output:
            v = output["gen_leader_worker_idx"][0]
            try:
                gen_metrics["gen_leader_worker_idx"] = (
                    int(v[0]) if isinstance(v, list) else int(v)
                )
            except Exception as e:
                print(f"Error extracting gen_leader_worker_idx: {e}")
        if "truncated" in output:
            gen_metrics["_response_truncated"] = output["truncated"]

        return assistant_message, input_lengths, gen_metrics

    def _aggregate_rollout_metrics(
        self, completions: list[Completion], all_sample_metrics: list[dict]
    ) -> dict[str, Any]:
        """Aggregate per-sample metrics across all completions."""
        # Prepare lists of values for each metric.
        total_reward = [c.reward for c in completions]
        turn_count = [m["turn_count"] for m in all_sample_metrics]
        # token metrics
        total_tokens = [m["total_tokens"] for m in all_sample_metrics]
        assistant_tokens = [m["assistant_tokens"] for m in all_sample_metrics]
        env_tokens = [m["env_tokens"] for m in all_sample_metrics]
        # truncated metrics
        truncated = [c.truncated for c in completions]
        terminated = [m["terminated"] for m in all_sample_metrics]
        max_turns_reached = [m["max_turns_reached"] for m in all_sample_metrics]

        # max_gen_tokens_per_turn: Diagnostic for long single generations
        max_gen_tokens_per_turn = [
            max(m["turn_gen_tokens"]) if m["turn_gen_tokens"] else 0
            for m in all_sample_metrics
        ]

        # Aggregate metrics across all samples.
        n = len(all_sample_metrics)
        rollout_metrics: dict[str, Any] = {
            **calculate_single_metric(total_reward, n, "total_reward"),
            # turn metrics
            "total_turns": sum(turn_count),
            **calculate_single_metric(turn_count, n, "turns_per_sample"),
            "turns_per_sample/p95": pct(turn_count, 95),
            "turns_per_sample/p99": pct(turn_count, 99),
            # token metrics
            **calculate_single_metric(total_tokens, n, "total_tokens_per_sample"),
            **calculate_single_metric(assistant_tokens, n, "gen_tokens_per_sample"),
            **calculate_single_metric(env_tokens, n, "env_tokens_per_sample"),
            # max_gen_tokens_per_turn: Diagnostic for long single generations
            "max_gen_tokens_per_turn/max": max(max_gen_tokens_per_turn),
            "max_gen_tokens_per_turn/mean": sum(max_gen_tokens_per_turn) / n,
            "max_gen_tokens_per_turn/p95": pct(max_gen_tokens_per_turn, 95),
            # truncated metrics
            "truncation_rate": sum(truncated) / n,
            "natural_termination_rate": sum(terminated) / n,
            "max_turns_reached_rate": sum(max_turns_reached) / n,
        }

        if "per_worker_token_counts" in all_sample_metrics[0]:
            per_worker_token_counts: dict[int, int] = {}
            for m in all_sample_metrics:
                for k, v in m["per_worker_token_counts"].items():
                    per_worker_token_counts[k] = per_worker_token_counts.get(k, 0) + v
            rollout_metrics["per_worker_token_counts"] = per_worker_token_counts

        # Per-turn token histograms (flat across all turns, distinct from the
        # per-sample histograms emitted via calculate_single_metric above).
        rollout_metrics["histogram/gen_tokens_length"] = [
            t for m in all_sample_metrics for t in m["turn_gen_tokens"]
        ]
        rollout_metrics["histogram/input_tokens_length"] = [
            t for m in all_sample_metrics for t in m["turn_input_tokens"]
        ]
        rollout_metrics["histogram/total_tokens_length"] = [
            t for m in all_sample_metrics for t in m["turn_total_tokens"]
        ]

        # Necessary for downstream nemo rl logging/printing.
        rollout_metrics["mean_gen_tokens_per_sample"] = rollout_metrics[
            "gen_tokens_per_sample/mean"
        ]
        return rollout_metrics


class AsyncNemoGymRolloutImpl:
    """Manages per-prompt NeMo-Gym rollouts, producing a PromptGroupRecord per call.

    Each run_rollout takes one prompt and returns the requested number of completions
    batched through a single NeMo-Gym run_rollouts call.
    """

    def __init__(
        self,
        tokenizer: TokenizerType,
        max_seq_len: int,
        generation_config: GenerationConfig,
        max_rollout_turns: int,
        **kwargs: Any,
    ) -> None:
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len
        self._max_rollout_turns = max_rollout_turns
        self._generation_config = generation_config

        self._validate_init_params()

    async def run_rollout(
        self,
        input_sample: DatumSpec,
        *,
        env_handles: dict[str, EnvironmentInterface],
        num_generations_per_prompt: int,
        on_sibling_complete: Optional[SiblingCompleteCallback] = None,
        restored_completions: Optional[dict[int, Completion]] = None,
    ) -> PromptGroupRecord:
        """Run the requested number of rollouts for one prompt.

        Args:
            input_sample: A single prompt (one DatumSpec entry).
            env_handles: Environments used to execute the rollouts.
            num_generations_per_prompt: Number of completions to generate.
            on_sibling_complete: Optional callback invoked as each sibling completes.

        Returns:
            PromptGroupRecord with the requested number of completions.
        """
        timer = Timer()
        timer_prefix = "timing/rollout"
        timer.start(f"{timer_prefix}/total")

        rollout_inputs = self._build_inputs(
            input_sample,
            num_generations_per_prompt=num_generations_per_prompt,
        )
        completions, prompt_message_log, rollout_metrics = await self._run_rollouts(
            rollout_inputs,
            timer,
            timer_prefix,
            env_handles=env_handles,
            on_sibling_complete=on_sibling_complete,
            restored_completions=restored_completions,
        )

        timer.stop(f"{timer_prefix}/total")
        rollout_metrics.update(timer.get_timing_metrics("sum"))

        return PromptGroupRecord(
            prompt_idx=input_sample["idx"],
            prompt=prompt_message_log,
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

        # Validate max_rollout_turns.
        assert self._max_rollout_turns == 1, (
            "`max_rollout_turns` is not supported in NeMo-Gym path! "
            "Please set `max_rollout_turns` to 1."
        )

    def _build_inputs(
        self,
        input_sample: DatumSpec,
        *,
        num_generations_per_prompt: int,
    ) -> list[dict]:
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
        for i in range(num_generations_per_prompt):
            row = copy.deepcopy(template_row)
            row["_rowidx"] = i
            rows.append(row)
        return rows

    async def _run_rollouts(
        self,
        inputs: list[dict],
        timer: Timer,
        timer_prefix: str,
        *,
        env_handles: dict[str, EnvironmentInterface],
        on_sibling_complete: Optional[SiblingCompleteCallback] = None,
        restored_completions: Optional[dict[int, Completion]] = None,
    ) -> tuple[list[Completion], LLMMessageLogType, dict[str, Any]]:
        """Dispatch rows to NeMo-Gym; return completions, prompt, and metrics."""
        nemo_gym_env = env_handles["nemo_gym"]
        restored_completions = dict(restored_completions or {})
        invalid_restored_indices = set(restored_completions) - set(range(len(inputs)))
        if invalid_restored_indices:
            raise ValueError(
                "restored NeMo-Gym generation indices are outside the rollout group: "
                f"{sorted(invalid_restored_indices)}"
            )
        if restored_completions and on_sibling_complete is None:
            raise ValueError(
                "restored NeMo-Gym completions require a sibling completion callback"
            )

        with timer.time(f"{timer_prefix}/run_rollouts"):
            if on_sibling_complete is None:
                results, env_timing_metrics = await nemo_gym_env.run_rollouts.remote(
                    inputs, self._tokenizer, timer_prefix
                )
                completions = [self._result_to_completion(result) for result in results]
                prompt_message_log = results[0]["input_message_log"]
            else:
                missing_inputs = [
                    row for row in inputs if row["_rowidx"] not in restored_completions
                ]
                generated_results: dict[int, dict] = {}
                generated_completions: dict[int, Completion] = {}
                env_timing_metrics: dict[str, Any] = {}
                if missing_inputs:
                    (
                        generated_results,
                        generated_completions,
                        env_timing_metrics,
                    ) = await self._stream_and_persist_rollouts(
                        missing_inputs,
                        timer_prefix,
                        env_handles=env_handles,
                        on_sibling_complete=on_sibling_complete,
                    )

                completions_by_index = {
                    **restored_completions,
                    **generated_completions,
                }
                missing_indices = set(range(len(inputs))) - set(completions_by_index)
                if missing_indices:
                    raise RuntimeError(
                        "NeMo-Gym rollout recovery ended without completions for "
                        f"generation indices {sorted(missing_indices)}"
                    )
                completions = [
                    completions_by_index[generation_index]
                    for generation_index in range(len(inputs))
                ]
                if generated_results:
                    prompt_message_log = next(iter(generated_results.values()))[
                        "input_message_log"
                    ]
                else:
                    first_completion = completions[0]
                    if not first_completion.message_log:
                        raise ValueError(
                            "restored NeMo-Gym completion has an empty message log"
                        )
                    prompt_message_log = copy.deepcopy(first_completion.message_log[:1])

            # All N rollouts share the same input prompt; tensorize one copy.
            _tensorize_by_key(prompt_message_log, "token_ids")

        # Compute rollout metrics.
        with timer.time(f"{timer_prefix}/compute_metrics"):
            rollout_metrics = self._compute_rollout_metrics(
                completions, inputs[0]["agent_ref"]["name"]
            )

        rollout_metrics.update(env_timing_metrics)

        return completions, prompt_message_log, rollout_metrics

    async def _stream_and_persist_rollouts(
        self,
        inputs: list[dict],
        timer_prefix: str,
        *,
        env_handles: dict[str, EnvironmentInterface],
        on_sibling_complete: SiblingCompleteCallback,
    ) -> tuple[dict[int, dict], dict[int, Completion], dict[str, Any]]:
        """Stream Gym rows, submit sibling writes, and restore input order."""
        nemo_gym_env = env_handles["nemo_gym"]
        expected_row_indices = {row["_rowidx"] for row in inputs}
        if len(expected_row_indices) != len(inputs):
            raise ValueError("NeMo-Gym rollout inputs contain duplicate row indices")
        results: dict[int, dict] = {}
        completions_by_index: dict[int, Completion] = {}
        received_row_indices: set[int] = set()
        env_timing_metrics: dict[str, Any] = {}
        persist_tasks: list[asyncio.Task[None]] = []
        stream_completed = False
        persist_results: tuple[None | BaseException, ...] = ()
        try:
            async for result_ref in nemo_gym_env.stream_rollouts.options(
                num_returns="streaming"
            ).remote(inputs, self._tokenizer, timer_prefix):
                rowidx, result, timing_metrics = await result_ref
                if not isinstance(rowidx, int) or rowidx not in expected_row_indices:
                    raise ValueError(
                        f"NeMo-Gym returned unexpected row index {rowidx!r}; "
                        f"expected one of {sorted(expected_row_indices)}"
                    )
                if rowidx in received_row_indices:
                    raise ValueError(f"NeMo-Gym returned duplicate row index {rowidx}")
                received_row_indices.add(rowidx)
                results[rowidx] = result
                completion = self._result_to_completion(result)
                completions_by_index[rowidx] = completion
                persist_tasks.append(
                    asyncio.create_task(on_sibling_complete(rowidx, completion))
                )
                if timing_metrics is not None:
                    env_timing_metrics = timing_metrics
            stream_completed = True
        finally:
            if persist_tasks:
                persist_results = await asyncio.gather(
                    *persist_tasks,
                    return_exceptions=True,
                )

        if stream_completed:
            persist_errors = [
                result
                for result in persist_results
                if isinstance(result, BaseException)
            ]
            if persist_errors:
                raise persist_errors[0]

        missing_row_indices = expected_row_indices - received_row_indices
        if missing_row_indices:
            raise RuntimeError(
                "NeMo-Gym rollout stream ended before rows "
                f"{sorted(missing_row_indices)} arrived"
            )

        return results, completions_by_index, env_timing_metrics

    def _result_to_completion(self, result: dict) -> Completion:
        """Convert one run_rollouts result dict into a Completion."""
        # Tensorize token fields.
        _tensorize_by_key(result["message_log"], "token_ids")
        _tensorize_by_key(
            [m for m in result["message_log"] if m["role"] == "assistant"],
            "generation_logprobs",
        )

        # Calculate truncation.
        truncated = (
            sum(len(m["token_ids"]) for m in result["message_log"]) == self._max_seq_len
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
        # Prepare lists of values for each metric.
        total_reward = [c.reward for c in completions]
        turn_count = [
            sum(1 for m in c.message_log if m["role"] == "user") for c in completions
        ]
        # token metrics
        total_tokens = [
            sum(len(m["token_ids"]) for m in c.message_log) for c in completions
        ]
        assistant_tokens = [
            sum(len(m["token_ids"]) for m in c.message_log if m["role"] == "assistant")
            for c in completions
        ]
        # max_gen_tokens_per_turn: Diagnostic for long single generations
        max_gen_tokens_per_turn = [
            max(
                (
                    len(m["token_ids"])
                    for m in c.message_log
                    if m["role"] == "assistant"
                ),
                default=0,
            )
            for c in completions
        ]
        # truncated metrics
        truncated = [c.truncated for c in completions]

        # Aggregate metrics across all samples.
        n = len(completions)
        rollout_metrics: dict[str, Any] = {
            **calculate_single_metric(total_reward, n, "total_reward"),
            # turn metrics
            **calculate_single_metric(turn_count, n, "turns_per_sample"),
            "turns_per_sample/p95": pct(turn_count, 95),
            "turns_per_sample/p99": pct(turn_count, 99),
            # token metrics
            **calculate_single_metric(total_tokens, n, "total_tokens_per_sample"),
            **calculate_single_metric(assistant_tokens, n, "gen_tokens_per_sample"),
            **calculate_single_metric(
                max_gen_tokens_per_turn, n, "max_gen_tokens_per_turn"
            ),
            "max_gen_tokens_per_turn/p95": pct(max_gen_tokens_per_turn, 95),
            # truncated metrics
            "natural_termination_rate": sum(not t for t in truncated) / n,
            "truncation_rate": sum(truncated) / n,
        }

        # Agent-level metrics.
        agent_extras = [c.env_extras for c in completions]
        for key in agent_extras[0].keys():
            values = [
                float(r[key])  # type: ignore
                for r in agent_extras
                if isinstance(r.get(key), (bool, int, float))
            ]
            if values:
                rollout_metrics.update(
                    calculate_single_metric(values, n, f"{agent_name}/{key}")
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


class RolloutManager:
    """Routes to AsyncRolloutImpl (native async) or AsyncNemoGymRolloutImpl (NeMo-Gym), and pushes results to a TQReplayBuffer."""

    def __init__(
        self,
        tokenizer: TokenizerType,
        *,
        env_handles: dict[str, EnvironmentInterface],
        val_env_handles: dict[str, EnvironmentInterface],
        max_seq_len: int,
        max_rollout_turns: Optional[int] = None,
        policy_generation: Optional[GenerationInterface] = None,
        generation_config: Optional[GenerationConfig] = None,
        use_nemo_gym: bool = False,
        tq_buffer: Optional[TQReplayBuffer] = None,
        checkpoint_io_policy: Optional[RolloutCheckpointIOPolicy] = None,
    ) -> None:
        if not use_nemo_gym:
            rollout_cls = AsyncRolloutImpl
            assert policy_generation is not None, (
                "policy_generation is required for the native async path"
            )
            if max_rollout_turns is None:
                max_rollout_turns = 999999  # use AsyncRolloutImpl's default value
        else:
            rollout_cls = AsyncNemoGymRolloutImpl
            assert generation_config is not None, (
                "generation_config is required for the NeMo-Gym path"
            )

        self._impl: AsyncRolloutImpl | AsyncNemoGymRolloutImpl = rollout_cls(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            max_rollout_turns=max_rollout_turns,  # type: ignore
            policy_generation=policy_generation,  # type: ignore
            generation_config=generation_config,
        )
        self._tokenizer = tokenizer
        self._env_handles = env_handles
        self._val_env_handles = val_env_handles
        self._tq_buffer = tq_buffer
        self._weight_version: int = 0
        self._checkpoint_io_policy = checkpoint_io_policy
        self._checkpoint_io_semaphore: Optional[asyncio.Semaphore] = None

    def set_weight_version(self, version: int) -> None:
        """Set the weight_version used for rollout tags.

        Args:
            version: Trainer weight version to stamp on future rollout tags.
        """
        self._weight_version = int(version)

    async def run_rollout(
        self,
        input_sample: DatumSpec,
        *,
        num_generations_per_prompt: int,
        is_validation: bool = False,
        checkpoint_work: Optional[RolloutWorkItem] = None,
        checkpoint_writer: Any = None,
    ) -> PromptGroupRecord:
        """Run one prompt against the selected environments.

        Completed Gym siblings are persisted when checkpoint work and a writer
        are provided together.
        """
        assert num_generations_per_prompt >= 1, (
            "num_generations_per_prompt must be >= 1"
        )
        if (checkpoint_work is None) != (checkpoint_writer is None):
            raise ValueError(
                "checkpoint_work and checkpoint_writer must be provided together"
            )
        env_handles = self._val_env_handles if is_validation else self._env_handles
        if checkpoint_work is None:
            return await self._impl.run_rollout(
                input_sample,
                env_handles=env_handles,
                num_generations_per_prompt=num_generations_per_prompt,
            )
        work = checkpoint_work
        if self._checkpoint_io_policy is None:
            raise ValueError(
                "checkpoint_io_policy is required when checkpointing is enabled"
            )
        if not isinstance(self._impl, AsyncNemoGymRolloutImpl):
            raise NotImplementedError(
                "completed-sibling persistence currently supports only NeMo-Gym"
            )
        if work.num_generations != num_generations_per_prompt:
            raise ValueError(
                "checkpoint work expects "
                f"{work.num_generations} generations, but RolloutManager "
                f"was asked to generate {num_generations_per_prompt}"
            )

        async def _persist(generation_index: int, completion: Completion) -> None:
            await self._persist_completed_sibling(
                work,
                checkpoint_writer,
                generation_index,
                completion,
            )

        restored_completions = await self._load_completed_siblings(
            work,
            checkpoint_writer,
        )
        return await self._impl.run_rollout(
            input_sample,
            env_handles=env_handles,
            num_generations_per_prompt=num_generations_per_prompt,
            on_sibling_complete=_persist,
            restored_completions=restored_completions,
        )

    async def _load_completed_siblings(
        self,
        work: RolloutWorkItem,
        checkpoint_writer: Any,
    ) -> dict[int, Completion]:
        """Load validated sibling checkpoints before dispatching Gym work."""
        policy = self._require_checkpoint_io_policy()
        async with self._get_checkpoint_io_semaphore(policy):
            loaded = await self._load_records_with_retry(
                checkpoint_writer,
                work,
                policy,
            )
            if not isinstance(loaded, dict):
                raise TypeError("checkpoint load_completed must return a dictionary")

            completions: dict[int, Completion] = {}
            for generation_index, record in loaded.items():
                if not isinstance(record, CompletedSiblingRecord):
                    raise TypeError(
                        "checkpoint load_completed values must be "
                        "CompletedSiblingRecord instances"
                    )
                if generation_index != record.generation_index:
                    raise ValueError(
                        "checkpoint dictionary key does not match record generation "
                        f"index: {generation_index!r} != {record.generation_index}"
                    )
                self._validate_loaded_record(record, work)
                completions[generation_index] = _copy_completion_to_cpu(
                    record.completion
                )
            return completions

    async def _load_records_with_retry(
        self,
        checkpoint_writer: Any,
        work: RolloutWorkItem,
        policy: RolloutCheckpointIOPolicy,
    ) -> Any:
        """Retry transient checkpoint reads with a bounded acknowledgement wait."""
        total_attempts = policy.max_load_retries + 1
        last_error: TimeoutError | StorageUnavailableError | None = None
        for attempt_index in range(total_attempts):
            try:
                return await asyncio.wait_for(
                    self._invoke_checkpoint_writer_method(
                        checkpoint_writer.load_completed,
                        work,
                    ),
                    timeout=policy.load_timeout_s,
                )
            except (TimeoutError, StorageUnavailableError) as error:
                last_error = error
                if attempt_index + 1 < total_attempts:
                    if policy.load_retry_backoff_s > 0:
                        await asyncio.sleep(policy.load_retry_backoff_s)
                    continue
                break

        assert last_error is not None
        raise StorageUnavailableError(
            "checkpoint writer did not return completed siblings for "
            f"{work.run_id!r}/{work.group_id!r} after {total_attempts} attempts"
        ) from last_error

    def _validate_loaded_record(
        self,
        record: CompletedSiblingRecord,
        work: RolloutWorkItem,
    ) -> None:
        """Reject a writer response that is incompatible with dispatched work."""
        expected = {
            "run_id": work.run_id,
            "group_id": work.group_id,
            "prompt_id": work.prompt_id,
            "policy_version": work.policy_version,
            "prompt_fingerprint": work.prompt_fingerprint,
            "sampling_fingerprint": work.sampling_fingerprint,
            "tokenizer_fingerprint": work.tokenizer_fingerprint,
        }
        actual = {
            "run_id": record.run_id,
            "group_id": record.group_id,
            "prompt_id": record.prompt_id,
            "policy_version": record.policy_version,
            "prompt_fingerprint": record.prompt_fingerprint,
            "sampling_fingerprint": record.sampling_fingerprint,
            "tokenizer_fingerprint": record.tokenizer_fingerprint,
        }
        mismatches = {
            field: (actual[field], expected_value)
            for field, expected_value in expected.items()
            if actual[field] != expected_value
        }
        if mismatches:
            raise IncompatibleCheckpointError(
                "checkpoint writer returned a record incompatible with rollout work: "
                f"{mismatches}"
            )
        if record.attempt_id > work.attempt_id:
            raise IncompatibleCheckpointError(
                f"checkpoint attempt {record.attempt_id} is newer than dispatched "
                f"attempt {work.attempt_id} for {work.run_id}/{work.group_id}"
            )
        if record.generation_index >= work.num_generations:
            raise IncompatibleCheckpointError(
                f"loaded generation_index {record.generation_index} is outside "
                f"checkpoint group size {work.num_generations}"
            )

    async def _persist_completed_sibling(
        self,
        work: RolloutWorkItem,
        checkpoint_writer: Any,
        generation_index: int,
        completion: Completion,
    ) -> None:
        """Persist one immutable CPU sibling and wait for its acknowledgement."""
        if not 0 <= generation_index < work.num_generations:
            raise ValueError(
                f"generation_index {generation_index} is outside checkpoint group "
                f"size {work.num_generations}"
            )
        policy = self._require_checkpoint_io_policy()
        async with self._get_checkpoint_io_semaphore(policy):
            durable_completion = _copy_completion_to_cpu(completion)
            record = CompletedSiblingRecord(
                schema_version=ROLLOUT_CHECKPOINT_SCHEMA_VERSION,
                run_id=work.run_id,
                group_id=work.group_id,
                prompt_id=work.prompt_id,
                generation_index=generation_index,
                attempt_id=work.attempt_id,
                policy_version=work.policy_version,
                prompt_fingerprint=work.prompt_fingerprint,
                sampling_fingerprint=work.sampling_fingerprint,
                tokenizer_fingerprint=work.tokenizer_fingerprint,
                phase="SIBLING_COMPLETE",
                completion=durable_completion,
                sample_metrics={},
            )
            await self._persist_record_with_retry(
                checkpoint_writer,
                record,
                policy,
            )

    async def _persist_record_with_retry(
        self,
        checkpoint_writer: Any,
        record: CompletedSiblingRecord,
        policy: RolloutCheckpointWritePolicy,
    ) -> None:
        """Retry transient persistence failures while retaining one CPU record."""
        total_attempts = policy.max_retries + 1
        last_error: TimeoutError | StorageUnavailableError | None = None
        for attempt_index in range(total_attempts):
            try:
                ack = await asyncio.wait_for(
                    self._invoke_checkpoint_writer_method(
                        checkpoint_writer.persist_completed,
                        record,
                    ),
                    timeout=policy.write_timeout_s,
                )
            except (TimeoutError, StorageUnavailableError) as error:
                last_error = error
                if attempt_index + 1 < total_attempts:
                    if policy.retry_backoff_s > 0:
                        await asyncio.sleep(policy.retry_backoff_s)
                    continue
                break

            if not isinstance(ack, PersistAck):
                raise TypeError(
                    "checkpoint persist_completed must return a PersistAck, got "
                    f"{type(ack).__name__}"
                )
            if ack.logical_key != record.logical_key:
                raise ValueError(
                    "checkpoint acknowledgement key does not match the record: "
                    f"{ack.logical_key!r} != {record.logical_key!r}"
                )
            return

        assert last_error is not None
        raise StorageUnavailableError(
            "checkpoint writer did not durably acknowledge "
            f"{record.logical_key!r} after {total_attempts} attempts"
        ) from last_error

    def _require_checkpoint_io_policy(self) -> RolloutCheckpointIOPolicy:
        """Return the configured policy or fail before checkpoint I/O."""
        if self._checkpoint_io_policy is None:
            raise RuntimeError("checkpoint I/O policy is not configured")
        return self._checkpoint_io_policy

    def _get_checkpoint_io_semaphore(
        self,
        policy: RolloutCheckpointIOPolicy,
    ) -> asyncio.Semaphore:
        """Return the manager-wide bound for concurrent checkpoint I/O."""
        if self._checkpoint_io_semaphore is None:
            self._checkpoint_io_semaphore = asyncio.Semaphore(policy.max_pending_writes)
        return self._checkpoint_io_semaphore

    async def _invoke_checkpoint_writer_method(
        self,
        method: Any,
        *args: Any,
    ) -> Any:
        """Invoke a method on either a Ray handle or an in-process writer."""
        remote = getattr(method, "remote", None)
        if remote is not None:
            result = remote(*args)
        elif inspect.iscoroutinefunction(method):
            result = method(*args)
        else:
            result = await asyncio.to_thread(method, *args)
        if inspect.isawaitable(result):
            return await result
        return result

    async def generate_and_push(
        self,
        input_sample: DatumSpec,
        *,
        num_generations_per_prompt: int,
        target_step: Optional[int] = None,
        checkpoint_work: Optional[RolloutWorkItem] = None,
        checkpoint_writer: Any = None,
    ) -> None:
        """Reserve a buffer slot, run one prompt's rollout, then commit the slot.

        Args:
            input_sample: A single prompt (one DatumSpec entry).
            num_generations_per_prompt: Number of completions to generate.
            target_step: Training step this rollout targets; stamped on the buffer slot for StalenessSampler.force_in_order.
        """
        assert num_generations_per_prompt >= 1, (
            "num_generations_per_prompt must be >= 1"
        )
        assert self._tq_buffer is not None, (
            "generate_and_push requires tq_buffer to be set at __init__"
        )
        if (checkpoint_work is None) != (checkpoint_writer is None):
            raise ValueError(
                "checkpoint_work and checkpoint_writer must be provided together"
            )
        if checkpoint_work is not None:
            if not isinstance(self._impl, AsyncNemoGymRolloutImpl):
                raise NotImplementedError(
                    "completed-sibling persistence currently supports only NeMo-Gym"
                )
            if checkpoint_work.num_generations != num_generations_per_prompt:
                raise ValueError(
                    "checkpoint work expects "
                    f"{checkpoint_work.num_generations} generations, but "
                    "RolloutManager was asked to generate "
                    f"{num_generations_per_prompt}"
                )
        start_version = self._weight_version
        group_id = None
        if checkpoint_work is not None:
            if checkpoint_work.policy_version != start_version:
                raise ValueError(
                    "checkpoint work policy version does not match RolloutManager: "
                    f"{checkpoint_work.policy_version} != {start_version}"
                )
            if target_step is None:
                target_step = checkpoint_work.target_step
            elif target_step != checkpoint_work.target_step:
                raise ValueError(
                    "target_step does not match checkpoint work: "
                    f"{target_step} != {checkpoint_work.target_step}"
                )
            group_id = checkpoint_work.group_id
        group_id = self._tq_buffer.reserve(
            weight_version=start_version,
            target_step=target_step,
            group_id=group_id,
        )

        record = await self.run_rollout(
            input_sample,
            num_generations_per_prompt=num_generations_per_prompt,
            checkpoint_work=checkpoint_work,
            checkpoint_writer=checkpoint_writer,
        )
        end_version = self._weight_version

        await self._tq_buffer.commit(
            group_id,
            record,
            start_weight_version=start_version,
            end_weight_version=end_version,
        )


def _copy_completion_to_cpu(completion: Completion) -> Completion:
    return Completion(
        message_log=_copy_tree_to_cpu(completion.message_log),
        env_extras=_copy_tree_to_cpu(completion.env_extras),
        truncated=completion.truncated,
        reward=completion.reward,
    )


def _copy_tree_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _copy_tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_tree_to_cpu(item) for item in value)
    return copy.deepcopy(value)
