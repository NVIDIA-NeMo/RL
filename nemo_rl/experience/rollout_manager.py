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
from typing import Any, Optional

import torch
from transformers import PreTrainedTokenizerBase
from wandb import Table

from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.interfaces import Completion, PromptGroupRecord
from nemo_rl.experience.rollouts import (
    _calculate_single_metric,
    _tensorize_by_key,
    calculate_rewards,
)
from nemo_rl.models.generation.interfaces import (
    GenerationConfig,
    GenerationDatumSpec,
    GenerationInterface,
)
from nemo_rl.utils.timer import Timer

TokenizerType = PreTrainedTokenizerBase


class AsyncRolloutImpl:
    """Manages per-prompt multi-turn rollouts, producing a PromptGroupRecord per call.

    Each run_rollout takes one prompt and returns num_generations_per_prompt completions
    generated concurrently via asyncio.gather.
    """

    def __init__(
        self,
        tokenizer: TokenizerType,
        env_handles: dict[str, EnvironmentInterface],
        num_generations_per_prompt: int,
        max_seq_len: int,
        policy_generation: GenerationInterface,
        max_rollout_turns: int = 999999,
        **kwargs: Any,
    ) -> None:
        self._tokenizer = tokenizer
        self._env_handles = env_handles
        self._num_generations_per_prompt = num_generations_per_prompt
        self._max_seq_len = max_seq_len
        self._max_rollout_turns = max_rollout_turns
        self._policy_generation = policy_generation

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
            results = list(
                await asyncio.gather(
                    *[
                        self._run_single_rollout(input_sample, traj_idx)
                        for traj_idx in range(self._num_generations_per_prompt)
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
        self, input_sample: DatumSpec, traj_idx: int
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
                calculate_rewards, sample_batch, self._env_handles
            )

            # Update reward and termination statistics
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

        # Aggregate metrics across all samples.
        n = len(all_sample_metrics)
        rollout_metrics: dict[str, Any] = {
            **_calculate_single_metric(total_reward, n, "total_reward"),
            # turn metrics
            "total_turns": sum(turn_count),
            **_calculate_single_metric(turn_count, n, "turns_per_sample"),
            # token metrics
            **_calculate_single_metric(total_tokens, n, "total_tokens_per_sample"),
            **_calculate_single_metric(assistant_tokens, n, "gen_tokens_per_sample"),
            **_calculate_single_metric(env_tokens, n, "env_tokens_per_sample"),
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
        # per-sample histograms emitted via _calculate_single_metric above).
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

    Each run_rollout takes one prompt and returns num_generations_per_prompt completions
    batched through a single NeMo-Gym run_rollouts call.
    """

    def __init__(
        self,
        tokenizer: TokenizerType,
        env_handles: dict[str, EnvironmentInterface],
        num_generations_per_prompt: int,
        max_seq_len: int,
        generation_config: GenerationConfig,
        max_rollout_turns: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self._tokenizer = tokenizer
        self._env_handles = env_handles
        self._num_generations_per_prompt = num_generations_per_prompt
        self._max_seq_len = max_seq_len
        self._max_rollout_turns = max_rollout_turns
        self._generation_config = generation_config

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

        # Validate max_rollout_turns.
        assert self._max_rollout_turns is None, (
            "`max_rollout_turns` is not supported in NeMo-Gym path!"
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
        nemo_gym_env = self._env_handles["nemo_gym"]

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
        # truncated metrics
        truncated = [c.truncated for c in completions]

        # Aggregate metrics across all samples.
        n = len(completions)
        rollout_metrics: dict[str, Any] = {
            **_calculate_single_metric(total_reward, n, "total_reward"),
            # turn metrics
            **_calculate_single_metric(turn_count, n, "turns_per_sample"),
            # token metrics
            **_calculate_single_metric(total_tokens, n, "total_tokens_per_sample"),
            **_calculate_single_metric(assistant_tokens, n, "gen_tokens_per_sample"),
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


class RolloutManager:
    """Routes to AsyncRolloutImpl (native async) or AsyncNemoGymRolloutImpl (NeMo-Gym), and pushes results to a TQReplayBuffer."""

    def __init__(
        self,
        tokenizer: TokenizerType,
        env_handles: dict[str, EnvironmentInterface],
        num_generations_per_prompt: int,
        max_seq_len: int,
        max_rollout_turns: Optional[int] = None,
        policy_generation: Optional[GenerationInterface] = None,
        generation_config: Optional[GenerationConfig] = None,
        use_nemo_gym: bool = False,
        tq_buffer: Optional[TQReplayBuffer] = None,
    ) -> None:
        assert num_generations_per_prompt >= 1, (
            "num_generations_per_prompt must be >= 1"
        )

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
            env_handles=env_handles,
            num_generations_per_prompt=num_generations_per_prompt,
            max_seq_len=max_seq_len,
            max_rollout_turns=max_rollout_turns,  # type: ignore
            policy_generation=policy_generation,  # type: ignore
            generation_config=generation_config,
        )
        self._tokenizer = tokenizer
        self._num_generations_per_prompt = num_generations_per_prompt
        self._tq_buffer = tq_buffer
        self._weight_version: int = 0

    def set_weight_version(self, version: int) -> None:
        """Set the weight_version used for rollout tags.

        Args:
            version: Trainer weight version to stamp on future rollout tags.
        """
        self._weight_version = int(version)

    async def run_rollout(self, input_sample: DatumSpec) -> PromptGroupRecord:
        return await self._impl.run_rollout(input_sample)

    async def generate_and_push(self, input_sample: DatumSpec) -> None:
        """Run one prompt's rollout and write the N completions through the buffer.

        Args:
            input_sample: A single prompt (one DatumSpec entry).
        """
        assert self._tq_buffer is not None, (
            "generate_and_push requires tq_buffer to be set at __init__"
        )
        record = await self.run_rollout(input_sample)
        # TODO @yukih: record start and end weight version of the rollout instead of just end
        await self._tq_buffer.add(record, weight_version=self._weight_version)
