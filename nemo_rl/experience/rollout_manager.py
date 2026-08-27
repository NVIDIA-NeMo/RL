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
import enum
import json
from dataclasses import dataclass, field
from typing import Any, Optional

import ray.exceptions
import torch
from transformers import PreTrainedTokenizerBase
from wandb import Table

from nemo_rl.algorithms.async_utils.replay_buffer import (
    PostWriteEnrichmentError,
    TQReplayBuffer,
)
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.failures import (
    FailureClass,
    GenerationUnavailable,
    GymTransportError,
    RolloutDataFailure,
    RolloutFailure,
    RolloutRedispatchExhausted,
    RolloutTimeout,
    classify_rollout_failure,
)
from nemo_rl.experience.interfaces import Completion, PromptGroupRecord
from nemo_rl.experience.metric_utils import calculate_single_metric, pct
from nemo_rl.experience.rollouts import (
    EffortLevelsConfig,
    _apply_effort_shaping,
    _attach_routed_experts_to_message_log_prefix,
    _dummy_routed_experts_for_tokens,
    _effort_shaping_metrics,
    _find_routed_experts_template,
    _tensorize_by_key,
    attach_static_multimodal_payload,
    calculate_rewards,
)
from nemo_rl.models.generation.interfaces import (
    GenerationConfig,
    GenerationDatumSpec,
    GenerationInterface,
)
from nemo_rl.utils.timer import Timer

TokenizerType = PreTrainedTokenizerBase


def _contains_post_write_enrichment_error(error: BaseException) -> bool:
    """Whether an error, including a rollback ExceptionGroup, is post-write."""
    if isinstance(error, PostWriteEnrichmentError):
        return True
    if isinstance(error, BaseExceptionGroup):
        return any(
            _contains_post_write_enrichment_error(child) for child in error.exceptions
        )
    return False


class RolloutOutcome(str, enum.Enum):
    """How :meth:`RolloutManager.generate_and_push` finished for one prompt."""

    # The prompt group reached the replay buffer.
    COMMITTED = "committed"
    # The prompt was given up on within a budget: its data-failure budget within
    # max_skipped_prompts, or its infrastructure budget within
    # max_consecutive_dropped_prompts. No group was committed, so the caller owns
    # releasing its backpressure permit and crediting the step's shortfall.
    SKIPPED = "skipped"


@dataclass(frozen=True)
class RolloutRetryPolicy:
    """Retry budgets for one prompt, resolved from ``async_rl.rollout_failure``.

    The attempt budgets are **required**. They previously defaulted to 1/1/1, which
    contradicted ``RolloutFailureConfig``'s 5/2/3 and put a second set of defaults in
    the codebase -- a reader here came away believing the shipped budget was 1. The only
    place a retry default lives is ``RolloutFailureConfig``; callers that need the
    historical no-retry behaviour ask for it by name via :meth:`single_attempt`.
    """

    # Attempts for infrastructure failures. 1 means no retry.
    max_infra_attempts: int
    # Attempts for deterministic, prompt-specific failures. 1 means no retry.
    max_data_attempts: int
    # Attempts to re-dispatch only the NeMo-Gym rows that never arrived, before the
    # whole prompt group is retried. 1 means no row-level retry.
    max_gym_row_attempts: int
    # These three do not contradict RolloutFailureConfig, so they keep their defaults.
    backoff_base_s: float = 1.0
    max_backoff_s: float = 30.0
    # Run-wide cap on prompts that may exhaust their data budget and be dropped,
    # enforced across every generate_and_push call. 0 means none may be: the first
    # exhaustion propagates the original failure.
    max_skipped_prompts: int = 0
    # Cap on CONSECUTIVE prompts that may exhaust their infra budget and be dropped;
    # any commit resets the run of failures. 0 means none may be, so the first
    # exhaustion raises RolloutRedispatchExhausted as it did before this budget existed.
    max_consecutive_dropped_prompts: int = 0

    @classmethod
    def single_attempt(cls, **overrides: Any) -> "RolloutRetryPolicy":
        """The historical no-retry policy, with optional overrides.

        An explicit choice for callers constructing a ``RolloutManager`` directly, who
        must not silently gain retries -- not a second set of defaults.
        """
        budgets: dict[str, Any] = {
            "max_infra_attempts": 1,
            "max_data_attempts": 1,
            "max_gym_row_attempts": 1,
        }
        budgets.update(overrides)
        return cls(**budgets)

    def __post_init__(self) -> None:
        # A zero budget would mean "never attempt the rollout at all", which no caller
        # wants and which would leave the retry loop with nothing to report.
        if (
            self.max_infra_attempts < 1
            or self.max_data_attempts < 1
            or self.max_gym_row_attempts < 1
        ):
            raise ValueError(
                "RolloutRetryPolicy attempt budgets must be >= 1; got "
                f"max_infra_attempts={self.max_infra_attempts}, "
                f"max_data_attempts={self.max_data_attempts}, "
                f"max_gym_row_attempts={self.max_gym_row_attempts}"
            )

    def backoff_for(self, attempt: int) -> float:
        """Return the delay before infra attempt ``attempt`` + 1 (1-based attempts)."""
        return min(self.backoff_base_s * 2 ** (attempt - 1), self.max_backoff_s)


@dataclass
class RolloutStats:
    """Counters describing what the retry policy has been doing.

    Read by the SingleController for logging. A stall or a rising redispatch count is
    the only externally visible sign that the fleet is degrading, so these are not
    optional bookkeeping.
    """

    committed: int = 0
    skipped: int = 0
    # Infra re-dispatches: the fleet is degrading. Kept apart from data retries because
    # conflating them defeats the whole point of the two-budget split -- an operator
    # watching redispatch_total climb needs to know whether the cluster is sick or the
    # dataset is.
    redispatches_by_reason: dict[str, int] = field(default_factory=dict)
    # Retries of a deterministic, prompt-specific failure.
    data_retries_by_reason: dict[str, int] = field(default_factory=dict)
    # Prompts that ran out of data budget entirely.
    data_failures_by_reason: dict[str, int] = field(default_factory=dict)
    # Prompts that ran out of INFRA budget and were dropped rather than failing the run.
    # Distinct from redispatches_by_reason, which counts attempts that were retried: a
    # fleet that recovers shows redispatches with this flat, and the two diverging is
    # what says the outage outlasted the per-prompt budget.
    infra_drops_by_reason: dict[str, int] = field(default_factory=dict)
    # Longest run of consecutive infra drops seen so far. The live counter resets on
    # every commit, so without this high-water mark a run that came within one prompt of
    # aborting is indistinguishable from one that never dropped anything.
    max_consecutive_infra_drops: int = 0
    # NeMo-Gym row-level re-dispatches. These recover a partial prompt group without
    # redoing the whole thing, so they never reached the counters above and gym could
    # retry rows all run with redispatch_total sitting flat.
    gym_row_redispatches: int = 0

    def record_redispatch(self, reason: str) -> None:
        self.redispatches_by_reason[reason] = (
            self.redispatches_by_reason.get(reason, 0) + 1
        )

    def record_data_retry(self, reason: str) -> None:
        self.data_retries_by_reason[reason] = (
            self.data_retries_by_reason.get(reason, 0) + 1
        )

    def record_data_failure(self, reason: str) -> None:
        self.data_failures_by_reason[reason] = (
            self.data_failures_by_reason.get(reason, 0) + 1
        )

    def record_infra_drop(self, reason: str, consecutive: int) -> None:
        self.infra_drops_by_reason[reason] = (
            self.infra_drops_by_reason.get(reason, 0) + 1
        )
        self.max_consecutive_infra_drops = max(
            self.max_consecutive_infra_drops, consecutive
        )

    def record_gym_row_redispatch(self, rows: int = 1) -> None:
        self.gym_row_redispatches += rows

    def as_metrics(self) -> dict[str, float]:
        """Flatten into a metric dict for the SingleController logger."""
        # Every family gets an aggregate, not just per-exception series: alerting on
        # "any data failure" should not require knowing the exception names up front.
        metrics: dict[str, float] = {
            "rollout/committed_total": float(self.committed),
            "rollout/skipped_total": float(self.skipped),
            "rollout/redispatch_total": float(
                sum(self.redispatches_by_reason.values())
            ),
            "rollout/data_retry_total": float(
                sum(self.data_retries_by_reason.values())
            ),
            "rollout/data_failures_total": float(
                sum(self.data_failures_by_reason.values())
            ),
            "rollout/gym_row_redispatch_total": float(self.gym_row_redispatches),
            "rollout/infra_drops_total": float(
                sum(self.infra_drops_by_reason.values())
            ),
            "rollout/max_consecutive_infra_drops": float(
                self.max_consecutive_infra_drops
            ),
        }
        for reason, count in self.redispatches_by_reason.items():
            metrics[f"rollout/redispatch_total/{reason}"] = float(count)
        for reason, count in self.data_retries_by_reason.items():
            metrics[f"rollout/data_retry_total/{reason}"] = float(count)
        for reason, count in self.data_failures_by_reason.items():
            metrics[f"rollout/data_failures_total/{reason}"] = float(count)
        for reason, count in self.infra_drops_by_reason.items():
            metrics[f"rollout/infra_drops_total/{reason}"] = float(count)
        return metrics


@dataclass(frozen=True)
class RolloutTimeouts:
    """Deadlines for the blocking waits inside one rollout.

    Resolved from ``async_rl.rollout_failure.nemo_gym.rollout_timeout_s`` and
    ``async_rl.rollout_failure.native.{generation,env}_timeout_s``, which own the
    user-facing defaults. ``None`` means no deadline,
    reproducing the historical behaviour of waiting indefinitely.
    """

    rollout_s: Optional[float] = None
    generation_s: Optional[float] = None
    env_s: Optional[float] = None


def _classify_generation_failure(
    exc: Exception, *, prompt_idx: Any, traj_idx: int
) -> RolloutFailure:
    """Wrap a generation error in the typed failure its class implies.

    The original exception is preserved as ``__cause__``; the prompt and trajectory
    coordinates are attached because a raw generation traceback does not say which
    rollout it belonged to.

    Any Ray-boundary error is infrastructure *here*, by context. An exception raised
    inside a still-living generation worker -- vLLM ``EngineDeadError``, a CUDA OOM --
    arrives as a bare ``RayTaskError`` whose cause the boundary degraded, so
    ``classify_rollout_failure`` can only fall through to DATA and the prompt gets two
    attempts instead of the five the fleet-failure path is built for.

    Scoped to this call site rather than widened in ``classify_rollout_failure``,
    deliberately. Globally, "unrecognized means DATA" is the right default: it is about
    exceptions we can inspect and do not recognize, and flipping it would retry genuine
    bugs everywhere. Here we have information the classifier does not -- this exception
    came from a *generation* RPC, so "that shard could not serve it" is the correct
    reading whatever the destroyed cause was, and re-dispatching to another shard is
    exactly the right response. A real bug in the worker still surfaces, chained, once
    the bounded infra budget runs out.

    This also makes the two paths agree: part 2/4's ``_generate_on_shard`` already maps
    ``ray.exceptions.RayError`` to ``GenerationUnavailable``, so without this the same
    exception classified INFRA there and DATA here.

    Args:
        exc: The exception raised while generating a turn.
        prompt_idx: Index of the prompt whose rollout failed.
        traj_idx: Index of the failing generation within the prompt group.

    Returns:
        ``GenerationUnavailable`` for infrastructure failures (retriable on another
        shard), ``RolloutDataFailure`` otherwise.
    """
    detail = f"prompt_idx={prompt_idx} traj_idx={traj_idx}: {type(exc).__name__}: {exc}"
    if (
        isinstance(exc, ray.exceptions.RayError)
        or classify_rollout_failure(exc) is FailureClass.INFRA
    ):
        return GenerationUnavailable(f"generation unavailable for {detail}")
    return RolloutDataFailure(f"generation failed for {detail}")


async def _gather_cancelling_siblings(coros: list[Any]) -> list[Any]:
    """Gather coroutines, cancelling the remainder as soon as one fails.

    ``asyncio.gather`` propagates the first exception but leaves the other awaitables
    running detached. On the rollout path those keep occupying generation capacity for
    a prompt group whose result is already being discarded, so they are cancelled and
    drained before unwinding.

    Args:
        coros: Coroutines to run concurrently.

    Returns:
        Their results, in input order.
    """
    tasks = [asyncio.ensure_future(coro) for coro in coros]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class _Deadline:
    """``asyncio.timeout`` that reports expiry as a typed :class:`RolloutTimeout`.

    A bare ``asyncio.timeout`` surfaces expiry as ``TimeoutError``, which is
    indistinguishable from a ``TimeoutError`` raised by the wrapped code itself. This
    consults ``expired()`` so only a real deadline breach is relabelled, and anything
    else propagates untouched.

    ``seconds=None`` disables the deadline, matching ``asyncio.timeout`` semantics.
    """

    def __init__(self, seconds: Optional[float], description: str) -> None:
        self._seconds = seconds
        self._description = description
        self._timeout: Optional[asyncio.Timeout] = None

    async def __aenter__(self) -> "_Deadline":
        self._timeout = asyncio.timeout(self._seconds)
        await self._timeout.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> Optional[bool]:
        assert self._timeout is not None
        try:
            return await self._timeout.__aexit__(exc_type, exc, tb)
        except TimeoutError as timeout_error:
            if not self._timeout.expired():
                raise
            raise RolloutTimeout(
                f"{self._description} exceeded {self._seconds}s"
            ) from timeout_error


class AsyncRolloutImpl:
    """Manages per-prompt multi-turn rollouts, producing a PromptGroupRecord per call.

    Each run_rollout takes one prompt and returns num_generations_per_prompt completions
    generated concurrently via asyncio.gather.
    """

    def __init__(
        self,
        tokenizer: TokenizerType,
        task_to_env: dict[str, EnvironmentInterface],
        num_generations_per_prompt: int,
        max_seq_len: int,
        max_rollout_turns: int,
        policy_generation: GenerationInterface,
        timeouts: RolloutTimeouts = RolloutTimeouts(),
        **kwargs: Any,
    ) -> None:
        self._tokenizer = tokenizer
        self._task_to_env = task_to_env
        self._num_generations_per_prompt = num_generations_per_prompt
        self._max_seq_len = max_seq_len
        self._max_rollout_turns = max_rollout_turns
        self._policy_generation = policy_generation
        self._timeouts = timeouts

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
            results = await _gather_cancelling_siblings(
                [
                    self._run_single_rollout(input_sample, traj_idx)
                    for traj_idx in range(self._num_generations_per_prompt)
                ]
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

            # Generate response for this sample using async generation.
            # A failure here must not be absorbed: returning a partial completion
            # would commit a zero-reward row that still counts toward this prompt
            # group's GRPO baseline, silently biasing every sibling's advantage.
            try:
                (
                    assistant_message,
                    input_lengths,
                    gen_metrics,
                ) = await self._generate_response(
                    current_message_log,
                    current_stop_strings,
                )
            except Exception as e:
                raise _classify_generation_failure(
                    e, prompt_idx=input_sample["idx"], traj_idx=traj_idx
                ) from e

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
            #
            # The deadline frees this rollout, not the thread: Python cannot kill a
            # running thread, so a hung env call keeps occupying a thread-pool slot
            # until its own ray.get returns. Unblocking the rollout is still what
            # matters -- otherwise it holds a max_inflight_prompts permit forever.
            async with _Deadline(self._timeouts.env_s, "environment step"):
                env_output = await asyncio.to_thread(
                    calculate_rewards, sample_batch, self._task_to_env
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

            env_message: dict[str, Any] = {
                "role": env_output.observations[0]["role"],
                "content": env_obs_content,
                "token_ids": tokenized_obs,
            }
            routed_template = _find_routed_experts_template(current_message_log)
            if routed_template is not None:
                env_message["routed_experts"] = _dummy_routed_experts_for_tokens(
                    tokenized_obs, routed_template
                )
            current_message_log.append(env_message)

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
        async with _Deadline(self._timeouts.generation_s, "generation turn"):
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
        if "routed_experts" in output:
            routed_experts = output["routed_experts"][0]
            prefix_length = _attach_routed_experts_to_message_log_prefix(
                message_log, routed_experts
            )
            if prefix_length != input_len:
                raise RuntimeError(
                    "message_log token length does not match generation input_length "
                    f"({prefix_length} != {input_len})."
                )
            assistant_message["routed_experts"] = routed_experts[input_len:total_len]

        # Calculate generation metrics
        gen_metrics: dict[str, Any] = {}
        if "gen_leader_worker_idx" in output:
            v = output["gen_leader_worker_idx"][0]
            try:
                gen_metrics["gen_leader_worker_idx"] = (
                    int(v[0]) if isinstance(v, list) else int(v)
                )
            except (IndexError, TypeError, ValueError) as e:
                # Load-accounting metric only -- a malformed value must not fail the
                # rollout, but the catch stays narrow so a real error still surfaces.
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

    Each run_rollout takes one prompt and returns num_generations_per_prompt completions
    batched through a single NeMo-Gym run_rollouts call.
    """

    def __init__(
        self,
        tokenizer: TokenizerType,
        task_to_env: dict[str, EnvironmentInterface],
        num_generations_per_prompt: int,
        max_seq_len: int,
        max_rollout_turns: int,
        generation_config: GenerationConfig,
        mask_env_flagged_samples: bool = True,
        # Optional so direct construction does not have to carry the resiliency wiring;
        # RolloutManager always passes both explicitly.
        timeouts: Optional[RolloutTimeouts] = None,
        retry_policy: Optional[RolloutRetryPolicy] = None,
        # Shared with the owning RolloutManager so row-level re-dispatches are visible
        # in the same counters as everything else. None when constructed directly.
        stats: Optional[RolloutStats] = None,
        # Length-based reward shaping for low-effort prompts; None disables it.
        effort_config: Optional[EffortLevelsConfig] = None,
        **kwargs: Any,
    ) -> None:
        self._tokenizer = tokenizer
        self._task_to_env = task_to_env
        self._num_generations_per_prompt = num_generations_per_prompt
        self._max_seq_len = max_seq_len
        self._max_rollout_turns = max_rollout_turns
        self._generation_config = generation_config
        self._mask_env_flagged_samples = mask_env_flagged_samples
        self._timeouts = timeouts if timeouts is not None else RolloutTimeouts()
        self._max_gym_row_attempts = (
            retry_policy
            if retry_policy is not None
            else RolloutRetryPolicy.single_attempt()
        ).max_gym_row_attempts
        self._stats = stats
        self._effort_config = effort_config

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
        completions, prompt_message_log, rollout_metrics = await self._run_rollouts(
            rollout_inputs, timer, timer_prefix
        )
        source_message_log = input_sample["message_log"]
        attach_static_multimodal_payload(prompt_message_log, source_message_log)
        for completion in completions:
            attach_static_multimodal_payload(completion.message_log, source_message_log)

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

    async def _stream_rows(
        self,
        nemo_gym_env: Any,
        pending: list[dict],
        results: list[Optional[dict]],
        total_rows: int,
        timer_prefix: str,
    ) -> Optional[dict[str, Any]]:
        """Dispatch ``pending`` rows and fill their slots in ``results`` as they land.

        Args:
            nemo_gym_env: The NeMo-Gym environment actor handle.
            pending: Rows still awaiting a result; each carries its original ``_rowidx``.
            results: Full-length result list, mutated in place.
            total_rows: Size of the original prompt group, used to validate row indices.
            timer_prefix: Timer namespace forwarded to the environment.

        Returns:
            The environment's timing metrics, or None if the stream ended without them.
        """
        dispatched = {row["_rowidx"] for row in pending}
        received: set[int] = set()
        env_timing_metrics: Optional[dict[str, Any]] = None

        async for result_ref in nemo_gym_env.run_rollouts.options(
            num_returns="streaming"
        ).remote(pending, timer_prefix):
            rowidx, result, timing_metrics = await result_ref
            # Validated against the original group, not the pending subset: on a
            # re-dispatch the row keeps its original index so results stay ordered.
            if not isinstance(rowidx, int) or not 0 <= rowidx < total_rows:
                raise ValueError(
                    f"NeMo-Gym returned invalid row index {rowidx!r} for "
                    f"{total_rows} inputs"
                )
            if rowidx not in dispatched:
                raise ValueError(
                    f"NeMo-Gym returned row index {rowidx}, which was not dispatched "
                    f"in this attempt ({sorted(dispatched)})"
                )
            if rowidx in received:
                raise ValueError(f"NeMo-Gym returned duplicate row index {rowidx}")
            received.add(rowidx)
            results[rowidx] = result
            if timing_metrics is not None:
                env_timing_metrics = timing_metrics

        return env_timing_metrics

    async def _run_rollouts(
        self,
        inputs: list[dict],
        timer: Timer,
        timer_prefix: str,
    ) -> tuple[list[Completion], LLMMessageLogType, dict[str, Any]]:
        """Dispatch rows to NeMo-Gym; return completions, prompt, and metrics.

        Rows that never arrive are re-dispatched on their own rather than by redoing the
        whole group. NeMo-Gym's stream dies on the first failing row, so one bad row
        takes every later row with it; at num_generations_per_prompt=16 a naive whole
        group retry pays 16 generations to recover one. Completed rows are kept across
        attempts, which is the same shape as the legacy collector's pending-group retry.
        """
        nemo_gym_env = self._task_to_env["nemo_gym"]
        total_rows = len(inputs)
        # Re-dispatch maps NeMo-Gym's echoed _rowidx back onto the original group, so
        # the rows must carry the index _build_inputs stamped on them. Checked here
        # because the alternative is a KeyError several frames deeper.
        for position, row in enumerate(inputs):
            if row.get("_rowidx") != position:
                raise ValueError(
                    f"NeMo-Gym input row {position} carries _rowidx="
                    f"{row.get('_rowidx')!r}; rows must be stamped with their own "
                    "position for re-dispatch to preserve ordering"
                )

        # Run generation and restore input order as results stream back.
        with timer.time(f"{timer_prefix}/run_rollouts"):
            results: list[dict | None] = [None for _ in inputs]
            env_timing_metrics: dict[str, Any] = {}
            # One deadline for the whole prompt group, re-dispatches included -- it is
            # the group that has a budget, not each attempt. It also spans the stream
            # rather than each await: NeMo-Gym yields rows as they finish, so a
            # per-await budget would reset every time a fast row landed and never fire
            # for the slow one holding the group up.
            # Kept across attempts so the failure below can name the transport error that
            # actually lost the rows. An intermediate attempt can absorb an INFRA error
            # and a later one end the stream cleanly-but-short, and without this the
            # operator reads "rows missing" with no cause attached at exactly the moment
            # they need one.
            # Exception, not BaseException: the only writer is the `except Exception`
            # below, and a wider annotation makes the `raise ... from last_error` at the
            # end unverifiable.
            last_error: Optional[Exception] = None
            async with _Deadline(self._timeouts.rollout_s, "NeMo-Gym prompt group"):
                for attempt in range(1, self._max_gym_row_attempts + 1):
                    pending = [row for row in inputs if results[row["_rowidx"]] is None]
                    if not pending:
                        break
                    if attempt > 1:
                        print(
                            f"NeMo-Gym: re-dispatching {len(pending)}/{total_rows} "
                            f"row(s) (attempt {attempt}/{self._max_gym_row_attempts})",
                            flush=True,
                        )
                        # Row re-dispatches are invisible in redispatch_total -- they
                        # recover a partial group instead of retrying the prompt -- so
                        # gym could retry rows all run with every counter flat.
                        if self._stats is not None:
                            self._stats.record_gym_row_redispatch(len(pending))
                    try:
                        timing_metrics = await self._stream_rows(
                            nemo_gym_env, pending, results, total_rows, timer_prefix
                        )
                    except Exception as error:
                        last_error = error
                        # Only transport-shaped failures are worth another dispatch; a
                        # prompt NeMo-Gym cannot serve fails the same way every time.
                        if (
                            classify_rollout_failure(error) is not FailureClass.INFRA
                            or attempt == self._max_gym_row_attempts
                        ):
                            raise
                    else:
                        if timing_metrics is not None:
                            env_timing_metrics = timing_metrics

            missing = [i for i, result in enumerate(results) if result is None]
            if missing:
                failure = GymTransportError(
                    "NeMo-Gym rollout stream ended before all rows arrived; missing "
                    f"rows {missing} of {total_rows} after "
                    f"{self._max_gym_row_attempts} attempt(s)"
                )
                # Narrowed before the raise: pyrefly rejects an Optional in a `from`
                # clause, even though `raise ... from None` is legal at runtime.
                if last_error is None:
                    raise failure
                raise failure from last_error

            completed_results = [result for result in results if result is not None]
            # Shape rewards for low-effort prompts before completions are built.
            shaping = _apply_effort_shaping(
                completed_results, inputs, self._effort_config
            )
            # All N rollouts share the same input prompt; tensorize one copy.
            prompt_message_log = completed_results[0]["input_message_log"]
            _tensorize_by_key(prompt_message_log, "token_ids")
            # Convert results to completions.
            completions = [
                self._result_to_completion(result) for result in completed_results
            ]

        # Compute rollout metrics.
        with timer.time(f"{timer_prefix}/compute_metrics"):
            rollout_metrics = self._compute_rollout_metrics(
                completions, inputs[0]["agent_ref"]["name"]
            )
            # Same helper the batched path uses, so the two cannot drift apart.
            rollout_metrics.update(_effort_shaping_metrics(shaping))

        rollout_metrics.update(env_timing_metrics)

        return completions, prompt_message_log, rollout_metrics

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

        # Same gate as the batched path: when masking is off, drop the env
        # mask flag so later batch building never sees it.
        if not self._mask_env_flagged_samples:
            (result["full_result"].get("instance_config") or {}).pop(
                "mask_sample", None
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
        task_to_env: dict[str, EnvironmentInterface],
        num_generations_per_prompt: int,
        max_seq_len: int,
        max_rollout_turns: int = 1,
        policy_generation: Optional[GenerationInterface] = None,
        generation_config: Optional[GenerationConfig] = None,
        use_nemo_gym: bool = False,
        mask_env_flagged_samples: bool = True,
        tq_buffer: Optional[TQReplayBuffer] = None,
        timeouts: Optional[RolloutTimeouts] = None,
        retry_policy: Optional[RolloutRetryPolicy] = None,
        effort_config: Optional[EffortLevelsConfig] = None,
    ) -> None:
        assert num_generations_per_prompt >= 1, (
            "num_generations_per_prompt must be >= 1"
        )
        # Resolved before the impl is built: the NeMo-Gym impl reads its row-retry
        # budget out of it at construction time, and shares the counters so its
        # row-level re-dispatches land in the same place as everything else.
        self._retry_policy = (
            retry_policy
            if retry_policy is not None
            else RolloutRetryPolicy.single_attempt()
        )
        self._stats = RolloutStats()

        if not use_nemo_gym:
            rollout_cls = AsyncRolloutImpl
            assert policy_generation is not None, (
                "policy_generation is required for the native async path"
            )
        else:
            rollout_cls = AsyncNemoGymRolloutImpl
            assert generation_config is not None, (
                "generation_config is required for the NeMo-Gym path"
            )

        self._impl: AsyncRolloutImpl | AsyncNemoGymRolloutImpl = rollout_cls(
            tokenizer=tokenizer,
            task_to_env=task_to_env,
            num_generations_per_prompt=num_generations_per_prompt,
            max_seq_len=max_seq_len,
            max_rollout_turns=max_rollout_turns,
            policy_generation=policy_generation,  # type: ignore
            generation_config=generation_config,
            # Only used by AsyncNemoGymRolloutImpl; AsyncRolloutImpl ignores it.
            mask_env_flagged_samples=mask_env_flagged_samples,
            # None means "no deadlines", which is what async_rl's own defaults resolve
            # to; callers that have a config pass the resolved values in.
            timeouts=timeouts if timeouts is not None else RolloutTimeouts(),
            # Only the NeMo-Gym impl reads these; the native impl absorbs them via kwargs.
            retry_policy=self._retry_policy,
            stats=self._stats,
            effort_config=effort_config,
        )
        self._tokenizer = tokenizer
        self._num_generations_per_prompt = num_generations_per_prompt
        self._tq_buffer = tq_buffer
        self._weight_version: int = 0
        # Run-wide, shared across concurrent generate_and_push calls. Safe as a plain
        # int: every caller runs on the SingleController's single event loop.
        self._skipped_prompts: int = 0
        # Infra drops since the last commit. Shared for the same reason, and shared
        # deliberately: the question it answers -- "is the fleet still answering
        # anyone?" -- is about the fleet, not about one prompt's history.
        self._consecutive_infra_drops: int = 0

    @property
    def stats(self) -> RolloutStats:
        """Counters describing retry/skip activity so far."""
        return self._stats

    def set_weight_version(self, version: int) -> None:
        """Set the weight_version used for rollout tags.

        Args:
            version: Trainer weight version to stamp on future rollout tags.
        """
        self._weight_version = int(version)

    async def run_rollout(self, input_sample: DatumSpec) -> PromptGroupRecord:
        return await self._impl.run_rollout(input_sample)

    async def generate_and_push(
        self,
        input_sample: DatumSpec,
        *,
        target_step: Optional[int] = None,
        inflight_registry: Optional[dict[str, tuple[asyncio.Task[None], int]]] = None,
    ) -> RolloutOutcome:
        """Roll out one prompt and commit it, re-dispatching on infrastructure failure.

        No prompt is discarded for infrastructure reasons. An infra failure means the
        fleet is unwell, not the prompt, so the attempt is retried -- and because each
        retry re-enters generation-shard selection, it naturally lands somewhere else
        without this method needing to know anything about shard health. Exhausting the
        infra budget therefore means the failure follows the prompt across the whole
        fleet, which is reported as fleet-wide failure rather than absorbed.

        Deterministic failures get their own, much smaller budget: another shard would
        reject the prompt identically, so retrying mostly burns time. One retry is still
        worth taking because a shard under memory pressure can return an empty
        generation that looks deterministic and is not.

        Args:
            input_sample: A single prompt (one DatumSpec entry).
            target_step: Training step this rollout targets; stamped on the buffer slot for StalenessSampler.force_in_order.
            inflight_registry: Optional controller-owned mapping from group ID to
                its dispatch task and start weight version.

        Returns:
            ``COMMITTED`` when the group reached the buffer, or ``SKIPPED`` when the
            prompt was given up on within a budget: its data budget within
            ``max_skipped_prompts``, or its infra budget within
            ``max_consecutive_dropped_prompts``. A ``SKIPPED`` prompt committed nothing,
            so the caller owns both its backpressure permit and the shortfall for the
            training step it was stamped for.

        Raises:
            RolloutRedispatchExhausted: The infra budget ran out and the fleet has not
                committed anything since ``max_consecutive_dropped_prompts`` drops ago.
            RolloutDataFailure: The data budget ran out beyond ``max_skipped_prompts``.
        """
        assert self._tq_buffer is not None, (
            "generate_and_push requires tq_buffer to be set at __init__"
        )
        policy = self._retry_policy
        infra_attempts = 0
        data_attempts = 0
        last_infra_error: Optional[Exception] = None

        # The loop condition is the infrastructure budget, so running out of it exits
        # here rather than raising from inside the handler. The data budget is tracked
        # separately and terminates from within, since exhausting it is a statement
        # about the prompt rather than about the fleet.
        while infra_attempts < policy.max_infra_attempts:
            start_version = self._weight_version
            # Reserved inside the loop so each attempt owns a fresh group_id: rows a
            # failed attempt may have written cannot then collide with the retry's.
            group_id = self._tq_buffer.reserve(
                weight_version=start_version, target_step=target_step
            )
            try:
                # Registered per ATTEMPT, not per prompt: each retry reserves a fresh
                # group_id, so the controller's registry must follow the attempt that
                # actually owns the slot it might abort.
                if inflight_registry is not None:
                    current_task = asyncio.current_task()
                    assert current_task is not None
                    inflight_registry[group_id] = (current_task, start_version)
                # Unregister before commit so cancellation cannot interrupt it.
                try:
                    record = await self.run_rollout(input_sample)
                finally:
                    if inflight_registry is not None:
                        inflight_registry.pop(group_id, None)
                end_version = self._weight_version
                await self._tq_buffer.commit(
                    group_id,
                    record,
                    start_weight_version=start_version,
                    end_weight_version=end_version,
                )
            except Exception as error:
                # A failed rollout must not leave an unready slot that can block an
                # in-order sampler. commit() rolls back any DataPlane rows it wrote.
                # Cleanup failure must not mask the error that caused it.
                try:
                    await self._tq_buffer.remove_group(group_id)
                except Exception as cleanup_exc:
                    print(
                        f"  warn: remove_group({group_id}) cleanup failed: {cleanup_exc!r}",
                        flush=True,
                    )
                # The rollout itself succeeded. Re-running generation cannot repair
                # a required downstream stage (for example MOPD teacher inference),
                # and would spend the rollout retry budget on the wrong subsystem.
                if _contains_post_write_enrichment_error(error):
                    raise
                reason = type(error).__name__

                if classify_rollout_failure(error) is FailureClass.INFRA:
                    infra_attempts += 1
                    last_infra_error = error
                    if infra_attempts >= policy.max_infra_attempts:
                        break
                    self._stats.record_redispatch(reason)
                    # The backpressure permit is held across this sleep, so the wait is
                    # capped by max_backoff_s rather than growing without bound.
                    await asyncio.sleep(policy.backoff_for(infra_attempts))
                    continue

                data_attempts += 1
                if data_attempts >= policy.max_data_attempts:
                    self._stats.record_data_failure(reason)
                    if self._skipped_prompts >= policy.max_skipped_prompts:
                        # At the default of 0 this fires on the first exhaustion and the
                        # original failure propagates unchanged -- one knob, and its
                        # zero value is the old "fail_fast" without a second key to
                        # contradict it.
                        if policy.max_skipped_prompts == 0:
                            raise
                        raise RolloutDataFailure(
                            f"skipped {self._skipped_prompts} prompts and this one also "
                            f"exhausted its data budget, exceeding max_skipped_prompts="
                            f"{policy.max_skipped_prompts}; the dataset or "
                            "sequence-length configuration is likely wrong"
                        ) from error
                    self._skipped_prompts += 1
                    print(
                        f"skipping prompt idx={input_sample['idx']} after "
                        f"{data_attempts} deterministic failure(s) ({reason}: {error})",
                        flush=True,
                    )
                    self._stats.skipped += 1
                    return RolloutOutcome.SKIPPED
                # A data retry, NOT a re-dispatch: the fleet is fine, this prompt is
                # suspect. Recording it as a re-dispatch made rollout/redispatch_total --
                # documented above as the sign the fleet is degrading -- climb for bad
                # data, which is the one distinction the two budgets exist to draw.
                self._stats.record_data_retry(reason)
                continue
            except BaseException:
                # Cancellation and other non-Exception exits: clean up, never retry.
                try:
                    await self._tq_buffer.remove_group(group_id)
                except Exception as cleanup_exc:
                    print(
                        f"  warn: remove_group({group_id}) cleanup failed: {cleanup_exc!r}",
                        flush=True,
                    )
                raise

            self._stats.committed += 1
            # A commit proves the fleet is answering, which is exactly the claim the
            # consecutive budget is testing, so it clears the run of drops. Placed on
            # the success path rather than in the infra handler so that a prompt which
            # succeeded on a retry also counts -- the fleet recovered either way.
            self._consecutive_infra_drops = 0
            return RolloutOutcome.COMMITTED

        # The infrastructure budget ran out. The same failure followed the prompt across
        # repeated shard selections, which says the fleet is broken rather than the
        # prompt.
        #
        # The budget is >= 1 (enforced in RolloutRetryPolicy), so the loop ran at least
        # once and can only have exited through the infra branch's break.
        assert last_infra_error is not None
        reason = type(last_infra_error).__name__
        self._consecutive_infra_drops += 1
        if self._consecutive_infra_drops > policy.max_consecutive_dropped_prompts:
            raise RolloutRedispatchExhausted(
                f"prompt idx={input_sample['idx']} exhausted its infrastructure retry "
                f"budget after {infra_attempts} attempt(s) "
                f"(max_infra_attempts_per_prompt="
                f"{policy.max_infra_attempts}), and this was drop "
                f"{self._consecutive_infra_drops} with no rollout committed in between, "
                f"exceeding max_consecutive_dropped_prompts="
                f"{policy.max_consecutive_dropped_prompts}; the generation fleet is not "
                f"recovering. Last failure was {reason}: {last_infra_error}"
            ) from last_infra_error

        # Under the budget: give up on this prompt and let the run continue. The caller
        # owns the backpressure permit for a SKIPPED outcome, and -- because the prompt
        # may have been stamped for a specific training step that will now never fill --
        # owns crediting the shortfall so the train pump can close that step short.
        self._stats.record_infra_drop(reason, self._consecutive_infra_drops)
        print(
            f"dropping prompt idx={input_sample['idx']} after {infra_attempts} "
            f"infrastructure failure(s) ({reason}: {last_infra_error}) "
            f"[consecutive drop {self._consecutive_infra_drops}/"
            f"{policy.max_consecutive_dropped_prompts}]",
            flush=True,
        )
        self._stats.skipped += 1
        return RolloutOutcome.SKIPPED
