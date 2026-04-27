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
import os
import re
import uuid
from typing import Any, Dict, List, NotRequired, Optional, Tuple, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.metrics import calculate_pass_rate_per_prompt
from nemo_rl.environments.utils import chunk_list_to_workers


_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

_JUDGE_SYSTEM_PROMPT = """\
You are evaluating a customer service AI agent. You will be given the domain rules and \
policies the agent must follow, the customer's original request, and the full conversation.

Score the agent from 0.0 to 1.0 based on:
- Whether the customer's request was correctly fulfilled
- Whether all domain policies and rules were followed
- Whether communication was clear and professional

Respond with JSON only: {"score": <float 0-1>, "reasoning": "<one sentence>"}"""


class TauBenchEnvConfig(TypedDict):
    """Configuration for TauBenchEnvironment.

    Attributes:
        num_workers: Number of Ray worker actors for parallel execution.
        env_name: tau-bench domain to run ("retail" or "airline").
        task_split: Dataset split ("train", "test", or "dev").
        user_strategy: How the simulated user behaves ("llm", "react", "verify", "reflection").
        user_model: LiteLLM model string for the user simulator (e.g. "openai/gpt-4o-mini").
        user_base_url: Optional base URL override for the user model's OpenAI-compatible API.
        user_api_key: API key for the user model server.
        max_steps: Maximum tool-call steps per episode before forced termination.
        judge_model: LiteLLM model string for LLM-as-judge scoring (None disables judging).
        judge_base_url: Optional base URL override for the judge model's API.
        judge_api_key: API key for the judge model server.
        judge_weight: Blend weight: 0.0 = pure tau-bench reward, 1.0 = pure judge score.
    """

    num_workers: int
    env_name: str
    task_split: str
    user_strategy: str
    user_model: str
    user_base_url: NotRequired[str | None]
    user_api_key: NotRequired[str]
    max_steps: NotRequired[int]
    judge_model: NotRequired[str | None]
    judge_base_url: NotRequired[str | None]
    judge_api_key: NotRequired[str]
    judge_weight: NotRequired[float]


class TauBenchEnvMetadata(TypedDict):
    """Per-episode state carried through nemo-rl's metadata channel.

    Attributes:
        task_index: Which tau-bench task index this episode is running.
        episode_id: Unique ID assigned on the first step; None before that.
        step_count: Number of environment steps taken in this episode so far.
        tau_reward: tau-bench task-completion reward, set when the episode terminates.
        judge_score: LLM-judge score, set when the episode terminates (None if judging is off).
    """

    task_index: int
    episode_id: str | None
    step_count: int
    tau_reward: NotRequired[float | None]
    judge_score: NotRequired[float | None]


@ray.remote  # pragma: no cover
class TauBenchWorker:
    """Executes tau-bench environment steps for a chunk of rollout samples.

    Each active episode gets its own ``tau_bench.Env`` instance, stored by episode_id,
    so that database state persists correctly across multi-turn rollouts.
    """

    def __init__(
        self,
        env_name: str,
        task_split: str,
        user_strategy: str,
        user_model: str,
        user_base_url: Optional[str],
        user_api_key: str,
        max_steps: int,
        judge_model: Optional[str],
        judge_base_url: Optional[str],
        judge_api_key: str,
    ) -> None:
        self._env_name = env_name
        self._task_split = task_split
        self._user_strategy = user_strategy
        self._user_model = user_model
        self._max_steps = max_steps
        self._judge_model = judge_model
        self._judge_base_url = judge_base_url
        self._judge_api_key = judge_api_key
        self._active_envs: Dict[str, Any] = {}

        # Configure LiteLLM (used by tau-bench's user simulator) once at startup.
        if user_base_url:
            os.environ["OPENAI_BASE_URL"] = user_base_url
        os.environ["OPENAI_API_KEY"] = user_api_key

    def _make_env(self, task_index: int) -> Any:
        """Create and reset a tau-bench Env for a specific task index."""
        from tau_bench.envs import get_env

        env = get_env(
            env_name=self._env_name,
            user_strategy=self._user_strategy,
            user_model=self._user_model,
            task_split=self._task_split,
            task_index=task_index,
        )
        env.reset(task_index=task_index)
        return env

    def _parse_action(self, agent_text: str) -> Any:
        """Convert agent-generated text into a tau_bench Action.

        Text wrapped in ``<tool_call>...</tool_call>`` is parsed as a JSON tool
        invocation.  Anything else is treated as a ``respond`` action (the agent's
        final message to the user).
        """
        from tau_bench.types import Action

        match = _TOOL_CALL_RE.search(agent_text)
        if match:
            try:
                payload = json.loads(match.group(1).strip())
                return Action(
                    name=payload["name"],
                    kwargs=payload.get("arguments", payload.get("kwargs", {})),
                )
            except (json.JSONDecodeError, KeyError):
                pass
        return Action(name="respond", kwargs={"content": agent_text.strip()})

    def _call_judge(
        self,
        conversation: List[Dict[str, str]],
        domain_rules: str,
        task_instruction: str,
    ) -> float:
        """Score a completed episode using an LLM judge.

        Returns a float in [0, 1].  Returns 0.0 on any error so that a judge
        misconfiguration degrades gracefully rather than crashing training.
        """
        import requests

        convo_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in conversation
        )
        user_prompt = (
            f"DOMAIN RULES:\n{domain_rules}\n\n"
            f"CUSTOMER REQUEST:\n{task_instruction}\n\n"
            f"CONVERSATION:\n{convo_text}"
        )
        base_url = (self._judge_base_url or "https://api.openai.com").rstrip("/")
        try:
            resp = requests.post(
                f"{base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._judge_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._judge_model,
                    "messages": [
                        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.0,
                },
                timeout=60,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return float(json.loads(content).get("score", 0.0))
        except Exception:
            return 0.0

    def execute(
        self,
        message_batch: List[str],
        metadata_batch: List[TauBenchEnvMetadata],
        judge_weight: float,
    ) -> Tuple[
        List[Dict[str, str]],
        List[bool],
        List[float],
        List[TauBenchEnvMetadata],
    ]:
        """Run one environment step for each sample in this worker's chunk.

        Args:
            message_batch: Latest assistant response texts, one per sample.
            metadata_batch: Per-episode state dicts, one per sample.
            judge_weight: Blending weight between tau-bench reward and judge score.

        Returns:
            Tuple of (observations, terminateds, rewards, updated_metadata).
        """
        observations: List[Dict[str, str]] = []
        terminateds: List[bool] = []
        rewards: List[float] = []
        new_metadata: List[TauBenchEnvMetadata] = []

        for agent_text, meta in zip(message_batch, metadata_batch):
            task_index: int = meta["task_index"]
            step_count: int = meta.get("step_count", 0)
            episode_id: str = meta.get("episode_id") or str(uuid.uuid4())

            if episode_id not in self._active_envs:
                self._active_envs[episode_id] = self._make_env(task_index)
            tau_env = self._active_envs[episode_id]

            action = self._parse_action(agent_text)
            env_response = tau_env.step(action)
            step_count += 1

            done = bool(env_response.done) or step_count >= self._max_steps
            reward = 0.0
            tau_reward: Optional[float] = None
            judge_score: Optional[float] = None

            if done:
                reward_result = tau_env.calculate_reward()
                tau_reward = float(reward_result.reward)
                if self._judge_model and judge_weight > 0.0:
                    rules = "\n".join(getattr(tau_env, "rules", []))
                    tasks = getattr(tau_env, "tasks", [])
                    instruction = tasks[tau_env.task_index].instruction if tasks else ""
                    raw_messages = getattr(tau_env, "messages", [])
                    convo = [
                        {"role": m["role"], "content": m["content"]}
                        for m in raw_messages
                        if m.get("role") in ("user", "assistant")
                    ]
                    judge_score = self._call_judge(convo, rules, instruction)
                    reward = (1.0 - judge_weight) * tau_reward + judge_weight * judge_score
                else:
                    reward = tau_reward
                del self._active_envs[episode_id]

            observations.append(
                {"role": "environment", "content": str(env_response.observation)}
            )
            terminateds.append(done)
            rewards.append(reward)
            new_metadata.append(
                {
                    "task_index": task_index,
                    "episode_id": episode_id,
                    "step_count": step_count,
                    "tau_reward": tau_reward,
                    "judge_score": judge_score,
                }
            )

        return observations, terminateds, rewards, new_metadata


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class TauBenchEnvironment(EnvironmentInterface[TauBenchEnvMetadata]):
    """Multi-turn customer-service environment backed by tau-bench.

    The agent communicates with a simulated user across multiple turns, calling
    domain tools (e.g. order management, flight rescheduling) to complete tasks.
    Rewards blend tau-bench's deterministic task-completion check with an optional
    LLM-as-judge score for policy compliance and communication quality.

    **Tool call format**: the policy must wrap tool calls in ``<tool_call>`` tags::

        <tool_call>{"name": "cancel_order", "arguments": {"order_id": "O123"}}</tool_call>

    Any generation that does not contain a ``<tool_call>`` block is treated as a
    final ``respond`` action (the agent sending a message to the user), which may
    trigger a user-simulator turn or end the episode.

    **Stop strings**: ``["</tool_call>"]``.

    **Data format**: each dataset sample's ``extra_env_info`` must contain at minimum
    ``{"task_index": <int>, "episode_id": null, "step_count": 0}``.  The initial
    ``message_log`` should be constructed by calling ``tau_bench.envs.get_env`` and
    ``env.reset(task_index=N)`` during data preparation to ensure prompt/env
    consistency.
    """

    def __init__(self, cfg: TauBenchEnvConfig) -> None:
        self.cfg = cfg
        self._num_workers = cfg["num_workers"]
        self._judge_weight = float(cfg.get("judge_weight", 0.0))

        self._workers = [
            TauBenchWorker.options(
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote(
                env_name=cfg["env_name"],
                task_split=cfg["task_split"],
                user_strategy=cfg["user_strategy"],
                user_model=cfg["user_model"],
                user_base_url=cfg.get("user_base_url"),
                user_api_key=cfg.get("user_api_key", "dummy"),
                max_steps=cfg.get("max_steps", 30),
                judge_model=cfg.get("judge_model"),
                judge_base_url=cfg.get("judge_base_url"),
                judge_api_key=cfg.get("judge_api_key", "dummy"),
            )
            for _ in range(self._num_workers)
        ]

    def step(
        self,
        message_log_batch: List[LLMMessageLogType],
        metadata_batch: List[TauBenchEnvMetadata],
    ) -> EnvironmentReturn[TauBenchEnvMetadata]:
        """Run one environment step for each sample in the batch.

        Only the latest assistant message is forwarded to tau-bench; earlier turns
        are already captured in the tau-bench env's internal message history.

        Args:
            message_log_batch: Full conversation histories; only the last message
                               (the most recent assistant turn) is used.
            metadata_batch: Per-episode state, including task_index and episode_id.

        Returns:
            EnvironmentReturn with tool results or user responses as observations.
            Rewards are non-zero only on terminal steps.
        """
        message_batch = [ml[-1]["content"] for ml in message_log_batch]

        chunked_messages = chunk_list_to_workers(message_batch, self._num_workers)
        chunked_metadata = chunk_list_to_workers(metadata_batch, self._num_workers)

        futures = [
            self._workers[i].execute.remote(
                chunked_messages[i],
                chunked_metadata[i],
                self._judge_weight,
            )
            for i in range(self._num_workers)
        ]
        results = ray.get(futures)

        observations: List[Dict[str, str]] = []
        terminateds_list: List[bool] = []
        rewards_list: List[float] = []
        new_metadata: List[TauBenchEnvMetadata] = []

        for obs, term, rew, meta in results:
            observations.extend(obs)
            terminateds_list.extend(term)
            rewards_list.extend(rew)
            new_metadata.extend(meta)

        next_stop_strings = [["</tool_call>"]] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=new_metadata,
            next_stop_strings=next_stop_strings,
            rewards=torch.tensor(rewards_list, dtype=torch.float32),
            terminateds=torch.tensor(terminateds_list, dtype=torch.bool),
            answers=None,
        )

    def shutdown(self) -> None:
        for worker in self._workers:
            ray.kill(worker)

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, Dict[str, Any]]:
        """Compute aggregate metrics over a completed rollout batch."""
        rewards = batch["rewards"]
        if rewards.ndim > 1:
            rewards = rewards[:, 0]
        rewards = rewards * batch["is_end"]

        tau_rewards: List[float] = []
        judge_scores: List[float] = []
        for meta in batch.get("extra_env_info", []):
            if meta and meta.get("tau_reward") is not None:
                tau_rewards.append(float(meta["tau_reward"]))
            if meta and meta.get("judge_score") is not None:
                judge_scores.append(float(meta["judge_score"]))

        metrics: Dict[str, Any] = {
            "tau_bench/task_completion_rate": rewards.mean().item(),
            "tau_bench/pass_at_k": calculate_pass_rate_per_prompt(
                batch["text"], rewards
            ),
            "tau_bench/fraction_properly_ended": batch["is_end"].float().mean().item(),
        }
        if tau_rewards:
            metrics["tau_bench/mean_tau_reward"] = sum(tau_rewards) / len(tau_rewards)
        if judge_scores:
            metrics["tau_bench/mean_judge_score"] = (
                sum(judge_scores) / len(judge_scores)
            )

        return batch, metrics
