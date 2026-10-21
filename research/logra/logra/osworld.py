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

"""OSWorld rollout capture using NeMoRL's token-exact vLLM HTTP responses.

Screenshot compaction starts a new training segment. All segments retain the
same episode identity and terminal reward; each generated token trains once.
"""

import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import (
    ThreadPoolExecutor,
)
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from threading import Condition, Lock
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import torch
from pydantic import BaseModel, Field

# Retain these imports for existing checkpoints and callers of logra.osworld.
from logra.rollout_pool import (
    RollingTaskPool as RollingTaskPool,
)
from logra.rollout_pool import (
    RolloutBatch as RolloutBatch,
)
from logra.rollout_pool import (
    RolloutPoolState as RolloutPoolState,
)
from logra.rollout_pool import (
    RolloutTicket as RolloutTicket,
)
from logra.rollout_pool import (
    TaskSampler as TaskSampler,
)
from logra.token_capture import (
    CuaSegment as CuaSegment,
)
from logra.token_capture import (
    TurnRecorder as TurnRecorder,
)
from logra.token_capture import (
    align_image_placeholders as align_image_placeholders,
)

if TYPE_CHECKING:
    from logra.cua import TaskBaseline
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class OSWorldConfig(BaseModel, extra="forbid"):
    """RFC0040 desktop protocol; paths refer to separately installed OSWorld assets."""

    train_data: Path
    validation_data: Path
    train_task_base: Path
    validation_task_base: Path
    setup_cache: Path
    template_capabilities: Path
    task_requirements: Path
    credentials_file: Path | None = None
    template_weights: dict[str, float] = {
        "osworld-v1": 1.0,
        "osworld-slim": 1.0,
        "osworld-v2": 1.0,
    }
    task_compatible_templates: bool = False
    evaluation_template: str = "osworld-v1"
    # Legacy prod4 consumes the completed verifier score. The native diagnostic
    # additionally treats context exhaustion as a policy failure.
    score_mode: Literal["error_adjusted", "verifier"] = "error_adjusted"
    concurrency: Annotated[int, Field(gt=0)] = 64
    max_steps: Annotated[int, Field(gt=0)] = 200
    max_tokens: Annotated[int, Field(gt=0)] = 8192
    evaluation_max_tokens: Annotated[int, Field(gt=0)] = 16384
    temperature: Annotated[float, Field(gt=0)] = 1.0
    top_p: Annotated[float, Field(gt=0, le=1)] = 1.0
    evaluation_temperature: Annotated[float, Field(gt=0)] = 0.5
    evaluation_top_p: Annotated[float, Field(gt=0, le=1)] = 0.9
    action_wait: Annotated[float, Field(ge=0)] = 3.0
    evaluation_action_wait: Annotated[float, Field(ge=0)] = 5.0
    startup_wait: Annotated[float, Field(ge=0)] = 10.0
    settle_wait: Annotated[float, Field(ge=0)] = 5.0
    task_timeout: Annotated[int, Field(gt=0)] = 3600
    model_timeout: Annotated[float, Field(gt=0)] = 900.0
    block_size: Annotated[int, Field(gt=0)] = 8
    boundary_images: Annotated[int, Field(gt=0)] = 3
    prompt_password: str = "password"
    desktop_password: str = ""


@contextmanager
def sandbox_credentials(path: Path) -> Iterator[None]:
    """Read credentials inside desktop processes, outside model-worker environments.

    The driver validates this file before creating any Ray workers. Each desktop
    actor then exposes the values locally for one episode and restores its environment.
    """
    data = json.loads(path.read_text())
    endpoint, key = data["endpoint"], data["apikey"]
    if (
        not isinstance(endpoint, str)
        or not endpoint
        or not isinstance(key, str)
        or not key
    ):
        raise ValueError("Sandbox credential file requires endpoint and apikey strings")
    url = "https://" + endpoint
    values = {
        "AENV_MODE": "cell2",
        "AENV_URL": url,
        "AENV_API_KEY": key,
        "E2B_API_KEY": key,
        "E2B_API_URL": url,
        "E2B_SANDBOX_URL": url,
        "E2B_DOMAIN": endpoint,
    }
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@dataclass
class DesktopTask:
    key: str
    prompt_key: str
    domain: str
    task: dict[str, Any]


def read_tasks(path: Path, task_base: Path) -> list[DesktopTask]:
    """Preserve the input order and repetitions used for task oversampling."""
    tasks = []
    cache = {}
    with path.open() as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row["verifier_metadata"]
            task = metadata.get("osworld_task")
            if task is None:
                task_path = task_base / metadata["task_config_path"]
                if task_path not in cache:
                    cache[task_path] = json.loads(task_path.read_text())
                task = cache[task_path]
            instruction = metadata["instruction"]
            if instruction != task["instruction"]:
                raise ValueError(f"Task and prompt instructions disagree: {row['id']}")
            # Historical baselines identify the original prompt, not its row or segment.
            key = hashlib.sha1(instruction.encode()).hexdigest()[:16]
            tasks.append(DesktopTask(row["id"], key, metadata["domain"], task))
    if not tasks:
        raise ValueError(f"No tasks in {path}")
    return tasks


@dataclass
class Episode:
    task: DesktopTask
    template: str
    score: float
    valid: bool
    error: str | None
    segments: list[CuaSegment]
    artifact_dir: Path | None = None
    verifier_score: float | None = None
    context_exhausted: bool = False
    error_adjusted_score: float | None = None


@dataclass
class _DesktopContext:
    config: OSWorldConfig
    template: str
    feedback: list[Any] = field(default_factory=list)
    context_exhausted: bool = False
    capture_error: Exception | None = None
    model_error: str | None = None


_DESKTOP: ContextVar[_DesktopContext] = ContextVar("logra_osworld_episode")


def make_agent(**kwargs: Any) -> Any:
    """Gym factory: original RFC0015 agent with the historical K8/S3 image window."""
    # The separately installed OSWorld runtime is needed only for desktop tasks.
    from desktop_env.execution_feedback import render_feedback_batch
    from mm_agents.nvidia.nemotron_agent import NemotronAgent

    context = _DESKTOP.get()
    config = context.config

    class Agent(NemotronAgent):
        max_image_history_length: int

        def predict(
            self, instruction: str, obs: dict[str, Any], **predict_kwargs: Any
        ) -> tuple[str, list[str], dict[str, Any]]:
            step = len(self.observations)
            start = (
                0
                if step < config.block_size
                else max(
                    1,
                    (step // config.block_size) * config.block_size
                    - config.boundary_images
                    + 1,
                )
            )
            self.max_image_history_length = step + 1 - start
            if context.feedback:
                obs["execution_feedback_batch_text"] = render_feedback_batch(
                    context.feedback
                )
                context.feedback.clear()
            return super().predict(instruction, obs, step_idx=step)

    kwargs.pop("client_password")
    kwargs.pop("log_context")
    return Agent(
        **kwargs, password=config.prompt_password, prompt_format_variant="rfc0015_v1"
    )


def make_environment(**kwargs: Any) -> Any:
    """Gym factory: retain original action feedback and desktop settling waits."""
    # Numerical and protocol tests do not require the external desktop runtime.
    from desktop_env.desktop_env import DesktopEnv
    from desktop_env.execution_feedback import CommandExecutionFeedback

    context = _DESKTOP.get()

    class Environment(DesktopEnv):
        def reset(self, *args: Any, **reset_kwargs: Any) -> Any:
            result = super().reset(*args, **reset_kwargs)
            self.controller.retry_times = 1
            context.template = self.provider.template
            time.sleep(context.config.startup_wait)
            return result

        def step(self, *args: Any, **step_kwargs: Any) -> Any:
            obs, reward, done, info = super().step(*args, **step_kwargs)
            if not done and (obs is None or obs.get("screenshot") is None):
                raise ConnectionError(
                    "Desktop did not return a screenshot after the action"
                )
            feedback = info.get("execution_feedback") if info is not None else None
            if feedback is not None:
                context.feedback.append(
                    CommandExecutionFeedback.model_validate(feedback)
                )
            return obs, reward, done, info

        def evaluate(self) -> float:
            time.sleep(context.config.settle_wait)
            return super().evaluate()

    return Environment(**kwargs)


class GenerationGate:
    """Finish active model requests before refitting; hold new requests meanwhile."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._paused = False
        self._active = 0

    @contextmanager
    def request(self) -> Iterator[None]:
        with self._condition:
            self._condition.wait_for(lambda: not self._paused)
            self._active += 1
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    @contextmanager
    def pause(self) -> Iterator[None]:
        with self._condition:
            if self._paused:
                raise RuntimeError("Generation is already paused for a weight update")
            self._paused = True
        try:
            with self._condition:
                self._condition.wait_for(lambda: self._active == 0)
            yield
        finally:
            with self._condition:
                self._paused = False
                self._condition.notify_all()


@dataclass
class DesktopRollouts:
    config: OSWorldConfig
    processor: Any
    model_name: str
    endpoints: list[str]
    output_dir: Path
    capabilities: dict = field(init=False)
    requirements: dict = field(init=False)
    gate: GenerationGate = field(default_factory=GenerationGate, init=False)

    def __post_init__(self) -> None:
        if not self.endpoints or any(not url for url in self.endpoints):
            raise ValueError("CUA requires native vLLM async HTTP endpoints")
        self.capabilities = json.loads(self.config.template_capabilities.read_text())[
            "templates"
        ]
        self.requirements = {
            row["id"]: row["needs"]
            for row in json.loads(self.config.task_requirements.read_text())
        }
        if not self.config.template_weights or any(
            (not math.isfinite(weight) or weight <= 0)
            for weight in self.config.template_weights.values()
        ):
            raise ValueError("Desktop template weights must be finite and positive")

    def template_for(self, task: DesktopTask, seed: int, evaluation: bool) -> str:
        if evaluation:
            return self.config.evaluation_template
        weights = self.config.template_weights
        requirements = (
            self.requirements.get(task.task["id"])
            if self.config.task_compatible_templates
            else None
        )
        candidates = list(weights)
        if requirements is not None:
            compatible = [
                name
                for name in weights
                if name in self.capabilities
                and all(
                    self.capabilities[name]["caps"].get(req) for req in requirements
                )
            ]
            # Match the old provider's full-mixture fallback when compatibility is unknown.
            if compatible:
                candidates = compatible
        return random.Random(seed).choices(
            candidates, [weights[name] for name in candidates]
        )[0]

    def run(
        self, tasks: list[DesktopTask], *, step: int, seed: int, evaluation: bool
    ) -> list[Episode]:
        if self.config.credentials_file is None:
            raise ValueError(
                "Set osworld.credentials_file before running desktop tasks"
            )
        # Local smoke path. Production uses one CPU actor per active desktop.
        with sandbox_credentials(self.config.credentials_file):
            return [
                self._episode(task, step, seed + index, index, evaluation)
                for index, task in enumerate(tasks)
            ]

    def _episode(
        self, task: DesktopTask, step: int, seed: int, index: int, evaluation: bool
    ) -> Episode:
        # HTTP and desktop dependencies are optional outside actual rollouts.
        import requests
        from desktop_env.providers.agentenv import provider
        from responses_api_agents.osworld_agent.client import run_osworld_task

        config = self.config
        recorder = TurnRecorder(self.processor)
        context = _DesktopContext(config, self.template_for(task, seed, evaluation))
        endpoint = (
            self.endpoints[index % len(self.endpoints)].rstrip("/")
            + "/chat/completions"
        )
        step_dir = (
            self.output_dir / ("eval" if evaluation else "train") / f"step-{step}"
        )
        step_dir.mkdir(parents=True, exist_ok=True)
        # Resuming may replay a step; retain its previous desktop attempt logs.
        artifacts: Path = Path(tempfile.mkdtemp(prefix=f"{index}-", dir=step_dir))
        context_token = _DESKTOP.set(context)
        template_token = provider.PRESET.set(context.template)

        with requests.Session() as session:

            def request(
                messages: list[dict[str, Any]], payload: dict[str, Any]
            ) -> dict[str, Any]:
                cached = recorder.cached_response(payload)
                if cached is not None:
                    return cached
                body = recorder.request_body(payload)
                body["model"] = self.model_name
                # R3 and token bundles come from the native NeMoRL HTTP server.
                try:
                    with self.gate.request():
                        response = session.post(
                            endpoint, json=body, timeout=config.model_timeout
                        )
                    if response.status_code == 400 and any(
                        word in response.text.lower()
                        for word in (
                            "context length",
                            "maximum model length",
                            "max_model_len",
                        )
                    ):
                        context.context_exhausted = True
                    response.raise_for_status()
                except requests.RequestException as error:
                    # The desktop agent retries and may swallow the final error.
                    # Keep service failures distinct from unsuccessful policy actions.
                    status = (
                        error.response.status_code
                        if error.response is not None
                        else "unavailable"
                    )
                    context.model_error = (
                        f"Model service {type(error).__name__}: {status}"
                    )
                    raise
                else:
                    context.model_error = None
                choice = response.json()["choices"][0]
                message = choice["message"]
                if choice["finish_reason"] == "length" and not message.get(
                    "generation_token_ids"
                ):
                    context.context_exhausted = True
                    raise ValueError("Policy exhausted the context window")
                try:
                    return recorder.record(payload, message)
                except (ValueError, KeyError, TypeError) as error:
                    context.capture_error = error
                    raise

            try:
                result = run_osworld_task(
                    task.task,
                    lambda *_: "",
                    provider_name="agentenv",
                    env_class_path="logra.osworld.make_environment",
                    agent_class_path="logra.osworld.make_agent",
                    runner_name="nemotron_v3_nano_omni_agent",
                    messages_model_fn=request,
                    client_password=config.desktop_password,
                    cache_dir=str(artifacts / "cache"),
                    setup_cache_dir=str(config.setup_cache),
                    max_steps=config.max_steps,
                    task_timeout=config.task_timeout,
                    model_timeout=config.model_timeout,
                    sleep_after_execution=(
                        config.evaluation_action_wait
                        if evaluation
                        else config.action_wait
                    ),
                    policy_model_name=self.model_name,
                    policy_max_tokens=(
                        config.evaluation_max_tokens
                        if evaluation
                        else config.max_tokens
                    ),
                    policy_temperature=(
                        config.evaluation_temperature
                        if evaluation
                        else config.temperature
                    ),
                    policy_top_p=(
                        config.evaluation_top_p if evaluation else config.top_p
                    ),
                    reward_mode="raw",
                    log_context={"output_dir": str(artifacts)},
                )
            finally:
                provider.PRESET.reset(template_token)
                _DESKTOP.reset(context_token)
        if context.capture_error is not None:
            raise RuntimeError(
                "Native rollout token/route capture failed"
            ) from context.capture_error
        # Old CUA treats context exhaustion as a policy failure, not a sandbox failure.
        # Its wall-time limit still evaluates the final state; preserve that valid score.
        valid = context.context_exhausted or (
            context.model_error is None
            and (
                result.runtime_eligible
                or (
                    result.evaluation_completed
                    and result.runtime_admission_reason == "timeout"
                )
            )
        )
        error_adjusted_score = (
            0.0 if context.context_exhausted or not valid else result.score
        )
        score = (
            (result.score if valid else 0.0)
            if config.score_mode == "verifier"
            else error_adjusted_score
        )
        error = (
            result.error
            if context.context_exhausted
            else context.model_error or result.error
        )
        episode = Episode(
            task,
            context.template,
            score,
            valid,
            error,
            recorder.segments,
            artifact_dir=cast(Path, artifacts),
            verifier_score=float(result.score),
            context_exhausted=context.context_exhausted,
            error_adjusted_score=float(error_adjusted_score),
        )
        (artifacts / "episode.json").write_text(
            json.dumps(
                {
                    "task": task.key,
                    "template": episode.template,
                    "score": score,
                    "verifier_score": float(result.score),
                    "error_adjusted_score": float(error_adjusted_score),
                    "score_mode": config.score_mode,
                    "context_exhausted": context.context_exhausted,
                    "model_error": context.model_error,
                    "evaluation_completed": result.evaluation_completed,
                    "runtime_admission_reason": result.runtime_admission_reason,
                    "valid": valid,
                    "error": episode.error,
                    "segments": len(recorder.segments),
                    "action_tokens": sum(
                        int(part.token_mask.sum()) for part in recorder.segments
                    ),
                },
                indent=2,
            )
        )
        return episode


class _DesktopWorker:
    """One desktop at a time: upstream agents and artifact loggers are process-global."""

    def __init__(self, *args: Any) -> None:
        self.rollouts = DesktopRollouts(*args)
        self._busy = Lock()
        self._paused: AbstractContextManager[None] | None = None

    def ready(self) -> int:
        return os.getpid()

    def episode(
        self, task: DesktopTask, step: int, seed: int, index: int, evaluation: bool
    ) -> Episode:
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("A desktop worker cannot run two episodes at once")
        try:
            path = self.rollouts.config.credentials_file
            if path is None:
                raise ValueError("A private sandbox credential file is required")
            with sandbox_credentials(path):
                episode = self.rollouts._episode(task, step, seed, index, evaluation)
            if not evaluation:
                for segment in episode.segments:
                    segment.prepare_media(self.rollouts.processor)
                    segment.compress_media()
        finally:
            self._busy.release()
        return episode

    def pause(self) -> None:
        self._paused = self.rollouts.gate.pause()
        self._paused.__enter__()

    def resume(self) -> None:
        if self._paused is not None:
            self._paused.__exit__(None, None, None)
            self._paused = None


@dataclass
class IsolatedDesktopRollouts:
    """Spread desktop processes across Ray; keep GPU inference in native workers.

    Each actor runs one episode. Its second thread only handles refit pause/resume.
    Small driver threads wait on results; they never execute the desktop library.
    """

    config: OSWorldConfig
    processor: Any
    model_name: str
    endpoints: list[str]
    output_dir: Path
    workers: int
    _actors: list[Any] = field(default_factory=list, init=False)
    _available: Queue[Any] = field(default_factory=Queue, init=False)

    def __enter__(self) -> "IsolatedDesktopRollouts":
        import ray

        # The model workers are already initialized without sandbox keys. Only
        # these CPU actors read the private file, for the duration of an episode.
        worker_type = ray.remote(num_cpus=0.5, max_concurrency=2)(_DesktopWorker)
        processor = ray.put(self.processor)
        try:
            for _ in range(self.workers):
                actor = worker_type.options(scheduling_strategy="SPREAD").remote(
                    self.config,
                    processor,
                    self.model_name,
                    self.endpoints,
                    self.output_dir,
                )
                self._actors.append(actor)
                self._available.put(actor)
            ray.get([actor.ready.remote() for actor in self._actors])
        except BaseException:
            self.close()
            raise
        return self

    def close(self) -> None:
        import ray

        for actor in self._actors:
            ray.kill(actor, no_restart=True)
        self._actors.clear()

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _episode(
        self, task: DesktopTask, step: int, seed: int, index: int, evaluation: bool
    ) -> Episode:
        import ray

        actor = self._available.get()
        try:
            episode = ray.get(actor.episode.remote(task, step, seed, index, evaluation))
        finally:
            self._available.put(actor)
        return episode

    @contextmanager
    def pause(self) -> Iterator[None]:
        import ray

        try:
            ray.get([actor.pause.remote() for actor in self._actors])
            yield
        finally:
            ray.get([actor.resume.remote() for actor in self._actors])

    def run(
        self, tasks: list[DesktopTask], *, step: int, seed: int, evaluation: bool
    ) -> list[Episode]:
        with ThreadPoolExecutor(max_workers=self.config.concurrency) as executor:
            futures = [
                executor.submit(self._episode, task, step, seed + i, i, evaluation)
                for i, task in enumerate(tasks)
            ]
            return [future.result() for future in futures]

    @contextmanager
    def stream(
        self,
        tasks: list[DesktopTask],
        sampler: "TaskSampler",
        *,
        batch_size: int,
        inflight: int,
        seed: int,
        start_step: int,
        resume: "RolloutPoolState | None",
    ) -> Iterator["RollingTaskPool"]:
        def generate(ticket: "RolloutTicket") -> Episode:
            return self._episode(
                tasks[ticket.task_index],
                ticket.dispatch_step,
                ticket.seed,
                ticket.serial,
                False,
            )

        pool = RollingTaskPool(
            generate,
            sampler,
            batch_size=batch_size,
            inflight=inflight,
            seed=seed,
            start_step=start_step,
            resume=resume,
        )
        try:
            yield pool
        finally:
            pool.close()


def make_training_batch(
    episodes: list[Episode],
    baseline: "TaskBaseline",
    processor: Any,
    dp_size: int,
) -> "BatchedDataDict[Any]":
    """Train every segment once; pad with masked rows to keep EP collectives aligned."""
    # Load native batching only when constructing a training batch.
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    rows = [(episode, segment) for episode in episodes for segment in episode.segments]
    if not rows or not any(
        episode.valid and segment.token_mask.any() for episode, segment in rows
    ):
        raise RuntimeError("This rollout batch has no usable action tokens")
    # The old baseline counts compacted segments, not unique episodes. Form all
    # advantages before updating the counts, including repeated task/template keys.
    advantages = baseline.advantages(
        [episode.task.prompt_key for episode, _ in rows],
        [episode.template for episode, _ in rows],
        torch.tensor([episode.score for episode, _ in rows]),
        torch.tensor([episode.valid for episode, _ in rows]),
    )
    # Only decompression runs concurrently; the shared processor is not called.
    compressed_segments = {
        id(segment): segment
        for _, segment in rows
        if segment._compressed_media is not None
    }
    if compressed_segments:
        with ThreadPoolExecutor(
            max_workers=min(8, len(compressed_segments))
        ) as workers:
            list(
                workers.map(
                    lambda segment: segment.prepare_media(processor),
                    compressed_segments.values(),
                )
            )
    batches = []
    for (episode, segment), advantage in zip(rows, advantages, strict=True):
        # Production actors already prepared these pixels; local smoke tests
        # use the same preprocessing lazily.
        media = segment.prepare_media(processor)
        batches.append(
            {
                "input_ids": segment.input_ids.unsqueeze(0),
                "input_lengths": torch.tensor([segment.input_ids.numel()]),
                "token_mask": segment.token_mask.unsqueeze(0),
                "sample_mask": torch.tensor([float(episode.valid)]),
                "advantages": torch.full_like(
                    segment.token_mask, float(advantage)
                ).unsqueeze(0),
                "generation_logprobs": segment.generation_logprobs.unsqueeze(0),
                "routed_experts": segment.routed_experts.unsqueeze(0),
                **media,
            }
        )
    for _ in range((-len(batches)) % dp_size):
        padding = dict(batches[0])
        padding["sample_mask"] = torch.zeros(1)
        batches.append(padding)
    return BatchedDataDict.from_batches(
        batches,
        {
            "input_ids": processor.tokenizer.pad_token_id,
            "routed_experts": -1,
        },
    )


def summarize_episodes(episodes: list[Episode]) -> dict[str, float]:
    """Keep task means and the historical equal-domain metric separately named."""
    if not episodes:
        raise ValueError("Cannot score an empty evaluation")
    domains: dict[str, list[float]] = {}
    for episode in episodes:
        domains.setdefault(episode.task.domain, []).append(episode.score)
    domain_scores = {
        name: sum(scores) / len(scores) for name, scores in domains.items()
    }
    metrics = {
        "score": sum(ep.score for ep in episodes) / len(episodes),
        "success_rate": sum(ep.score >= 1 for ep in episodes) / len(episodes),
        "domain_macro_score": sum(domain_scores.values()) / len(domain_scores),
        "infrastructure_failure_rate": sum(not ep.valid for ep in episodes)
        / len(episodes),
        "tasks": float(len(episodes)),
        **{"domain/" + name: value for name, value in domain_scores.items()},
    }

    # Preserve the model/error-adjusted score and the untouched verifier output.
    # They are different metrics; neither silently replaces the other.
    verifier_scores = [
        ep.verifier_score for ep in episodes if ep.verifier_score is not None
    ]
    if len(verifier_scores) == len(episodes):
        metrics.update(
            {
                "verifier_score": sum(verifier_scores) / len(episodes),
                "verifier_success_rate": sum(score >= 1 for score in verifier_scores)
                / len(episodes),
                "score_override_rate": sum(
                    raw != ep.score
                    for raw, ep in zip(verifier_scores, episodes, strict=True)
                )
                / len(episodes),
                "context_exhaustion_rate": sum(ep.context_exhausted for ep in episodes)
                / len(episodes),
            }
        )
    adjusted = [
        ep.error_adjusted_score
        for ep in episodes
        if ep.error_adjusted_score is not None
    ]
    if len(adjusted) == len(episodes):
        metrics["error_adjusted_score"] = sum(adjusted) / len(episodes)
        metrics["error_adjusted_success_rate"] = sum(
            score >= 1 for score in adjusted
        ) / len(episodes)
        if len(verifier_scores) == len(episodes):
            metrics["error_adjustment_rate"] = sum(
                raw != value
                for raw, value in zip(verifier_scores, adjusted, strict=True)
            ) / len(episodes)
    return metrics
