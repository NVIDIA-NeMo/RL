# Profile synchronous vLLM rollouts with a plugin

NeMo RL can drive an optional profiler plugin around complete synchronous vLLM
rollout attempts. One profiled attempt includes every generation turn and
`finish_generation()`. Reward processing, policy scoring, validation rollouts,
and policy training are outside this lifecycle.

The initial integration supports `tensor_parallel_size=1`,
`pipeline_parallel_size=1`, `expert_parallel_size=1`, and
`async_engine=false`. NeMo RL rejects a configured rollout profiler for other
vLLM topologies before allocating generation workers. Parallel and asynchronous
rollouts can move or overlap GPU work outside one profiled actor and do not have
a single begin/end boundary, so they are not supported by this interface.

## Plugin contract

The profiler package must expose a class with the following interface:

```python
from typing import Any


class MyRolloutProfiler:
    def __init__(self, *, rank: int) -> None:
        """Initialize one profiler instance for this generation-worker rank."""

    def begin_engine_initialization(self) -> Any:
        """Open the one-time vLLM engine-initialization window."""

    def end_engine_initialization(self, token: Any) -> None:
        """Close the engine-initialization window identified by token."""

    def begin_rollout(self, *, step_id: int | str) -> None:
        """Start profiling one complete synchronous rollout attempt."""

    def finish_rollout(self) -> None:
        """Finish a successful rollout attempt."""

    def abort_rollout(self, *, reason: str) -> None:
        """Abort an open rollout attempt after a worker or caller error."""

    def close(self) -> None:
        """Validate and release profiler resources during worker shutdown."""
```

NeMo RL creates one profiler instance in each model-owning
`VllmGenerationWorker`, before model loading begins, and passes the worker's
dense Ray rank as a keyword argument. ModelOpt synchronous vLLM workers inherit
the same integration.

`begin_engine_initialization()` and `end_engine_initialization()` wrap
`vllm.LLM(...)` construction. A plugin can use that separate window to observe
one-time engine setup or compiled graph creation without charging it to measured
rollout attempts. NeMo RL passes the exact token returned by
`begin_engine_initialization()` back to the corresponding end call.

The legacy and TransferQueue synchronous GRPO trainers invoke
`begin_rollout()` and `finish_rollout()` around each complete dynamic-sampling
attempt. The `step_id` has the form `stepN/attemptM`. Failures invoke
`abort_rollout()` without masking the original rollout error, and orderly worker
shutdown invokes `close()`.

## Install the plugin

Install the profiler package and its runtime dependencies in the vLLM
generation-worker environment. Installing it only in the NeMo RL driver or a
policy-worker environment is insufficient because rollout generation runs in a
separate Ray actor environment.

The profiler's module must be importable from `python-VllmGenerationWorker`. If
it writes files, their destination must be writable from every selected worker
and mounted at the same absolute path on every node.

## Enable the plugin

Set `NRL_ROLLOUT_PROFILER_CLASS` to the profiler's fully qualified class path
before launching NeMo RL:

```bash
export NRL_ROLLOUT_PROFILER_CLASS=my_profiler.package.MyRolloutProfiler

uv run python examples/run_grpo.py \
  --config examples/configs/grpo_math_8B_megatron.yaml \
  grpo.max_num_steps=3
```

NeMo RL forwards the driver's environment to vLLM actors. The class path can
instead be supplied through `policy.generation.vllm_cfg.env_vars`; that scoped
value takes precedence over the driver environment.

If the configured module or class is missing, its constructor fails, or it does
not implement the required methods, worker initialization fails instead of
silently running without profiling. When the environment variable is unset or
empty, NeMo RL does not import a profiler package and generation behavior is
unchanged.

Profiler-specific capture ranges, rank selection, output paths, and runtime
constraints remain the responsibility of the selected plugin.
