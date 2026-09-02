# Profile vLLM rollouts with a plugin

NeMo RL can drive an optional profiler plugin around complete vLLM rollout
attempts. One profiled attempt includes every generation turn and
`finish_generation()`. Reward processing, policy scoring, validation rollouts,
and policy training are outside this lifecycle.

The integration supports `tensor_parallel_size>=1`,
`pipeline_parallel_size=1`, and either `expert_parallel_size=1` or
`expert_parallel_size=tensor_parallel_size` with synchronous or asynchronous
vLLM engines. NeMo RL rejects a configured rollout profiler for unsupported
pipeline- or expert-parallel topologies before allocating generation workers.

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
        """Start profiling one complete rollout attempt."""

    def finish_rollout(self) -> None:
        """Finish a successful rollout attempt."""

    def abort_rollout(self, *, reason: str) -> None:
        """Abort an open rollout attempt after a worker or caller error."""

    def close(self) -> None:
        """Validate and release profiler resources during worker shutdown."""
```

For a synchronous TP1 engine, NeMo RL creates the profiler in the model-owning
`VllmGenerationWorker`. For TP>1 or an asynchronous engine, NeMo RL selects a
generic vLLM worker subclass and creates one profiler inside every GPU worker.
The latter is necessary because the outer NeMo RL actor only coordinates the
engine and does not launch its GPU kernels. Every profiler receives a unique,
dense rollout rank formed from the owning NeMo RL worker's rank and the internal
vLLM worker rank. The built-in NIXL vLLM worker is supported; another custom
`vllm_kwargs.worker_cls` cannot currently be combined with rollout profiling.
Synchronous TP1 ModelOpt workers inherit the outer-actor integration. ModelOpt
modes that replace vLLM's internal GPU worker are not currently supported by
the TP/async path.

`begin_engine_initialization()` and `end_engine_initialization()` wrap
vLLM worker and engine startup through model warmup and compiled graph creation.
A plugin can use that separate window to observe one-time setup without charging
it to measured rollout attempts. NeMo RL passes the exact token returned by
`begin_engine_initialization()` back to the corresponding end call.

The legacy and TransferQueue synchronous GRPO trainers invoke
`begin_rollout()` and `finish_rollout()` around each complete dynamic-sampling
attempt. The `step_id` has the form `stepN/attemptM`. Failures invoke
`abort_rollout()` without masking the original rollout error, and orderly worker
shutdown invokes `close()`.

## Install the plugin

Install the profiler package and its runtime dependencies in the vLLM
generation-worker environment. For TP>1 and asynchronous engines, it must also
be importable by vLLM's internal GPU-worker processes. Installing it only in the
NeMo RL driver or a policy-worker environment is insufficient.

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
