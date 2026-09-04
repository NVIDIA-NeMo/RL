# Profile Megatron policy training with a plugin

NeMo RL can drive an optional profiler plugin around complete Megatron policy
training updates. A profiled step includes forward and backward execution,
gradient reduction, and the optimizer step. Evaluation calls, rollout
generation, reward processing, and policy scoring are outside this lifecycle.

## Plugin contract

The profiler package must expose a class with the following interface:

```python
class MyPolicyProfiler:
    def __init__(self, *, rank: int) -> None:
        """Initialize one profiler instance for this distributed rank."""

    def begin_train_step(self) -> None:
        """Start profiling a policy-training step."""

    def finish_train_step(self) -> None:
        """Finish a successful policy-training step."""

    def abort_train_step(self, *, reason: str) -> None:
        """Abort an open step after a worker or caller error."""

    def close(self) -> None:
        """Validate and release profiler resources during worker shutdown."""
```

NeMo RL creates one profiler instance per Megatron policy worker after
distributed setup, passing the worker's distributed rank as a keyword argument.
ModelOpt Megatron policy workers inherit the same integration.

Monolithic `train()` calls invoke `begin_train_step()` and
`finish_train_step()` around each successful, non-evaluation update. The split
training API opens the profiler in `begin_train_step()`, leaves it open across
all `train_microbatch()` calls, and finishes it in `finish_train_step()`.
Worker errors call `abort_train_step()` with a diagnostic reason without
masking the original error. NeMo RL guarantees at most one finish or abort
callback for each begun profiler step, including when a caller invokes the
worker's explicit abort operation after an error. Orderly shutdown calls
`close()`.

## Install the plugin

Install the profiler package and its runtime dependencies in the Megatron
policy-worker environment. Installing it only in the NeMo RL driver environment
is insufficient because training runs in Ray actors with isolated Python
environments.

The profiler's module must be importable from the worker environment. If it
uses output files, their directory must be writable from every selected worker
and mounted at the same absolute path on every node.

## Enable the plugin

Set `NRL_POLICY_PROFILER_CLASS` to the profiler's fully qualified class path
before launching NeMo RL:

```bash
export NRL_POLICY_PROFILER_CLASS=my_profiler.package.MyPolicyProfiler

uv run python examples/run_grpo.py \
  --config examples/configs/grpo_math_8B_megatron.yaml \
  grpo.max_num_steps=3
```

NeMo RL forwards the driver's environment to policy actors. If the configured
module or class is missing, its constructor fails, or it does not implement the
required methods, worker initialization fails instead of silently running
without profiling. When the environment variable is unset or empty, NeMo RL
does not import a profiler package and training behavior is unchanged.

Profiler-specific capture ranges, rank selection, output paths, and runtime
constraints remain the responsibility of the selected plugin.
