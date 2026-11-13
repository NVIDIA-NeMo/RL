# Error Handling

## Use Specific Exceptions

When using try-except blocks, limit the except to the smallest set of errors possible.

**Don't:**
```python
try:
    open(path, "r").read()
except:
    print("Failed to open file")
```

**Do:**
```python
try:
    open(path, "r").read()
except FileNotFoundError:
    print("Failed to open file")
```

## Avoid Meaningless Exception Wrappers

Before adding `try`/`except`/`finally`, identify what the wrapper changes when the operation fails:

- If the original exception already reports the problem clearly and terminates the run, let it propagate. Catching it only to print the same error and re-raise adds no useful behavior. Do not turn an upstream bug into a warning just to delay the crash.
- If the intent is fault tolerance, trace the failure through the caller, training driver and affected workers. Catching a local exception is not recovery when required state is still missing or another worker or the driver will terminate the whole run anyway. Add a handler only when it provides a working recovery path.
- Use `finally` for required cleanup on both success and failure, not as an extra wrapper with no meaningful cleanup to perform.

Keep wrappers that perform required cleanup or add actionable context the original exception lacks, even when the run must still stop. Those have a concrete purpose; they do not provide fault tolerance.

Examples from reviews:

- [#4520](https://github.com/NVIDIA-NeMo/RL/pull/4520#discussion_r4336019725): controller-side `try`/`finally` blocks performed cleanup before the exception still ended the run, with no retry or rollback. An abort RPC could also replace the original error. The fix removed those wrappers while retaining worker-side hook restoration.
- [#2700](https://github.com/NVIDIA-NeMo/RL/pull/2700#discussion_r3537704562): the handler still raised, so the review requested removing `try`/`except` until actual fault tolerance was supported.

## Fail Loud, Not Silent

When adding a config option or branch, ask: what does the worst plausible misconfiguration do? If it silently produces *wrong results* rather than an error, add a setup-time `assert`/`raise` so it fails loudly at startup instead of corrupting a run.

**Examples of silent-wrong worth guarding:**
- An advantage estimator that needs a real `prev_logprobs` while a loss flag zeroes it (advantage degrades to garbage with no error).
- An agent routed to a teacher alias that has no worker group (cryptic `KeyError` mid-run instead of a setup-time check).
- A backend/quantization setting silently ignored for a sub-component.

Prefer failing at `setup()` time over a deep-in-the-loop crash; prefer a crash over silent garbage. If you truly can't validate, surface a logged warning rather than nothing.
