---
name: error-handling
description: Error handling guidelines for NeMo-RL. Covers exception specificity, minimal try bodies, and else blocks.
when_to_use: Writing or reviewing exception handling; 'try-except', 'catch all exceptions', 'bare except', 'how to handle errors', during code review.
---

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

## Fail Loud, Not Silent

When adding a config option or branch, ask: what does the worst plausible misconfiguration do? If it silently
produces *wrong results* rather than an error, add a setup-time `assert`/`raise` so it fails loudly at startup
instead of corrupting a run.

**Examples of silent-wrong worth guarding:**
- An advantage estimator that needs a real `prev_logprobs` while a loss flag zeroes it (advantage degrades to
  garbage with no error).
- An agent routed to a teacher alias that has no worker group (cryptic `KeyError` mid-run instead of a
  setup-time check).
- A backend/quantization setting silently ignored for a sub-component.

Prefer failing at `setup()` time over a deep-in-the-loop crash; prefer a crash over silent garbage. If you
truly can't validate, surface a logged warning rather than nothing.
