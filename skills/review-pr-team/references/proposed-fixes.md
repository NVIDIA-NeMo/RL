# Proposed fixes

A true finding with a harmful fix is worse than no finding.

- ★ **Test every proposed fix on the runs that pass today, including the PR's own tests on a scratch copy with all suggestions applied.** An EXIT trap turned green shards red and `coverage -p` wiped data (#4216); seven cases so far.
- **Check that every variable a fix reads is assigned before the insertion point and not reset each loop.** Three lanes proposed the same guard that could never fire (#2884).
- **Before rejecting a fix because "it leaves B broken", check B under every candidate fix.** The test was red under all of them (#4020). Details: [dont-disqualify-a-fix-with-a-pre-existing-failure](details/dont-disqualify-a-fix-with-a-pre-existing-failure.md).
- **Before suggesting try/except, list the inputs that raise; if every one means the data is already broken, crashing is right.** (#3748) Details: [dont-propose-a-swallow-if-every-trigger-is-a-bug](details/dont-propose-a-swallow-if-every-trigger-is-a-bug.md).
- **Compare a degraded mode to the real no-knob baseline, not to the full feature; a timeout without recovery beats an unbounded hang.** The finding was a docs ask (#3591).
- **When other callers delete a key, read what they did with it first; copy the use, not just the cleanup.** A bare `pop` would have dropped a metric four loops report (#3883).
- **Prefer a guard that checks the thing directly (stat the file) over negating a string prefix.** (#4020)
- **Before widening a type to `Optional`, check whether the value can be None there; often the honest fix is narrowing the producer.** (#3367) Details: [widening-a-type-can-be-the-wrong-direction](details/widening-a-type-can-be-the-wrong-direction.md).
- **A design sketch must define every helper it calls and drop every field the prototype does not need.** Five dicts became `Metric(values, reduce)` (#4068).
