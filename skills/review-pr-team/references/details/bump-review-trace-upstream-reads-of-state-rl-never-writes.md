# bump-review-trace-upstream-reads-of-state-rl-never-writes

> On a Megatron-Bridge/MLM submodule bump, sweep the upstream commits for NEW reads of state that NeMo-RL never writes (train_state counters, config fields) — green L1 CI cannot catch these, because nothing in CI resumes

A submodule bump's worst bugs are not signature breaks (the import sweep finds those, and CI fails on them).
They are upstream code that starts READING a piece of shared state that only upstream's own training loop
keeps up to date. NeMo-RL drives Bridge from its own loop, so those fields sit at their defaults, and the new
read silently uses the default.

**Why:** On #4363 (Bridge 1f8873b→ec83553), Bridge d4ad7a02 started setting
`scheduler.num_steps = train_state.consumed_train_samples` on checkpoint load when
`override_opt_param_scheduler` is true. NeMo-RL never writes `consumed_train_samples` (it ticks the scheduler
itself), so every resume restarted the LR schedule from warmup — no error, all CI green, because no CI lane
resumes. The clue was in the PR itself: the same Bridge commit forced a test mock to grow `fp8_param_gather` /
`fp4_param_gather`, i.e. a new read of fields the mock did not have.

**How to apply:** For every bump, have the submodule expert list, per upstream commit that touches
checkpoint load/save, scheduler, or train state: which attributes it newly READS, and then `grep -rn <attr>
nemo_rl/` for a writer. Zero writers + a reachable code path = a finding. Treat any test-mock field the bump
PR had to add as a pointer to exactly such a new read, and trace what else that commit changed. State the
symptom only as far as it was reproduced (e.g. scheduler-level CPU repro), per
`dont-invent-the-symptom-you-couldnt-run` (see [verify-external-behavior](../verify-external-behavior.md)).
**Seen:** #4363 MB-F1 (team)
