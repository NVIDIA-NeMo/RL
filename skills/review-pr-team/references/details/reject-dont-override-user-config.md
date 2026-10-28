# reject-dont-override-user-config

> Code that rewrites or drops a value the user set is a silent ignore; ask for a raise at setup instead

**Why:** after an override, the config says one thing and the run does another, and nothing tells the
user. Overrides also spread: the mismatch they create usually needs a second bypass somewhere else to
keep the run going, and that bypass hides the problem a second time.

**How to apply:**

1. Look for code that changes a configured value for one mode: `replace(params, top_k=None, ...)`,
   forcing a field, zeroing a threshold. Treat it as "the user's setting is silently ignored", even
   when the PR description says it is intended.
2. Ask to revert the override and reject the combination at setup, in the existing validator next to
   the nearest sibling check, following a repo precedent that already raises for a similar case. Guard
   optional blocks (`generation_config is not None`) in the check.
3. Find the second bypass the override forced (a skipped threshold, a relaxed assert) and ask to remove
   it with the override.
4. State it plainly in the comment ("silently ignores ..."); this is not a `nit:`.

**Seen:** #4615 OPD full mode overrode the user's top-k/top-p in the logprob pass while the rollout stayed filtered, which then needed a threshold bypass in the single controller; the ask was to raise in `_validate_opd_full_config` next to the temperature check (team)

Related: `typeddict-missing-key-failfast-is-convention` (see [typeddict-missing-key-failfast-is-convention](typeddict-missing-key-failfast-is-convention.md)).
