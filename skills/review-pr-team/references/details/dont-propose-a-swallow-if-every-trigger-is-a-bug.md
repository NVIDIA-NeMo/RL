# dont-propose-a-swallow-if-every-trigger-is-a-bug

> Before suggesting try/except (or a warning) around a call, list the inputs that actually raise — if every one means the data upstream is already broken, crashing is correct and catches the bug earlier; a sibling's try/except is only precedent if it faces the same situation

# Before proposing try/except around a call, enumerate what actually raises

**Caught by the user, PR #3748 (`nemo_rl/utils/logger.py:193`).**

That PR turned `TensorboardLogger.log_histogram` from a no-op `return` into a real
`self.writer.add_histogram(...)`. I filed a suggestion to wrap it in
`try/except Exception` + a printed warning, arguing "a logger shouldn't kill a run it
only observes," and cited two siblings in the same file as precedent.

The user pushed back: *"if it failed we should crash b/c the code is buggy."* They were
right. I had confirmed the exact raising inputs — all-NaN, all-inf, `.size == 0` — and
every single one means the upstream data is already broken. There is **no legitimate
input that raises**: values below the smallest bin edge still land in the bin bounded by
`0`, and identical values bin fine. NaN advantages reaching the histogram means NaN into
the optimizer one step later, so crashing at the logger catches it *earlier*, not
spuriously.

My precedent argument was also weaker than I made it sound, and I should have noticed
while writing it:

- `log_metrics` catches around `add_scalar` because it loops over **hundreds of
  heterogeneous metrics** — one bad key shouldn't kill the other 200. A single dedicated
  call is not that situation.
- `WandbLogger.log_histogram` doesn't swallow either; it **falls back** to logging a
  scalar. By the user's own standard that fallback is arguably the buggy one.

Neither sibling was "swallow and continue," which is what I proposed.

## The rule

Before suggesting `try/except` around a call, **enumerate the inputs that actually raise**
and classify each one:

- **Every trigger is a real bug upstream** -> do NOT propose a swallow. Crashing is
  correct, and it is usually catching the bug *closer to its source* than whatever would
  have failed next. At most, propose a clearer error message or an assert that names the
  offending key — never a silent warning.
- **Some trigger is legitimate input** -> now you have a finding, and the guard belongs at
  the boundary where that input enters, not wrapped around the writer.

Corollary on precedent: "the sibling has a try/except" is only an argument if the sibling
faces the *same* situation. A loop over many heterogeneous items has a real reason to
swallow one failure; a single call does not. Check what the sibling is actually defending
against before citing it, and check whether it swallows or falls back — those are
different designs.

Corollary on this repo specifically: NeMo-RL's own guidance leans **fail-loud** (see the
"Fail-loud check" in the review skill, and `typeddict-missing-key-failfast-is-convention.md`).
A suggestion that converts a hard failure into a printed warning is swimming against the
repo's convention and needs a much stronger justification than "logging shouldn't crash."

Tell that I was rationalizing: I wrote "I could not construct an everyday config that
triggers this" **into the comment itself** and staged it anyway. That sentence is the
finding failing its own test — if no real config reaches it, the honest move is to drop
it, not to ship it with a hedge attached.

See also `fail-fast-is-not-strictly-worse.md` (same family: compare against the real
baseline, not against an idealized graceful mode) and
`dead-guard-is-not-a-finding.md`.
