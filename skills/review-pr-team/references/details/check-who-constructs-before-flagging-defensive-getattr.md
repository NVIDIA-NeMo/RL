# check-who-constructs-before-flagging-defensive-getattr

> Before asking to remove or "standardize" a defensive getattr/hasattr, list every way the attribute can be missing — who builds the object (object.__new__ in tests skips __init__), the guarded variable's declared type (hasattr(None, ...) is False, so on an Optional it is also the None check), enclosing None checks, and short-circuit order

A `getattr(self, "_x", False)` next to a plain `self._y`, or a `hasattr(obj, "method")` after an ABC
now supplies the method, looks like leftover clutter. **The one check:** before saying it is
unneeded, list every way the attribute can be missing at that site. If any is real, the guard is
doing work and the finding goes.

1. **Who builds the object?** Grep for who *constructs* the class, not just who subclasses it.
   `object.__new__(Cls)` in a unit test skips `__init__` and sets only what the test needs. The
   `getattr` default is what keeps that test working.
   `grep -rn "object.__new__\|__new__(.*<ClassName>" tests/` and `grep -rn "<ClassName>(" `.
2. **What is the guarded variable's declared type?** If it is `Optional[...]`, `hasattr(None, "m")`
   is `False`, so the `hasattr` is also the None check. Removing it turns a quiet skip into
   `AttributeError: 'NoneType' object has no attribute ...`.
3. **Is there really a None check around it?** Read the enclosing block. Don't assume one exists.
4. **What is the evaluation order?** In `if getattr(self, "_x", False) and not self._y:` the plain
   access runs only when the first term is True. Defensive on the guard, plain on the guarded term
   is correct, not inconsistent.
5. **Does the guard mean the same thing at every site?** If not, one edit for all of them is wrong.

If the only safe version of the ask is swapping `hasattr` for `is not None` — same behavior, in files
outside the diff, no live bug — drop it. It is a guard-for-guard swap the author can do nothing
useful with.

**Why:**
- PR #3490: I staged a nit to replace `getattr(self, "_colocated_reshard_eligible", False)`
  (megatron_worker.py:413) with plain access, arguing the mixin's one host class always sets it.
  `tests/unit/models/policy/test_megatron_worker.py:646` builds the mixin with
  `object.__new__(MegatronGenerationMixin)`, sets neither reshard attribute, and calls
  `prepare_for_generation()`. My suggestion would have broken a green test. Design-reviewer proposed
  it, devil-advocate endorsed it, and it passed my audit; I had checked `__getstate__` but never who
  instantiates the class. Deleted before publishing.
- PR #3883: the PR moved `snapshot_step_metrics` / `get_step_metrics` onto `GenerationInterface`
  with no-op defaults. Design-reviewer filed a finding (score 48) to delete five surviving `hasattr`
  guards, with an action line saying "keep the existing `is not None` checks". There were none:
  `policy_generation` is `Optional[GenerationInterface]` (grpo.py:2816, :4197). Devil-advocate caught
  it. The guard also meant three different things across the five sites: a real None guard
  (`grpo.py:3240`/`:4875`, `grpo_sync.py:788`), already redundant (`sync_rollout_actor.py:227`, after
  `if self.policy_generation is not None:` at 226), and always true (`ppo.py:1755`, because
  `ppo.py:1221-1223` replaces a None `policy_generation` with `policy`).

Related: `dead-guard-is-not-a-finding` (see [true-but-not-a-finding](../true-but-not-a-finding.md)) (a guard that looks pointless usually is not),
`grep-the-idiom-before-calling-a-convention-wrong` (see [true-but-not-a-finding](../true-but-not-a-finding.md)),
`challenge-the-challenger-das-premises-need-verifying-too` (see [lane-adjudication](../lane-adjudication.md)).

**Seen:** #3490 getattr kept a test's `object.__new__` mixin working (user) · #3883 hasattr was the None check on an Optional (team)
