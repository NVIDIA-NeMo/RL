# partial-fix-check-the-sibling-branch

> When a PR fixes one arm of an if/elif, check every sibling arm for the same defect — and check whether the unfixed arm's recipes are in a suite that actually ran, since "it passed CI" often means "that arm was never exercised"

A fix scoped to one branch of a conditional is the highest-yield thing to check in a large PR. The
author's own comment and PR body will usually *name the scope* ("gated models (e.g. Qwen3-MoE)",
"for gated MoEs") — treat that phrasing as a pointer to the arms they did **not** revisit, not as
reassurance. Read the sibling arms and ask whether the root cause discriminates on the same
condition the code does.

Two checks that turn this from a hunch into a finding:
1. **Does the upstream failure key off the branch condition, or off something else?** If the library
   decides behaviour from the *tensor/value itself* (e.g. `is_fused = loaded_weight.dim() == 3`)
   rather than from which mapping entry matched, then the author's branch condition is irrelevant to
   the hazard and every arm producing that shape is affected.
2. **Was the hazard introduced by this change?** Grep the OLD dependency version for the mechanism.
   If it doesn't exist there, the bump/refactor is what makes the unfixed arm dangerous — that moves
   it from "pre-existing, out of scope" to squarely in scope.

Then check coverage before believing green CI: resolve which recipes exercise the unfixed arm and
which suite file they live in. Recipes in `release*.txt` / `performance*.txt` do **not** run in
nightly, so "all tests pass" can coexist with "that arm was never executed." Also check whether the
unit test covering it asserts *behaviour* or only *emitted names* — a name-only assertion keeps
passing through a layout bug.

**Why:** In PR #3280 the gated MoE arm of `_batch_fused_modelopt_moe_weights` was fixed to emit
per-expert 2-D shards; the non-gated arm 13 lines above still emitted the full 3-D `[E, I, K/2]`
tensor under a per-expert name. vLLM 0.25.1 sets `is_fused` purely from `loaded_weight.dim() == 3`,
so the fix's rationale applied identically to both arms. v0.20.0 had no `routed_experts.py` and no
such heuristic, so the bump introduced the hazard. And the only recipes hitting the non-gated arm
lived in `release_gb200.txt`, not nightly — never run for the PR — while the unit test covering that
arm asserted emitted names only and would keep passing. This became the top finding of the review.

**How to apply:** for every `if/elif` the diff modifies exactly one arm of, read the other arms in
full; grep the pinned OLD dependency version for the mechanism to establish scope; and map the
unfixed arm's recipes to their suite file before accepting CI green as evidence. Related:
[compare-guards-with-sibling-bypass-path](compare-guards-with-sibling-bypass-path.md) (same instinct applied to guard lists),
`verify-what-a-raise-guard-measures` (see [does-the-failure-really-happen](../does-the-failure-really-happen.md)).
