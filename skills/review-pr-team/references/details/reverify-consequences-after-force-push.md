# reverify-consequences-after-force-push

> After the head moves (force-push, rebase, new commit), re-derive each finding's failing case end to end at the new head — read the whole function for a new early exit, re-check each consequence and each sync point; "the flagged lines are unchanged" is not a verification

**The one check:** when the head moves, re-walk each finding from trigger to symptom at the new head.
Grepping the flagged lines and seeing them unchanged proves nothing. Fixes often land somewhere
else: a new guard above the buggy code, or a hardened consequence below it.

Re-derive, in this order:
1. **Read the whole enclosing function at the new head**, not a grep window around the anchor.
2. **Look for a new `break` / `return` / `raise` / `continue` on the path into the flagged code.**
   Authors often fix a bug by rejecting the bad state at the top of the block, because that is
   smaller and safer than repairing it at the bottom. So the flagged region stays byte-identical
   while the defect is gone.
3. **Re-check each consequence separately.** A finding is a chain: trigger → interleaving →
   consequence. Authors often harden the consequence (a rollback, TaskGroup / structured
   concurrency, passing values instead of re-looking them up) while the trigger survives. Then the
   finding stands at lower severity and must be reworded to credit the fix.
4. **Re-check each reachability claim against the sync points.** "Also reachable with slow X" needs
   an interleaving that accounts for every await, drain, barrier and gate. A per-step drain can make
   the slow path impossible. Don't reason from timing intuition.
5. **Re-run the arithmetic or the saved simulation** against the new control flow. Cheapest check,
   and decisive.
6. Only then report "still valid". If you cannot run it, say "statically verified" — never imply
   execution.

The usual signals do not save you. GitHub re-anchors pending comments, so a stale comment still
*looks* well placed. The fixing commit may have a generic title (`fix(sft): validate direct packed
inputs`) that does not name your finding. Getting it wrong costs three things: you tell the user a
merge-blocker is live, you leave a comment asking for work already done, and any public explainer
page now describes a bug that no longer exists.

**Why:**
- PR #3220: the evict-unready/commit-race comment claimed orphaned DP rows plus a "slow rollouts"
  reachability path. After the rebase, `commit` had gained a BaseException rollback that clears
  written rows (consequence fixed), and re-tracing `_train_pump` showed the per-step
  `_sync_weights` drain makes the slow-rollout path impossible. The only live trigger was a separate
  bug (unwired `set_weight_version`). The user asking "did you confirm it?" is what caught it.
- PR #3380: the truncation branch in `megatron_sft_packed.py` was unchanged between heads, but a new
  `if len(pack_tokens) == pack_length: break` two lines above it closed the exact-fill case. I had
  told the user the blocker was still live, left a stale comment, and published an explainer page
  for a fixed bug.

Related: [partial-fix-check-the-sibling-branch](partial-fix-check-the-sibling-branch.md) (the mirror case: a fix lands in one arm, not its
siblings), `stalled-review-rereview-the-new-head` (see [stacks-rebases-and-rereviews](../stacks-rebases-and-rereviews.md)), `upstack-fix-kills-the-finding-check-merge-base` (see [stacks-rebases-and-rereviews](../stacks-rebases-and-rereviews.md)),
[deleted-ci-guard-verify-replacement](deleted-ci-guard-verify-replacement.md).

**Seen:** #3220 consequence fixed by rollback, slow path blocked by drain (user) · #3380 new early `break` above unchanged code (team)
**Seen:** #4607 head moved 62647d9→2f6b955 between preview and staging (rebase + host-memory rework + lens pin to main): re-checking made one finding worse (stale snapshot), fixed one (dead keys), and exposed a new false rationale (team)
