# stacked-pr-check-threads-before-flagging

> Right before staging, fetch every published AND pending comment on this PR, its base PR, its stack siblings, and any review posted since you started (GraphQL — REST hides pending ones), plus the stack's tracking issue, and drop every finding already raised or answered

**The one check:** right before staging (and again before offering to publish), fetch every
existing comment that could already make your point, and dedup each staged finding against it:
- this PR's threads, including author replies;
- the base PR's threads and every merged or open sibling in the stack;
- reviews other people published while the team was running;
- the stack's shared tracking issue.

Phase 1 reads existing reviews once, at the start. A team review runs long enough for all of this to
change underneath it.

**Use GraphQL, not REST.** A PENDING review's comments are invisible to REST:
`gh api repos/NVIDIA-NeMo/RL/pulls/<n>/comments` returns `[]` — not an error, an empty array that
looks exactly like "no threads here". Your own staged review on the base PR is the most likely thing
to be missed, because it is the one still pending.

```bash
gh api graphql -f query='query { repository(owner:"NVIDIA-NeMo", name:"RL") {
  pullRequest(number: <N>) { reviews(last:5) { nodes {
    databaseId state author{login}
    comments(first:100){ totalCount nodes { path line body } } } } } }'
gh pr view <N> --json reviews                       # fresh published reviews
gh api repos/NVIDIA-NeMo/RL/pulls/<N>/comments --paginate --jq '.[].body' | grep -c <symbol>
```

What to do with a match:
- Same ask on the same code already published → drop ours; point at it in the body.
- A concern raised and resolved on the base ("Done in <sha>") → closed, not new.
- Ours proposes a different fix on the same lines → do not post both (two competing suggestions
  leave the author choosing); raise the difference with the user.
- In a stack-wide audit, record "already raised on #N (link)" or "new" per item before drafting.
  Already-raised items become a one-line pointer, never a second full comment.
- Never modify the base PR's pending review: it is the user's staged state.

**"Unused in this PR" may have its consumer in a later PR.** The author may already have said so in
a thread. Check threads first; checking the later PRs' code is
`upstack-fix-kills-the-finding-check-merge-base` (see [stacks-rebases-and-rereviews](../stacks-rebases-and-rereviews.md)).

**After a stacked PR rebases, re-check CI registration both ways.** A runner line
(`run_test ... ./tests/functional/foo.sh`) and its script often live in different PRs. The base
often drops the runner line, to be re-added by the child with the script. Confirm the rebased child
re-added it: `grep -rn <script-name> tests/ .github/` must return a runner, not only the
script. If the head moved under your own staged review, edit the staged comments in place — see
`rebase-edit-staged-comments-in-place` (see [stacks-rebases-and-rereviews](../stacks-rebases-and-rereviews.md)). Do not delete and recreate the review.

**Before filing a new issue or loose end**, read the stack's tracking issue. When the base PR is
still open or conflicting, say in the body that findings come from static review at the current head.

**Why:**
- PR #3219: a staged nit said the `_dispatched_rollouts` comment ("used by downstream drain paths")
  was inaccurate. The author (RayenTian) had already replied in that very thread that #3266 consumes
  it, and #3266's diff showed two readers.
- PR #3371 (base #3369): the REST check returned nothing, and three findings went toward staging.
  All three were already in a PENDING review on #3369 with 10 comments. A fourth staged comment on
  #3369 asked for a test on a function #3371 was about to be told to delete.
- PR #4266 (2026-09-29): a dead-code audit of #4264→#4265→#4266 produced three items; two
  (`attempt_uuid`, `completed_execution_acknowledgements()`) were already published on #4265
  (r4119321795, r4109065749). Devil-advocate caught it.
- PR #3267 (#2819 split): a merge-order concern was rediscovered, but the user had raised it on
  base #3266 and RayenTian had resolved it ("Done in ed283dfcb"). Reading tracking issue #2625 was
  also needed to confirm a real gap (router replay on SC) was new before filing #3327.
- PR #3267 after rebase: the gym SC functional test was registered at
  `L1_Functional_Tests_SingleController.sh:38` before; the base dropped the line and the rebased
  child added the script but not the registration, so the new test ran nowhere.
- PR #4214: another reviewer published a 5-comment review (same data-loss bug, same race, same test
  ask) about 20 minutes into the run. Our 3 inline comments all duplicated it and were deleted by
  hand, leaving only a body-level ask the other reviewer had not covered.

Related: [recommend-one-fix-not-a-menu](recommend-one-fix-not-a-menu.md), [tracking-issue-body-not-comment](tracking-issue-body-not-comment.md),
[deleted-ci-guard-verify-replacement](deleted-ci-guard-verify-replacement.md).

**Seen:** #3219 author already answered in thread (team) · #3371 duplicates of the base's pending review (team) · #4266 duplicates of sibling #4265 (team) · #3267 concern already resolved on base #3266 (user) · #3267 runner registration lost in rebase (team) · #4214 duplicates of another reviewer's fresh review (user)
