# stale-branch-check-main-for-parallel-duplication

> For a PR that de-duplicates logic, diff its branch point against current main — a parallel workstream may have grown ANOTHER copy while the PR sat, which strengthens the PR and belongs as an FYI, not an action

When reviewing a PR whose value is "extract duplicated logic into one helper", check how far
behind `main` the branch is (`git log --oneline HEAD..origin/main`) and grep current `main` for
the predicate being consolidated. In a repo with several parallel workstreams (NeMo-RL's
single-controller / data-plane stacks land in rapid sequence), a PR that sits for weeks often
finds that another stack has independently grown a fresh copy of the same logic.

Frame that as **FYI / tracking-issue**, not as an action for this author: the other copy is in a
file the PR does not touch, owned by a different workstream, so demanding a rebase-plus-refactor
is scope creep. It is still worth saying, because it is direct evidence the extraction was the
right call.

**Why:** PR #3087 extracted `_resolve_logprob_skip_flags` from `grpo_train`/`async_grpo_train`
and added `grpo_train_sync` as a third consumer. The branch was 5 commits behind `main`, two of
which were the SC stack; #3266 had landed a *fourth* copy in `SingleControllerActor.__init__`
(`_policy_logprobs_required` / `_reference_logprobs_required`), in inverted `_required` polarity,
re-inlining `opd._skip_prev_logprobs` rather than calling it — plus the same
`if policy_required or reference_required:` guard around `prepare_for_lp_inference` that #3087
adds for `grpo_sync`. The user's instinct to check `main` surfaced this; my first write-up
over-reached with "**Action:** rebase and point SC at the helper" and included a `self.`-prefixed
snippet on a comment anchored in `grpo.py`, which read as a change to the module-level function
under review. The user caught both: the module-level `grpo.py`/`grpo_sync.py` code was correct as-is,
and the right disposition was "FYI, no action needed, heads-up that this differs from the SC path".

**How to apply:** (1) run `git log --oneline HEAD..origin/main` early and read the titles for
adjacent workstreams; (2) grep `origin/main` (not the PR head) for the consolidated predicate and
for whether previously-dormant code has since been wired — SC gained
`examples/run_grpo_single_controller.py` on `main` while #3087 sat, invalidating a
"no production entrypoint" claim made from the PR head; (3) keep cross-file observations as FYI
with a pointer to the tracking issue (#3336 for SC/sync cleanup); (4) when a snippet targets a
DIFFERENT file than the comment's anchor, say so in the snippet itself
(`# single_controller.py on main — not this PR's code`) or omit it. Related:
[tracking-issue-body-not-comment](tracking-issue-body-not-comment.md), `upstack-fix-kills-the-finding-check-merge-base` (see [stacks-rebases-and-rereviews](../stacks-rebases-and-rereviews.md)).
