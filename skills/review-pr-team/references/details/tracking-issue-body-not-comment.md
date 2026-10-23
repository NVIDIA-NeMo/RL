# tracking-issue-body-not-comment

> Cross-cutting gaps found during review go into the tracking issue's BODY bullet list (not an issue comment) + a non-actionable FYI inline comment on the PR

When a PR review surfaces a gap that belongs to a broader effort with a tracking issue (e.g.
#2625, the Single Controller / Async RL cleanup tracking issue), the disposition the user wants is:
(1) edit the tracking issue's BODY, adding a `- [ ]` bullet to the known-missing-items list in
the house style — bold title, evidence permalinks at pinned SHAs, cross-reference related items
in the same list, and a "(Flagged during #NNNN review)" suffix; and (2) add an inline review
comment on the PR anchored at the relevant diff line, explicitly framed as FYI-only ("nothing
to do in this PR") pointing at the tracking issue. Do NOT post the gap as an issue comment.

**Why:** In the PR #3295 review, the SC-port gap (use_generation_logprobs_as_prev exists only in
legacy grpo_train, not grpo_sync.py/single_controller.py) was first posted as a comment on #2625;
The user asked to delete it and put it in the body's bullet list instead, plus an FYI review comment
on the PR. Checkboxes in the body are the actionable ledger (items get checked off or split into
issues); comments scroll away.

**How to apply:** fetch the full issue body (`gh issue view N --json body`), insert the bullet
before the trailing "Add anything else here as it comes up." anchor, `gh issue edit N --body-file`;
then add the PR comment (via addPullRequestReviewThread if the review is still pending). Related:
[stacked-pr-check-threads-before-flagging](stacked-pr-check-threads-before-flagging.md) (check the tracking issue BEFORE filing anything new).

**Scope the tracking issue to the feature's actual applicability.** Follow-up correction on the
same #3295 item: the feature is sync-only by nature (incompatible with async's required IS
correction), so it did NOT belong in #2625 ("[Single Controller / Async RL] cleanup tracking
issue") at all — the user had the bullet removed and a companion tracking issue created for the sync
dataplane (#3336, "[Single Controller / Sync RL] cleanup tracking issue", seeded with the
grpo_sync.py port item), with the PR's FYI comment repointed. Before filing into a tracking
issue, check the issue's scope line against where the feature can actually apply; create a
sibling tracking issue (mirroring the house style) when none fits.
