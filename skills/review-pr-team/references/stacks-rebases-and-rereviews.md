# Stacks, rebases and re-reviews

Stacked PRs, moving heads and second-round reviews.

- ★ **Right before staging, fetch every published and pending comment on this PR, its base, its stack siblings and the tracking issue (GraphQL, since REST hides pending ones), and drop what is already raised.** Six cases, e.g. a fresh outside review duplicating all three inline comments (#4214). Details: [stacked-pr-check-threads-before-flagging](details/stacked-pr-check-threads-before-flagging.md).
- ★ **Before posting on a stacked PR, check each later PR with `merge-base --is-ancestor` and `git show <upstack-head>:<path>`; if a later PR fixes it, delete the comment, and if none does, say it ships dead through the stack.** (#4207, #3220)
- ★ **When the head moves, re-walk each finding from trigger to symptom at the new head; "the flagged lines are unchanged" proves nothing.** A new `break` two lines above closed the bug on #3380. Details: [reverify-consequences-after-force-push](details/reverify-consequences-after-force-push.md).
- **After a rebase under a staged review, compare only the PR's own +/- lines; prose anchors usually remap (a multi-line suggestion range does not: redo the apply-check in [suggestion-blocks](suggestion-blocks.md)), but prose line numbers, cross-file permalinks and fix snippets go stale; edit in place.** (#3374, #4106)
- **If a staged comment predicted a silent merge break, check it first after the author rebases; it may now be a live bug that can take a suggestion, and then re-anchor the comment on the broken lines instead of editing it in place.** (#4191) Details: [predicted-rebase-hazard-recheck-first](details/predicted-rebase-hazard-recheck-first.md).
- **A review that stalled for days is new work: re-fetch, re-anchor, and read the new commit messages first.** The author had shipped a better fix than the one staged (#4090).
- **On re-review, a fix that meets the substance of an "e.g." suggestion is fixed, not partially fixed.** (#3308)
- **For `pytest.raises(match=...)` in a stack, pick a substring the upstack rewrite keeps and that is unique to this raise.** (#3369)
- **A conflict comment earns its place only by naming what landed on main, especially new CI gates; use `git merge-tree --write-tree`.** (#3352)
- **Before "this starts with no error", read the target branch head and any stacked child PR, not just the diff base.** The target already raised on the bad node count.
