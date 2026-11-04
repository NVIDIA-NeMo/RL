# predicted-rebase-hazard-recheck-first

> When a staged review warns about a conflict-free merge that breaks code, and the author then rebases, check that hazard first — it can land exactly as predicted, and the staged comment's "when you rebase" wording is now wrong

If a finding says "this file merges with no conflict during the rebase, and the result is broken", the author gets no conflict
marker to stop them. Warning them in a comment they haven't seen yet (the review is still pending) protects nothing. So when the
head moves under a staged review, re-derive the predicted merge result at the NEW head before anything else. If it landed,
rewrite the comment as a live bug. It can now carry a committable suggestion block, because the broken code is in the PR's diff.
Re-anchor it, don't edit it in place: the old anchor covered only part of the region the fix now spans.

In the same pass, go through every "after the rebase…" finding one by one. Several may now be fixed by the author (delete
those), and prose like "at head X is limited to Y" may have become false.

**Why:** On PR #4191 the round-2 review predicted that `openai_server_utils.py` would auto-merge into an `UnboundLocalError`.
The author rebased and squashed while the review was still staged, and the break landed byte for byte. Meanwhile, two other
rebase findings had been fixed by the author, and one doc comment's supporting sentence had become false.

**How to apply:** on any head move, run the payload diff (the PR's own +/- lines, old vs new base) and re-check every rebase
finding first. Snapshot the live comment bodies (`live_before.json`) before editing, because a session restart can wipe the
scratch folder. Everything must be rebuildable from the live review. See `rebase-edit-staged-comments-in-place` (see [stacks-rebases-and-rereviews](../stacks-rebases-and-rereviews.md)).
