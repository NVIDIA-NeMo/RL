# compat-switch-mutation-test-for-one-path

> When a PR keeps the old code path behind "size 1 / flag off", mutation-test whether any test reaches it before asking to delete it

A PR that adds a batched/new path often keeps the old one behind a setting ("batch size 1 keeps the original per-call path"). Check two things: does the new path at that setting already do the same work, and does any test still reach the old path? Prove the second with mutations, not by reading: (1) route the old setting through the new path, (2) turn off a check that only exists in the old path. If the tests stay green both times, the old path is untested, and "keep one path" becomes a strong finding with its own evidence.

**Why:** On #4508 the shared test fixture had quietly moved to the batched default, so 18 older tests no longer touched the per-call loop. Both mutants left 62/62 tests green. That turned "two loops repeat the same steps" (arguable) into "the old loop is code no test reaches" (not arguable). Deleting it also orphaned a pre-existing sink method, so count the callers that die with the path.
**How to apply:** Run the two mutants on a copy of the tree. In the comment, give the delete list including newly orphaned callees, plus one assert that keeps the old setting covered through the new path (run it: passes after, fails today). Also add a "Context — no action" line when the compatibility setting is not a real escape hatch (both paths run the new code). Related: `proposed-fixes-need-their-own-adversarial-pass` (see [proposed-fixes](../proposed-fixes.md)).
**Seen:** #4508 C2 (team)
