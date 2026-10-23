# grep-the-symbol-not-just-the-lines

> A wrong function name paired with correct line numbers is nearly undetectable — verify the identifier exists, not just that the cite resolves

When a note, brief, or teammate report cites a symbol *and* a file:line, verify **both
independently**: `grep -rn '<symbol>'` to confirm the name exists, and read the line to
confirm it says what the sentence claims. Checking only the line number is not enough.

**Why:** on PR #3591 a prior-session note referenced `validate_nccl_reshard_config`. No such
symbol exists in the tree — the real function is `check_nccl_reshard_refit_support`. But the
line cite beside it (`nccl_reshard_utils.py:736-748`) was *correct* and landed inside that
real function's body. So every reader who spot-checked the citation got a confirmation and
moved on. The wrong name survived **five appearances across four participants and two
sessions**, including through the team lead's own briefing message, and in one instance it
directly caused a real, live bug to be dismissed as unreachable: a lane reasoned "the reshard
transport validates this at setup" from the *name*, when the function that name implied does
not exist and the real one is never called on the single-controller path.

The failure mode is specific and worth naming: **a wrong identifier with right line numbers
defeats the verification we normally do**, because our habit is to click the link and see
whether the code is there. It is. The name is what's wrong.

**How to apply:**
- Before repeating a symbol name from any inherited artifact (handoff, brief, teammate report),
  run `grep -rn '<symbol>'`. If it returns nothing, stop and find the real name.
- This applies hardest to *reachability* arguments — "X is validated at setup", "Y rejects that
  config". Those turn entirely on a specific function actually being called, so confirm both
  that it exists and that it has a caller on the path in question (`grep -rn` for call sites,
  not just the `def`).
- When you inherit a stale artifact, correct it **at source** as well as in the current
  transcript, or the next session inherits it again.
- Related: `verify-permalink-lines-programmatically` (see [permalinks-and-anchors](../permalinks-and-anchors.md)) catches a permalink that does not
  resolve; this catches one that resolves and still misleads. Run both.
- See also `why-is-this-line-here-look-it-up-first` (see [comment-writing](../comment-writing.md)).
