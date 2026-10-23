# apply-the-suggestion-at-its-own-anchor

> Before staging AND again before publishing after any head move, print the exact lines each suggestion block would replace at the current head, apply it, and diff — an anchor one line too long, or one GitHub remapped after a push, deletes real code and still parses

**The one check:** for every ```suggestion, print the exact lines it will replace at the CURRENT
head next to the replacement text, then apply it and compare. Reading the block and thinking "that
looks right" does not catch a wrong range.

Make it mechanical, as an `applied` column in the pre-POST audit:
- Rebuild the file as `lines[:start-1] + suggestion + lines[end:]`, `ast.parse` it, and FAIL if any
  function, class or `return` disappears that you did not declare as expected.
- For an insert-only block on a blank line, also run `ruff format --check` on the rebuilt file:
  the anchored blank line plus the block's own leading/trailing blanks can add up to 3 blank lines.
- Declare intentional removals explicitly (a rename drops the old test name) rather than turning the
  check off.
- A permalink range cited by a lane is NOT an anchor range. Permalinks often include a trailing line
  for context. Read the last anchored line yourself (`sed -n '<end>p'`).
- The same check catches a new symbol anchored on an existing symbol's line, where the suggestion
  swaps a function header and silently adopts the old body.

**Run it again whenever the head moves between staging and publishing.** GitHub re-anchors pending
comments onto the new head, and a multi-line comment's start and end can move by different amounts.
A suggestion written for the old lines then "replaces" whatever sits in the new range, which can be
code the author just added. After a head move: (1) `git diff old..new --stat` to see which anchored
files changed, (2) the replace-these-lines check, (3) grep every comment and the review body for
names the author renamed or removed, (4) only then publish.

**Why:**
- PR #4112: a suggestion replacing a `try/except` was about to be staged at lines 197-**203** of
  `rollout_reassembler_actor.py`. Line 203 is `return actors`. Both ranges parse:

      anchor 197-202: parses OK | non-bare returns in factory: 1
      anchor 197-203: parses OK | non-bare returns in factory: 0   <-- return deleted

  No syntax error, no lint failure. `create_rollout_reassembler_actors` would return `None`, and the
  crash would land later at `single_controller.py:311` (`list(None)`). The range came from a lane's
  permalink (`#L197-L203`) that I copied without checking line 203.
- PR #4266 (2026-09-29): the stack was rewritten minutes before publishing. The vLLM-guard
  suggestion was remapped from `setup.py` L1118-1119 to L1112-1118, which now held the author's
  moved `snapshot_attempt_interval_s` check; one click on "Commit suggestion" would have deleted it.
  The same rewrite renamed the Gym flags to `rollout_checkpointing.gym.mode`, which made four
  comments and the body stale and showed a new bug: the PR's own test script and docs still used the
  old keys.

- PR #4670 (2026-10-14): a lane's verified test block, anchored on blank L769, carried two trailing
  blank lines; with L770 also blank that is 3 before the next `def`, and `ruff format` failed. The
  lane had tested a different layout. Rebuilding the file from the suggestion caught it.

Related: `verify-permalink-lines-programmatically` (see [permalinks-and-anchors](../permalinks-and-anchors.md)), `suggestion-fence-must-target-the-anchored-file` (see [suggestion-blocks](../suggestion-blocks.md)),
`rebase-edit-staged-comments-in-place` (see [stacks-rebases-and-rereviews](../stacks-rebases-and-rereviews.md)), [reverify-consequences-after-force-push](reverify-consequences-after-force-push.md).

**Seen:** #4112 suggestion range deleted `return actors` (team) · #4266 remapped suggestion over the author's moved check (team) · #4670 insert-only test block left 3 blank lines (team)
