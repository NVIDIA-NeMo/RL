# sibling-fixes-must-use-each-others-names

> When one comment introduces a name (constant, helper, type), every other comment whose fix touches the same value must use that name and link to it — otherwise the review asks for X and re-adds the literal X replaces

Before staging, list every new name a comment's fix introduces (`UNKNOWN_ROLLOUT_ENVIRONMENT`,
a helper, a type). Grep every other staged comment's code blocks for the literal or pattern
that name replaces. Each hit must use the new name, link to the comment that defines it, and
stop being a ```suggestion block (it now has a prerequisite). Also add the hit's line to the
defining comment's list of sites, with a link back.

**Why:** On PR #4068, the tag-constant comment asked for `UNKNOWN_ROLLOUT_ENVIRONMENT` to
replace every hand-typed `"unknown"`, while the seeding comment's suggestion was
`rollout_environment = "unknown"` — a new hand-typed copy. The user caught it. The per-comment
audit cannot see this; it is a property of the set of comments.
**How to apply:** Run it as a cross-comment pass after the editor, alongside the body-refs
check: for each defined name, `grep` the other bodies for the literal it replaces. Related:
`convergence-on-finding-is-not-convergence-on-fix` (see [proposed-fixes](../proposed-fixes.md)), [derive-key-lists-dont-restate-them](derive-key-lists-dont-restate-them.md).
**Seen:** #4068 r4212860753 vs r4075502693 (user)
