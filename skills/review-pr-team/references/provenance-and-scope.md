# Provenance and scope

Decide what this PR actually changed and whether the ask belongs to this author.

- **`git fetch` and derive the base with `git merge-base` before any "PR-introduced" claim; a stale base makes every provenance check confirm the wrong answer.** On #3837 another author's merged PR sat inside the diff; a commit ending `(#NNNN)` in the range is the tell.
- **For "sibling A has it, B doesn't", diff each new file against its own parent, not against each other.** Adding the assert on #3652 would have made the MXFP8 gate stricter than its BF16 baseline.
- **For a half-old, half-new finding, say in the first line which part is old and which is new, and that the ask covers only the new part.** (#4326)
- **Admit an out-of-diff finding only if the diff created the exposure: would it still break if the diff were reverted?** If yes, file an issue instead (#4216).
- **Before "the migration missed one", find the commit that created the line (`git log --diff-filter=A`).** A `_v2=false` pin was deliberate gemma3 v1 coverage (#4196). Details: [leftover-pin-archaeology-before-calling-it-drift](details/leftover-pin-archaeology-before-calling-it-drift.md).
- **Check the PR description's concrete claims with `git log -S`; zero commits means the body describes code that never existed.** (#2957)
- **Run `git show <sha> --stat` on every commit the PR body credits; after a rebase it may be comments only, and its before/after numbers then prove nothing.** (#3280)
- **For a de-duplication PR, grep current main for a new copy; report it as FYI, not as an ask.** (#3087) Details: [stale-branch-check-main-for-parallel-duplication](details/stale-branch-check-main-for-parallel-duplication.md).
- **When a PR drops an old on-disk format, size the affected users from the last release's code default plus its shipped configs.** Default DPO/RM runs at v0.7.0 wrote v1 checkpoints (#4273). Details: [dropped-format-check-previous-release-defaults](details/dropped-format-check-previous-release-defaults.md).
- **Out-of-scope gaps go into the tracking issue's body checklist, plus an FYI inline comment; pick a tracking issue whose scope fits.** (#3295) Details: [tracking-issue-body-not-comment](details/tracking-issue-body-not-comment.md).
