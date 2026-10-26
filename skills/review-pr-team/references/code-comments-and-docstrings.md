# Code comments and docstrings

What to check in the comments and docstrings a PR adds, edits or drops.

- **A comment states a WHY the code cannot show: flag 3+ line comment blocks inside function bodies (not on declarations) and comments that narrate the debugging, timings or library internals (including on declarations), listing each with a one-line replacement.** That history belongs in the commit message or the PR description. Five long blocks were already wrong before merge on #4052. Details: [flag-long-in-function-comments](details/flag-long-in-function-comments.md).
- **A docstring describes what the code is now, not its history or the paths it does not take.** Sentences like "this used to handle X" or "Y never reaches here because …" go stale on the next refactor; that context belongs in the PR description or a review-thread reply.
- **A comment dropped in a refactor whose reason still holds at the new location should be restored there.** Check each removed `#` line against where its code moved; a lost "why we do this" comment is expensive to rediscover (#1757).
