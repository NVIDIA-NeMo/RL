# Code comments and docstrings

How to write the comments and docstrings in a change.

- **A comment states a WHY the code cannot show: do not write 3+ line comment blocks inside function bodies (not on declarations) or comments that narrate the debugging or timings (including on declarations).** That history belongs in the commit message or the PR description. Five long blocks were already wrong before merge on #4052. Source: [flag-long-in-function-comments](../../review-pr-team/references/details/flag-long-in-function-comments.md).
- **A docstring describes what the code is now, not its history or the paths it does not take.** Sentences like "this used to handle X" or "Y never reaches here because …" go stale on the next refactor; that context belongs in the PR description or a review-thread reply.
- **In a refactor, keep a comment whose reason still holds at the new location; move it there with its code.** Check each removed `#` line against where its code moved; a lost "why we do this" comment is expensive to rediscover (#1757). Source: [keep-comments-through-a-refactor](../../review-pr-team/references/details/keep-comments-through-a-refactor.md).
