# Comments, docstrings and docs

How to write the comments, docstrings and docs in a change.

- **A comment states a WHY the code cannot show: do not write 3+ line comment blocks inside function bodies (not on declarations) or comments that narrate the debugging or timings (including on declarations).** That history belongs in the commit message or the PR description. Five long blocks were already wrong before merge on #4052. Review Details: [flag-long-in-function-comments](../../review-pr-team/references/details/flag-long-in-function-comments.md).
- **A docstring describes what the code is now, not its history or the paths it does not take.** Sentences like "this used to handle X" or "Y never reaches here because …" go stale on the next refactor; that context belongs in the PR description or a review-thread reply.
- **In a refactor, keep a comment whose reason still holds at the new location; move it there with its code.** Check each removed `#` line against where its code moved; a lost "why we do this" comment is expensive to rediscover (#1757). Review Details: [keep-comments-through-a-refactor](../../review-pr-team/references/details/keep-comments-through-a-refactor.md).
- **A new comment that explains a config key needs that key to exist: grep the pinned upstream for it.** On #3785 the comment's conclusion was right, but the key it relied on had never existed.
- **After a rename, move or deletion, grep comments, docstrings and `docs/` for the old name or the old behaviour, including files the diff does not touch.** On #3724 `docs/guides/refit.md`, which the diff did not touch, still described a refit guard the PR had deleted. Review Details: [diff-removals-are-a-work-list](../../review-pr-team/references/details/diff-removals-are-a-work-list.md).
- **One feature gets one name across every doc surface: doc H1, `docs/index.md` card, README and file name.** After a rename, grep for the old name; one PR shipped three names for one feature (#2823).
- **A user guide section documents only what changes for the reader, as tables of new and renamed keys.** (#4068)
