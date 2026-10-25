# Review threads and pending review

GitHub mechanics not already in Phase 5.

- **`updatePullRequestReview` cannot add a body to a pending review that has none; pass the body at submit (`-F body=@file` on the events endpoint).** (#3592)
- **A pending review is invisible to the author; never credit a later commit to an unpublished comment.** (#3453) Details: [pending-review-is-invisible-to-author](details/pending-review-is-invisible-to-author.md).
- **In `gh api`, only `-F` expands `@file`; `-f body=@x` posts the literal string. Read the body back after every mutation.**
- **Re-run the pre-POST audit on the exact payload file just before POST; use versioned file names so a teammate cannot overwrite your patched file.** (#3566) Details: [audit-the-file-you-are-about-to-post](details/audit-the-file-you-are-about-to-post.md).
- **Before replying to an existing thread, compare `gh api user` with each comment's author.** A colleague's agent-written comment is not ours to "self-correct" (#2957).
