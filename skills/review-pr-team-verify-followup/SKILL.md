---
name: review-pr-team-verify-followup
description: After a published PR review, check the author's fixes, scan the new commits for major issues, close fixed threads, stage follow-ups
user-invocable: true
argument-hint: <pr-number> [owner/repo]
---

# Verify follow-up on a published review

Run this when the author has pushed fixes after your review. It checks every open thread you
opened against evidence, closes the ones that are cleanly fixed, and turns partial fixes into
short follow-up comments the author can resolve with one click. It also checks the new commits for
major issues they introduced, since nobody has reviewed them yet. It works for reviews from
`/review-pr-team` or by hand.

Default repo: `NVIDIA-NeMo/RL`. "You" / "the reviewer" = the `gh`-authenticated user
(`gh api user --jq .login`).

## Rules that apply to every step

- **Confirm before every GitHub write.** Show exactly what will be posted, then ask. One
  outward-facing action per question; if the user's "yes" lists steps, do only those steps.
- **Evidence, not the author's word — but read the word first.** A reply saying "fixed" proves
  nothing on its own. Still read every reply before judging: it tells you which fix they chose, where
  it lives, and what they skipped and why. Use it as a map, then check the code at the new head.
- **Enumerate with GraphQL**, never REST (REST under-reports pending and threaded comments).
- **One pending review per user per PR.** If one exists, it may hold the user's own drafts: add to
  it, never delete it, and re-read it live before every change. A comment you don't recognize is
  the user's. A comment that disappeared was deleted on purpose.
- **Anchor new comments against the CURRENT head** (`addPullRequestReviewThread` reads `line`
  against the current head, not the review's commit).
- **SHAs in replies are the full 40 characters, plain text, no backticks**, so GitHub auto-links them.
- **Never merge** anything.
- Plain, short words. No: load-bearing, non-trivial, orthogonal, leverage, semantics, canonical,
  idempotent, delta, by construction.

## Workflow

### 1. Collect the state

1. PR head, base, commits, CI status:
   `gh pr view N --repo R --json headRefOid,baseRefName,commits,mergeStateStatus` and `gh pr checks N --repo R`.
2. Your review threads with GraphQL: for each thread, `id`, `isResolved`, `isOutdated`, `path`, `line`,
   and its comments (`databaseId`, `author.login`, `body`, `url`, `originalCommit.oid`). Keep threads
   whose first comment is yours and `isResolved` is false.
3. Whether you already have a PENDING review on the PR (`reviews(states: PENDING)`). If so, read it
   live and remember its comment IDs: they are off-limits unless the user says otherwise.
4. The reviewed commit = the review's `commit_id` (or each thread's `originalCommit`).

### 2. Validate each thread

For each thread, do these four things **before** looking at diffs. Skipping them is how a quick
grep sweep produces wrong verdicts in both directions (on #3380 one pass called a fixed thread "not
addressed" and an unmet ask "fixed").

1. **Read the author's reply in the thread.** It usually names the fix they picked and the file:line,
   or explains why they skipped part of the ask. A reply that pushes back is input to your verdict,
   not noise to ignore.
2. **Re-read your original comment and list every fix it offered** ("(a) … or (b) …", "a script —
   or a description"). Check each one. Finding that option (a) is absent proves nothing if they did (b).
3. **Check the problem, not the line you flagged.** Ask "can two packed datasets still collide
   silently?", not "is `task_name` still a constant?". An unchanged flagged line does not mean
   unfixed: the fix often lands somewhere else (a guard above it, a check at setup, a config check).
   When you can, re-run the failing case from the comment.
4. **Judge against the user's real need, not your comment's wording.** If the wording is met but the
   need is not (the docs describe the format, but nobody could produce the file from them), the verdict
   is **partial**.

Then find what changed:

- **Normal push:** `git log <reviewed>..<head>` and diff each new commit against the lines the thread is about.
- **Rebase or force-push** (the reviewed commit is not an ancestor of the head): compare the PR's own
  change before and after. Take `git diff <old-base> <old-head> -- <file>` against
  `git diff <new-base> <new-head> -- <file>`, looking only at the `+`/`-` lines. Base drift is not the
  author's change. Also re-check anything the review predicted a rebase would break: it may have landed.
- **Applied suggestion blocks:** diff the hunk the author applied and read its `-` lines. A one-click
  apply can swap a function header or delete code. `-1 +35` on a file you expected only to grow is the tell.
- **Re-run the check** the comment was about: the lint hook, the unit test, the grep, the failing command.
  Fixing the symptom is not the same as fixing the cause.
- Also note: has CI run on the new head?

Verdicts:

| Verdict | When |
|---|---|
| **fixed** | The problem the comment was about is gone, the check passes, and it is obvious. Matching the comment's wording is not enough. |
| **partial** | Some of the ask landed; a specific piece (a file:line, a test, a doc copy) is left. Also: fixed, but the fix created a new small problem. Also: the wording is met but the user's need is not. |
| **not addressed** | Nothing relevant changed — confirmed after reading the reply and checking every offered fix. One grep cannot prove absence. |
| **moot** | The code the comment was about is gone, or the premise no longer holds. Treat like fixed if the risk is gone; say why. |

Show the user a table (thread link, one-line finding, **what the author's reply claims**, verdict,
evidence with permalinks at the new head, the commit that did it), plus CI status. Putting the
author's claim next to your verdict makes any disagreement between them visible.

### 2b. Check the new commits for new major issues

Closing the old threads is only half the job: the fix commits are new code that nobody has reviewed,
and they are where a new bug most often lands. Review what changed since the reviewed commit (for a
rebase, the PR's own change before and after, as above), plus anything that code now reaches.

- **Major only:** bugs and missing tests for changed behaviour. Style and nits were the first
  review's job.
- **Don't re-raise an open thread.** That is step 2's verdict.
- Each finding cites file:line at the new head and is verified before it is shown, same as a
  `/review-pr-team` finding.

Show them under the step 2 table as a separate list. **Stop and ask for confirmation** before step 3;
the user picks which new findings to stage.

### 3. Close the fixed threads (published directly)

Do this BEFORE staging any pending review: while you have a pending review on the PR, GitHub
rejects direct replies through the REST `/replies` endpoint. If the user already has a pending
review, ask how to proceed instead of working around it.

For each **fixed** (or moot) thread, after the user confirms the list:

```bash
gh api repos/$R/pulls/$N/comments/$ROOT_COMMENT_ID/replies --method POST -f body="fixed in $FULL_SHA"
```

```bash
gh api graphql -f query='mutation($t: ID!) { resolveReviewThread(input: {threadId: $t}) { thread { isResolved } } }' -f t="$THREAD_ID"
```

The reply is exactly `fixed in <sha>`, nothing else. For moot threads, one short sentence saying
why it no longer applies is fine. Verify each thread shows `isResolved: true` afterward.

### 4. Stage follow-ups for partial fixes (PENDING)

Create ONE pending review (or add to the user's existing one) holding **one new inline comment
per partial item and per new finding the user picked in step 2b**, anchored on the current head at
the line where the remaining work is. Do not reply inside the old thread yet. A new finding is a
normal `/review-pr-team` comment; the **Fixed:** / **Still open:** shape below is for partial items.

Each follow-up is written for the author, who has not read your notes. Follow the same comment rules as
`/review-pr-team`:
- an actionability line with the comment's ID (`**C<n> · 1 action item.**`). A follow-up takes the
  review's next free number and never reuses a retired one;
- a bold one-line TL;DR saying what is still left;
- **Fixed:** what already landed, with a permalink;
- **Still open:** exactly what remains, with a permalink to every place;
- one `### AI-1` with a `suggestion` block, or one clause saying why it can't be one;
- permalinks pinned to the new head, and a link back to the original thread.

Cut anything not worth another round. Drop leftovers that your own earlier suggestion caused, and nits.

Mechanics: create the pending review with the REST `POST /pulls/N/reviews` without `event`
(`commit_id` = current head), or add threads to an existing one with GraphQL
`addPullRequestReviewThread` (`pullRequestReviewId` = the review's node id, plus `startLine`
for multi-line anchors). Check the count with GraphQL `comments.totalCount`.

Show the user the pending review link and each comment. **The user publishes it**, or tells you to:
`gh api repos/$R/pulls/$N/reviews/$REVIEW_ID/events --method POST -f event=COMMENT`, passing
`-f body=...` in that same call if the review has a body, because a pending body can be lost at submit.

### 5. After the follow-up review is published: close the old threads

A pending comment's link only works for its author, so this step must wait until the review is
published. Confirm the review state is `COMMENTED` first. Then, for each partial thread, after confirming with the user:

```bash
gh api repos/$R/pulls/$N/comments/$ROOT_COMMENT_ID/replies --method POST \
  -f body="partially addressed in $FULL_SHA — follow-up: $NEW_COMMENT_URL"
```

Then resolve the old thread with `resolveReviewThread`. The follow-up comment is now the one open
place for the remaining work.

### 6. Report

Tell the user:
- which threads were closed as fixed (and by which commit);
- which follow-ups were posted (with links);
- which were **not addressed** (no reply was posted on those, by design);
- which new findings from step 2b were staged, and which the user dropped;
- the CI status on the new head.
