# pending-review-is-invisible-to-author

> A PENDING review is a private draft the author cannot see — never narrate a later commit as "the author addressed my comment", and confirm submitted_at before reasoning about what feedback reached them

A PENDING review and its inline comments are visible ONLY to the review author. Before reasoning
about whether feedback "landed", establish which state you are in:

```bash
gh api repos/<owner>/<repo>/pulls/$PRNUM/reviews \
  --jq '.[] | "id=\(.id) state=\(.state) submitted_at=\(.submitted_at)"'
gh api repos/<owner>/<repo>/pulls/$PRNUM/comments --jq 'length'   # PUBLISHED comments only
```

`submitted_at: null` / `state: PENDING` / published-comment count `0` all mean the same thing: the
author has never seen a word of it. Convergent evidence is worth gathering — a reviewer resuming a
session days later genuinely will not remember whether they hit publish.

Two consequences:
- **Never write "the author did exactly what the comment asked"** for a draft review. If they could
  not see it, any matching commit is independent convergence (usually them reading red CI), and
  crediting your unsent comment is a false causal claim in a user-facing summary.
- **Deleting a pending review retracts nothing public.** Say so when proposing it — "delete and
  re-stage" sounds destructive and invites a reasonable "wait, I thought I submitted that?"

Also, when re-reviewing after the head moves, re-fetch the file list (`gh pr view $PRNUM --json files`)
BEFORE re-verifying individual findings. A finding whose file left the diff entirely is dead on
arrival — GitHub cannot anchor a comment to it — so that check cheaply invalidates whole findings
before you spend effort re-tracing them.

**A re-review must AMEND the existing draft, not stage a new one.** GitHub allows exactly one
pending review per user, so once a prior session has staged one there is no second draft to create —
and deleting + re-POSTing throws away comments that already survived an adversarial pass. Edit in
place with GraphQL (REST `PATCH /pulls/comments/<id>` returns 404 while the review is PENDING):

```
updatePullRequestReviewComment(input:{pullRequestReviewCommentId:$id, body:$body})  # one comment
updatePullRequestReview(input:{pullRequestReviewId:$id, body:$body})                # the review body
```

Both work on a PENDING review and leave `state=PENDING` untouched — verified on #2689. One caveat:
`updatePullRequestReview` only *edits* an existing body. On a pending review with no body it fails
with "Could not edit a review with a missing body"; then hold the body in a file and pass it at
submit time via the REST events endpoint (`-F body=@summary.md`) — see
`pending-review-body-only-at-submit` (see [review-threads-and-pending-review](../review-threads-and-pending-review.md)). Fetch the
current bodies first and apply string replacements programmatically rather than retyping them; that
preserves the permalinks and suggestion blocks you are not changing. Afterwards re-assert
`state=PENDING` and the comment count via GraphQL before telling the user it is staged.

**Why:** On PR #3453 (nccl 2.30.7 bump) a draft review from a prior session blocked on the
base-image half (CUDA 13.2→13.3 broke both container builds). At re-review the Dockerfile change was
gone and CI was green, and the summary asserted the author had acted on the comment — but the review
was still PENDING with zero published comments, so they had never seen it; they had reverted off the
red builds themselves. The user caught it with "wait is my review pending? i thought i already
submitted" — the assertion, not the review mechanics, was the error.

**How to apply:** at the start of ANY re-review or "did they address my comments?" task, query review
state and published-comment count first, and state the answer explicitly before analysing the diff.
Attribute an author's change to your feedback only when that feedback was actually published. Related:
[reverify-consequences-after-force-push](reverify-consequences-after-force-push.md), `rebase-can-void-a-credited-fix-commit` (see [provenance-and-scope](../provenance-and-scope.md)),
the skill's "Teams are MANDATORY" section.
