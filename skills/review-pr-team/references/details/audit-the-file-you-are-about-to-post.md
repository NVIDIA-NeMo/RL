# audit-the-file-you-are-about-to-post

> Re-run the pre-POST audit against the exact payload file you are about to send, never trusting an earlier green result — a teammate can regenerate that file underneath you

The pre-POST audit is only valid for the bytes on disk **at the moment you POST**. Re-run it
against the exact file you are about to send, every time, even when it went green minutes ago.

**Why:** on PR #3566 the comment-editor lane wrote `review.json`, I audited it green, then patched
three missing "not a one-click suggestion" clauses into it. The lane then regenerated `review.json`
at the same path with its own final version — silently discarding my three patches. Nothing warned
me. The only reason it was caught is that I re-ran the audit before POSTing instead of trusting the
earlier CLEAN result; the rerun came back with 3 failures and different line numbers, which is what
revealed the file had changed underneath me. Had I trusted the first green, the review would have
gone out with three action items that state a fix in prose and never say why it is not a committable
suggestion.

A fixed filename shared by two writers is a mailbox, not an artifact. The same hazard applies to
anchor line numbers: the regenerated payload used *different* anchors (config.py 378 -> 367,
test_vllm_generation.py 195 -> 191), so a patch keyed on `(path, line)` silently matched nothing.

**How to apply:**
1. Immediately before the POST, re-run the audit, the anchor check, and the apply-alone test on the
   payload file itself — not on your memory of them, and not on a teammate's report that its own
   builder audited before writing.
2. After the POST, verify from the server, not the file: GraphQL `comments.totalCount` on the
   PENDING review must equal the payload's comment count (REST under-reports on pending reviews).
3. Prefer versioned payload paths (`review.v2.json`) so a regeneration cannot overwrite a patched
   file; point the POST at one specific version.
4. If you patch a payload keyed on `(path, line)`, re-verify the keys still match after any
   regeneration — a no-op patch reports success and changes nothing.
5. **Verify a patch by re-reading the target text, not by trusting the patch's own exit status.**
   This is the sharper half: a `(path, line)` patch that silently matches nothing is worse than the
   overwrite, because both writers then believe the change is in. On #3566 the anchors moved between
   generations (`config.py` 378 -> 367, `test_vllm_generation.py` 195 -> 191), so the patch matched
   zero comments and still printed "patched 3 comments" logic-side until the keys were corrected.
   Assert the new text is present in the file afterwards.

Related: `verify-permalink-lines-programmatically` (see [permalinks-and-anchors](../permalinks-and-anchors.md)),
`challenge-the-challenger-das-premises-need-verifying-too` (see [lane-adjudication](../lane-adjudication.md)), `reread-skill-from-disk-before-staging`
