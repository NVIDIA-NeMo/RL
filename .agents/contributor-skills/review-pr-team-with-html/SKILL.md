---
name: review-pr-team-with-html
description: /review-pr-team plus a public /pr-html-concise-public explainer, built before staging
argument-hint: "<pr-number>"
user-invocable: true
---

# Team PR Review + HTML Explainer

Run `/review-pr-team <pr-number>`, and use what the review learned to build a public
explainer page (or pages) with `/pr-html-concise-public`. The page shows what the PR
does and, if the review found one, the major bug.

This skill only changes **when** things happen. Follow both skills in full for **how**.
Everything in them still applies: the devil-advocate gate, the pre-POST audit, the page
limits, the public-site check.

## Workflow

1. **Review.** Run `/review-pr-team <pr-number>` through Phase 3 (collation), including
   the devil-advocate hard gate. Findings are final only after that gate.

2. **Build the page(s) — before the Phase 4 "Stage all?" question.** Now that you
   understand the PR and its confirmed findings, run `/pr-html-concise-public <pr-number>`:
   - Pick the page set with its Step 0 table. A new feature gets the comparison page
     first. A **major bug** is a Critical finding that devil-advocate confirmed, and it
     gets its own page showing the failure and the fix. Minor findings stay in the review
     comments only.
   - Use only confirmed findings. Never put a finding on the page that devil-advocate
     disputed or that you dropped.
   - The review comments do not exist yet, so leave out the comment links for now
     (step 5 adds them).
   - Upload as its "Site setup and upload" section says, and cross-link the pages if there are several.

3. **Show the preview.** Print the normal Phase 4 findings preview, and add the page
   link(s) at the top. Then ask the Phase 4 question (Stage all / Discuss individually /
   Cancel).
   - If the user cancels, ask whether to take the page(s) down. They are already public.

4. **Stage.** When staging the pending review, add one line to the end of the review
   body: `Visual explainer: <url>` (one line per page).

5. **Add the comment links, then re-upload.** Once the pending review exists:
   - Get each staged comment's ID:
     `gh api repos/NVIDIA-NeMo/RL/pulls/<pr>/reviews/<review-id>/comments --jq '.[] | {id, path, line}'`
   - For each page section that explains a finding, add the `.cref` link from
     `/pr-html-concise-public` pointing at
     `https://github.com/NVIDIA-NeMo/RL/pull/<pr>#discussion_r<id>`.
   - Re-upload to the **same filename**, so the URL in the review body stays correct.
   - These links only work for others after the review is published. That is expected.

6. **Print the links.** As soon as steps 4 and 5 are done, end your message with this
   block, so the user never has to dig back through the history for the URLs:

   ```
   **Links**
   - Pending review: https://github.com/NVIDIA-NeMo/RL/pull/<pr>#pullrequestreview-<review-id>
   - Explainer: <page-url>          (one line per page)
   ```

   Use the real review ID from staging and the final uploaded page URL(s). Put the block
   at the very end of the message, after any other text.

7. **Publish.** Continue the normal Phase 5 flow and ask Publish / Edit / Cancel. If an
   edit changes a finding that a page shows, update the page and re-upload before
   publishing. Print the **Links** block again after any edit, and after publishing.
