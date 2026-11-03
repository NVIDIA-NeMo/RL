# Permalinks and anchors

Links and anchors must be right, not just resolvable.

- ★ **Never hand-write a line number: script every permalink and anchor to assert non-blank, contains the expected token, and (anchors) is a right-side line inside a diff hunk.** One out-of-hunk anchor 422s the whole POST (#3813); an off-by-one on a sibling YAML key inverted a comment (#3470).
- **Comments added later with `addPullRequestReviewThread` anchor at the current head, not the review's commit.** (#3590)
- **Check that each cited line supports the sentence, not just that it resolves.** A "CI already covers it" link pointed at a note saying it never ran (#3591).
- **Grep every symbol name you repeat from a brief or report; a wrong name with right line numbers passes the click test.** (#3591) Details: [grep-the-symbol-not-just-the-lines](details/grep-the-symbol-not-just-the-lines.md).
- **Each `### AI-n` that changes code off the anchor links every target line inside the AI itself.** (#4481) Details: [ai-section-links-its-own-targets](details/ai-section-links-its-own-targets.md).
