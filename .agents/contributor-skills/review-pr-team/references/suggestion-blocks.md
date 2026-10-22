# Suggestion blocks

Make every ```suggestion safe to click.

- ★ **For every suggestion, rebuild the file at the current head with the block applied, `ast.parse` it, fail if a def/class/return vanished, and `ruff format --check` insert-only blocks; redo it after any head move.** A one-line-too-long range deleted `return actors` and still parsed (#4112). Details: [apply-the-suggestion-at-its-own-anchor](details/apply-the-suggestion-at-its-own-anchor.md).
- **A suggestion only applies to the comment's own anchored lines; code for another file goes in a plain fence.** Audit: fail when a suggestion block names a path other than the comment's (#3371).
- **A multi-line suggestion range must sit inside one diff hunk, or GitHub previews it as all-new lines; use a ```diff block instead.** (#4670)
- **When the same fix applies to several separate lines, put one small suggestion on each line.** A multi-line comment shows at its last line (#4687).
- **When one comment introduces a name, every other comment's fix must use it.** One comment added `UNKNOWN_ROLLOUT_ENVIRONMENT` while another re-typed `"unknown"` (#4068). Details: [sibling-fixes-must-use-each-others-names](details/sibling-fixes-must-use-each-others-names.md).
- **For "the PR added X and dropped Y" findings, say exactly that in the TL;DR and show the fix as a ```diff block.** (#4129)
