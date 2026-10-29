# dropped-format-check-previous-release-defaults

> When a PR drops support for an old on-disk format (checkpoint layout, config key), size the affected users from the LAST RELEASE's code default plus its shipped configs — not just configs that set the flag explicitly

When a PR stops reading an old on-disk format, the finding turns on who still has files in that
format. Answer it at the last release tag, in two steps:
1. Read the code default for the switch at that tag (`git show <tag>:<file>`), not today's default.
2. List the shipped configs at that tag that leave the switch unset — they got the code default —
   as well as the ones that set it explicitly.

**Why:** On #4273 (deleting the DTensor v1 checkpoint reader, base r0.8.0), one lane said v1
checkpoints exist "only for users who explicitly set `_v2: false`". At v0.7.0, `lm_policy.py` used
`.get("_v2", False)`, and the shipped `dpo.yaml` and `rm.yaml` never set `_v2`. So every default DPO
and RM run wrote v1 checkpoints, on top of the 4 recipes with an explicit `_v2: false`. The narrower
count would have made the finding look like a corner case.

**How to apply:** For any "support for X removed" PR, especially on a release branch, grep the
previous release tag for both the explicit setting and configs that omit it, and quote the default
line. Prefer asking for a clear error that names the old format and the last release that reads it,
over asking to restore the old reader: CI usually can no longer produce the old format, so restored
code would go untested. Related: `count-who-relies-on-a-default-before-arguing-about-it` (see [claims-searches-and-numbers](../claims-searches-and-numbers.md)),
`scope-test-did-the-diff-create-the-exposure` (see [provenance-and-scope](../provenance-and-scope.md)), `show-the-breaking-case`.

**Seen:** #4273 v1 checkpoint converter (team)
