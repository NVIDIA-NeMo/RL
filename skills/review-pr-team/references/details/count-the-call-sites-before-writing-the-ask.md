# count-the-call-sites-before-writing-the-ask

> If an ask says "at the N call sites", grep and count them, and permalink each one — an uncounted N contradicts itself later in the same comment

When a comment asks the author to change a helper AND its callers, the ask must name every call
site with its own permalink and its own `after` snippet. `grep -n '<symbol>' <file>` first, count,
then write.

**Why:** on PR #4207 a published comment said "the three xtoken call sites" in its TL;DR and "at
both xtoken sites" two paragraphs later, with no link on the line that told the author where to
act. There were two. The count was inherited from a lane report and never checked; the user read the
comment, could not tell where to add the keyword, and asked for permalinks — which is when the
contradiction surfaced.

**How to apply:**
- Grep the symbol, count the sites, and give each one a permalink plus the exact resulting line.
  "at both call sites" is not actionable; two linked snippets are.
- Re-read the finished comment for internal consistency — a number in the TL;DR and a number in the
  **Action:** line must agree. A comment that contradicts itself reads as untrustworthy even when
  the underlying finding is right.
- **Check your own comments against each other for conflicting suggestions.** The same review had a
  committable `suggestion` on the exact line a second comment asked to edit differently; applying
  both in either order stomps one. When two comments touch one line, the later one must carry the
  combined result and say so.

See [apply-the-suggestion-at-its-own-anchor](apply-the-suggestion-at-its-own-anchor.md), [derive-key-lists-dont-restate-them](derive-key-lists-dont-restate-them.md).
