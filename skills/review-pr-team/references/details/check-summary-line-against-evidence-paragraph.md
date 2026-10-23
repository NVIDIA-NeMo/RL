# check-summary-line-against-evidence-paragraph

> A finding can be correct in the paragraph that proves it and inverted in the sentence that sells it — and the sentence is what ships, gets quoted back, and gets the whole finding discounted

After writing a finding, re-read its **headline, TL;DR and Action line against its own evidence
paragraph**, as separate claims. The evidence is where the work happened; the summary is what an
author reads, quotes back, and checks. They drift apart, and a summary that contradicts its own
evidence gets the entire finding discounted — including the correct part.

Concrete shapes this took on one review (PR #3837), all with sound underlying work:

- Ran the imports correctly, then labelled the result an **"import cycle"**. There was none; the
  author could have disproved it with one `python -c`, taking a real placement finding with it.
- Ran a `diff` correctly, then described an offset artifact as **"differs at the first line"**.
- Established that a forbidding clause survived 57 commits *after* the code it forbade landed,
  then wrote **"the PR removed the guard rail and then did the thing it named"** — the reverse
  ordering, disprovable with one `git log`, and it would have read as an accusation of
  deliberate suppression.
- Escalated a finding to medium on a mechanism firing repeatedly, without opening the default
  (`replacement_reserve_prompts: NonNegativeInt = 1`) that stops it after the first time.
- Ran a real `git log --oneline -- <file>` and drew from it a claim it does not answer: that
  counts commits that *touched the file*, not how long a contradiction stood.

Two rules that fall out and generalise:

1. **Prefer a check that reads state directly over one that counts or derives.**
   `git show <parent>:<file> | grep -c "<clause>"` returning 1 settles a question in one command;
   an ordinal computed from `git log --reverse` was wrong twice from arithmetic and would have
   been invalidated a third time by a base correction.
2. **When a diff DELETES contract text, read the deleted clause for what it permitted and what it
   forbade separately.** They can point in opposite directions and only one may have been meant
   to go. Then check when the affected code landed relative to the deletion: deletion provenance
   decides whether the finding is real, code provenance decides what tone it deserves.

**Why:** across that review every process failure had this shape — the mechanism held, the
sentence selling it did not — and every one was caught by a teammate re-reading the claim rather
than by the author of it re-reading their own work. The adversarial agent alone made five, and
the most consequential (the import-cycle label) would have killed a genuine finding.

**How to apply:** treat the summary line as a claim needing its own verification pass, not as
prose. Before staging, read each finding's first line with the body covered and ask "is this
exactly what my evidence shows, no stronger and not reversed?" Related:
`verify-merge-base-before-any-provenance-claim` (see [provenance-and-scope](../provenance-and-scope.md)), `prefer-base-independent-figures` (see [claims-searches-and-numbers](../claims-searches-and-numbers.md)),
`show-the-breaking-case`.
