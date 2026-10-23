# audit-checks-form-a-reader-checks-meaning

> The pre-POST audit can only verify a TL;DR exists and is bolded — read every TL;DR back as someone who has not read the code before staging

The mechanical pre-POST audit verifies *form*: a TL;DR exists, it is bolded, no banned words, links
resolve, suggestions are byte-identical. It cannot verify that a sentence is understandable. Add a
deliberate human-simulation pass: re-read each TL;DR as someone who has never opened the file, and
ask what they would have to already know to parse it.

**Why:** On PR #4112 the user read a staged TL;DR and said "I still don't get it". It had passed the
audit cleanly. The line was:

> Both parametrize cases fail at the same faked import, so the test cannot tell a direct missing
> import from a transitive one; drop the second case and rename it.

Three defects, none catchable mechanically:
1. **"faked import"** — my shorthand, from an hour in the code, for "the test hijacks Python's
   import machinery so the import throws instead of running".
2. **"direct vs transitive"** — the reader must already hold the distinction to see why it matters;
   the comment never said what it was.
3. **It stated the conclusion** ("cannot tell X from Y") instead of the one concrete fact underneath
   it: the two runs differ only by a string inside an error message.

The rewrite leads with the fact — "runs twice but does the same thing both times; only the text
inside the fake error changes" — and adds the contrast the original never stated (what the two cases
were *meant* to be, and that the real second case would get further before failing). That contrast
was the actual finding, and it had been missing.

The user asking one question surfaced defects in BOTH staged comments that six agents, an adversarial
pass, and a passing audit had all cleared.

**How to apply:**
- Before staging, read each TL;DR alone and point at every noun. Any noun you only know because you
  read the code is either replaced with what the thing plainly is, or explained where it first
  appears. Real code identifiers are fine — they are greppable and permalinked. Invented English is
  not.
- Test: "if the reader believed only my TL;DR, would they still have to guess why this is a bug?"
  If yes, rewrite it as the causal fact.
- If the user asks you to explain a staged comment, that is a defect report on the comment. Fix it
  in place; do not just explain it in chat and leave the comment as it was.
- An analogy can be the right tool for explaining in chat and the wrong one for the comment — the PR
  author is an expert who needs the plain mechanism, not a metaphor.

- A TL;DR that asks the author to keep X and drop Y must name X first. "Y is unused" plus a second
  clause about refactoring reads as two findings; lead with what stays, then what goes.

**Seen:** #4112 two TL;DRs (user); #4750 C2 "last_entropy is unused…" — the user asked "are you saying
we only need cur_adaptive_ppo_clip_k?"; rewritten to lead with the one value needed (user).

Related: [label-finding-class-usability-vs-correctness](label-finding-class-usability-vs-correctness.md), [check-summary-line-against-evidence-paragraph](check-summary-line-against-evidence-paragraph.md).
