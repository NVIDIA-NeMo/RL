# count-the-artifact-before-calling-it-incomplete

> Before flagging "the PR added X but didn't update table/list Y", count how much of the full population Y already covers — a spot-checked artifact establishes no convention to violate, an enumeration does; the same count decides two sibling findings opposite ways

"The PR adds a new N but doesn't add the matching row/bullet to Y" is a cheap, satisfying finding.
It is only valid if Y is an **enumeration**. If Y is a **spot-check**, there is no convention to
violate and the finding is noise. The deciding evidence is one number: how many of the full
population does Y already cover?

Compute it before writing the comment:
`grep -c '<the thing being enumerated>' <source>` vs. the entry count in Y.
- Low coverage (Y samples a few) → DROP. "Exactly two rows, and the PR adds a third with no row"
  is a *misleading framing* of 2-of-17 — it reads as exhaustive-minus-one when it is a sample.
- High coverage (Y covers nearly all, possibly grouped thematically) → the finding stands, and the
  count is the argument that it isn't a nit.

**Why:** On PR #3619 (MTP gating) I staged two structurally identical findings and got one wrong.
Both said "this PR adds a `real_quant` violation to `check_nccl_reshard_refit_support` and didn't
update the artifact that lists them."
- Against `tests/unit/weight_sync/test_nccl_reshard_utils.py`'s parametrize table: `devil-advocate`
  DISPUTED it. The function has **17** `violations.append` sites; the table covers **2**. No
  convention. I had framed it as "the table has exactly two rows… the PR adds a THIRD" — true
  sentence, false implication. Dropped.
- Against `docs/design-docs/nccl-reshard-refit.md`'s requirements list: DA reached for the same
  2-of-17 argument, then **withdrew it** after I mapped all 17 sites to the 7 bullets — **15 are
  covered** (one bullet spells out all five ETP/PP-layout checks, another all four precision ones).
  That is an enumeration with one pre-existing lapse, and this PR widened the gap to two. Confirmed
  as the strongest finding in the review.

Same PR, same shape, same author, opposite verdicts — decided purely by the coverage count. Note
also that three agents (rl-expert, test-agent, bug-finder, conf 88-90) all wanted the *test-table*
version posted; convergence again measured shared sampling, not truth.

**How to apply:** treat "artifact Y is missing an entry" as unproven until you have counted Y's
coverage of the population, and put the count in the comment ("every other check here has a bullet")
rather than the audit trail ("I mapped all 17 and 15 are covered" — that is showing your work at the
author's expense; DA cut it). Grouped/thematic entries still count as coverage — map them, don't
compare raw entry counts. When a prior finding was killed by this argument, re-derive the count for
the sibling rather than assuming it transfers; it often doesn't.
Related: `grep-the-idiom-before-calling-a-convention-wrong` (see [true-but-not-a-finding](../true-but-not-a-finding.md)), `grep-the-diff-before-claiming-an-oversight` (see [true-but-not-a-finding](../true-but-not-a-finding.md)),
`dead-guard-is-not-a-finding` (see [true-but-not-a-finding](../true-but-not-a-finding.md)).
