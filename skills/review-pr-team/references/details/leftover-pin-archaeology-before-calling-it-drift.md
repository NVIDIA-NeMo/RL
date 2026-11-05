# leftover-pin-archaeology-before-calling-it-drift

> Before calling a leftover flag/pin "the migration missed one", git-archaeology the line — if it was born that way in a commit whose payload is the thing the line selects, it is deliberate coverage, not drift

On any sweep-style PR ("move every remaining X to Y", "delete the last Z"), a grep will turn up
leftovers outside the diff. **Finding the string is the start of the work, not the end.** Before
writing "the migration missed this one", establish why the line is there:

1. `git log --diff-filter=A -- <path>` to find the creation commit — **not** `git log --follow`,
   which happily walks into a different file's history through a rename and hides the real one.
2. `git show <sha> -- <path>` and look for `new file mode`. If the line is present in the initial
   body, it was written deliberately, not left behind.
3. Read that commit's *payload*. If it fixes the very thing the line selects, the line is coverage.
4. Check whether the coverage is unique: `grep -rln <model/feature> <test dir>`.

**Why:** On PR #4196 (flip every DTensor v1 config to v2) three lanes and I independently flagged
`tests/functional/grpo_fsdp2.sh:24`'s `policy.dtensor_cfg._v2=false` as "the one the migration
missed", with a `suggestion` block flipping it to `true`. The archaeology says otherwise: the file
was created whole by `986d48aff2` *"fix: fix gemma3 (#2185)"* — by the same author — born with that
pin, and that commit's payload is a gemma3 fix **inside the v1 worker** (`is_gemma3` detection, the
CP assert, tied `embed_tokens`/`lm_head` handling in `dtensor_policy_worker.py`). It runs
`google/gemma-3-4b-it` and is the only functional test touching gemma. So it is deliberate v1
coverage, in the same category as the v1 unit-test variants the same lanes had already correctly set
aside (`tests/unit/excluded_unit_tests.sh` literally says *"exclude v1 (use_v2=False) variants of
kept tests"*). They drew the right line and then put this file on the wrong side of it.

The suggested fix was worse than the finding: flipping the pin to `true` silently converts a v1
regression test into a v2 one. That is a coverage decision for the author, never a one-click
`suggestion`.

Two further traps this exposed, both real:
- **A "leftover" in a stack is usually the stack's problem, not the base PR's.** My reframe
  ("#4207 will break this L1 gate") was itself wrong — #4207 *deletes* that file and its lane wiring.
  See `failed-search-is-not-a-negative-fact` (see [claims-searches-and-numbers](../claims-searches-and-numbers.md)) for how I got that wrong (truncated diff read).
- **Deliberate coverage that a stack deletes is still worth one line — on the PR that deletes it.**
  #4207 removes the repo's only functional gemma coverage with no replacement. Raise it there.

**How to apply:** for every "you missed one" / "leftover" finding, the comment must be able to answer
"why was this line written?" before it is posted. If the answer is "a commit that was fixing exactly
this path", downgrade or drop. Bias the archaeology budget by blast radius: a one-line pin in a CI
script that gates a lane is worth the three commands. Related:
`why-is-this-line-here-look-it-up-first` (see [comment-writing](../comment-writing.md)) (same instinct, applied to an unexplained line *inside*
the diff), `dead-guard-is-not-a-finding` (see [true-but-not-a-finding](../true-but-not-a-finding.md)), `check-provenance-before-calling-asymmetry-pr-introduced` (see [provenance-and-scope](../provenance-and-scope.md)).
