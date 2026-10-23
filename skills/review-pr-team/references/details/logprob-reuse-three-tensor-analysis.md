# logprob-reuse-three-tensor-analysis

> Reviewing logprob-reuse/IS changes in NeMo-RL — map each loss term to the three logprob tensors (curr/prev/generation) and check the degenerate-consumer list; sync reuse = verl bypass_mode, async reuse is semantically invalid

NeMo-RL's ClippedPGLossFn consumes three logprob tensors: `curr` (live trainer forward per
minibatch update), `prev_logprobs` (frozen proximal/old anchor), `generation_logprobs` (engine's
processed sampling logprobs). Term map (loss_functions.py @ a2c85bae8): PPO ratio =
exp(curr − prev) (L523); off-policy IS weight = exp(prev − generation), detached (L575), gates
TIS/icepop/seq-mask-tis (L592-665, which assert IS-correction on at L288); diagnostics
token_mult_prob_error / gen_kl / policy_kl / js all compare prev vs generation; seq_logprob_error
masking (grpo.py:2387) also keys off |generation − prev|.

For any PR that changes where `prev_logprobs` comes from, walk EVERY consumer of exp(prev−gen)
and classify: mathematically-forced exclusion vs implementation-forced vs silently-degenerate.
E.g. prev:=generation (PR #3295) ⇒ IS weight ≡ 1 (exclusion is a mathematical necessity —
double-counting, verl bypass_mode's documented reasoning), TIS exclusion is only
implementation-forced (slime keeps a separate train_log_probs_for_tis forward, loss.py L912/L930
@ aaf5c209), and seq_logprob_error masking degenerates SILENTLY (flag-worthy: guard didn't cover
it). Sync reuse is always exact, because of how it is built (prev and generation are the same policy, θ_gen, through
two implementations); async reuse is semantically invalid, not just unimplemented (async prev =
current trainer policy ≠ stale engine policy; the IS weight π_current/π_stale IS the staleness
correction, and async hard-requires IS-correction on at grpo.py:3798).

**Why:** The user's core review question on #3295 ("does reuse make the ratio always 1?") resolves
instantly with this map: reuse sets only prev; curr is fresh per update, so the ratio never
degenerates — it's the IS weight that collapses, which is the intended trade.

**How to apply:** for logprob-source PRs, produce the term map first, then check guard coverage
against the full degenerate-consumer list (loss terms AND filters/diagnostics — the silent ones
are filters like seq_logprob_error whose no-op looks like "no sequences masked"). Verify external
precedent from pinned source (verl core_algos.py bypass_mode @ 2b47a68b), never memory.
