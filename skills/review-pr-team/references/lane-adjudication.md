# Lane adjudication

How the leader weighs teammates' claims, agreement and the adversarial pass.

- ★ **Before you accept, relay or repeat any teammate claim (including a devil-advocate kill or a "this matters because" reason), check the one fact it rests on.** A relayed misread parametrize list nearly killed a real finding on #3813.
- **Agreement counts only if lanes used different evidence; same diff, same brief is one data point.** Neither lane had read the next PR in the stack on #3371.
- **A score rising as two lanes re-check the same premise measures care, not truth; check that something reads the value.** (#4108)
- **When one lane says "can't be computed" and another gives a number, compute it yourself.** The "unparseable" lane had missed `GPUS_PER_NODE` defaulting to 8 (#2884).
- **A bare call site can still pass an argument: check whether a middle layer fills it from its own default.** `lm_policy` forwarded `refit_timeout_s=None` to every worker, breaking DTensor runs (#3591).
- **If both adversarial lanes stay silent after two nudges each, run the pass yourself, close their scopes with scripts, say so in the preview, and stop waiting.** Never self-fill the comment-editor (#3372).
- **A correction sent to a running background agent breaks its return channel; put the explicit `SendMessage(to="team-lead", …)` instruction in the correction itself.** (#3490)

(The "never stage before every adversarial pass reports" rule is already the Phase 3 HARD GATE.)
