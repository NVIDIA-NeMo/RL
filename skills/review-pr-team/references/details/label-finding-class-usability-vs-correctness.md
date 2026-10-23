# label-finding-class-usability-vs-correctness

> Say in the first line whether a finding is a correctness bug, a usability/fail-loud ask, or something that can change training results — a detailed failure walkthrough reads as "you have a bug" even when the code is correct and the real ask is a log line

A vivid breaking-case walkthrough (config → permalinked steps → error string) is the right tool for
a correctness bug, but applied to a **usability** finding it misleads: the author reads a trace
ending in a fatal error and concludes their code is broken, when the code is correct and the actual
ask is "print the value you already computed" or "document this knob."

State the class up front, in one clause, alongside the existing PR-introduced vs pre-existing
label: *"Usability gap (not a correctness bug): …"*. Then keep the trace, but proportionate — two
sentences, not five numbered steps.

Corollary — **admit when you could not build a guard.** If you considered a validation/assert and
it isn't feasible, say so and say why ("`cluster.num_nodes` lives inside the user's `COMMAND`, so
`ray.sub` has nothing to compare against"). That preempts the obvious "why not just assert?" reply
and is honest that a log line is a weaker remedy than a hard failure.

**Why:** In the PR #3388 review my top finding walked five permalinked steps to a fatal
`ResourceInsufficientError`. The user read it and asked, essentially, "so this is just pointing out
that a user might misconfigure, and asking for a log line telling them what `num_nodes` should have
been?" — which was exactly right, and exactly what the comment failed to say. It was ~4.4k chars
for two small asks; trimmed to ~2.2k with the class stated in the first line, it lost nothing.

**A third class: "can change training results."** If a finding can change the numbers the
trainer sees (advantages, loss, logprobs), its first line says so in training terms ("runs with
different X can converge differently"), not in mechanism terms ("rows are grouped by prompt
tokens"). The A/B that would show it becomes a body-level ask, not a nice-to-have. A real fix for
the cause beats correcting the comment that over-claims. On PR #4326 the duplicate-prompt finding
first led with the grouping mechanism, asked only to fix a comment, and listed the A/B as "nice to
have". The user made convergence the main concern, moved the A/B into a **Convergence** section, and
pointed the fix at issue #4371 (key the baseline by group id).

**How to apply:** Before posting, ask "if the author does nothing, is the *code* wrong, or is a
*user* likely to hurt themselves?" Only the first earns a full breaking-case treatment. For the
second, lead with the class, compress the trace, and name the remedy honestly (log / doc / guard —
and if a guard is impossible, say that). Pairs with the comment skeleton in /review-pr-team and
`show-the-breaking-case`.
**Seen:** #3388 num_nodes walkthrough read as a bug (user) · #4326 C4 duplicate prompts led with mechanism, not convergence (user)
