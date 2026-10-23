# verify-the-consumer-not-just-the-validator

> When a finding turns on what a value MEANS (inclusive/exclusive, unit, base), check the code that consumes it — a validator's internal consistency proves nothing

When a finding is about the *meaning* of a value — a range bound being inclusive or
exclusive, a unit, a 0- vs 1-based index — settle it by reading the code that **consumes**
the value, not the code that validates it. A validator can be perfectly self-consistent and
still disagree with the thing it feeds.

**Why:** On PR #3369 a lane argued that `port_range_high: 65536` was correct and told the
team "do not let anyone file that as a bug." Its premise was true — NeMo-RL's own validator
rejects `low >= high`, so it reads the pair half-open, and 65536 exclusive gives a max
usable port of 65535. But it never checked the consumer. Gym picks with
`randint(port_range_low, port_range_high)`, inclusive on both ends, so 65536 is drawable;
`socket.bind(("", 65536))` raises `OverflowError`, which is *not* an `OSError`, so Gym's
`except OSError` retry loop does not catch it and the actor dies. The real defect was not an
off-by-one in either module — it was that **the two sides disagree about what the pair means,
and the validating side is the one that accepts the value.** Two lanes had it half-right and
neither stated it that way.

**How to apply:** for any "this bound/unit/index is wrong" finding, name the consumer and
link the exact line that reads the value. Run the boundary case if you can (`bind(65536)`
takes one line). Then state the finding as the *disagreement between the two sides*, not as
an error in one of them — that framing survives review, and "off-by-one" invites the author
to relitigate. Also state the honest scope limit: this mismatch was harmless at every value
except exactly 65536, and saying so made the comment more trustworthy, not less.
Related: `settle-python-semantics-empirically` (see [verify-external-behavior](../verify-external-behavior.md)), `verify-what-a-raise-guard-measures` (see [does-the-failure-really-happen](../does-the-failure-really-happen.md)).

**Corollary — a masked finding folds into its masker.** If finding A makes finding B
unreachable (here: the ports never reach Gym at all, so the bound never matters), post ONE
comment with B as a trailing sentence. Two comments on the same three lines, one of them
unreachable, reads as incoherent and gets both discounted.
