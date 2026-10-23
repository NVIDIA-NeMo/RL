# widening-a-type-can-be-the-wrong-direction

> When a PR widens a parameter to Optional to silence a mismatch, check whether the value can actually be None on that path — the honest fix is often to narrow the producer instead, and "make the callee match the caller" can propagate a lie

A PR widens a signature (`list[str]` -> `list[Optional[str]]`) because the caller's declared type is
wider. The reflex finding is "you widened the parameter but not the struct it feeds — widen that too,
for consistency." **That reflex can be exactly backwards.** Before proposing it, ask two questions:

1. **Can the value actually be `None` on this path?** Producers are often declared `Optional` for
   reasons that do not apply to the caller you are looking at (a sync branch, a backend that is gated
   off). Check the gates.
2. **Does the body handle the wider case?** A function that accepts `Optional` and then passes the
   value straight into a non-optional field is not "more permissive" — it accepts something it cannot
   process. That is the real defect, and it is sharper than the consistency argument.

If the answer to 1 is no, widening the struct **records a falsehood** ("this API accepts None") that
the next reader will believe. Narrowing is the honest fix, and the mismatch's real source is usually a
sentinel value upstream.

**Why:** On PR #3367 the review staged "widen `NemoGymConfig.base_urls` to `List[Optional[str]]` so it
matches the new signatures" at confidence 80, through six agents and an adversarial pass. The user asked
the one question nobody had: *"list of Optional[str] means it could be a list of all Nones. is that
actually valid?"* It is not. `should_use_nemo_gym` asserts `async_engine` + `expose_http_server`, the
deferred vLLM path binds ports in `__init__` so URLs exist before model load (the class docstring says
"populated immediately from reserved ports"), and Gym's own pydantic field would reject a `None` at
server startup. The widening's stated justification — in the commit message — was contradicted by the
code's own docstring. The recommended fix would have shipped an annotation asserting the opposite of
the truth.

The root cause was upstream and invisible from the diff: three backends return `[None]` to mean "no
server" while Megatron returns `[]` after filtering. That sentinel is what forces `Optional` through
the whole chain. Filed as a separate issue rather than asked for on the PR, since those files were not
in the diff.

**How to apply:** For any "the types don't line up" finding, decide the *direction* before writing.
Grep the producer for where the `Optional` actually originates and whether a mixed list is ever
produced — usually the value is all-real or all-`None`, never mixed, which means a sentinel and not a
genuine optional. Look for a sibling backend/implementation that already does it the narrow way and
cite it; that turns a taste argument into a precedent. And read the commit message's own justification
against the code — an author's stated reason is a checkable claim, and when it is wrong that is a
better finding than the typing nit. Related:
[dont-propose-a-swallow-if-every-trigger-is-a-bug](dont-propose-a-swallow-if-every-trigger-is-a-bug.md) (same family: the proposed fix encodes something
untrue), [check-upstream-mandate-before-flagging-asymmetry](check-upstream-mandate-before-flagging-asymmetry.md), `dead-guard-is-not-a-finding` (see [true-but-not-a-finding](../true-but-not-a-finding.md))
(the guard version of this: do not propose an assert for a state that cannot occur).
