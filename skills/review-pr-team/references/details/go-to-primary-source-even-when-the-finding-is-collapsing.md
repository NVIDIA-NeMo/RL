# go-to-primary-source-even-when-the-finding-is-collapsing

> When a finding's severity turns on one binary fact and the source is public, read the source — especially when the finding looks like it is dying

If a finding's severity depends on a single yes/no fact about an external library, and that
library's source is public, **go read it**. Do this even when — especially when — the finding
appears to be collapsing and the cheap move is to downgrade and move on.

**Why:** on PR #3591 a finding about NCCL communicator teardown had been argued for several
rounds from *documentation* and a Python wrapper. Two mechanisms for it had already been refuted,
the confidence had drifted to ~55-60, and the team lead **stood the lane down** on the reasoning
that the finding no longer rested on NCCL semantics. The lane pushed back, fetched
`NVIDIA/nccl` at the pinned tag, and settled the question in four HTTP fetches:

- a split child communicator gets **its own** abort flag unless `splitShare` is set;
- `splitShare` defaults to 0;
- `ncclCommAbort` sets only the parent's flag plus a `childAbortFlag` that is explicitly
  unlinked the moment child init completes (`/* unlink child abort flag. */`).

Conclusion: the abort does **not** reach a split child, so a rank blocked on one is never
released — an unbounded hang, on the exact path the PR existed to make recoverable. It went from
a ~57%-confidence note to the review's most severe finding, and no exception-translation fix
reached it. The lead's stand-down was wrong and the lane was right to ignore it.

The general shape: **"I cannot prove a mechanism" is not "there is no mechanism."** Downgrading
on the absence of evidence, when the evidence is one `curl` away, is the error.

**How to apply:**
- Pin the version first (`uv.lock`, `pyproject.toml`), then read at that exact tag. Quote code,
  permalink the line, never argue from a doc sentence when source is available — a doc sentence
  about "aborting an in-flight split" was over-read into "abort does not propagate to children",
  which was the right conclusion from the wrong evidence.
- Vendored/installed sources count too: vLLM's real behaviour was settled by reading the wheel in
  the local `uv` cache, and nccl4py's by unpacking the pinned wheel.
- As a lead: before standing a lane down, ask whether the question it was given has a cheap
  decisive answer. If it does, let it finish.
- "Conclusion right, evidence wrong" is the more dangerous error than being flatly wrong: it
  survives review because the conclusion checks out, and breaks later when someone leans on the
  reasoning. Correct the evidence even when the verdict stands.
