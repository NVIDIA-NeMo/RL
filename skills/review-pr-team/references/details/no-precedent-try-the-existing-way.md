# no-precedent-try-the-existing-way

> A mechanism with no precedent in the repo is a finding to raise, not something to accept because it works: name the precedent's path in the ask, and accept the new way only when the existing one does not work

**Why:** On #3613 two unprecedented mechanisms passed several review passes because they worked. The chaos harness forced a private Ray cluster (`RAY_ADDRESS=local`) where the sibling chaos script attaches to the existing one, and it broke on `ray.sub` nodes. A `tests/unit` file unit-tested the functional harness, which nothing else in `tests/unit` does. Both were switched to the existing way.

**How to apply:**
1. Settle "is there precedent" with `git grep` on `origin/main`, never from memory.
2. If there is none, find the nearest sibling that solves the same problem and ask whether the PR can use its way. Name the sibling's path, with a permalink, in the ask.
3. Accept the new way only when the existing one does not work.
4. Before claiming "this is not how we do it", run the disconfirming grep across all siblings; one hit kills the claim.

**Seen:** #3613
