# stall-comments-link-page-and-ask-for-test

> A comment about a stall/hang/wasted pause must link its explainer page and ask for a unit test that fails today and passes once fixed

When a review comment describes a stall — something that hangs, times out, or pauses work for nothing — it must
(1) link the explainer page for it, if one exists, and (2) ask for a unit test that fails today and passes once the
issue is fixed, described at the behavior level (inputs, what to trigger, what to assert).

For the test to be meaningful, the comment has to say what "fixed" means, so it needs a concrete fix, not just
"please measure". Check that the test is writable against the repo's fakes: a fix that calls `ray.cancel(ref)` could
not be tested with a fake actor that returns plain coroutines until the cancel was put behind a small method the test
can replace. And never let a hang-style test hang: wrap it in `asyncio.wait_for(..., timeout=...)`, because the repo
has no global pytest timeout and a hang stalls the CI shard.

**Why:** On PR #4266 (2026-09-29) the step-during-Gym-pause comment only asked for a discard rate. The user asked that
stall comments link their page and ask for a unit test that passes once fixed. Devil-advocate then found the first test
wording was not writable (fake actor, no ObjectRef) and claimed a hang "until the prepare deadline" that nothing on
the RL side enforces.

**How to apply:** For every stall finding: page link + fix + test. Run the test wording past the devil-advocate for
"can this be written with the existing fakes?" and "what does it do today, exactly?" before posting. Related:
`show-the-breaking-case`, `proposed-fixes-need-their-own-adversarial-pass` (see [proposed-fixes](../proposed-fixes.md)).
