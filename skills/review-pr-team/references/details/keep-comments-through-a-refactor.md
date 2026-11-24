# keep-comments-through-a-refactor

> A comment a refactor dropped whose reason still holds at the new location is a finding; ask the author to restore it there

When code is moved, split or renamed, the explanatory comments often do not come with it. A comment that explains a non-obvious choice (a workaround, an ordering constraint, an upstream bug, a packaging quirk, the intent of a field) almost always still applies at the new site.

**Why:** the reason is usually invisible in the code itself, so a dropped comment loses context that is expensive to rediscover. It is cheap to keep and comes up on almost every refactor PR, sometimes four times in one.

**How to apply:**

1. On a refactor PR, list the removed comment lines: `git diff <base>...<head> | grep '^- *#'`.
2. For each one, find where its code moved and check whether the comment moved with it.
3. If the reason still holds there, ask the author to restore it at the new location, one comment per site. If the code changed and the comment no longer describes it, that is the opposite case: the comment should be updated or deleted, not restored.

**Seen:** #1757 four dropped comments in `dtensor_policy_worker_v2.py` · #1709 two `setup.py` comments explaining why a step is needed · #2078 · #1628 · #2642 · #2524 field-intent comments dropped in a TypedDict to `@dataclass` refactor
