# dead-code-repo-wide-check

> A symbol is dead only when a repo-wide search at the PR head finds no production consumer, direct or indirect

A named thing that exists to be consumed: a function or method, a class, a config
key, a metric, a dataclass field, a parameter that receives a value, or a return value. It is dead
when nothing in production code reads or calls it. Two shapes, checked the same way:

- the diff removed its last caller and kept the definition;
- the diff added it and nothing uses it yet.

A caller that exists only in tests does not keep it alive: the test then guards code no user runs.

**Why:** dead code never fails. It imports, its tests pass, and an end-to-end run cannot catch it
because it never runs. It still costs: it drifts away from the code that replaced it, and the next
reader assumes it matters. It is also the finding a per-file review misses most, because the
evidence (no caller) lives in files the PR did not touch.

**How to apply:**

1. Search the whole repo at the PR head, not the diff:
   ```bash
   git grep -n "<name>" <HEAD_SHA> -- . ':!3rdparty'
   ```
2. Look for indirect access before concluding there is none: `getattr(obj, "<name>")`, `hasattr`,
   `vars(...)`, a string-keyed registry or dispatch table, YAML keys read through a dict, and
   `mock.create_autospec` / `spec=` in tests.
3. Check whether it is part of an interface: an ABC method, a protocol, or a hook another backend
   overrides. An unused implementation of a required method is not dead.
4. For a config key, resolve every config that inherits it through `defaults:` before calling it
   dead; a parent with `enabled: false` proves nothing if a child turns the feature back on.
5. For a metric, trace it to the logger, not just into a dict: a metric that never reaches
   `log_metrics` is dead even though a unit test asserts its key.
6. For a tuple-returning helper, check what the callers keep: if every caller discards the same
   slot, that slot is dead and the signature can narrow.

In the comment, name the symbol, say the search was repo-wide at the head, and ask for deletion in
this PR. When this diff made it dead, point at the removal that did it. When the PR is one step of a
series and a later PR will consume it, a `# TODO` naming that PR is an acceptable answer.

**Seen:** #3741 `_result_to_completion` left with two unit-test callers after `_run_rollouts` switched to the batch method (user) · #3770 six effort metrics folded into a dict nothing on the SingleController path reads, unit tests asserting the keys (team) · #2997 every caller discarded the IP from `_get_current_node_ip_and_free_port()` (user) · #4196 the review called a YAML block dead because its file had `enabled: false`, but a child recipe re-enabled the feature and needed the block (user)

Related: [diff-removals-are-a-work-list](diff-removals-are-a-work-list.md), `grep-the-symbol-not-just-the-lines` (see [grep-the-symbol-not-just-the-lines](grep-the-symbol-not-just-the-lines.md)).
