# Claims searches and numbers

"nothing", "only", "every", counts and figures are the claims an author can disprove in one grep.

- **An empty search or a truncated read cannot prove absence; read the exact file:line the other lane cited, and list files with `gh pr view --json files` before saying a PR "does not touch X".** The devil's advocate made four wrong disputes this way on #4108; `head -120` of a diff hid a deleted file on #4196.
- **Before saying "module X is not imported on this path", read every parent `__init__.py`.** An eager package `__init__` made the existing probe cover everything on #4112, killing four findings.
- **A grep locates a symbol; read the file before saying what it shows.** The template project's point was `worker_extension_cls_fqn`, not the string that matched (#3408).
- **Any "only/every/none/always" claim needs the complement enumerated; prefer deleting the quantifier.** "The only unguarded `ray.kill`" was false: 25 sites, one unguarded in `grpo.py` (#4112).
- **Before "this param is redundant, its output is identical", name the code that stops running if it is deleted.** The `deepseek_r1` param was the only CI path filling `_reasoning_start_suffixes` (#3422).
- **Before "the PR forgot to add a row to Y", count how much of the population Y covers.** 2 of 17 is a sample, not a convention; 15 of 17 is (#3619). Details: [count-the-artifact-before-calling-it-incomplete](details/count-the-artifact-before-calling-it-incomplete.md).
- **For a new default, count the call sites that rely on it; zero or one makes "remove the default" a free deletion.** (#3367)
- **When a comment pairs two counts, make sure both count the same set.** "25 tests never run" next to "79 passed" read as 79 orphaned (#3592).
- **Prefer figures that survive a base change.** A count between two in-PR commits stayed right; ordinals from the merge base went stale three times (#3837).
- **A number in a config comment may be an extrapolation; find where it came from before building severity on it.** "~500s at 405B" was 1.9s × 270; real refits are under 30s (#3929).
- **Two CI failures near a threshold are a reason to ask for reruns before calling it a regression.** The next run on #4129 passed.
