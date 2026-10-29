# diff-removals-are-a-work-list

> Every named thing a diff removes is a lead: grep it at the head and classify what survives

**Why:** reviews read the `+` lines, because that is what the author asks you to look at. The
removals are where unreviewed defects hide, and they often surface in a file the PR never touched,
so no per-file reviewer is assigned to it. None of them produce an error: dead code still imports, a
stale `--deselect` is silently ignored by pytest, and an `Any`-typed caller hides an ABC change. A
passing end-to-end run does not rule them out.

**How to apply:** collect every `-` line that removes a named thing (a `def`, a `class`, a test
function, a call site, an ABC declaration, a config key, a comment carrying an issue number or SHA).
For each one, search at the PR head:

```bash
git grep -n "<removed_name>" <HEAD_SHA>
```

Then classify what survives:

| What the PR removed | What survives | Finding |
|---|---|---|
| every call site of a function | the definition | dead code |
| a test function | its name in a list such as `tests/unit/excluded_unit_tests.sh` | stale entry; nothing fails |
| a declaration on an ABC | the method on one concrete class only | contract downgrade; other backends are no longer required to implement it |
| a comment carrying an issue link | nothing | the last in-repo record of why something was disabled is gone |
| a guard or validation | a `docs/` page describing what it rejected | the doc now describes behaviour that no longer exists |

For a removed guard, grep `docs/` for the thing it rejected, not the function name: docs describe a
rejection in prose and rarely name the function.

**Check each hit before reporting it.** A mechanical sweep produces false positives in bulk. Resolve
class-scoped test IDs (`path::Class::method`) and parametrized IDs (`name[param]`) to the real `def`
before calling an entry stale, and confirm each one by hand.

**Seen:** #3724 the refit guard for grouped-MoE MXFP8 was deleted, but `docs/guides/refit.md` (not in the diff) still said refit rejects those slabs (team) · #3741 `_result_to_completion` was removed, but a wiring comment still pointed at it (team)

Related: [dead-code-repo-wide-check](dead-code-repo-wide-check.md).
