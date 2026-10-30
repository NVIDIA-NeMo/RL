# Dead and outdated code

What a diff leaves behind: code nothing uses any more, and comments or docs that no longer match the code.

- **Dead code is a named thing that exists to be consumed and has no production consumer: a function, method, class, config key, metric, dataclass field, parameter, or unread return value.** It covers both a symbol that lost its last caller in this diff and a new symbol nothing reads. A test-only caller does not count as a consumer. Details: [dead-code-repo-wide-check](details/dead-code-repo-wide-check.md).
- **Before calling something dead, grep the whole repo at the PR head, not the diff, and check indirect access (`getattr`, `hasattr`, `vars`, string dispatch, `spec=`) and whether it is part of an ABC or interface another backend implements.** A search of the changed files misses callers in files the PR never touched.
- **Treat every name the diff removes as a work list: grep each one at the head.** Callers removed but the definition kept is dead code; a test removed but its name still in a registry (a `--deselect` entry) fails silently; an ABC declaration removed and kept only on the concrete class is a contract downgrade. Details: [diff-removals-are-a-work-list](details/diff-removals-are-a-work-list.md).
- **When a refactor removes a no-op at one call site, check the sibling call sites for the same no-op.** The cleanup usually means the operation was a no-op everywhere.
