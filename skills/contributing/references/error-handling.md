# Error handling

How to catch errors, and when to raise them.

- **When using try-except blocks, limit the except to the smallest set of errors possible.** Contributing Details: [error-handling](details/error-handling.md).
- **Do not add meaningless `try`/`except`/`finally` wrappers around code that already fails loudly, or claim fault tolerance when the whole training run still crashes.** Contributing Details: [error-handling](details/error-handling.md#avoid-meaningless-exception-wrappers).
- **Fail loud, not silent: when adding a config option or branch, ask what the worst plausible misconfiguration does. If it silently produces wrong results rather than an error, add a setup-time `assert`/`raise` so it fails loudly at startup instead of corrupting a run.** Contributing Details: [error-handling](details/error-handling.md).
