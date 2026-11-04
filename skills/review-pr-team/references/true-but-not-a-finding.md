# True but not a finding

Correct observations that still should not be posted, and checks to run before "this is unneeded/wrong".

- **An assert that can never fire is a working sanity check; post only if it lets a live bug through, and lead with that bug.** The reviewer deleted a "tautological check" comment on #3378.
- **When a finding's premise dies, delete the whole comment; do not keep a smaller piece.** The trimmed version on #3633 asked for a no-op.
- **A soft guideline ("usually", "a smell unless") is not a finding without a concrete cost.** `default_factory=dict` on a telemetry field was pruned on #4294.
- **A code comment that is true for its main caller is not "inaccurate" because it is false on another path.** (#3618)
- **"The author missed X" is a second claim; grep the diff, docstrings and commit messages first.** The author had a docstring about the V2 executor on #3280.
- **To call a mechanism "not how we do it", run the disconfirming grep across all siblings; one hit kills it.** One recipe exported both env vars, refuting three agents (#2689).
- **Before asking to remove a `getattr`/`hasattr`, list every way the attribute can be missing: `object.__new__` in tests, an `Optional` type, evaluation order.** (#3490, #3883) This is about a defensive read of an attribute that can really be missing; TEST F's ban is on a `getattr` default standing in for a required class property. Details: [check-who-constructs-before-flagging-defensive-getattr](details/check-who-constructs-before-flagging-defensive-getattr.md).
- **If you cannot see the artifact a value came from (an out-of-tree recipe), post only the type/shape problem and hand the value back.** Two lanes proposing two different "correct" values is the sign (#2689).
