# dont-disqualify-a-fix-with-a-pre-existing-failure

> Before rejecting a simpler fix because "it leaves X failing", check whether X fails under every candidate fix — if so it is a separate problem, not an argument

When you reject a simple, central fix because it "leaves some other thing broken", first ask
whether that other thing breaks under **every** candidate fix, including the one you prefer. If it
does, it is an independent problem and must not be used to disqualify the simpler option.

**Why:** On PR #4020 the finding was that `create_local_venv` had no guard against being handed a
bare interpreter. The obvious fix was a passthrough inside that one function. I argued against it
because it would leave `test_mcore_py_executable` red — that test asserts a venv exists at
`{tempdir}/test_venv/bin/python`. But that assertion is false under the flag *regardless* of which
fix is chosen; the test needed a `skipif` either way. I let a pre-existing, fix-independent failure
masquerade as a defect in one specific fix, and steered the author toward two scattered call-site
guards instead of one central one. The user pushed back with "why not just change
`create_local_venv` … this way we don't need so many if statements all over the place", and was
right. Compounding it: a devil-advocate pass had *also* endorsed my framing, and separately claimed
the central guard "would have to be added to two functions" — it had missed that
`create_local_venv_on_each_node` reaches `create_local_venv` via `_env_builder`, so one guard
really did cover everything.

**How to apply:** For any "fix A is wrong because it leaves B broken" claim, write out B's status
under fix A, under fix B, and under no fix. Only if B is broken *specifically by* fix A is it an
argument. Otherwise state B separately as its own item. And before claiming a central guard cannot
cover all callers, trace the call graph to a leaf — helpers routinely funnel into one another
(`X_on_each_node` -> `_env_builder` -> `X`), which is exactly what makes the central fix correct.
Related: `challenge-the-challenger-das-premises-need-verifying-too` (see [lane-adjudication](../lane-adjudication.md)),
[verify-the-consumer-not-just-the-validator](verify-the-consumer-not-just-the-validator.md).
