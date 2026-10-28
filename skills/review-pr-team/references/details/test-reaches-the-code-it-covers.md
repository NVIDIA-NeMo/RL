# test-reaches-the-code-it-covers

> A green test proves nothing about code it never reaches; check that fakes, `spec=` mocks and skips leave the path under test running

**Why:** these three shapes all keep CI green while the code a PR adds or fixes goes untested.
Nothing raises, and the test's own assertions still pass, so the gap only shows when you trace what
the test actually executes.

**How to apply:**

1. **Fakes.** When the code under test installs a hook, wrapper or patch, the test's fake must call
   through it. A fake whose `forward` sets the attribute the hook would have set tests the fake, not
   the hook; a regression that removes the hook stays green.
2. **`spec=` mocks and a new predicate.** When a PR adds a predicate method (`blocks_training()`,
   `needs_refit()`) and branches on it, grep `tests/` for `MagicMock(spec=<Class>)` and
   `create_autospec(<Class>)`. `spec=` only limits which attributes exist; the new method returns a
   `MagicMock`, which is truthy, so every existing test that mocks the class now takes the new branch:
   ```python
   m = MagicMock(spec=Fake)        # Fake.blocks_training() returns False
   bool(m.blocks_training())       # True
   ```
   Ask to pin the return value in the shared mock helper and to add a variant for the other side.
3. **Skips.** Map a new regression test's `pytest.skip` / `importorskip` conditions against the CI
   shard that collects it: read the shard script's `uv run --extra ...` line, not the author's local
   run. A skip described as "where X is absent" may fire every time in CI, and the skipped half is
   often the one the fix was for. An `N passed, M skipped` line where `M` matches one class's cases is
   the sign.

**Seen:** #2884 `MagicMock(spec=MegatronGeneration)` made `blocks_training()` truthy, so all 14 async GRPO tests took the colocated branch and one silently skipped `refit_policy_generation` (team) · #3785 the only shard that collects the new test runs without the `vllm` extra, so the cases for the one field the PR added to the vLLM config always skipped (team)

Related: `importorskip-is-a-lane-guard-not-a-verdict` (see [importorskip-is-a-lane-guard-not-a-verdict](importorskip-is-a-lane-guard-not-a-verdict.md)).
