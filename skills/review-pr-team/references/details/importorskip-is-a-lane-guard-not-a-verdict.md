# importorskip-is-a-lane-guard-not-a-verdict

> A pytest.importorskip at the top of a test file says which CI lane runs it, not whether it can run locally — stub the module and run the suite

`tests/unit/data/test_energon_*.py` open with `pytest.importorskip("megatron.core")`. That is a
**CI lane marker**, not a statement that the tests need Megatron. The modules under test never
import `megatron.core`. Injecting a placeholder `megatron.core` module bypasses the skip and the
suite runs on a CPU-only venv in ~10s per file.

Recipe that worked (PR #4108): build a venv with `megatron-energon==<the uv.lock pin>`, torch,
transformers, PyAV, soundfile, librosa; **copy the test files out of `tests/`** first, or the
session-scoped autouse fixture in `tests/unit/conftest.py` starts Ray and writes
`unit_results.json` into the worktree. Do NOT stub `triton` — torch `_inductor` chases it forever.

Control for the "it's just a platform artifact" objection: if some files pass 100% in the same
interpreter with the same stub, the stub is not the cause of the failures.

**Why:** On PR #4108, five lanes — including the leader and devil-advocate — reasoned about 4,300
lines of new data-processing code without running the test suite that ships with it, because one
`importorskip` line read as "unrunnable here." Two lanes eventually disbelieved it and found **29
of the PR's own new tests failing at head**, including 5 source-side bugs invisible to reading. The
leader had already staged a 15-comment review whose body stated that this ground was uncovered and
that nothing had been run. Both statements were false; the review had to be deleted and rebuilt.

**How to apply:** Treat "the tests can't run in this environment" as a premise to verify, not a
fact — the same standard as any other claim (see `verify-lane-collection-claims-yourself` (see [repo-conventions-ci-and-tests](../repo-conventions-ci-and-tests.md))). It
cost ten minutes to disprove. Before writing "no correctness review of X" into a review body, try
running X's tests first. Related: [run-pinned-ruff-directly-when-precommit-cannot-run](run-pinned-ruff-directly-when-precommit-cannot-run.md),
`macos-run-energon-mcore-tests`.
