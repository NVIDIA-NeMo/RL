# run-gh-pr-checks-before-any-lint-or-test-claim

> Run `gh pr checks` early — a local tool run is not the CI job, and NeMo-RL's Lint job also runs pyrefly plus a whitelist rule that a passing local ruff will never show you

**Run `gh pr checks <n> --repo NVIDIA-NeMo/RL` in the setup phase, before saying anything about
lint or tests.** It costs one call and it is the only thing that tells you what the repo's own
gates think.

A local tool run is **not** the CI job. NeMo-RL's `Lint check` runs `pre-commit` (ruff x3) **and**
`pyrefly`, **and** a third step with no local equivalent at all — `.github/workflows/cicd-main.yml`
"Check if any files with zero errors not in whitelist", which lists every file `pyrefly` reports
zero errors for and fails if one is missing from `pyrefly.toml`'s `project-includes`.

That inverse rule is the trap. Everyone checks the forward direction — *are the new source files
whitelisted?* — because that is what the contributor skills talk about. CI enforces the other
direction, and a change that makes an **unrelated, untouched** file type-clean turns the build red.

**Why:** on PR #3837 a seven-lane review with two adversarial rounds reported "linters are clean"
and "pyrefly.toml: no gaps", and the `Lint check` was red the whole time. The PR added
`"nemo_gym.*"` to `replace-imports-with-any`, which made `examples/nemo_gym/__init__.py` — not in
the diff, not touched by the author — resolve its imports to `Any`, drop to zero errors, and
therefore require a whitelist entry it did not have. The lane that owned `pyrefly.toml` checked the
five new source files were listed (they were) and stopped. Nobody ran `gh pr checks`. The failure
was visible in three seconds.

Second thing that call gives you free: **whether any tests ran at all.** That same head reported
`CI quality check — No tests run: Label CI:L*`. Writing "the tests pass" when no lane has been
triggered is worse than saying nothing.

**How to apply:**
1. `gh pr checks <n>` in setup, alongside the diff fetch. Record which checks are red.
2. For a red check, `gh run view --job <id> --log` and grep past the git/credential noise —
   `--log-failed` often returns only teardown lines and hides the real error.
3. Never write "lint is clean" from a local `uvx ruff` run. Say what you ran and what CI says, and
   when they disagree, CI wins and the disagreement is itself the finding.
4. Treat a red gate caused by the PR as a first-tier finding: it blocks merge, it is cheap to fix,
   and the author usually cannot see the cause from the job name alone.

Related: [run-pinned-ruff-directly-when-precommit-cannot-run](run-pinned-ruff-directly-when-precommit-cannot-run.md) (how to lint locally when
`pre-commit` cannot build a venv — necessary, but not sufficient on its own),
[deleted-ci-guard-verify-replacement](deleted-ci-guard-verify-replacement.md), `verify-lane-collection-claims-yourself` (see [repo-conventions-ci-and-tests](../repo-conventions-ci-and-tests.md)).
