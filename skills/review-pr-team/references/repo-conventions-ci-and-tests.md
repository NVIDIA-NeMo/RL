# Repo conventions: CI and tests

NeMo-RL CI and test facts.

- **Run `gh pr checks` in setup before any lint or test claim; the Lint job also fails when a file becomes pyrefly-clean but is not in `project-includes`.** Seven lanes said "clean" while Lint was red (#3837). Details: [run-gh-pr-checks-before-any-lint-or-test-claim](details/run-gh-pr-checks-before-any-lint-or-test-claim.md).
- **If `pre-commit` cannot run, use `uvx ruff@<pinned>` and run all three hooks; plain `ruff check` misses `I001`.** (#2957) Details: [run-pinned-ruff-directly-when-precommit-cannot-run](details/run-pinned-ruff-directly-when-precommit-cannot-run.md).
- **Before "no lane runs this test", grep function-level marks and read the L0 lane scripts' `--ignore` lists.** (#3331)
- **A module-level `pytest.importorskip` is a lane marker; stubbing the module may let the tests run.** 29 of the PR's own tests failed at head (#4108). Details: [importorskip-is-a-lane-guard-not-a-verdict](details/importorskip-is-a-lane-guard-not-a-verdict.md).
- **A PR that adds a `nightly.txt` entry must bump the GPU-hour cap by that entry's cost, in the test name, assert and message together; never raise it ahead of need.** The module runs under `-x` in the CPU config checks job, whose failure cancels the rest of a PR run (#4024). Details: [nightly-recipe-must-bump-gpu-hour-cap](details/nightly-recipe-must-bump-gpu-hour-cap.md).
- **Unit tests use the session Ray cluster from `tests/unit/conftest.py`; ask to delete a module fixture that calls `ray.init`/`ray.shutdown` just to get a cluster, because its shutdown stops the shared cluster for every later module.** A test that truly needs a fresh cluster must say why (#4052). Details: [unit-tests-use-session-ray-fixture](details/unit-tests-use-session-ray-fixture.md).
- **A tuple-param fixture's ids are named after the fixture, and a `--deselect` that matches nothing is silent; check `excluded_unit_tests.sh` by running pytest.** (#4207)
- **A functional test's real config is recipe YAML + base script overrides + wrapper overrides.** KL was 0.01, not 0 (#4507).
- **Check that a new test's parameters exercise the feature, not its no-op case (e.g. an M-to-N feature at M=N=1).** (#2797, #3739)
- **When a PR deletes a CI check, compare its trigger conditions with the claimed replacement.** `cicd-main.yml` jobs wait for `/ok to test` (#3235). Details: [deleted-ci-guard-verify-replacement](details/deleted-ci-guard-verify-replacement.md).
- **To test a binary the PR installs in the image, ask for a unit test in the existing lane, not a new CI step.** (#4375) Details: [check-in-the-image-ci-already-builds](details/check-in-the-image-ci-already-builds.md).
- **New files and tests follow their siblings' location and shape; also flag unit tests that assert nothing and YAMLs with local model, tokenizer or dataset paths.** (#3286)
- **Check that a test actually reaches the code it covers: a fake that bypasses the wrapper, a `spec=` mock that makes a new predicate truthy, or a skip that always fires in CI all hide it.** (#2884, #3785) Details: [test-reaches-the-code-it-covers](details/test-reaches-the-code-it-covers.md).
