# Repo conventions: CI and tests

NeMo-RL CI and test facts.

- **Run `gh pr checks` in setup before any lint or test claim; the Lint job also fails when a file becomes pyrefly-clean but is not in `project-includes`.** Seven lanes said "clean" while Lint was red (#3837).
- **If `pre-commit` cannot run, use `uvx ruff@<pinned>` and run all three hooks; plain `ruff check` misses `I001`.** (#2957)
- **Before "no lane runs this test", grep function-level marks and read the L0 lane scripts' `--ignore` lists.** (#3331)
- **A module-level `pytest.importorskip` is a lane marker; stubbing the module may let the tests run.** 29 of the PR's own tests failed at head (#4108).
- **A PR that adds a `nightly.txt` entry must bump the GPU-hour cap by that entry's cost, in the test name, assert and message together; never raise it ahead of need.** The module runs first under `-x` and kills the shard (#4024).
- **Unit tests should use the session Ray cluster from `tests/unit/conftest.py`; a module that calls `ray.init`/`ray.shutdown` itself needs a comment saying why it needs its own cluster.** (#4052)
- **A tuple-param fixture's ids are named after the fixture, and a `--deselect` that matches nothing is silent; check `excluded_unit_tests.sh` by running pytest.** (#4207)
- **A functional test's real config is recipe YAML + base script overrides + wrapper overrides.** KL was 0.01, not 0 (#4507).
- **When a PR deletes a CI check, compare its trigger conditions with the claimed replacement.** `cicd-main.yml` jobs wait for `/ok to test` (#3235).
- **To test a binary the PR installs in the image, ask for a unit test in the existing lane, not a new CI step.** (#4375)
