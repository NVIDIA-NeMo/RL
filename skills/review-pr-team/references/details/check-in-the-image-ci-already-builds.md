# check-in-the-image-ci-already-builds

> When asking for a CI check of a tool the PR installs in the image, ask for a plain unit test in the existing lane, not a new CI step or a second image

If the PR adds a binary to `docker/Dockerfile` and you want CI to run it (a config validator, a linter), ask for a plain unit test that calls the installed binary. Do not ask for a separate CI step that pulls the tool's own image.

**Why:** L0 unit tests already run inside the `release` image CI builds from `docker/Dockerfile` (`.github/workflows/cicd-main.yml`, `build-container` → `target: release`). A unit test there uses the exact pinned version with no extra pull and no second version pin to drift. It also fails if a later Dockerfile change drops the binary, which nothing else checks. On PR #4375 the review first asked for `docker run grafana/alloy:v1.18.0 validate …` as a new CI step; the user asked why not just use the build image.
**How to apply:** Check which Dockerfile stage the binary lands in and that the unit-test lane uses that image. Then suggest `subprocess.run(["/usr/local/bin/<tool>", …], check=True)` in `tests/unit/`, failing rather than skipping when the binary is missing (see [importorskip-is-a-lane-guard-not-a-verdict](importorskip-is-a-lane-guard-not-a-verdict.md)).
**Seen:** #4375 C7 alloy validate (user)
