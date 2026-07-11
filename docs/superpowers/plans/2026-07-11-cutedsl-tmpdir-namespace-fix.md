# CuTeDSL TMPDIR Namespace Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the batch host and Pyxis container temporary directories valid in their respective namespaces, even when the container login environment supplies a stale `TMPDIR`.

**Architecture:** The batch process uses the host-mounted source directory `${HOST_RUNTIME_DIR}/tmp`; `TMPDIR` is no longer propagated through Pyxis. Every container shell overrides `TMPDIR` to `${CONTAINER_RUNTIME_DIR}/tmp` as its first inner command, after login initialization, then logs and validates the mounted directory before doing work.

**Tech Stack:** Bash, SLURM, Pyxis, pytest, Python 3.13, static HTML/JSON experiment reports.

## Global Constraints

- Apply the namespace split to both functional and matrix payloads.
- Cover every `srun` container phase, including runtime bootstrap.
- Preserve container-side `TMPDIR` in generated provenance.
- Record jobs `1910599` and `1911208` in the committed incident timeline.
- Run full relevant tests and checks, create one signed commit, and do not push or submit jobs.

Local execution note: the repository's `uv` environment is Linux-only, so the
macOS worktree used `python3 -m pytest` for the same test files and `uvx ruff`
for lint/format verification. The Pyrefly hook remains a Linux gate.

---

### Task 1: Encode the host/container TMPDIR contract

**Files:**

- Modify: `tests/test_oci_cutedsl_wrapper.py`
- Test: `tests/test_oci_cutedsl_wrapper.py`

**Interfaces:**

- Consumes: the marked `CUTEDSL_RUNTIME_TMPDIR` host initialization block and every `"${SRUN[@]}" bash` payload body.
- Produces: regression coverage requiring a host-valid `TMPDIR`, excluding `TMPDIR` from `--container-env`, and overriding stale login values inside every container shell.

- [x] **Step 1: Write failing regression tests**

Update the Pyxis environment test to require only `VIRTUAL_ENV`, `UV_PROJECT_ENVIRONMENT`, and `NVTE_CUDA_ARCHS`. Update the runtime TMPDIR test to execute the host initialization block with a stale inherited value and require `${HOST_RUNTIME_DIR}/tmp`. Extract each container shell preamble, require its first command to be `export TMPDIR="${CONTAINER_RUNTIME_DIR}/tmp"`, execute it with a stale initial `TMPDIR`, and require the container-mounted path.

- [x] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
uv run pytest tests/test_oci_cutedsl_wrapper.py \
  -k 'pyxis_explicitly_overrides_image_runtime_environment or payloads_override_stale_tmpdir_for_every_profile' -q
```

Expected: failures show that the host currently receives `/runtime/tmp`, Pyxis still propagates `TMPDIR`, and the first inner command is not the post-login override.

### Task 2: Split host and container TMPDIR namespaces

**Files:**

- Modify: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_functional.sbatch`
- Modify: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch`
- Test: `tests/test_oci_cutedsl_wrapper.py`

**Interfaces:**

- Consumes: `${HOST_RUNTIME_DIR}`, `${CONTAINER_RUNTIME_DIR}`, and each existing Pyxis `SRUN` array.
- Produces: host `TMPDIR=${HOST_RUNTIME_DIR}/tmp`; container `TMPDIR=${CONTAINER_RUNTIME_DIR}/tmp` established after shell startup.

- [x] **Step 1: Implement the minimal wrapper change**

Set the host-side export to:

```bash
export TMPDIR="${HOST_RUNTIME_DIR}/tmp"
```

Remove `TMPDIR` from `--container-env`. At the start of every container body, before `set -euo pipefail`, add:

```bash
export TMPDIR="${CONTAINER_RUNTIME_DIR}/tmp"
set -euo pipefail
printf '[INFO] Container temporary directory: TMPDIR=%s uid=%s gid=%s\n' \
    "${TMPDIR}" "$(id -u)" "$(id -g)"
[[ "${TMPDIR}" == "${CONTAINER_RUNTIME_DIR}/tmp" ]]
[[ -d "${TMPDIR}" && -w "${TMPDIR}" ]]
```

Keep the runtime-environment logger after this preamble so metadata and generated provenance record the container value.

- [x] **Step 2: Run the focused tests and confirm GREEN**

Run the focused command from Task 1 and require all selected tests to pass.

- [x] **Step 3: Run the complete wrapper test file**

Run:

```bash
uv run pytest tests/test_oci_cutedsl_wrapper.py -q
```

Expected: all wrapper tests pass.

### Task 3: Record live-job evidence

**Files:**

- Modify: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/report/incidents.json`
- Modify: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/report/public/index.html`
- Test: `tests/test_cutedsl_report.py`

**Interfaces:**

- Consumes: job `1910599` silent bootstrap failure and job `1911208` host/container diagnostic evidence.
- Produces: deterministic committed root-cause timeline entries describing symptom, boundary evidence, root cause, tested fix, and pending verification.

- [x] **Step 1: Add a failing aggregate-report expectation**

Require the committed incident data and rendered aggregate to contain both job IDs and the diagnosed stale post-login `TMPDIR` boundary.

- [x] **Step 2: Run the focused report test and confirm RED**

Run:

```bash
uv run pytest tests/test_cutedsl_report.py::test_aggregate_report_uses_local_assets_and_incident_timeline -q
```

Expected: failure because the committed incident timeline is empty.

- [x] **Step 3: Add incidents and render the aggregate**

Add bounded structured entries for jobs `1910599` and `1911208`, then run:

```bash
uv run --no-project python experiments/cutedsl_qwen3_30ba3b_oci_1n4g/render_cutedsl_report.py \
  --render-aggregate experiments/cutedsl_qwen3_30ba3b_oci_1n4g/report
```

If the renderer does not expose a direct aggregate-only option, invoke its documented equivalent without refreshing from absent local result directories.

- [x] **Step 4: Run the focused report test and confirm GREEN**

Run the command from Step 2 and require it to pass.

### Task 4: Verify and commit

**Files:**

- Verify all modified files.

**Interfaces:**

- Consumes: Tasks 1–3.
- Produces: one locally verified signed commit.

- [x] **Step 1: Run full relevant checks**

Run:

```bash
uv run pytest tests/test_oci_cutedsl_wrapper.py tests/test_cutedsl_report.py -q
uv run pre-commit run --files \
  tests/test_oci_cutedsl_wrapper.py \
  tests/test_cutedsl_report.py \
  experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_functional.sbatch \
  experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch \
  experiments/cutedsl_qwen3_30ba3b_oci_1n4g/report/incidents.json \
  experiments/cutedsl_qwen3_30ba3b_oci_1n4g/report/public/index.html \
  docs/superpowers/plans/2026-07-11-cutedsl-tmpdir-namespace-fix.md
```

Expected: zero test failures and zero pre-commit failures.

- [x] **Step 2: Inspect the final diff and commit**

Run `git diff --check`, inspect `git diff`, stage only the task files, and create:

```bash
git commit -s -m "fix: override CuTeDSL tmpdir after container login"
```

Do not push and do not submit jobs.
