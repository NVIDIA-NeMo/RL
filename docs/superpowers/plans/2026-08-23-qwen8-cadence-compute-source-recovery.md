# Qwen3-8B Cadence Compute-Visible Source Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run the complete 13-arm Qwen3-8B static/always/fixed/adaptive matrix from an immutable source tree that is visible inside every OCI-HSG Pyxis container.

**Architecture:** Package the exact signed recursive-clean repository as a SHA256-bound archive on Lustre. Each array task extracts its own copy under `/raid/scratch`, explicitly mounts that copy into the nested Pyxis container, and runs the existing fail-closed arm lifecycle from the extracted Git checkout. A one-arm startup canary must exercise this exact staging and nested-container path before a new exactly-once 13-arm array is allowed.

**Tech Stack:** Python 3.13, Bash, unittest, Git, tar, SHA256, Slurm, Pyxis, Ray, NeMo-RL.

**Spec:** `research/qwen3_8b_draft_cadence_200step/README.md`

## Global Constraints

- Preserve the existing 13-arm workload, product behavior, checkpoints, CUDA Graph buckets, W&B project, and cadence schedules unchanged.
- Never reuse or overwrite the failed `20260823-d9f3a89ac` result root or its exactly-once ledger.
- Source archive identity, product SHA, harness SHA, result root, account, and array mapping must be immutable and recorded before submission.
- Run `sbatch --test-only` before every actual canary or matrix submission.
- Submit the full matrix as one atomic Slurm array only after the exact-path startup canary passes.
- Monitor actual jobs at no less than 60-second cadence for at least five minutes.

---

### Task 1: Reproduce and lock the source-visibility failure

**Files:**
- Modify: `research/qwen3_8b_draft_cadence_200step/tests/test_launch.py`
- Modify: `research/qwen3_8b_draft_cadence_200step/run_arm.sh`

**Interfaces:**
- Consumes: existing `build_submission` and arm runner contract.
- Produces: a failing contract that rejects an unmounted `/home` working tree and accepts only `/home` or task-owned `/raid/scratch` source roots inside the container.

- [x] Add a test asserting the staged runner uses `/raid/scratch/q8c200-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}-r${SLURM_RESTART_COUNT:-0}` and explicitly mounts the extracted source path.
- [x] Run the focused test and record the expected RED caused by the existing `/home`-only path.
- [x] Make the smallest runner validation change needed for an extracted `/raid/scratch` checkout.
- [x] Run the focused test and the complete research suite GREEN.

### Task 2: Add immutable staged-array rendering

**Files:**
- Create: `research/qwen3_8b_draft_cadence_200step/staged_launch.py`
- Create: `research/qwen3_8b_draft_cadence_200step/tests/test_staged_launch.py`

**Interfaces:**
- Consumes: `build_arms()`, the existing arm runner, a full product SHA, source archive path/SHA256, result root, and W&B generation suffix.
- Produces: `render_staged_array_script(...) -> str` and `build_staged_array_argv(...) -> tuple[str, ...]`.

- [x] Write tests for exact 13-arm mapping, archive checksum verification, job/task-specific scratch extraction, explicit source mount, preflight-only canary mode, segment 1, and no `/home` chdir dependency.
- [x] Run the tests and record RED because the module does not exist.
- [x] Implement the minimal renderer and argument builder.
- [x] Run focused and full research suites, Ruff, format, Bash syntax, ShellCheck, compile, and diff checks.

### Task 3: Freeze and review the recovery head

**Files:**
- Modify: `research/qwen3_8b_draft_cadence_200step/README.md`
- Modify: `research/qwen3_8b_draft_cadence_200step/PREPARATION_REPORT.md`

**Interfaces:**
- Consumes: the GREEN staged-source contracts.
- Produces: one signed+DCO commit and an exact source-archive construction recipe.

- [x] Document the failed array IDs and causal Pyxis working-directory error.
- [x] Document archive creation, SHA verification, canary, and new-generation rules.
- [ ] Run all local verification commands freshly.
- [ ] Create a signed+DCO commit, verify its signature, and push only the recovery branch.

### Task 4: Run the exact-path startup canary

**Files:**
- Create remotely under the new result root: `source.tar`, `source.tar.sha256`, `manifest.json`, and the rendered staged array script.

**Interfaces:**
- Consumes: exact pushed recovery SHA and rendered staged script.
- Produces: a durable canary receipt proving extraction, Git/submodule identity, explicit source mount, nested Pyxis workdir, container import, and product preflight.

- [ ] Create and checksum the recursive-clean source archive on the login node.
- [ ] Validate the archive by extracting it into a temporary login-node directory and rerunning signed-source/submodule checks.
- [ ] Pass `sbatch --test-only` for the exact canary command.
- [ ] Submit one preflight-only canary and monitor it to terminal.
- [ ] Stop without matrix submission if any source, mount, container, or product preflight gate fails.

### Task 5: Submit and monitor the recovered 13-arm matrix

**Files:**
- Create remotely under the new result root: test-only ledger, actual intent, submission ledger, scheduler logs, and per-arm result directories.

**Interfaces:**
- Consumes: successful canary receipt and the same immutable archive/script/product identities.
- Produces: one exactly-once Slurm array covering baseline plus DFlash/DSpark static, always, fixed-5/10/20, and adaptive arms.

- [ ] Run all 13 `sbatch --test-only` checks and atomically write the test-only ledger.
- [ ] Validate the manifest, canary, and ledger before actual submission.
- [ ] Submit one `0-12` array and write the intent/submission ledgers atomically.
- [ ] Monitor the grouped job set for at least five minutes at 60-second cadence.
- [ ] Report exact job IDs, states, first gates, W&B URLs, and any causal failure without claiming performance until terminal evidence exists.
