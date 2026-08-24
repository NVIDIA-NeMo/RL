# Qwen3-30B-A3B SWE rollout benchmark implementation plan

**Goal:** Add a contract-validated, exactly-once OCI-HSG benchmark harness for
the approved five-arm PR #3733 rollout-only comparison.

**Architecture:** A checked-in JSON manifest owns immutable identities and arm
definitions. A typed Python CLI validates the manifest, derives CUDA Graph
coverage, renders NeMo-RL commands, and maintains atomic submission records. A
second typed Python entrypoint performs OCI-HSG Slurm test-only, submission, and
monitoring transitions only after the preflight passes.

**Tech stack:** Python 3.13 standard library, pytest, Bash, NeMo-RL OmegaConf
recipes, Slurm/Pyxis.

**Spec:** `experiments/qwen3_30ba3b_swe_rollout_pr3733/DESIGN.md`

## Global constraints

- Work from PR head `b580dd8927b88c996470d315e74d57bf0cb4090e`.
- Keep baseline, DFlash K5/K7, and DSpark K5/K7 matched except for method, K,
  draft checkpoint, and derived CUDA Graph shapes.
- Never submit from test-only mode or before canary success.
- Record each profile/arm submission exactly once.
- Use W&B entity `nvidia` and project `sna-specdec`.
- Do not claim PR-level 30B GPU proof.

### Task 1: Contract tests

**Files:**

- Create: `tests/unit/test_qwen3_30ba3b_swe_rollout_benchmark.py`

- [x] Write tests that load the real manifest and invoke the real CLI.
- [x] Assert exact identities, five full arms, two canary arms, matched sampling
  and topology, rollout-only overrides, and W&B routing.
- [x] Assert DFlash and DSpark CUDA Graph token shapes from literal expected
  lists.
- [x] Assert a second claim for the same profile/arm fails.
- [x] Observe contract RED before adding the manifest and each lifecycle stage.

### Task 2: Manifest and planner

**Files:**

- Create: `experiments/qwen3_30ba3b_swe_rollout_pr3733/benchmark_matrix.json`
- Create: `experiments/qwen3_30ba3b_swe_rollout_pr3733/benchmark.py`

- [x] Add a typed manifest loader that rejects unknown/missing arms, identity
  drift, unmatched common settings, unsupported K values, and incomplete CUDA
  Graph coverage.
- [x] Render baseline with `speculative_config=null`; render speculative arms
  with the exact method, checkpoint, and K.
- [x] Derive DFlash query width `K+1`; derive this DSpark checkpoint's draft
  width `K` and target verification width `K+1`.
- [x] Add atomic exclusive reservations, submissions, and completion records.
- [x] Re-run the focused local pytest fallback and require zero failures.

### Task 3: OCI-HSG launcher

**Files:**

- Create: `experiments/qwen3_30ba3b_swe_rollout_pr3733/submit.py`

- [x] Validate the clean source SHA, PR ancestry, container SHA256, target and
  drafter config hashes, data SHA256/line count, and absolute output paths.
- [x] Require successful `sbatch --test-only` with the identical contract before
  each actual submission.
- [x] Require successful baseline and DFlash K5 canary records before allowing
  the full matrix.
- [x] Submit through `ray.sub` with two nodes, four GPUs per node, an explicit
  account/partition/container, and profile-specific logs.
- [x] Monitor all submitted IDs with one filtered query per minute for a window
  of at least five minutes.
- [x] Add launcher tests using fake scheduler calls, first observe RED, then
  implement the minimum shell behavior and return to GREEN.

### Task 4: Verification and handoff

**Files:**

- Create: `experiments/qwen3_30ba3b_swe_rollout_pr3733/README.md`
- Create after execution: `experiments/qwen3_30ba3b_swe_rollout_pr3733/report/`

- [ ] Run focused pytest, shell syntax checks, pre-commit on changed files, and
  pyrefly if the new Python file is in the checked project include set.
- [ ] Obtain an independent read-only review and resolve findings with TDD.
- [ ] Commit all owned files with `git commit -s -S` and verify the signature,
  DCO sign-off, clean tree, and exact commit SHA.
- [ ] On OCI-HSG, create a clean recursive checkout at that SHA; verify
  FairShare and run test-only.
- [ ] Run the two-arm one-prompt canary and monitor for at least five minutes.
- [ ] Run the five-arm full matrix only after the canary gate succeeds, then
  aggregate matched generation throughput, time, acceptance rate, and mean
  accepted length without E2E-training claims.
