# Native MXFP8 Refit PR Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the validated native MXFP8 NCCL Reshard implementation into reviewer-sized stacked draft PRs, with concise visual explainers and matched BF16 versus MXFP8 evidence.

**Architecture:** Keep the refit wire contract independent of Megatron and vLLM. Add a Transformer Engine producer above that contract and a vLLM reload adapter below it. Put model recipes, compatibility fixes, and end-to-end evidence in the final validation PR so the three core PRs remain reusable and reviewable.

**Tech Stack:** Python 3.13, PyTorch DTensor, Transformer Engine, Megatron-Core, Megatron-Bridge, NCCL Reshard, vLLM 0.25.1, Ray, pytest, Pyrefly, Ruff, GitHub CLI, Archify, standalone HTML.

**Spec:** `docs/superpowers/specs/2026-08-27-native-mxfp8-nccl-reshard-refit-design.md`

## Global Constraints

- Rebuild every branch from the current `origin/main`; do not publish the large staging branch.
- Preserve existing BF16, BF16-to-MXFP8, and blockwise-FP8 behavior.
- Do not duplicate open PR #3281. The Megatron producer PR declares it as a runtime dependency until it merges.
- Do not copy open PR #3630 or #3724 into the native receiver PR. Link them only where Nano or Qwen3.5 validation needs their model-specific behavior.
- PR #3477 and PR #3545 are merged and are the supported base contracts.
- Do not stack on the large open PR #3651. Use the reload entrypoints already available from PR #3545 and isolate version-sensitive code behind `VllmRefitAdapter`.
- Keep experiment launchers, cluster names, internal paths, W&B project names, and unreleased model details out of public PR bodies and public HTML.
- Keep source and worktrees under `/home` on clusters, caches and builds under `/raid/scratch`, and only durable large outputs under `/lustre`.
- Use signed commits, normal fast-forward pushes, and draft PRs. Never rewrite a remote branch without separate approval.
- Before each external write, re-fetch the latest base/head SHAs and preview the exact PR payload.

---

### Task 1: Audit And Freeze The Stack Boundaries

**Files:**
- Read: `nemo_rl/weight_sync/nccl_reshard_utils.py`
- Read: `nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py`
- Read: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Read: `nemo_rl/models/generation/vllm/vllm_backend.py`
- Read: open PRs #3281, #3630, #3651, #3653, and #3724

**Interfaces:**
- Produces: one ownership manifest mapping every production file and test to exactly one PR.

- [ ] Fetch `origin/main` and all dependency PR heads.
- [ ] Compare the staging branch against current `origin/main`, not only its old merge base.
- [ ] Mark changes already provided by merged PRs or open dependencies.
- [ ] Record the immutable base SHA and staging SHA in the HTML evidence source.

### Task 2: Build PR 1, Ordered NCCL Refit Components

**Files:**
- Create: `nemo_rl/weight_sync/refit_components.py`
- Modify: `nemo_rl/weight_sync/nccl_reshard_utils.py`
- Modify: `nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py`
- Test: `tests/unit/weight_sync/test_refit_components.py`
- Test: `tests/unit/weight_sync/test_nccl_reshard_utils.py`
- Test: `tests/unit/weight_sync/test_nccl_reshard_weight_synchronizer.py`

**Interfaces:**
- Produces: `RefitComponentMeta`, normalized `weight`/`weight_scale` ordering, deterministic plan digest, and legacy one-component compatibility.
- Must not import Transformer Engine or vLLM.

- [ ] Create an isolated branch from current `origin/main`.
- [ ] Port only the component metadata and transport validation changes.
- [ ] Run focused unit tests and Ruff/Pyrefly checks.
- [ ] Self-review the exact branch diff.
- [ ] Push and open a draft PR based on `main`.

### Task 3: Build PR 2, Megatron Native MXFP8 Producer

**Files:**
- Create: `nemo_rl/models/policy/workers/mxfp8_refit_source.py`
- Create: `nemo_rl/models/policy/workers/native_mxfp8_inventory.py`
- Create: `nemo_rl/models/megatron/quantization_recipe.py`
- Modify: `nemo_rl/models/megatron/setup.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Test: corresponding Megatron and policy-worker unit tests

**Interfaces:**
- Consumes: PR 1 component contract and Transformer Engine native MXFP8 metadata.
- Produces: live canonical `weight` and `weight_scale` components without dequantization.

- [ ] Create an isolated branch from PR 1.
- [ ] Exclude the optimizer `fp8_recipe` patch already owned by PR #3281.
- [ ] Port source extraction, grouped-expert mapping, mixed BF16/native inventory, and optimizer-buffer synchronization.
- [ ] Verify that `fp8_param=false` and non-MXFP8 paths remain unchanged.
- [ ] Run focused tests, self-review, push, and open a draft PR based on PR 1.

### Task 4: Build PR 3, vLLM Native MXFP8 Receiver

**Files:**
- Create: `nemo_rl/models/generation/vllm/refit_adapter.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_backend.py`
- Modify only when required: `checkpoint_engine.py`, `quantization/fp8.py`, `quantization/mxfp8_utils.py`
- Test: corresponding vLLM backend, checkpoint-engine, and quantization tests

**Interfaces:**
- Consumes: PR 1 component plan and PR 2 producer metadata.
- Produces: capability-based vLLM destination binding and one begin/finish/abort reload lifecycle.

- [ ] Create an isolated branch from PR 2.
- [ ] Port only native component destination and lifecycle behavior.
- [ ] Remove changes owned by PR #3630 and #3724.
- [ ] Keep all vLLM 0.25.1 internals behind the adapter.
- [ ] Run focused tests, self-review, push, and open a draft PR based on PR 2.

### Task 5: Build PR 4, Recipes And End-To-End Evidence

**Files:**
- Modify: public performance recipes needed for Qwen3-30B-A3B and Nemotron3 Nano
- Add: public documentation and test-suite entries only after stable runs exist
- Exclude: internal launchers, cluster paths, internal checkpoints, and private W&B details

**Interfaces:**
- Consumes: PRs 1-3 and dependency PRs needed by each model.
- Produces: reproducible public configurations and validation tables.

- [ ] Create an isolated branch from PR 3.
- [ ] Add the smallest public recipe changes.
- [ ] Record Qwen second-step correctness and native refit evidence.
- [ ] Add Nano evidence only after Step 2 succeeds.
- [ ] Add matched BF16 versus MXFP8 policy/logprob data using the same step window.
- [ ] Run recipe validation, self-review, push, and open a draft PR based on PR 3.

### Task 6: Create Reviewer-Facing PR Bodies

**Files:**
- Create locally: `experiments/native_mxfp8_source_refit/pr_stack/pr_bodies/*.md`

**Interfaces:**
- Produces: concise PR title/body payloads with no AI-assistance attribution.

- [ ] Use one plain-language problem statement per PR.
- [ ] Show the owned data-flow segment in four to six lines.
- [ ] List focused tests and current integration evidence.
- [ ] State dependencies and remaining validation honestly.
- [ ] Preview every payload before `gh pr create` or `gh pr edit`.

### Task 7: Create And Validate Visual Explainers

**Files:**
- Create: `experiments/native_mxfp8_source_refit/pr_stack/native_mxfp8_refit_stack.json`
- Create: `experiments/native_mxfp8_source_refit/pr_stack/native_mxfp8_refit_stack.html`
- Create: one concise HTML page per draft PR

**Interfaces:**
- Produces: one overview diagram and four linked PR pages, each pinned to an immutable head SHA.

- [ ] Author the end-to-end data-flow with Archify.
- [ ] Validate and deliver with showcase quality.
- [ ] Run visual checks at all required desktop sizes.
- [ ] Inspect the screenshots for clipping, overlap, and unreadable text.
- [ ] Validate each concise explainer with `validate_explainer.py`.
- [ ] Open the overview page locally and provide clickable paths.

### Task 8: Finish Matched Performance Evidence

**Files:**
- Read: canonical W&B histories or durable exported CSV/JSON
- Update: PR 4 body and HTML evidence source

**Interfaces:**
- Produces: Step 2-19 mean policy-training time/TPS and logprob time/TPS for matched BF16 and MXFP8 runs.

- [ ] Wait for the BF16 20-step baseline to complete.
- [ ] Confirm both runs use the same model, GPU count, batch, rollout, logprob count, and step window.
- [ ] Export canonical histories and compute means from raw rows.
- [ ] Label partial-window numbers as preliminary.
- [ ] Update the PR body and HTML only with directly comparable numbers.

### Task 9: Final Verification And Handoff

**Files:**
- Verify: all four PR diffs, CI/check status, HTML artifacts, and integration evidence

**Interfaces:**
- Produces: draft PR URLs, HTML paths, test matrix, and explicit blockers.

- [ ] Re-fetch every base and head SHA.
- [ ] Confirm commit sign-offs and one correct CI label per PR.
- [ ] Confirm each stacked PR diff contains only its owned files.
- [ ] Report test results and skipped GPU coverage without overstating validation.
- [ ] Keep PRs in draft until Qwen and Nano Step 2 evidence is attached.
