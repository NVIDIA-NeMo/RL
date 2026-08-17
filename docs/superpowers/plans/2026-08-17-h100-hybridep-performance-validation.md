# H100 HybridEP Performance Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox completion tracking.

**Goal:** Enable HybridEP in every eligible H100 `n8g` MoE performance recipe, prove that the configurations run without OOM, and measure end-to-end performance against the previous default recipes.

**Architecture:** Build one clean experiment branch from a fixed latest-main revision, combine the x86 HybridEP dependency and sequence-packing compatibility PRs, then apply recipe-only HybridEP overlays. Run matched baseline/HybridEP jobs on the same software stack; materialize the baseline YAML from the fixed main commit at launch time so only dispatcher configuration differs.

**Tech Stack:** NeMo-RL, Megatron-Bridge/Megatron-LM, DeepEP HybridEP commit `17cfb817bccec3a9c247013360cc550c2bac441e`, pytest, Bash, SLURM/Ray, W&B, static HTML.

**Spec:** `experiments/h100-hybridep-performance-20260817/README.md`

---

### Task 1: Assemble the fixed experiment source stack

- [ ] Verify the experiment branch starts at the recorded `origin/main` SHA.
- [ ] Merge the current PR #3436 head and record its exact SHA.
- [ ] Merge the current PR #2964 head and record its exact SHA.
- [ ] Resolve conflicts without changing unrelated source behavior.
- [ ] Confirm `pyproject.toml` and `uv.lock` resolve x86 HybridEP to `17cfb817bccec3a9c247013360cc550c2bac441e`.

### Task 2: Apply the H100 recipe overlay

**Files:**

- Modify the 14 eligible YAML files under `examples/configs/recipes/llm/performance/` listed in the experiment spec.
- Add or update `tests/unit/tools/test_hybridep_default_8g_recipes.py`.

- [ ] Add the flex/HybridEP dispatcher fields to all eligible recipes.
- [ ] Add explicit single-domain H100 topology environment variables.
- [ ] Enable packed-input pre-padding only where PP=1 and MTP is disabled.
- [ ] Assert that dense EP=1 recipes remain unchanged.
- [ ] Assert that each eligible recipe resolves to the expected inherited values.

### Task 3: Add reusable, public-safe experiment tooling

**Files:**

- Create `experiments/h100-hybridep-performance-20260817/matrix.tsv`.
- Create `experiments/h100-hybridep-performance-20260817/submit.sh`.
- Create `experiments/h100-hybridep-performance-20260817/check_results.py` if local log classification is needed.
- Create `experiments/h100-hybridep-performance-20260817/stage_enroot_image.sbatch` from the approved container-staging template.

- [ ] Require private infrastructure values through environment variables.
- [ ] Use `--gpus-per-node=8`, omit subset-GPU exclusive allocations, and preserve each recipe's node count.
- [ ] Generate the baseline YAML from the fixed `main` commit in the same config directory so inherited paths resolve identically.
- [ ] Override only run length, checkpointing, logging destination, and W&B run identity.
- [ ] Support four 20-step A/B pairs and ten three-step HybridEP smoke jobs.
- [ ] Fail early on missing inputs and create per-run log directories.

### Task 4: Verify locally before submission

- [ ] Run the focused recipe unit tests.
- [ ] Run YAML/config resolution tests required by repository guidance.
- [ ] Run formatting/linting for newly added Python and shell tooling.
- [ ] Inspect the final diff for any private host, account, path, credential, or job identifier.
- [ ] Commit with sign-off and push only to the user's fork branch.

### Task 5: Stage and validate the runtime on CW H100

- [ ] Fetch/pull the pushed experiment commit into a Lustre worktree.
- [ ] Run `sbatch --test-only` for staging, a one-node GPU smoke, and representative experiment jobs.
- [ ] Stage the selected current NeMo-RL nightly image to an immutable filename with provenance metadata.
- [ ] Run a one-node GPU container smoke test.
- [ ] Record source, submodule, dependency, container, GPU, CUDA, and driver provenance in private experiment logs.

### Task 6: Run the four matched performance comparisons

- [ ] Submit baseline and HybridEP for DeepSeek V3.
- [ ] Submit baseline and HybridEP for Nemotron 3 Super.
- [ ] Submit baseline and HybridEP for Qwen3 235B-A22B.
- [ ] Submit baseline and HybridEP for Qwen3 30B-A3B.
- [ ] Monitor each submission for at least five minutes and stop clearly failed jobs.
- [ ] Confirm all successful comparisons reach step 20 without OOM.

### Task 7: Run remaining recipe smoke coverage

- [ ] Submit the ten non-canonical HybridEP recipes in resource-conscious waves.
- [ ] Monitor startup and at least three completed optimizer steps.
- [ ] Classify OOM, hang, scheduler, infrastructure, and application failures separately.
- [ ] Rerun only failures attributable to transient infrastructure.

### Task 8: Aggregate and publish results

- [ ] Export W&B history for each completed A/B run.
- [ ] Compute means over steps 2–20 inclusive and record the actual sample count.
- [ ] Calculate E2E step-time speedup and throughput gain per model family.
- [ ] Add policy/log-probability breakdowns where available.
- [ ] Update the existing HybridEP HTML report with concise model-by-hardware tables and graphs.
- [ ] Run a security scan over the generated page and publish via the existing Pages workflow.
- [ ] Report incomplete or non-comparable results explicitly rather than extrapolating.
