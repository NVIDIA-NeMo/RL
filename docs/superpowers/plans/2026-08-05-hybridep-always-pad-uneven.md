# HybridEP Always-On Uneven Dispatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Always enable Megatron-LM PR #5008's uneven-dispatch protection when NeMo-RL selects HybridEP, then validate the minimal upstream-only path for 20 steps on CW H100 and OCI-HSG GB200.

**Architecture:** NeMo-RL derives the MCore flag inside `_apply_moe_config`; no new user-facing configuration is introduced. Production code and unit tests form one cherry-pickable commit. Reproducible cluster launchers and manifests are separate experiment commits so the eventual PR can retain only the minimal production change.

**Tech Stack:** Python 3.13.14, pytest, NeMo-RL, Megatron-Bridge `573e088c`, Megatron-LM `6513e3e` including #5008, DeepEP HybridEP `17cfb817`, SLURM, Ray, CW H100, OCI-HSG GB200.

## Global Constraints

- Base all work on NeMo-RL main `80a84eb99d31c73aa659973ba06814a2d73ef8ce` or a newer main only after explicitly rebasing and recording the new SHA.
- Do not modify `data.py`, `train.py`, `.gitmodules`, submodule gitlinks, `worker_groups.py`, `venvs.py`, or canonical performance recipes.
- Set the MCore flag for every HybridEP backend selection, independent of sequence packing.
- Use upstream Megatron-Bridge/Megatron-LM submodules and verify MCore contains merge commit `81770cb015eab05785ecd540ba929d1400a52f67`.
- Use DeepEP HybridEP commit `17cfb817bccec3a9c247013360cc550c2bac441e` on both architectures.
- Commit and push exact source and launcher state before each SLURM submission; run `sbatch --test-only` and monitor for at least five minutes.
- Store logs and provenance under experiment-specific Lustre directories.

---

### Task 1: Add the always-on HybridEP model-config behavior with TDD

**Files:**
- Modify: `tests/unit/models/megatron/test_megatron_setup.py`
- Modify: `nemo_rl/models/megatron/setup.py`

**Interfaces:**
- Consumes: `PolicyConfig["megatron_cfg"]["moe_flex_dispatcher_backend"]`.
- Produces: `model_cfg.moe_hybridep_pad_uneven_dispatch_inputs = True` whenever the backend is `"hybridep"`.

- [ ] **Step 1: Write the failing HybridEP test**

Add this test to `TestApplyMoeConfig`:

```python
def test_hybridep_always_enables_uneven_dispatch_padding(self, monkeypatch):
    from nemo_rl.models.megatron.setup import _apply_moe_config

    monkeypatch.setenv("NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN", "8")
    monkeypatch.setenv("USE_MNNVL", "0")
    model_cfg = SimpleNamespace(
        moe_hybridep_pad_uneven_dispatch_inputs=False,
    )
    config = self._base_moe_cfg(
        expert_model_parallel_size=8,
        moe_flex_dispatcher_backend="hybridep",
    )

    _apply_moe_config(model_cfg, config)

    assert model_cfg.moe_hybridep_pad_uneven_dispatch_inputs is True
```

- [ ] **Step 2: Verify RED on Linux**

Run in a committed-source SLURM test job using the current NeMo-RL nightly:

```bash
uv run --extra mcore pytest --mcore-only -q \
  tests/unit/models/megatron/test_megatron_setup.py::TestApplyMoeConfig::test_hybridep_always_enables_uneven_dispatch_padding
```

Expected: FAIL because the value remains `False`.

- [ ] **Step 3: Add the non-HybridEP regression test**

```python
def test_non_hybridep_preserves_uneven_dispatch_padding_default(self):
    from nemo_rl.models.megatron.setup import _apply_moe_config

    model_cfg = SimpleNamespace(
        moe_hybridep_pad_uneven_dispatch_inputs=False,
    )
    config = self._base_moe_cfg(
        moe_flex_dispatcher_backend="deepep",
    )

    _apply_moe_config(model_cfg, config)

    assert model_cfg.moe_hybridep_pad_uneven_dispatch_inputs is False
```

- [ ] **Step 4: Implement the minimal production change**

Inside the existing HybridEP backend block in `_apply_moe_config`, before topology environment setup, add:

```python
model_cfg.moe_hybridep_pad_uneven_dispatch_inputs = True
```

- [ ] **Step 5: Verify GREEN and nearby regressions**

Run:

```bash
uv run --extra mcore pytest --mcore-only -q \
  tests/unit/models/megatron/test_megatron_setup.py::TestApplyMoeConfig
python -m py_compile nemo_rl/models/megatron/setup.py
git diff --check
```

Expected: all `TestApplyMoeConfig` tests pass, compilation passes, and the diff is clean.

- [ ] **Step 6: Commit the production change separately**

```bash
git add nemo_rl/models/megatron/setup.py \
  tests/unit/models/megatron/test_megatron_setup.py
git commit -s -m "fix: always pad uneven HybridEP dispatch inputs"
git push fork HEAD:sna/hybridep-always-pad-uneven-20260805
```

### Task 2: Create reproducible H100 and GB200 validation launchers

**Files:**
- Create: `experiments/hybridep-upstream5008-validation/README.md`
- Create: `experiments/hybridep-upstream5008-validation/cluster-cw.yaml`
- Create: `experiments/hybridep-upstream5008-validation/cluster-oci-hsg.yaml`
- Create: `experiments/hybridep-upstream5008-validation/submit-cw-qwen30.sh`
- Create: `experiments/hybridep-upstream5008-validation/submit-oci-qwen30.sh`

**Interfaces:**
- Consumes: the production commit from Task 1, immutable containers, DeepEP `17cfb817` wheels, model caches, and canonical Qwen3-30B-A3B performance recipes.
- Produces: one 20-step H100 job and one 20-step GB200 job with complete provenance.

- [ ] **Step 1: Record immutable cluster contracts**

Set CW to `cw-dfw-cs-001-login-01.nvidia.com`, `batch`, 4 nodes × 8 H100, one-hour walltime. Set OCI-HSG to `oci-hsg-cs-001-vscode-02`, `batch`, 4 nodes × 4 GB200, four-hour walltime. Record the selected FairShare account immediately before submission.

- [ ] **Step 2: Keep runtime configuration isolated from canonical recipes**

Each launcher must apply command-line overrides for:

```text
grpo.max_num_steps=20
checkpointing.enabled=false
policy.megatron_cfg.moe_token_dispatcher_type=flex
policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep
policy.megatron_cfg.moe_hybridep_num_sms=32
policy.sequence_packing.enabled=true
```

Do not add `moe_hybridep_pad_uneven_dispatch_inputs` to YAML; the production change derives it.

- [ ] **Step 3: Pin hardware topology**

CW must set ranks/domain `8`, combine chunk `128`, NVLink domain size `8`, and `USE_MNNVL=0`. OCI-HSG must set ranks/domain `16`, combine chunk `128`, NVLink domain size `72`, and `USE_MNNVL=1`.

- [ ] **Step 4: Add fail-closed provenance probes**

Before training, assert the NeMo-RL commit, upstream submodule origins/SHAs, MCore ancestry of `81770cb`, DeepEP module/version/commit artifact, Python version, GPU model/count, and effective `model_cfg.moe_hybridep_pad_uneven_dispatch_inputs=True`.

- [ ] **Step 5: Validate launchers and commit experiment artifacts separately**

Run `bash -n` on both scripts and `git diff --check`, then commit only the experiment files with:

```bash
git commit -s -m "test: add upstream HybridEP THD cluster validation"
git push fork HEAD:sna/hybridep-always-pad-uneven-20260805
```

### Task 3: Execute CW H100 validation

**Files:**
- Runtime output: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/hybridep-upstream5008-validation/cw-h100/`

**Interfaces:**
- Consumes: `submit-cw-qwen30.sh` and a Python-3.13.14-compatible current nightly.
- Produces: terminal SLURM status, Ray driver log, step metrics, and provenance manifest.

- [ ] **Step 1: Pull the pushed branch recursively into a fresh Lustre clone**
- [ ] **Step 2: Stage or verify a CPython 3.13.14 nightly and DeepEP `17cfb817` x86 wheel**
- [ ] **Step 3: Check FairShare and run `sbatch --test-only`**
- [ ] **Step 4: Submit the exact 4×8 H100 launcher and record the job ID**
- [ ] **Step 5: Monitor at least five minutes and triage any early failure**
- [ ] **Step 6: Monitor to terminal state and extract steps 2–20 averages when completed**

### Task 4: Execute OCI-HSG GB200 validation

**Files:**
- Runtime output: `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/hybridep-upstream5008-validation/oci-gb200/`

**Interfaces:**
- Consumes: `submit-oci-qwen30.sh`, the verified Python-3.13.14 image, and a DeepEP `17cfb817` aarch64 wheel or lock-native build.
- Produces: terminal SLURM status, Ray driver log, step metrics, and provenance manifest.

- [ ] **Step 1: Clone the pushed branch recursively into the fresh OCI-HSG Lustre path**
- [ ] **Step 2: Smoke-test the current image and build/stage DeepEP `17cfb817` for aarch64 if absent**
- [ ] **Step 3: Check FairShare and run `sbatch --test-only`**
- [ ] **Step 4: Submit the exact 4×4 GB200 launcher and record the job ID**
- [ ] **Step 5: Monitor at least five minutes and triage any early failure**
- [ ] **Step 6: Monitor to terminal state and extract steps 2–20 averages when completed**

### Task 5: Validate the simplification decision

**Files:**
- Modify: `docs/superpowers/specs/2026-08-05-hybridep-always-pad-uneven-design.md`
- Create or update: the existing HybridEP HTML experiment report repository.

**Interfaces:**
- Consumes: H100 and GB200 terminal results and metrics.
- Produces: a decision on replacing PR #2964 with the minimal production commit.

- [ ] **Step 1: Confirm both jobs completed 20/20 steps without hang, IMA, OOM, NaN, or Inf**
- [ ] **Step 2: Record step time, policy-training time, logprob time, throughput, and any observed always-on overhead**
- [ ] **Step 3: Record exact W&B links when available and exact Lustre log paths otherwise**
- [ ] **Step 4: State the merge recommendation**

If both platforms pass, recommend replacing PR #2964's NeMo pre-padding/custom-submodule stack with the Task 1 production commit. Keep ordinary THD padding-mask accuracy semantics as a separate future workstream.

- [ ] **Step 5: Verify the report and final branch state before claiming completion**

Run HTML validation, `git diff --check`, targeted tests, and `git status --short --branch`; report any non-production experiment commits separately from the cherry-pickable production commit.
