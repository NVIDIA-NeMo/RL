# NeMo 26.06 Policy-Training Feature Factorial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate CuTeDSL grouped-MLP fusion, full-iteration CUDA Graph, and expert-parallel A2A overlap into NeMo-RL policy training, then quantify their individual and combined effects on E2E, generation, logprob, policy training, and refit performance.

**Architecture:** Keep NeMo-RL's public `Policy` API unchanged. Add a portable cluster-profile layer around one common SLURM payload, integrate full-CG through a worker-owned forward/backward runner, integrate A2A through Bridge `CommOverlapConfig` and MCore's schedule-plan forward contract, and evaluate the three binary factors with matched workloads. Every run emits machine-readable events and metrics that render into a self-contained HTML report.

**Tech Stack:** Python 3.13.13, PyTorch 2.11/CUDA 13, NeMo-RL, Megatron-Bridge, Megatron Core, Transformer Engine 2.15, Cutlass DSL 4.5.2+, Ray, pytest, Bash, SLURM/Pyxis, Nsight Systems, GB200.

## Global Constraints

- Primary cluster: Pre-Tyche; backup clusters: AWS-DFW and Lyris.
- Source branch: `sna/nemo-2606-cutedsl-20260710`; all cluster checkouts must equal the submitted commit and be clean.
- Model/topology: `Qwen/Qwen3-30B-A3B`, TP1/PP1/CP1/ETP1/EP4, one node and four GB200 GPUs.
- Training shape: sequence length 1024, GBS4, MBS1, four microbatches, dynamic batching off, sequence packing off.
- Precision: MXFP8 for every factorial cell; grouped GEMM, TE op fuser, GLU interleave 32, and ETP1 remain fixed.
- Pre-Tyche and Lyris request a full four-GPU tray without `--gres` and use `--segment=1`. AWS-DFW uses `--gres=gpu:4` and no segment.
- Latest nightly images are runtime bases only. TE and Cutlass DSL come from the repository lockfile environment because the baked images do not contain them.
- Timing runs never include Nsight. Profile runs are separate and cannot contribute timing acceptance samples.
- Root-cause workflow for every failure: record symptom, reproduce, capture boundary diagnostics, state one hypothesis, test one change, and record verification evidence.
- No raw secrets, credentials, or large logs are copied into Git or HTML. Reports link to cluster log paths and include only bounded excerpts.
- Baseline and feature cells must use identical code, image, model, seed, input batch, topology, and measurement windows.

---

### Task 1: Portable cluster profiles and common SLURM payloads

**Files:**
- Create: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/cluster_profiles/pre_tyche.sh`
- Create: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/cluster_profiles/aws_dfw.sh`
- Create: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/cluster_profiles/lyris.sh`
- Create: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/lib/cluster_profile.sh`
- Create: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_cutedsl_functional.sh`
- Rename: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_oci_hsg.sh` to `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_functional.sbatch`
- Rename: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/benchmark_cutedsl_ab_oci_hsg.sh` to `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch`
- Modify: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_cutedsl_ab_replicates.sh`
- Modify: `tests/test_oci_cutedsl_wrapper.py`
- Create: `tests/test_cutedsl_cluster_profiles.py`

**Interfaces:**
- Consumes: `CUTEDSL_CLUSTER_PROFILE=pre_tyche|aws_dfw|lyris`.
- Produces: validated `CUTEDSL_SBATCH_ARGS`, immutable image path/SHA, functional and benchmark walltimes, and common payload environment.

- [ ] **Step 1: Write failing profile-contract tests**

```python
def test_pre_tyche_profile_uses_segment_without_gres(profile):
    assert profile["account"] == "coreai_dlalgo_llm"
    assert profile["partition"] == "batch"
    assert profile["segment"] == "1"
    assert profile["gres"] == ""


def test_aws_profile_uses_gres_without_segment(profile):
    assert profile["account"] == "nemotron_sw_post"
    assert profile["partition"] == "batch_long"
    assert profile["segment"] == ""
    assert profile["gres"] == "gpu:4"


def test_lyris_profile_uses_segment_without_gres(profile):
    assert profile["partition"] == "gb200"
    assert profile["segment"] == "1"
    assert profile["gres"] == ""
```

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run --no-project --with pytest pytest -q tests/test_cutedsl_cluster_profiles.py
```

Expected: FAIL because the profile files and loader do not exist.

- [ ] **Step 3: Implement declarative profiles and strict loader**

Each profile exports exactly:

```bash
CUTEDSL_PROFILE_NAME=pre_tyche
CUTEDSL_ACCOUNT=coreai_dlalgo_llm
CUTEDSL_PARTITION=batch
CUTEDSL_GRES=
CUTEDSL_SEGMENT=1
CUTEDSL_COMMENT=metrics
CUTEDSL_IMAGE=/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-2606-cutedsl-pre-tyche-20260711/containers/nemo_rl_nightly_20260711_2361407.sqsh
CUTEDSL_IMAGE_SHA256=dd32f77a0a6fb09710e31f87402f0433413b9c71120fe893297e2f46e32ce8be
CUTEDSL_FUNCTIONAL_TIME=02:00:00
CUTEDSL_BENCHMARK_TIME=05:00:00
```

The loader rejects unknown names, non-absolute images, malformed SHA values, unsupported segments, and conflicting GRES/segment settings. It adds `--gres` only for AWS and `--segment=1` for Pre-Tyche and Lyris.

- [ ] **Step 4: Convert existing wrappers to cluster-neutral payloads**

Remove static account, partition, GRES, image, and OCI path assumptions. Require the validated profile environment, preserve the locked node-local runtime, and record the effective profile in `metadata.json`.

- [ ] **Step 5: Verify GREEN and shell syntax**

```bash
uv run --no-project --with pytest pytest -q \
  tests/test_cutedsl_cluster_profiles.py \
  tests/test_oci_cutedsl_wrapper.py
bash -n experiments/cutedsl_qwen3_30ba3b_oci_1n4g/{cluster_profiles/*.sh,lib/*.sh,*.sh,*.sbatch}
```

Expected: all tests pass; fake `sbatch` tests prove exact argv for all three profiles and `--test-only` propagation.

- [ ] **Step 6: Commit**

```bash
git add -f experiments/cutedsl_qwen3_30ba3b_oci_1n4g tests/test_cutedsl_cluster_profiles.py tests/test_oci_cutedsl_wrapper.py
git commit -s -m "refactor: add portable CuTeDSL cluster profiles"
```

---

### Task 2: Structured experiment events and self-contained HTML report

**Files:**
- Create: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/lib/events.sh`
- Create: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/render_cutedsl_report.py`
- Create: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/report/public/index.html`
- Create: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/report/incidents.json`
- Create: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/report/run_index.tsv`
- Create: `tests/test_cutedsl_report.py`
- Modify: both common SLURM payloads from Task 1

**Interfaces:**
- Consumes: one `events.jsonl`, `status.json`, provenance, metric summaries, and bounded error excerpts per run.
- Produces: per-run `report.html` plus aggregate `report/public/index.html` with reproducibility links and root-cause timeline.

- [ ] **Step 1: Write failing renderer tests**

```python
def test_report_renders_failure_root_cause_and_verification(tmp_path):
    events = [
        {"timestamp_utc": "2026-07-11T18:00:00Z", "phase": "runtime_bootstrap", "status": "fail", "exit_code": 1, "message": "UV_PROJECT_ENVIRONMENT mismatch", "artifact": "slurm.out"},
        {"timestamp_utc": "2026-07-11T18:10:00Z", "phase": "runtime_diagnostic", "status": "pass", "exit_code": 0, "message": "both uv environment variables resolve to /runtime/venv", "artifact": "runtime_env.log"},
    ]
    html = render_fixture(tmp_path, events)
    assert "UV_PROJECT_ENVIRONMENT mismatch" in html
    assert "runtime_env.log" in html
    assert "Root cause" in html
```

Also test HTML escaping, chronological ordering, missing optional artifacts, successful runs, and bounded log excerpts.

- [ ] **Step 2: Verify RED**

```bash
uv run --no-project --with pytest pytest -q tests/test_cutedsl_report.py
```

Expected: FAIL because the renderer and event schema do not exist.

- [ ] **Step 3: Implement event writer and stdlib-only renderer**

Use this event schema:

```json
{"timestamp_utc":"2026-07-11T18:00:00Z","cluster":"pre_tyche","job_id":"123","phase":"gpu_smoke","status":"start","exit_code":null,"message":"four-GPU Transformer Engine smoke","artifact":"gpu_smoke.log"}
```

Required phases are `preflight`, `image_hash`, `runtime_bootstrap`, `config_validation`, `focused_tests`, `gpu_smoke`, `functional_grpo`, `timing`, `profile`, `metrics_export`, and `complete`. A failed run must additionally include a root-cause record with `symptom`, `evidence`, `root_cause`, `fix_commit`, and `verification_job`.

- [ ] **Step 4: Add report sections**

Render status, cluster/account/partition, code and image SHAs, model/topology, feature cell, component timing table, throughput table, Nsight evidence, job/log links, and chronological incident history. Keep HTML self-contained with no external assets.

- [ ] **Step 5: Verify GREEN**

```bash
uv run --no-project --with pytest pytest -q tests/test_cutedsl_report.py
python -m html.parser experiments/cutedsl_qwen3_30ba3b_oci_1n4g/report/public/index.html
```

Expected: tests pass and HTML parses without errors.

- [ ] **Step 6: Commit**

```bash
git add -f experiments/cutedsl_qwen3_30ba3b_oci_1n4g tests/test_cutedsl_report.py
git commit -s -m "feat: add CuTeDSL experiment HTML reporting"
```

---

### Task 3: Validate and measure the runnable CuTeDSL factor first

**Files:**
- Modify: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch`
- Modify: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_cutedsl_ab_replicates.sh`
- Modify: `tests/test_oci_cutedsl_wrapper.py`
- Modify: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/report/run_index.tsv`

**Interfaces:**
- Consumes: C=0/1 while G=0 and A=0.
- Produces: matched config diff, component metrics, ON/OFF Nsight evidence, and CuTeDSL effect with confidence interval.

- [ ] **Step 1: Preserve the exact single-factor control**

The OFF cell changes only:

```text
policy.megatron_cfg.env_vars.NVTE_CUTEDSL_FUSED_GROUPED_MLP="0"
```

The ON cell uses recipe default `"1"`. Grouped GEMM, op fuser, GLU interleave 32, MXFP8, and ETP1 remain identical.

- [ ] **Step 2: Add failing contracts for component metrics**

Require measured-step series for:

```text
timing/train/total_step_time
timing/train/generation
timing/train/get_logprobs
timing/train/policy_training
timing/train/prepare_for_generation/transfer_and_update_weights
performance/policy_training_tokens_per_sec_per_gpu
train/total_num_tokens
train/global_valid_toks
```

When a metric key differs in the current logger, the extractor must discover the exact timer key from `metrics.json` and record the resolved name in the manifest; it must not silently write zero.

- [ ] **Step 3: Run Pre-Tyche functional gate**

```bash
CUTEDSL_CLUSTER_PROFILE=pre_tyche \
  ./experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_cutedsl_functional.sh --test-only
CUTEDSL_CLUSTER_PROFILE=pre_tyche \
  ./experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_cutedsl_functional.sh
```

Monitor five minutes. On failure, stop performance submission, populate the incident record, reproduce minimally, and fix only after identifying root cause.

- [ ] **Step 4: Run timing and profiles separately**

Use five warmup plus twenty measured updates per arm. Submit at least three alternating paired replicates first; extend to six if replicate-median CV exceeds 5% or the paired CI remains inconclusive. Profile one separate ON/OFF pair at steps 6–7.

- [ ] **Step 5: Enforce CuTeDSL attribution**

The ON profile must contain fused CuTeDSL grouped-MLP kernels; OFF must lack them while retaining the TE grouped GEMM/op-fuser path. Functional loss, optimizer success, refit, and generated-policy parity must pass.

- [ ] **Step 6: Commit experiment records**

```bash
git add -f experiments/cutedsl_qwen3_30ba3b_oci_1n4g/report
git commit -s -m "perf: record Pre-Tyche CuTeDSL policy results"
```

---

### Task 4: Integrate full-iteration CUDA Graph into NeMo-RL policy training

**Files:**
- Modify: `nemo_rl/models/policy/__init__.py`
- Modify: `nemo_rl/models/megatron/train.py`
- Modify: `nemo_rl/models/megatron/data.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `tests/unit/models/megatron/test_megatron_setup.py`
- Modify: `tests/unit/models/policy/test_megatron_worker.py`
- Create: `tests/unit/models/megatron/test_full_iteration_cuda_graph.py`
- Modify: the performance recipe and matrix payload

**Interfaces:**
- Consumes: `cuda_graph_impl=none|full_iteration`, `cuda_graph_warmup_steps=3`, `cuda_graph_use_single_mempool=true`.
- Produces: a worker-owned cached `FullCudaGraphWrapper`, fixed-shape microbatch payloads, replay/capture counters, and explicit lifecycle invalidation.

- [ ] **Step 1: Write failing runner-injection tests**

```python
def test_megatron_forward_backward_uses_injected_runner():
    calls = []
    runner = lambda **kwargs: calls.append(kwargs) or []
    megatron_forward_backward(..., forward_backward_func=runner)
    assert len(calls) == 1


def test_full_graph_runner_is_reset_before_refit(worker):
    worker.training_runner = FakeRunner()
    worker.offload_before_refit()
    assert worker.training_runner.reset_calls == 1
```

Add guards for variable shapes, packing, dynamic batching, split training, CPU offload, and `empty_unused_memory_level > 0`.

- [ ] **Step 2: Verify RED**

```bash
uv run --group test pytest -q \
  tests/unit/models/megatron/test_full_iteration_cuda_graph.py \
  tests/unit/models/policy/test_megatron_worker.py -k 'cuda_graph or runner'
```

Expected: FAIL because runner injection and lifecycle ownership are absent.

- [ ] **Step 3: Implement graph-safe data and runner injection**

Add optional `forward_backward_func` to `megatron_forward_backward`. Serialize `ProcessedMicrobatch` into tensors plus invariant metadata, refresh persistent buffers before replay, and build `FullCudaGraphWrapper` once per stable signature. Log warmup, capture, replay, and recapture counts.

- [ ] **Step 4: Implement lifecycle reset**

Reset or discard graph state before offload, reload, refit, checkpoint restore, storage resize, and worker teardown. Evaluation, logprob, refit, and generation remain eager in the first supported slice.

- [ ] **Step 5: Verify GREEN and functional parity**

```bash
uv run --group test pytest -q \
  tests/unit/models/megatron/test_full_iteration_cuda_graph.py \
  tests/unit/models/policy/test_megatron_worker.py \
  tests/unit/models/megatron/test_megatron_setup.py
```

Then run a Pre-Tyche functional pair G=0/1 with C=0,A=0 and require one capture plus at least two replays, equal workload hash, optimizer success, and precision-appropriate loss/update parity.

- [ ] **Step 6: Commit**

```bash
git add nemo_rl/models tests/unit examples/configs/recipes/llm/performance
git commit -s -m "feat: enable full-iteration CUDA graphs for policy training"
```

---

### Task 5: Integrate expert-parallel A2A overlap into NeMo-RL

**Files:**
- Modify: `nemo_rl/models/policy/__init__.py`
- Modify: `nemo_rl/models/megatron/setup.py`
- Modify: `nemo_rl/models/megatron/train.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `tests/unit/models/megatron/test_megatron_setup.py`
- Create: `tests/unit/models/megatron/test_a2a_schedule_plan.py`
- Modify: the performance recipe and matrix payload

**Interfaces:**
- Consumes: `comm_overlap.overlap_moe_expert_parallel_comm=true|false`; keeps `delay_wgrad_compute=false` fixed.
- Produces: Bridge `CommOverlapConfig`, MCore `return_schedule_plan=True` forward contract, and schedule-plan-aware NeMo-RL loss binding.

- [ ] **Step 1: Write failing config and schedule-plan tests**

```python
def test_a2a_overlap_builds_bridge_comm_overlap_config():
    cfg = make_policy_config(overlap_moe_expert_parallel_comm=True)
    container = build_config_container(cfg)
    assert container.comm_overlap.overlap_moe_expert_parallel_comm is True
    assert container.comm_overlap.delay_wgrad_compute is False


def test_forward_step_returns_schedule_plan_and_same_loss_callable():
    plan, loss = forward_with_post_processing_fn(..., return_schedule_plan=True)
    assert plan is model.expected_schedule_plan
    assert callable(loss)
```

- [ ] **Step 2: Verify RED**

```bash
uv run --group test pytest -q \
  tests/unit/models/megatron/test_a2a_schedule_plan.py \
  tests/unit/models/megatron/test_megatron_setup.py -k 'a2a or schedule_plan'
```

Expected: FAIL because NeMo-RL does not expose `CommOverlapConfig` or the schedule-plan forward branch.

- [ ] **Step 3: Implement config lifecycle and guards**

Propagate the Bridge config before model construction. Reject PP>1, EP<=1, unsupported dispatcher, shared-expert overlap, full/MoE recompute, MTP>1, fewer than four microbatches, and unsupported split training. Do not silently fall back.

- [ ] **Step 4: Implement schedule-plan forward**

When MCore requests `return_schedule_plan=True`, call `model.build_schedule_plan(...)` and return the same NeMo-RL loss callable used by eager execution. Keep temperature scaling and MTP masks inside the bound callable so eager and overlap branches are numerically equivalent.

- [ ] **Step 5: Verify GREEN and functional parity**

```bash
uv run --group test pytest -q \
  tests/unit/models/megatron/test_a2a_schedule_plan.py \
  tests/unit/models/megatron/test_megatron_setup.py \
  tests/unit/models/policy/test_megatron_worker.py
```

Then run a Pre-Tyche A=0/1 functional pair with C=0,G=0. Require no hang, optimizer success, identical workload hash, and equal loss/update within MXFP8 tolerance.

- [ ] **Step 6: Commit**

```bash
git add nemo_rl/models tests/unit examples/configs/recipes/llm/performance
git commit -s -m "feat: enable expert-parallel A2A overlap for policy training"
```

---

### Task 6: Matched workload capture and three-factor experiment matrix

**Files:**
- Create: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/build_feature_matrix.py`
- Create: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/analyze_feature_matrix.py`
- Modify: `run_cutedsl_matrix.sbatch`
- Modify: `submit_cutedsl_ab_replicates.sh`
- Create: `tests/test_cutedsl_feature_matrix.py`

**Interfaces:**
- Consumes: factors C/G/A and one frozen workload payload per replicate.
- Produces: eight cells (`000` through `111`), per-step metrics, replicate summaries, factorial effects, confidence intervals, and acceptance decisions.

- [ ] **Step 1: Write failing factorial and workload tests**

Require eight unique cells, intended config diffs only, balanced/complementary order, identical workload hashes, five warmup plus twenty measured updates, and strict separation of timing/profile artifacts. Add a synthetic dataset with known main and interaction effects and require deterministic recovery with a fixed bootstrap seed.

- [ ] **Step 2: Verify RED**

```bash
uv run --no-project --with pytest pytest -q tests/test_cutedsl_feature_matrix.py
```

Expected: FAIL because the matrix builder and analyzer do not exist.

- [ ] **Step 3: Implement matched workload manifest**

Hash token IDs, masks, advantages/returns, sample IDs, sequence lengths, padding, microbatch shapes, total/valid tokens, and per-expert token counts. Refuse a matched comparison when hashes differ; label non-replay runs as observational.

- [ ] **Step 4: Implement the eight-cell matrix**

Use cells `000,100,010,001,110,101,011,111`. Keep companion controls fixed: alltoall dispatcher, shared-expert overlap false, delay-wgrad false, fixed shapes, and MXFP8. Each cell starts a fresh process/model.

- [ ] **Step 5: Implement statistical analysis**

Fit replicate-level `log(throughput) ~ C + G + A + C:G + C:A + G:A + C:G:A + replicate`. Convert coefficients to speedup ratios and use 10,000 replicate-level paired bootstrap samples. Do not pool steps as independent observations.

- [ ] **Step 6: Implement acceptance rules**

Each main effect requires point speedup >=1.02 and 95% CI lower bound >1.00. Full stack `111/000` requires point speedup >=1.05 and CI lower bound >=1.02. E2E non-regression requires CI lower bound >=0.99. Mark interactions material when absolute effect is at least 1% and the CI excludes 1.0.

- [ ] **Step 7: Verify GREEN and commit**

```bash
uv run --no-project --with pytest pytest -q tests/test_cutedsl_feature_matrix.py
git add -f experiments/cutedsl_qwen3_30ba3b_oci_1n4g tests/test_cutedsl_feature_matrix.py
git commit -s -m "perf: add matched NeMo 26.06 feature factorial"
```

---

### Task 7: Execute primary and backup cluster gates and publish the report

**Files:**
- Modify generated records under `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/report/`
- Modify: `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/README.md`

**Interfaces:**
- Consumes: validated common payloads and matrix analyzer.
- Produces: final component breakdown, feature effects, Nsight attribution, incident history, and portable static HTML.

- [ ] **Step 1: Run functional screening in dependency order**

On Pre-Tyche run `000`, `100`, `010`, `001`, then pairwise cells and `111`. A cell must pass correctness before its timing job is submitted. Use AWS first and Lyris second if Pre-Tyche is delayed or fails for cluster-specific reasons.

- [ ] **Step 2: Run timing blocks**

Start with three replicate blocks. Each cell uses five warmup and twenty measured updates. Extend to six replicate blocks when CV exceeds 5% or confidence remains inconclusive. Alternate cell order within each block.

- [ ] **Step 3: Run separate profiles**

Profile steps 6–7 after warmup. Require fused CuTeDSL kernel selection for C, CUDA graph replay and reduced CPU launch gaps for G, and reduced exposed A2A or at least five percentage points more compute/communication overlap for A.

- [ ] **Step 4: Report component effects**

For E2E, generation, logprob, policy training, and refit report raw per-step values, replicate median, p50/p95, normalized throughput, speedup, 95% CI, peak allocated/reserved memory, and initialization/capture cost. Explicitly state which components should not change directly; generation/refit changes may represent lifecycle side effects rather than kernel acceleration.

- [ ] **Step 5: Render and validate HTML**

```bash
python experiments/cutedsl_qwen3_30ba3b_oci_1n4g/render_cutedsl_report.py
uv run --no-project --with pytest pytest -q tests/test_cutedsl_report.py
git diff --check
```

Open the static page locally, verify all relative links, then commit only lightweight HTML/JSON/CSV/config/script records. Keep `.nsys-rep`, full logs, and model artifacts on cluster storage.

- [ ] **Step 6: Final verification and commit**

```bash
git add -f experiments/cutedsl_qwen3_30ba3b_oci_1n4g/report experiments/cutedsl_qwen3_30ba3b_oci_1n4g/README.md
git commit -s -m "docs: publish NeMo 26.06 policy performance report"
```

The final report must distinguish measured results from upstream expectations and must not claim the NeMo 26.06 pretraining 1.3–1.6x gain as a NeMo-RL result.
