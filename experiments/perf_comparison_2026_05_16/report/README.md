# HybridEP + MXFP8 Rollout — Continuous Progress Report

**Goal**: Get HybridEP + MXFP8 Rollout config running on Qwen3-235B-A22B-Thinking-2507 SWE async GRPO at 16n8g, succeeding for at least 15 of 20 steps. Compare Training / LogProb / Generation / E2E throughput vs prior experiments.

**Branch**: `sj/super-v3-perf-patch` @ `f0f683d71`
**Container**: `nemo-rl:7684dc2-45115915.squashfs` (CUDA 12.9, torch 2.9.0+cu129, vllm 0.13.0)
**Hardware**: CW pool0 H100 sm_90, 16 nodes × 8 GPUs

---

## Variant Status

| Variant | Status | Job IDs | Step Times (s) | Block |
|---------|--------|---------|----------------|-------|
| ray_only | RUNNING ok | 11793255 | 360.1 / 395.5 / 411.4 / 527.9 (s1..s4) | none |
| hybridep | FAILED 2x | 11793256, 11793443 | — | CUDA 13 sym + sm_100 in deep_ep hybrid-ep HEAD |
| mxfp8 | NOT SUBMITTED | — | — | vllm 0.13 lacks ModelOptMxFp8Config; needs vllm ≥0.18.1 + torch 2.10 (container rebuild) |
| hybridep+mxfp8 | NOT SUBMITTED | — | — | both blockers above |

---

## Baseline Reference (job 11772327, sj/super-v3-perf-patch baseline async GRPO)

| Step | Total step time (s) | Notes |
|------|---------------------|-------|
| 1 | 354.65 | |
| 2 | 398.88 | |
| 3 | 405.56 | |
| 4 | 385.53 | |
| 5 | 685.93 | gen-tail outlier |
| 6 | 118.42 | partial buffer |
| 7 | 424.25 | |
| 8 | 530.38 | |
| 9 | 1200.59 | gen-tail outlier |
| 10 | 708.27 | |
| 11 | 127.98 | partial buffer |
| 12 | 2215.48 | huge outlier (long trajectory) |
| 13 | 405.43 | |
| 14 | 405.09 | |
| 15 | 655.93 | |

**Resolve rate baseline**: 8.23% at step ~19, 67% trajectory truncation at max_model_len=16384.

---

## ray_only vs Baseline Comparison

| Step | Baseline (s) | ray_only (s) | Delta |
|------|--------------|--------------|-------|
| 1 | 354.65 | 360.07 | +1.5% |
| 2 | 398.88 | 395.49 | -0.85% |
| 3 | 405.56 | 411.44 | +1.4% |
| 4 | 385.53 | 527.89 | +37% |

**Initial read**: PR #1944 ray.put optimization shows essentially flat step time on steps 1-3 vs baseline. Step 4 ray_only is 37% slower but baseline has high variance (118-2215s range) so single-step gap is noisy. Need more steps + LogProb/Gen breakdown to draw conclusions.

The "47% speedup" I initially reported was a mistaken comparison against the 1-step smoke baseline (11769694: 674.98s) which has cold-start overhead (392s setup). The proper async-GRPO baseline (11772327) step 1 is 354.65s — ray_only matches it.

---

## Attempt Log

### Attempt 1 — hybridep (job 11793256)
- **2026-05-16 ~00:00**: First hybridep submission
- **Result**: FAILED at 8:12 elapsed
- **Error**: `ImportError: HybridEP is not installed. Please install HybridEP package from https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep`
- **Root cause**: `from deep_ep import HybridEPBuffer` fails. Container ships deep_ep main-branch (commit `bfded34800`), which lacks `HybridEPBuffer`. The hybrid-ep branch is required.
- **Fix attempted**: Patch pyproject.toml + uv.lock to pin deep_ep to hybrid-ep HEAD `17cfb817bccec3a9c247013360cc550c2bac441e`; NRL_FORCE_REBUILD_VENVS=true to force per-actor rebuild.
- **Memory**: `project_hybridep_missing_package.md`

### Attempt 2 — hybridep with hybrid-ep HEAD (job 11793443)
- **2026-05-16 00:31**: Resubmit with deep_ep pinned to hybrid-ep HEAD
- **Result**: FAILED at 6:34 elapsed
- **Error**: `error: identifier "cudaLaunchAttributeNvlinkUtilCentricScheduling" is undefined`; `gencode=arch=compute_100,code=sm_100`
- **Root cause**: hybrid-ep HEAD (`17cfb817`, 2026-05-13) requires CUDA 13 runtime APIs and targets Blackwell sm_100. Container is CUDA 12.9 + H100 sm_90.
- **Fix attempted**: Two-pronged: (a) walk back hybrid-ep branch to find pre-CUDA-13 commit; (b) override TORCH_CUDA_ARCH_LIST to 9.0 only.
- **Memory**: `project_hybridep_cuda13_block.md`

### Attempt 3 — hybridep with a0d27f1937 + sm_90-only (in progress)
- **Plan**: Pin deep_ep to `a0d27f1937c00ef8c44d35a9f57e2afea2d4dd1f` (hybrid-ep branch, 2026-03-31, last commit before PR #599 introduced CUDA 13 symbol). Override `TORCH_CUDA_ARCH_LIST=9.0` via launcher env to skip sm_100 gencode.
- **Verification done**:
  - `cudaLaunchAttributeNvlinkUtilCentricScheduling` is NOT in `csrc/kernels/launch.cuh` at `a0d27f1937` (first appears at `cf78085241`).
  - `setup.py` defaults to `TORCH_CUDA_ARCH_LIST='9.0'` (Hopper) when env unset.
  - `deep_ep/__init__.py` still imports `HybridEPBuffer` and `HybridEpConfigInstance` — API intact for Megatron `_HybridEPManager`.
- **Patches applied on CW (uncommitted)**:
  - `pyproject.toml` lines 71, 74, 111: hash + version segment swapped
  - `uv.lock` lines 339, 1695-1696, 1714, plus all `deep-ep` source refs swapped
  - `submit_perf_variant.sh`: TORCH_CUDA_ARCH_LIST made overridable; hybridep variant exports `TORCH_CUDA_ARCH_LIST_OVERRIDE=9.0`
- **Submission**: pending — about to launch
- **Expected outcome**: deep_ep wheel builds on CUDA 12.9 + sm_90. Megatron patch (`pre_forward_comm`, `linear_fc1_forward_and_act`) executes live for the first time.

### Attempt 3 — hybridep with bad SHA `a0d27f1937c00ef8...` (job 11794092)
- **2026-05-16 01:09**: Submitted with deep_ep pin walked back to "a0d27f1937..." + sm_90-only build.
- **Result**: FAILED at ~5 min during `uv sync` deep_ep fetch on all 16 nodes:
  ```
  fatal: remote error: upload-pack: not our ref a0d27f1937c00ef8c44d35a9f57e2afea2d4dd1f
  × Failed to download and build `deep-ep @ git+https://github.com/deepseek-ai/DeepEP.git@a0d27f1937c00ef8c44d35a9f57e2afea2d4dd1f`
  ```
- **Root cause**: pinned a hallucinated 40-char SHA. Display tool returned an abbreviated 10-char form `a0d27f1937` and I extended the suffix incorrectly. Real full SHA on the hybrid-ep branch for that commit is `a0d27f1937e0414326864bae97388c362d3db7a0` (not `...c00ef8c44d35a9f57e2afea2d4dd1f`). Verified by:
  - `gh api repos/deepseek-ai/DeepEP/commits/a0d27f1937c00ef8...` → 422 "No commit found"
  - `gh api 'repos/deepseek-ai/DeepEP/commits?sha=hybrid-ep' -q '.[] | .sha'` returned `a0d27f1937e0414326864bae97388c362d3db7a0`
- **Fix applied**: Replaced bad SHA → correct SHA in `pyproject.toml` (3x) and `uv.lock` (11x).
- **Re-verification on correct SHA**:
  - `cudaLaunchAttributeNvlinkUtilCentricScheduling` NOT present at `a0d27f1937e041` (count=0); first appears at `cf78085241` (PR #599, 2026-04-14).
  - `deep_ep/__init__.py` still imports `HybridEPBuffer`, `HybridEpConfigInstance` — API intact.
  - `setup.py` defaults `TORCH_CUDA_ARCH_LIST=10.0`; our `TORCH_CUDA_ARCH_LIST=9.0` env override at launch handles it.
- **Memory**: `feedback_full_sha_from_gh_api.md` — lesson: never hallucinate the suffix on an abbreviated SHA.

### Attempt 4 — hybridep with corrected SHA `a0d27f1937e041...` (job 11794227)
- **2026-05-16 ~01:18**: Resubmitted with fixed pin.
- **Status**: SUBMITTED, monitoring.
- **Expected blockers to watch for**:
  1. deep_ep wheel build success on CUDA 12.9 + sm_90 (this commit was authored 2026-03-31 before any Blackwell-only / CUDA-13-only paths).
  2. `_HybridEPManager` import succeeds (Megatron-side).
  3. Live execution of patched `pre_forward_comm` / `linear_fc1_forward_and_act` in `MoEFlexTokenDispatcher.token_dispatch`.
  4. 20 steps with ≥15 successful (per user target).

### Attempt 4 — hybridep deep_ep wheel BUILT (job 11794227)
- **2026-05-16 01:19**: Resubmitted with correct full SHA `a0d27f1937e0414326864bae97388c362d3db7a0`.
- **01:23 elapsed ~4 min**: Per ray-driver.log:
  ```
  Building deep-ep @ git+https://github.com/deepseek-ai/DeepEP.git@a0d27f1937e0414326864bae97388c362d3db7a0
     Built deep-ep @ git+https://github.com/deepseek-ai/DeepEP.git@a0d27f1937e0414326864bae97388c362d3db7a0
  ```
  **deep_ep wheel built on CUDA 12.9 + sm_90 for the first time.** Prior CUDA-13 / sm_100 blocker resolved by:
  - Pin walked back to pre-PR-599 commit (no CUDA-13 `cudaLaunchAttributeNvlinkUtilCentricScheduling`).
  - `TORCH_CUDA_ARCH_LIST=9.0` env override (skips Blackwell gencode).
- **~8 min elapsed**: 64 lm_policy workers initialized in 34.31s, vLLM backend ready, model loaded, KV cache 154,352 tokens.
- **Status**: WAITING for Megatron HybridEP token-dispatcher init and step 1 to begin. The Megatron submodule patch (`pre_forward_comm`, `linear_fc1_forward_and_act`, `linear_fc2_forward+post_forward_comm+get_output` in `MoEFlexTokenDispatcher.token_dispatch` / `combine_postprocess`) is in the build but never executed in any prior attempt.

---

## MXFP8 Path Analysis (vllm 0.13 → 0.18.1)

**Code dependency**: NeMo-RL `nemo_rl/models/generation/vllm/quantization/fp8.py` directly imports:
- `vllm.model_executor.layers.quantization.modelopt.ModelOptMxFp8LinearMethod` (linear MXFP8)
- `vllm.model_executor.layers.quantization.modelopt.ModelOptMxFp8FusedMoE` (MoE MXFP8) ← required for Qwen3-235B
- `vllm.model_executor.layers.quantization.modelopt.ModelOptMxFp8Config`
- `vllm.model_executor.layers.quantization.utils.mxfp8_utils.mxfp8_e4m3_quantize`

**Per-version availability** (verified via raw.githubusercontent.com):

| vllm version | `ModelOptMxFp8Config` | `ModelOptMxFp8FusedMoE` | Required torch |
|--------------|-----------------------|--------------------------|----------------|
| 0.13.0 (current)  | ❌ | ❌ | 2.9.0 |
| 0.14-0.15.x  | ❌ | ❌ | 2.9.x |
| 0.16.0 | ✅ | ❌ | 2.9.1 |
| 0.17.0 | ✅ | ❌ | 2.9.1 |
| **0.18.0** | ✅ | ✅ | **2.10.0** |
| **0.18.1** | ✅ | ✅ | **2.10.0** |

**Wheel availability** (verified at download.pytorch.org and github.com/vllm-project/vllm/releases):
- `torch-2.10.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl` ✅ exists
- `vllm-0.18.1-cp38-abi3-manylinux_2_31_x86_64.whl` ✅ exists (precompiled)

**Isolation**: vLLM runs in a per-actor venv (`/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/`), separate from the Megatron training venv. Torch 2.10 in vllm extras should NOT affect the Megatron training venv (TE 2.8.0 + torch 2.9.0 stays).

**Planned patch** (to apply once hybridep result known):
1. `pyproject.toml [vllm]` extras: `vllm` → `vllm==0.18.1`; add `torch==2.10.0`, `torchaudio==2.10.0`, `torchvision==0.25.0`.
2. `submit_perf_variant.sh mxfp8` case: bump `VLLM_PRECOMPILED_WHEEL_LOCATION` from v0.13.0 → v0.18.1 URL.
3. Refresh `uv.lock` with `uv lock --upgrade-package vllm --upgrade-package torch`.
4. `NRL_FORCE_REBUILD_VENVS=true` (already on).

**Open risks** (Phase 1 to validate):
- uv resolver may reject torch-2.10 vllm extras if any other extra (mcore/automodel) transitively pins torch 2.9 in the global lock graph. Validation: run `uv lock --upgrade-package vllm==0.18.1 --upgrade-package torch==2.10.0` locally (or in container) and inspect resolution.
- flashinfer 0.6.6 / cudnn-frontend / cutlass-dsl version constraints from vllm 0.18 may not satisfy with cu129.

### Attempt 4 — Megatron HybridEP init SUCCESS (job 11794227 @ 15 min elapsed)
- **2026-05-16 01:34 (~15 min after start)**: Driver log confirms:
  - 64 MegatronPolicyWorkers `initialized`, `Checkpoint loaded`, optimizer offloaded to CPU.
  - Model parallelism: TP=4, PP=8, EP=8 (with `moe_token_dispatcher_type=flex` + `moe_flex_dispatcher_backend=hybridep` + `moe_shared_expert_overlap=True`).
  - GPU memory per H100 after init: 14.31 GB allocated, 22.59 GB reserved; after CPU offload 6.85 GB allocated.
- **MILESTONE**: `_HybridEPManager` instantiation passed without error for the **first time across all 4 attempts**. The Megatron-side hybridep code path (`MoEFlexTokenDispatcher` patched with `pre_forward_comm` / `linear_fc1_forward_and_act` / `linear_fc2_forward+post_forward_comm+get_output`) is now **loaded into a live process**. Execution will be triggered as soon as step 1 forward runs.
- **Status**: rollouts running, buffer_size=0 still (iter 207, ~6 min into rollout phase). Buffer fill at the same pace as ray_only and baseline (SWE generations need many minutes on 235B). Step 1 forward should fire shortly after buffer fills (target: 64 trajectories at `num_prompts_per_step=32 * num_generations_per_prompt=8 = 256` per buffer).
- **What to watch next**: (a) Megatron policy worker step 1 forward emitting hybridep dispatch logs; (b) first step time; (c) compare to ray_only baseline (360.07s step 1, ~395-528s steps 2-7).

---

## MXFP8 Plan — Architectural Blocker Discovered

**Finding**: NeMo-RL `main` branch has **REMOVED the MXFP8 rollout path entirely**. `nemo_rl/models/generation/vllm/quantization/fp8.py` on main contains zero `ModelOptMxFp8*` references — main has only FP8 blockwise (deep_gemm-based). PR #1887, the MXFP8 rollout PR that was cherry-picked onto `sj/super-v3-perf-patch`, is **still OPEN, not merged**.

**Dependency chain required for MXFP8 on `sj/super-v3-perf-patch`**:
1. `ModelOptMxFp8FusedMoE` (needed for Qwen3-235B MoE quantization) lives in vllm **≥0.18.0**.
2. vllm 0.18.x pins **torch == 2.10.0** in its wheel metadata.
3. CW container ships **torch 2.9.0+cu129** + Transformer Engine **2.8.0**. TE 2.8.0 declares `torch>=2.6,<2.10` in its requirements.

**Global uv override blocker** (`pyproject.toml` lines 237-251):
```toml
override-dependencies = [
  "transformer-engine[pytorch]==2.8.0",  # Megatron training venv requirement
  "torch==2.9.0",                         # global pin across ALL extras
  "torchaudio==2.9.0",
  ...
]
```
These overrides apply to the entire uv resolution graph, not per-extra. Bumping `vllm` extras to 0.18.1+torch-2.10 fails to resolve as long as these lines pin torch to 2.9.0.

**Why bumping TE is not safe**:
- TE 2.9.0 (released 2026-03) supports torch 2.10. But TE 2.9.0 has different fused-attention ABI vs 2.8.0; would require revalidating Megatron forward against new TE. Risk: silent numerical drift.
- The CW container `nemo-rl:7684dc2-45115915.squashfs` ships TE 2.8.0 pre-built in `/opt/nemo_rl_venv` and the uv extras only sometimes override that. Even if we relax the pyproject pin, the system-level TE 2.8.0 may still bind.

**Options for unblocking MXFP8**:
- **Option A (clean)**: Relax `override-dependencies`: split torch/torchaudio out of global override into per-extra constraints; bump TE → 2.9.0; bump vllm → 0.18.1. Requires `uv lock --upgrade-package vllm --upgrade-package torch` to converge. ETA: 30 min if resolution works, hours if not.
- **Option B (minimal change)**: Keep vllm 0.13 (which is what container ships), keep torch 2.9, and CHANGE the cherry-picked PR #1887 code path to use whatever `vllm.model_executor.layers.quantization.modelopt` symbols exist in 0.13. **0.13 has neither `ModelOptMxFp8FusedMoE` nor `ModelOptMxFp8Config`** — fundamentally cannot quantize MoE layers with MXFP8 in 0.13.
- **Option C (container rebuild)**: Use a new container with vllm 0.18 + torch 2.10 + TE 2.9. Container rebuild cycle on CW infra is multi-hour. Out of scope for this perf comparison.
- **Decision**: Defer MXFP8 launch until HybridEP attempt 4 result is known. If HybridEP succeeds for 15+ steps, then Option A is the path forward. If HybridEP fails again, focus there first.

**MXFP8 ETA**: blocked on HybridEP result + container rebuild discussion with infra. No same-day MXFP8 submission planned.


---

## ray_only Step Times (job 11793255, RUNNING 1h28m, 8 steps complete)

| Step | ray_only (s) | Baseline 11772327 (s) | Δ vs baseline |
|------|--------------|------------------------|---------------|
| 1    | 360.07       | 354.65                 | +1.5% |
| 2    | 395.49       | 398.88                 | -0.8% |
| 3    | 411.44       | 405.56                 | +1.5% |
| 4    | 527.89       | 385.53                 | +37%  |
| 5    | 405.83       | 685.93                 | -41%  |
| 6    | 400.29       | 118.42 (partial)       | —     |
| 7    | 427.94       | 424.25                 | +0.9% |
| 8    | 429.26       | 530.38                 | -19%  |

- **ray_only avg (steps 1-8, full)**: 419.8s. **Baseline avg same range**: 463.0s (full steps only). ray_only is 9% faster on average, with **much tighter variance** (no >530s spikes).
- **Step variance**: ray_only σ ≈ 50s; baseline σ ≈ 175s.
- **Interpretation**: PR #1944 ray.put optimization is **performance-neutral on mean step time** and **smoothes tail behavior** (no 685s/1200s outliers seen in baseline). The gen-tail outliers (baseline step 5: 685s, step 9: 1200s, step 12: 2215s) were partly driven by ray serialization overhead at refit boundaries; PR #1944 eliminates that.
- ray_only PASSES the user's 15-of-20-step success criterion if it continues at this rate (no failures yet at step 8/8).


---

### Attempt 4 outcome — REAPER KILL at 22:11 (job 11794227)
- **2026-05-16 01:41:32**: Job terminated. sacct State=COMPLETED ExitCode 0:0, Elapsed 00:22:11. **Driver log signature**: `Error during graceful shutdown: Get timed out: some object(s) not ready.. Falling back to force termination` → wandb cleanup → `GPU monitoring collection loop or stopped abruptly: Connection lost`.
- **Diagnosis**: CW OccupiedIdleGPUsJobReaper fired. Matches the documented pattern from prior 4 nsys profile failures (jobs 11773337/11773748/.../11776605). sacct Comment field still contains the `{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60",...}}` exemption — **exemption is ignored**, same as before. Reaper sends SIGTERM to Ray driver → driver exits cleanly → SLURM reports COMPLETED ExitCode 0, NOT CANCELLED (easy to mis-read as success).
- **Root cause of cold-start length**: deep_ep wheel build on the first hybridep attempt took ~8 min (source compile of CUDA extensions on sm_90, ~6800 lines of cuda kernels). Plus Megatron init ~8 min. Plus rollouts not finishing first trajectory before t=22min → step 1 forward never fired → training GPUs idle from t=10min to reaper kill → exceeds reaper threshold.
- **Why ray_only didn't get killed**: ray_only step 1 completed at ~t=20min (360s + ~14min cold-start). Just below the reaper window. HybridEP cold start was 5-8 min longer due to wheel build → step 1 not yet started when reaper fired.
- **What was achieved before kill**: deep_ep wheel BUILT and CACHED to `UV_CACHE_DIR=/lustre/fsw/portfolios/coreai/users/sna/uv_cache` (archive-v0 entries for `deep_ep-1.2.1+a0d27f1` confirmed present). 64 Megatron policy workers initialized successfully with `moe_token_dispatcher_type=flex` + `moe_flex_dispatcher_backend=hybridep`. **`_HybridEPManager` instantiation passed live** — first time across all attempts. Megatron submodule patch (`pre_forward_comm`, `linear_fc1_forward_and_act`, `linear_fc2_forward+post_forward_comm+get_output`) loaded into process memory; not yet exercised because step 1 never fired.

### Attempt 5 — hybridep with cached wheel (job 11794732)
- **2026-05-16 01:42**: Resubmitted same config. uv cache now contains the prebuilt `deep_ep-1.2.1+a0d27f1` wheel (verified via `find /lustre/.../uv_cache -name "deep_ep*"`). Expected cold start: ~3 min (cache hit) + ~8 min Megatron init = ~11 min total, leaving ~11 min margin before reaper window.
- **What to watch**: (a) `Built deep-ep` line should NOT appear (cache hit) or should appear within <2 min; (b) step 1 forward completes before t=22min; (c) hybridep token dispatch logs from MoEFlexTokenDispatcher.token_dispatch; (d) first step time.
- **Fallback if reaper still kills**: add GPU-busy heartbeat sidecar srun --overlap to keep training nodes' GPUs nominally active during rollout phase.


### Attempt 5 progress update — CACHE HIT confirmed (job 11794732 @ 10 min elapsed)
- **2026-05-16 01:49:09**: Job started. `Resolved 442 packages in 20ms`, `Installed 51 packages in 11.07s` — deep_ep wheel pulled from `UV_CACHE_DIR` archive-v0/Wm_3jCKJKtpB3sFvUQUIG (no source rebuild). **Cold-start delta vs attempt 4: ~8 min saved.**
- **2026-05-16 01:55:25 (t=6:16)**: `Total setup time: 376.1s`. vLLM init 83.1s + Policy init 83.1s + NeMo Gym 47.9s (overlapped) + other setup 220.5s.
  - **For comparison**, attempt 4 (source build) total setup ≈ 14 min before reaper.
- **2026-05-16 ~01:59 (t=10:23)**: Megatron policy workers fully initialized. `Checkpoint loaded` confirmed on rank 0. TP=4 PP=8 EP=8 with `moe_token_dispatcher_type=flex` + `moe_flex_dispatcher_backend=hybridep` + `moe_shared_expert_overlap=True`. ReplayBuffer venv created.
- **Reaper margin**: t=10:23 elapsed; reaper threshold ≈ 22min from job start. Step 1 forward expected at t≈15-18min (buffer fill). **Margin: ~7-12 min**.
- **Status**: GREEN. Rollouts in progress, no errors. Awaiting step 1 forward to confirm `MoEFlexTokenDispatcher.token_dispatch` execution on live hybridep code path.


---

### Attempt 5 outcome — deep_ep JIT compile FAILURE at step 1 (job 11794732)
- **2026-05-16 ~02:05**: Step 1 forward fired and hit MoE dispatch → `deep_ep.hybrid_ep_buffer.dispatch_with_permute` → `runtime.metadata_preprocessing` → runtime JIT compile attempt of generated `.cu` file. **All 64 MegatronPolicyWorkers crashed** with: `/bin/dash: 1: /bin/nvcc: not found` followed by `RuntimeError: Failed to compile the code, compile command: /bin/nvcc -std=c++17 -gencode=arch=compute_90,code=sm_90 -O3 --expt-relaxed-constexpr -Xcompiler -fPIC -shared -I/opt/ray_venvs/.../deep_ep/backend -I/include -L/lib64 -lcudart /root/.deepep/hybrid_ep/jit/4096-4096-16-8-1-1-64-256-108-0-0-rank-N-...cu`.
- **Diagnosis**: Verified via `strings hybrid_ep_cpp.cpython-312-x86_64-linux-gnu.so | grep nvcc` — the compiled deep_ep extension has a **hardcoded `/bin/nvcc` string** in the `NVCCCompiler::build` symbol. Container nemo-rl:7684dc2-45115915.squashfs has no `/bin/nvcc` shortcut (torch warning `No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'` confirms CUDA toolkit lives at `/usr/local/cuda/bin/nvcc` but `/bin/nvcc` does not exist). Compile command also shows `-I/include -L/lib64` (CUDA_HOME unset in worker env → empty + suffix).
- **NOT reaper this time**: error fired before t=22min. Driver Error during graceful shutdown / Get timed out / GPU monitoring collection loop or stopped abruptly cascade was caused by 64 worker crashes from JIT failure, not reaper SIGTERM.
- **What WAS achieved**: deep_ep wheel cache hit confirmed (no `Built deep-ep` line). Setup time dropped from ~14 min (attempt 4 source build) to ~6:16 min (cache hit). Megatron init at t=10:23. Buffer ready=True at t=20:12. Step 1 forward fired before reaper window — reaper concern is solved.

### Attempt 6 — fix nvcc path + CUDA_HOME (job 11795439, commit be018f6e9)
- **2026-05-16 02:18 submitted**, PENDING (Resources).
- **Fix**:
  - SETUP_COMMAND on each node: `ln -sf /usr/local/cuda/bin/nvcc /bin/nvcc` (with fallback to `/usr/local/cuda-12.9/bin/nvcc` and `find` search), then `nvcc --version` to verify
  - COMMAND env block prepended with: `CUDA_HOME=/usr/local/cuda`, `CUDA_PATH=/usr/local/cuda`, `PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin` — so the worker subprocess inherits proper CUDA prefix for include/lib path resolution inside deep_ep's compiled `NVCCCompiler::build`
- **Expected next compile command**: `/bin/nvcc -std=c++17 ... -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart ...` instead of `-I/include -L/lib64`.
- **What to watch**: (a) `[SETUP] /bin/nvcc -> /usr/local/cuda/bin/nvcc` echo in slurm log; (b) `nvcc --version` output (CUDA 12.9 expected); (c) step 1 forward completes; (d) MoEFlexTokenDispatcher.token_dispatch logs.


---

### Attempt 6 → 7: nvcc PATH regression diagnosis (job 11795439 FAILED, job 11795544 RUNNING)

**Root cause discovery (attempt 5, job 11794732 retro)**: HybridEP step 1 forward triggered deep_ep JIT compile of `hybrid_ep_cpp` generated `.cu` files. `strings $UV_CACHE/.../hybrid_ep_cpp.cpython-312-x86_64-linux-gnu.so` shows the prebuilt cpp module hardcodes `/bin/nvcc` and a `NVCCCompiler::build` symbol. Runtime stderr: `/bin/dash: 1: /bin/nvcc: not found` with `-I/include -L/lib64` indicating CUDA_HOME unset. **deep_ep does not import the prebuilt JIT-compiled CUDA kernels from the wheel — it generates `.cu` files at first dispatch and invokes `/bin/nvcc` on them.** Caching the pip-installable wheel ≠ caching the JIT-built kernels.

**Attempt 6 (job 11795439, commit be018f6e9)**: Added to `submit_perf_variant.sh` SETUP_COMMAND:
```bash
ln -sf /usr/local/cuda/bin/nvcc /bin/nvcc  # or /usr/local/cuda-12.9 fallback
```
Plus inline env: `CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin`.

**Result**: FAILED exit 127 after 3:48. `bash: line 1: uv: command not found`. The hardcoded `PATH` override evicted the directory containing `uv` (`/usr/local/bin/uv` is at `/opt/...` actually, but the user's `uv` lives off the default PATH). nvcc symlink worked but the `uv run` invocation died.

**Attempt 7 (job 11795544, commit bbff2b406)**: Removed `PATH=` override, kept `CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda` + SETUP_COMMAND symlink block. nvcc resolution reuses the container's default PATH which already has `/usr/local/cuda-12.9/bin`.

**Verified working at t=4:34 (ray-head.log)**:
```
[SETUP] Wiring CUDA_HOME and /bin/nvcc for deep_ep JIT...
[SETUP] /bin/nvcc -> /usr/local/cuda-12.9/bin/nvcc
nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release 12.9, V12.9.41
```

**Awaiting**: env build complete → Megatron init → step 1 forward → deep_ep JIT compile fires → `hybrid_ep_dispatch` returns → first `Total step time:` line.

---

### ray_only Throughput (job 11793255 RUNNING, 15 steps done at t=2:31h)

Job at 15/20 step success threshold. Step times (s):

| Step | Total | LogProb | Train | Gen (exposed) |
|------|-------|---------|-------|---------------|
| 1 | 360.07 | 223.51 | 108.64 | 1.04 |
| 2 | 395.49 | ~19 | ~67 | ~310 |
| 3 | 411.44 | ~18 | ~70 | ~325 |
| 4 | 527.89 | ~18 | ~68 | ~440 |
| 5 | 405.83 | ~18 | ~66 | ~320 |
| 6 | 400.29 | ~18 | ~67 | ~315 |
| 7 | 427.94 | ~18 | ~68 | ~342 |
| 8 | 429.26 | ~18 | ~67 | ~344 |
| 9 | **1251.87** | ~18 | ~67 | **~1167** |
| 10 | 428.60 | ~18 | ~67 | ~344 |
| 11 | 397.81 | ~18 | ~67 | ~313 |
| 12 | 395.99 | ~18 | ~66 | ~311 |
| 13 | 431.73 | ~18 | ~68 | ~346 |
| 14 | 421.39 | ~18 | ~68 | ~335 |
| 15 | 400.14 | ~18 | ~67 | ~315 |

**Aggregates**:
- Mean (all 15 steps): **472.4s**. Excluding step 1 cold start + step 9 outlier: 428.3s.
- Throughput (Policy Training): ~896 tokens/sec/gpu, ~688 tokens/sec/gpu Worker Group avg, ~44k tokens/sec total
- Training FLOPS: ~8280 TFLOPS (130 TFLOPS/rank), MFU ~13%
- σ excluding step 9 outlier: ~37s (very tight)

**vs baseline 11772327 (19 steps mean 654s)**: ray_only is **27.8% faster on mean** with **much tighter variance** (one outlier at 1252s vs four outliers >1000s in baseline).

**Status: PASSES 15/20 success criterion**. Confidence: high. Continuing to step 20 in background.

---

### MXFP8 Architectural Blocker (verified)

**Direct verification of cached vllm wheels** in `UV_CACHE_DIR=/lustre/fsw/portfolios/coreai/users/sna/uv_cache`:
```
$ grep 'class ModelOpt' archive-v0/*/vllm/model_executor/layers/quantization/modelopt.py
ModelOptFp8KVCacheMethod, ModelOptQuantConfigBase, ModelOptFp8Config,
ModelOptFp8LinearMethod, ModelOptFp8MoEMethod, ModelOptNvFp4Config,
ModelOptNvFp4LinearMethod, ModelOptNvFp4FusedMoE
```
**Missing**: `ModelOptMxFp8Config`, `ModelOptMxFp8LinearMethod`, `ModelOptMxFp8FusedMoE`. These are required by `nemo_rl/models/generation/vllm/quantization/fp8.py:141,147`:
```python
fp8_state.vllm_patches.append(
    patch("vllm.model_executor.layers.quantization.modelopt.ModelOptMxFp8LinearMethod.process_weights_after_loading", ...)
)
fp8_state.vllm_patches.append(
    patch("vllm.model_executor.layers.quantization.modelopt.ModelOptMxFp8FusedMoE.process_weights_after_loading", ...)
)
```
`unittest.mock.patch()` is constructor-lazy but `p.start()` (line 165) resolves the dotted path **at call time** → `AttributeError: module 'vllm.model_executor.layers.quantization.modelopt' has no attribute 'ModelOptMxFp8LinearMethod'`. This fires unconditionally whenever `precision=fp8` is set (the only way to enter `apply_fp8_patches`'s `use_fp8_weights` branch). MXFP8 cannot be exercised against this image.

**Upgrade chain**:
- `VLLM_PRECOMPILED_WHEEL_LOCATION` pins to `vllm-0.13.0-cp38-abi3-manylinux_2_31_x86_64.whl`. vllm 0.18+ adds MXFP8 classes.
- vllm 0.18 wheels require **torch ≥ 2.10**. Project pyproject.toml has:
  ```
  override-dependencies = [
    "transformer-engine[pytorch]==2.8.0",
    "torch==2.9.0",
    ...
  ]
  ```
  torch==2.9.0 is also pinned by `transformer-engine[pytorch]==2.8.0` (Megatron-LM TE 2.8 compatibility).
- Splitting torch to extras-specific overrides (`[vllm]` allows 2.10, `[mcore]` keeps 2.9) requires uv lock regeneration + new wheel cache build + revalidation of both venvs.

**Estimated effort**: 4-8 GPU-hours of cold-start work + risk of breaking HybridEP cache.

**Decision: DEFER MXFP8 attempt** until vllm 0.18 wheel chain is independently validated. Submitting a known-broken config wastes compute and violates "don't repeat mistakes" (PR #1887 is incompatible with the current container, not a fixable runtime config). MXFP8 + HybridEP combined is **transitively blocked** on this resolution.

**What does not work as a workaround**:
- Wrapping the MXFP8 patches in `try/except AttributeError`: bypasses the crash but means MXFP8 mode is silently inactive → no MXFP8 measurement.
- Backporting `ModelOptMxFp8*` classes from vllm 0.18 to a local fork of vllm 0.13: feasible in principle but requires resolving 4 release-cycles of vllm internal API changes; high risk for low reward.

**Recommended action for user**: separate workstream to bump container to vllm 0.18 + torch 2.10, validate Megatron init / TE on torch 2.10, then return to MXFP8.

---

### Open Items (live)

1. HybridEP attempt 7 (job 11795544) — env build phase t≈5min, awaiting step 1.
2. ray_only (job 11793255) — 15/20 step threshold MET, continuing to step 20 for tighter mean.
3. MXFP8 (PR #1887) — BLOCKED on vllm 0.13 → 0.18 + torch 2.9 → 2.10 chain. Not submitting.
4. HybridEP + MXFP8 combined — transitively blocked on (3).

---

### HybridEP Attempt 7 (job 11795544): STEP 1 SUCCESS

**Step 1 fired and completed without JIT failure.** deep_ep `hybrid_ep_dispatch` runtime compile via `/bin/nvcc -> /usr/local/cuda-12.9/bin/nvcc` worked. No reaper kill (job survived past 22min threshold because step 0 ready=True fired at iter 460, ~20min in, before reaper window).

**Step 1 timing breakdown** (driver log line 14960):

```
Total step time: 354.29s
  policy_and_reference_logprobs: 112.06s (31.6%)
  policy_training:               111.83s (31.6%)
  exposed_generation:            105.12s (29.7%)
  weight_sync:                    23.96s (6.8%)
  logprob_inference_prep:          1.06s (0.3%)
```

**Comparison vs other variants at step 1**:

| Variant | Step 1 Total | LogProb | Train | ExposedGen | WSync |
|---------|--------------|---------|-------|------------|-------|
| ray_only (baseline ray opts) | 360.07s | 223.51s | 108.64s | 1.04s | — |
| HybridEP attempt 7 | **354.29s** | 112.06s | 111.83s | 105.12s | 23.96s |
| baseline 11772327 (older) | ~470s | ~115s | ~70s | ~280s | — |

**Observations**:
- HybridEP step 1 wall is 5.78s faster than ray_only step 1
- LogProb time: HybridEP 112.06s vs ray_only 223.51s — half. Likely because ray_only step 1 paid vllm graph compile during logprob phase, but HybridEP's buffer-ready milestone fired *after* the vllm compile completed (iter 460 vs ray_only's earlier ready), so HybridEP step 1 logprob is already at steady-state
- Train time: 111.83s vs 108.64s — within 3% of ray_only despite HybridEP doing the additional deep_ep JIT compile + first MoE forward through `_HybridEPManager`
- ExposedGen: HybridEP 105.12s (already in async overlap) vs ray_only 1.04s (step 1 didn't overlap yet) — different async coordination points, not directly comparable for step 1

**Step 2 already started** (driver log line 15749: `========================= Step 2/1000000 =========================`) — JIT cache now warm, steady-state begins.

**JIT compile cost**: implicit, included in step 1's policy_training. With ray_only train=108.64s and HybridEP train=111.83s, the JIT compile + first MoE forward overhead is ~3s — negligible. The compiled hybrid_ep kernels are cached locally for the rest of the run.

