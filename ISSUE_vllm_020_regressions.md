# vLLM 0.17.1 → 0.20.0 bump (PR #2384) introduces three distinct regressions

**Triggering commit:** `241ece552 chore: Upgrade vLLM from 0.17.1 to 0.20.0 (#2384)` (2026-05-22)

**Bisect status:** Confirmed by direct A/B run of the same recipe at `501cd12ce` (parent, vllm 0.17.1) and `241ece552` (vllm 0.20.0). Everything else held constant — same transformers (5.3.0), same mamba-ssm rev, same container, same hardware. Recipes are run **upstream-style** — `bash tests/test_suites/llm/<recipe>.sh` via the standard `user_run_slurm.sh` launcher with only the recipe's own Hydra defaults plus a `grpo.max_num_steps=3` smoke shortener.

---

## TL;DR

A single PR (#2384) — upgrading vLLM from 0.17.1 to 0.20.0 — breaks at least three independent code paths in the GRPO recipe suite:

| # | Recipes affected | Failure | Family |
|---|---|---|---|
| 1 | 4 (all FP8 recipes) | `NotImplementedError: "remainder_cuda" not implemented for 'Float8_e4m3fn'` | FP8 |
| 2 | 2 (nano-v2 mamba) | `ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (658)` | Mamba |
| 3 | 1 (moonlight automodel EP=8) | trainer-vs-generator weight desync — `gen_kl_error` ≈ 0.57 (target < 0.001); `token_mult_prob_error` ≈ 10⁹ (target ≈ 1) | Refit |

Net effect: 7 recipes in `tests/test_suites/nightly.txt` go from PASS → FAIL across the bump.

---

## Clean ablation matrix (all 6 cells, all wandb runs in `nemorl-probes-zhiyul`)

Three recipes × two consecutive main commits = six 1-node runs, every Hydra knob held constant except the pyproject-pinned vLLM version. All runs use `nightly-05242026.squashfs` with `NRL_FORCE_REBUILD_VENVS=true` so each commit's pyproject is honored. All runs invoke the recipe `.sh` directly through `user_run_slurm.sh`.

| Issue | pre-bump (`501cd12ce`, vllm **0.17.1**) | post-bump (`241ece552`, vllm **0.20.0**) |
|---|---|---|
| **FP8** (`grpo-qwen3-8b-base-1n8g-fp8-kvcache-megatron`) | ✅ COMPLETED @ 17:09, no `remainder_cuda` — [12105330 / y07rpz0b](https://wandb.ai/nvidia/nemorl-probes-zhiyul/runs/y07rpz0b) | ❌ FAIL `remainder_cuda` @ 2:36 — [12105331 / 88x8l1ax](https://wandb.ai/nvidia/nemorl-probes-zhiyul/runs/88x8l1ax) |
| **Mamba** (`grpo-nano-v2-12b-1n8g-megatron`) | ✅ COMPLETED @ 15:07, no `max_num_seqs` — [12105332 / oqrv8bmn](https://wandb.ai/nvidia/nemorl-probes-zhiyul/runs/oqrv8bmn) | ❌ FAIL `max_num_seqs > cache_blocks` @ 3:04 — [12105333 / y6syiic7](https://wandb.ai/nvidia/nemorl-probes-zhiyul/runs/y6syiic7) |
| **Refit** (`grpo-moonlight-16b-automodel-1n8g-ep8`) | ✅ COMPLETED @ 18:42<br>gen_kl=**0.000295**<br>token_mult_prob_error=**1.006**<br>reward=**0.584** — [12105334 / 0mdl5hq8](https://wandb.ai/nvidia/nemorl-probes-zhiyul/runs/0mdl5hq8) | ⚠️ Slurm-COMPLETED @ 21:49 but soft-broken:<br>gen_kl=**0.553** (1800× over)<br>token_mult_prob_error=**642M** (target ≈ 1)<br>reward=**0.023** (no learning) — [12105335 / i227tcmb](https://wandb.ai/nvidia/nemorl-probes-zhiyul/runs/i227tcmb) |

Every prediction held. The refit row is especially notable: Slurm exits cleanly with code 0 on both sides — only the recipe's closing `check_metrics.py` distinguishes pass from fail. Any CI gate that watches only exit codes will silently miss this regression.

---

## Reproduction (deterministic)

Each regression reproduces in a single 1-node run.

### Stack used for the bisect

- Repo HEAD: `main` at `4454fa64e`.
- Container: `nightly-05242026.squashfs` (rebuild via `NRL_FORCE_REBUILD_VENVS=true` resolves each commit's pyproject, so any recent container works).
- Launcher: stock `user_run_slurm.sh`.
- Hydra overrides: only `grpo.max_num_steps=3` (smoke) and `logger.wandb_*`.

### Recipe set

```
tests/test_suites/llm/grpo-llama3.1-8b-instruct-1n8g-megatron-fp8-rollouts.v3.sh   # FP8
tests/test_suites/llm/grpo-llama3.1-8b-instruct-2n8g-megatron-fp8-e2e.sh           # FP8
tests/test_suites/llm/grpo-qwen3-8b-base-1n8g-fp8-kvcache-megatron.sh              # FP8
tests/test_suites/llm/grpo-moonlight-16ba3b-4n8g-megatron-fp8-e2e.sh               # FP8
tests/test_suites/llm/grpo-nano-v2-12b-1n8g-megatron.sh                            # Mamba
tests/test_suites/llm/grpo-nano-v2-12b-2n8g-fsdp2tp1.sh                            # Mamba
tests/test_suites/llm/grpo-moonlight-16b-automodel-1n8g-ep8.sh                     # Refit
```

---

## Regression 1 — FP8 `remainder_cuda` NotImplementedError

### Symptom

`VllmGenerationWorker` crashes during weight load:

```
File "vllm/utils/deep_gemm.py", line ..., in should_use_deepgemm_for_fp8_linear
    and weight_shape[0] % N_MULTIPLE == 0
        ~~~~~~~~~~~~~~^~~~~~~~~~~~
NotImplementedError: "remainder_cuda" not implemented for 'Float8_e4m3fn'
```

Selected backend in the driver log:

```
INFO [__init__.py:389] Selected FlashInferFp8DeepGEMMDynamicBlockScaledKernel for Fp8LinearMethod
INFO [deep_gemm.py:116] DeepGEMM E8M0 disabled on current configuration.
```

### Evidence (clean A/B, `nemorl-probes-zhiyul`)

| Job | Commit | vLLM | Result | Wandb |
|---|---|---|---|---|
| 12105330 | `501cd12ce` (pre-bump) | **0.17.1** | ✅ COMPLETED, no `remainder_cuda` | [y07rpz0b](https://wandb.ai/nvidia/nemorl-probes-zhiyul/runs/y07rpz0b) |
| 12105331 | `241ece552` (the bump) | **0.20.0** | ❌ FAIL `remainder_cuda` @ 2:36 | [88x8l1ax](https://wandb.ai/nvidia/nemorl-probes-zhiyul/runs/88x8l1ax) |

### Hypothesis

vLLM 0.20 picks the new `FlashInferFp8DeepGEMMDynamicBlockScaledKernel` and calls `weight_shape[0] % N_MULTIPLE` on a tensor that's already `Float8_e4m3fn`. Torch's `remainder_cuda` op has no FP8 dispatch, so the inner kernel-selection check raises `NotImplementedError`.

---

## Regression 2 — Mamba `max_num_seqs > cache_blocks` ValueError

### Symptom

`VllmGenerationWorker` raises during CUDA-graph capture:

```
ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (658).
Each decode sequence requires one Mamba cache block, so CUDA graph capture cannot proceed.
Please lower max_num_seqs to at most 658 or increase gpu_memory_utilization.
```

### Evidence (clean A/B, `nemorl-probes-zhiyul`)

| Job | Commit | vLLM | Result | Wandb |
|---|---|---|---|---|
| 12105332 | `501cd12ce` (pre-bump) | **0.17.1** | ✅ COMPLETED, no `max_num_seqs` | [oqrv8bmn](https://wandb.ai/nvidia/nemorl-probes-zhiyul/runs/oqrv8bmn) |
| 12105333 | `241ece552` (the bump) | **0.20.0** | ❌ FAIL `max_num_seqs` @ 3:04 | [y6syiic7](https://wandb.ai/nvidia/nemorl-probes-zhiyul/runs/y6syiic7) |

### Hypothesis

vLLM 0.20 tightened Mamba cache-block accounting in `compilation.py:resolve_cudagraph_mode_and_sizes`. Default `max_num_seqs=1024` (which fit comfortably in 0.17.1) now exceeds the computed cache-block budget on a 12B Mamba-Transformer hybrid at `gpu_memory_utilization=0.6` and `max_num_seqs=1024`. Lowering to ≤ 658 (or raising GMU) is the documented vLLM workaround, but the **invariant changed silently** between versions.

---

## Regression 3 — Moonlight automodel EP=8 refit desync *(silent corruption — most dangerous)*

### Symptom

Recipe `grpo-moonlight-16b-automodel-1n8g-ep8` completes its smoke run but every training step shows trainer/generator weight desync:

```
train/gen_kl_error[step1]          = 0.553   (target < 0.001 → 1800× over)
train/token_mult_prob_error[step1] = 6.4e8   (target ≈ 1     → 8 orders of magnitude)
train/reward[step1]                = 0.023   (never improves over 30 steps)
train/grad_norm[step1]             = ~0.12   (PASSES — gradients flow normally)
```

This is *soft* failure — training runs to completion, weights update on the trainer, but the vLLM generator never picks up the new weights. Models that previously converged (PASSing this recipe on May 19) now produce garbage rollouts indefinitely.

### Evidence (clean A/B, `nemorl-probes-zhiyul`)

| Job | Commit | vLLM | gen_kl_error[1] | token_mult_prob_error[1] | reward[1] | Result | Wandb |
|---|---|---|---|---|---|---|---|
| 12105334 | `501cd12ce` (pre-bump) | **0.17.1** | **0.000295** | **1.006** | **0.584** | ✅ PASS | [0mdl5hq8](https://wandb.ai/nvidia/nemorl-probes-zhiyul/runs/0mdl5hq8) |
| 12105335 | `241ece552` (the bump) | **0.20.0** | **0.553** | 642,846,016 | 0.023 | ⚠️ Soft-broken | [i227tcmb](https://wandb.ai/nvidia/nemorl-probes-zhiyul/runs/i227tcmb) |

These commits differ by **exactly one PR**: `chore: Upgrade vLLM from 0.17.1 to 0.20.0 (#2384)`. Same transformers (5.3.0), same mamba-ssm rev, same container, same recipe yaml, same Hydra overrides.

### Hypothesis

vLLM 0.20 changed how weights are loaded / how `update_weights_from_collective` / `update_weights_via_ipc_zmq` apply staged tensors, *or* added a new compile-cache / graph-capture path that holds onto the initial weights past the first refit.

The recipe uses **DTensor + EP=8 + automodel + colocated vLLM**. The refit path (`update_weights_from_collective`) is exercised once per training step. If vLLM 0.20's worker is silently no-oping the broadcast for this sharding shape, the generator stays at iter 0 forever — which is exactly what the metrics show.

---

## Why no automated CI caught this

- GH Actions nightly cron (`CICD NeMo RL → L1_Functional_Tests_GPU`) runs `tests/functional/L1_Functional_Tests_GPU.sh`, which does **not** invoke `tests/test_suites/nightly.txt`. No nightly job in this repo runs any of the 7 affected recipes.
- The moonlight refit regression is the worst kind: training completes, no exception, only the *metrics assertions* in the recipe's closing `check_metrics.py` flag it. Without that block running in CI, the regression is invisible.

Recommend wiring `nightly.txt` (or at least the moonlight + FP8 + Mamba representative recipes) into the scheduled CI matrix so the next vLLM bump catches the same class of bugs at PR time, not at downstream sweep time.

---

## Proposed remediation paths

1. **Pin vLLM at 0.17.1 in `pyproject.toml`** until 0.20.0 issues are fixed upstream or worked around — simplest, but loses 0.20's perf wins.
2. **File / find upstream vLLM issues**:
   - FP8: `remainder_cuda` not registered for `Float8_e4m3fn` in `should_use_deepgemm_for_fp8_linear`.
   - Mamba: max_num_seqs vs cache_blocks invariant tightened without a default-tuning recommendation.
   - Refit: `update_weights_from_collective` not propagating for DTensor + EP MoE in colocated mode.
3. **Workarounds in NeMo-RL**:
   - Cap `max_num_seqs ≤ 658` in nano-v2 yamls (covers regression 2).
   - Guard `should_use_deepgemm_for_fp8_linear` selection in `nemo_rl/models/generation/vllm/vllm_backend.py` (covers regression 1).
   - Add a refit-correctness assertion: after `update_weights_from_collective`, sample a single token and assert `|generator_logprob − trainer_logprob| < ε`. Would catch regression 3 immediately even without a metric-check sweep.

---

## Repro one-liners (against current main + 05242026 container)

```bash
CONTAINER=/lustre/fsw/portfolios/coreai/users/zhiyul/enroot-images/nvcr.io#nvidian/nemo-rl:nightly-05242026.squashfs
HF_NIGHTLY=/lustre/fsw/portfolios/coreai/users/zhiyul/hf_nightly

repro () {
  local recipe=$1 ; local label=$2
  NNODES=1 TIME="1:00:00" CONTAINER="$CONTAINER" JOB_NAME="repro-$label" \
    COMMAND="NRL_FORCE_REBUILD_VENVS=true \
HF_HOME=${HF_NIGHTLY} HF_DATASETS_CACHE=${HF_NIGHTLY}/datasets \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
bash tests/test_suites/llm/${recipe}.sh \
  grpo.max_num_steps=3 \
  logger.wandb_enabled=true \
  logger.wandb.project=nemorl-probes-zhiyul \
  logger.wandb.name=${label}" \
    bash user_run_slurm.sh
}

# FP8 (regression 1):
repro grpo-qwen3-8b-base-1n8g-fp8-kvcache-megatron        repro_fp8

# Mamba (regression 2):
repro grpo-nano-v2-12b-1n8g-megatron                      repro_mamba

# Refit (regression 3 — the soft one):
repro grpo-moonlight-16b-automodel-1n8g-ep8               repro_refit
# After completion, inspect tests/test_suites/llm/grpo-moonlight-16b-automodel-1n8g-ep8/metrics.json:
#   gen_kl_error[step1]          ~0.0003 (PASS) or ~0.55 (broken)
#   token_mult_prob_error[step1] ~1.006  (PASS) or 1e8+  (broken)
```

To reproduce on `501cd12ce` (pre-bump, PASS-expected), create a worktree at that commit (e.g. `~/.claude/scripts/setup_debug_worktree.sh <name>` then `git reset --hard 501cd12ce` inside the worktree), restore the `3rdparty/` symlinks, and run the same three commands from the worktree directory.

---

## Reference jobs (all in `nemorl-probes-zhiyul`)

| Regression | pre-bump (`501cd12ce`, vllm 0.17.1) | post-bump (`241ece552`, vllm 0.20.0) |
|---|---|---|
| FP8 | 12105330 — [y07rpz0b](https://wandb.ai/nvidia/nemorl-probes-zhiyul/runs/y07rpz0b) | 12105331 — [88x8l1ax](https://wandb.ai/nvidia/nemorl-probes-zhiyul/runs/88x8l1ax) |
| Mamba | 12105332 — [oqrv8bmn](https://wandb.ai/nvidia/nemorl-probes-zhiyul/runs/oqrv8bmn) | 12105333 — [y6syiic7](https://wandb.ai/nvidia/nemorl-probes-zhiyul/runs/y6syiic7) |
| Refit | 12105334 — [0mdl5hq8](https://wandb.ai/nvidia/nemorl-probes-zhiyul/runs/0mdl5hq8) | 12105335 — [i227tcmb](https://wandb.ai/nvidia/nemorl-probes-zhiyul/runs/i227tcmb) |
