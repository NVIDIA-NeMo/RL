# Reproducing SWE2 Async-GRPO on **SGLang** (Qwen3-30B-A3B-Thinking)

Self-contained guide to run the multi-turn SWE-bench agentic GRPO recipe with the
**SGLang** generation backend, at parity with the vLLM baseline (rollout completeness +
throughput) and at **training-grade per-token logprob parity** with vLLM.

This is the SGLang counterpart of [`REPRO_swe2.md`](./REPRO_swe2.md) (the vLLM baseline).
Everything needed lives in **one clone** — you do **not** need to fetch any other PR.

---

## 0. TL;DR

```bash
# One clone has everything (RL + splice-fixed Gym + grafted SGLang backend).
git clone --recurse-submodules -b swe2-qwen-sglang git@github.com:Kh4L/NemoRL.git
cd NemoRL
export REPO_ROOT="$PWD"

# Credentials (not shipped) — see §3.
export HF_HOME=/your/hf/home HF_TOKEN=... WANDB_API_KEY=...

# 2-node smoke / parity (generation-only, no training):
SKIP_TRAINING=1 NUM_VLLM_REPLICAS=4 BACKEND=sglang \
  bash examples/swe_bench/run_grpo_swe2_scale_gen.sh

# Full convergence run on SGLang (16 nodes = 8 train + 8 gen, reproduces the baseline shape):
NUM_VLLM_REPLICAS=32 BACKEND=sglang \
  bash examples/swe_bench/run_grpo_swe2_scale_gen.sh
```

Expected: multi-turn rollouts complete (**8/8, contiguity failures 0**), **~193 gen tok/s**
with full CUDA graph (≈ vLLM), and SGLang↔vLLM per-token logprobs agree to within the model's
own bf16/MoE numerical noise floor (see §6).

---

## 1. What this is

| Item | Value |
|------|-------|
| Algorithm | Async GRPO (non-colocated generation) |
| Model | Qwen3-30B-A3B-Thinking-2507 (MoE, 30B total / ~3B active) |
| Generation backend | **SGLang** (vLLM is the baseline; this recipe swaps it for SGLang) |
| Init checkpoint | SWE1 `step_230_hf` (same as the vLLM baseline `dc3m70us`) |
| Env | `swe_agents` (OpenHands agent inside an apptainer/singularity sandbox) |
| Launcher | `examples/swe_bench/run_grpo_swe2_scale_gen.sh` (`BACKEND=sglang`) |
| Scheduler | SLURM (`sbatch` + `ray.sub`), enroot+pyxis container runtime |

The goal: make **SGLang** a usable generation backend for this multi-turn SWE-bench recipe,
proven equivalent to vLLM at (a) rollout completeness, (b) throughput, and (c) per-token
logprobs (which feed GRPO importance ratios).

---

## 2. Provenance — what's grafted vs. what's new

This branch set is a **graft**, assembled so a single clone is runnable end-to-end:

- **Base:** NeMo-RL `main` (already carries the *basic* SGLang backend).
- **Grafted from [NVIDIA-NeMo/RL#2447](https://github.com/NVIDIA-NeMo/RL/pull/2447)
  (`zhw/mxfp8_support`):** the *enhanced* SGLang backend our 30B-MoE recipe needs —
  Megatron→SGLang weight-refit (`megatron_sglang_weight_iterator.py`), non-colocated
  weight update, router, fault-tolerance, and the heavier `sglang_worker` / `sglang_generation`.
  #2447 is an open, evolving PR; this branch **pins a known-good graft of it** so you don't
  have to track #2447 yourself.
- **New in this branch set (the genuinely novel work):**
  1. **★ Gym-proxy token-splicing contiguity fix** (the load-bearing piece, in the Gym fork
     `responses_api_models/vllm_model/app.py`). Multi-turn SGLang rollouts broke a hard
     prefix-stability assert in `nemo_gym.py` on ~every tool-using turn (48/48 failures).
     Fix: build each turn's prompt as `prompt_{K-1} + gen_{K-1}(verbatim) + delta_K`,
     splicing the prior assistant's **exact sampled token IDs** instead of re-tokenizing
     (`_build_sglang_prompt_ids`, `_update_sglang_session_seq`, `_sglang_followup_fragment_ids`).
     Also: SGLang native `/generate` with `return_logprob=True`, and `skip_special_tokens=False`
     so `</think>` (id 151668) survives the multi-turn re-feed.
  2. **SWE2 SGLang launcher path** — `BACKEND=sglang` in `run_grpo_swe2_scale_gen.sh`.
  3. **CUDA-graph perf** — full CUDA graph ON (piecewise off; it crashes on torch-2.10/sglang)
     → **51 → ~193 tok/s, ≈ vLLM**.
  4. **Refit OOM / NCCL-deadlock mitigations** — `mem_fraction_static=0.55`,
     `NRL_REFIT_BUFFER_MEMORY_RATIO=0.018`, `pause_generation_mode=retract`.
- **Parity instrumentation** (sentinel-gated, harmless when off): in-proxy + in-worker
  teacher-force hooks used to *prove* logprob parity (§6). Not needed for training.

Pinned SHAs: **RL `c88030f`**, **Gym `50586ec`** (auto-resolved as the submodule).

---

## 3. Prerequisites

### 3.1 Cluster / runtime
A SLURM cluster with **enroot + pyxis** (so `ray.sub` runs `srun --container-image` natively).
Validated on CW-DFW (`cw-dfw-cs-001`, H100). 2 nodes for the smoke/parity run; 16 nodes for
the full convergence run.

### 3.2 Container (SWE training image — has the working hermes tool parser)
```
/lustre/fsw/portfolios/coreai/users/ruit/enroot-images/docker_images:ruit-swe_bench-6de99f772-x86_64-060326-mcore-apptainer.squashfs
```
Wired via `CONTAINER` (overridable). It bakes mcore + apptainer; the launcher overlays your
clone's `nemo_rl/` and `3rdparty/Gym-workspace/Gym` over the baked copies, so **your checkout
is what runs**.

### 3.3 Shared assets (absolute, world-readable on CW-DFW Lustre)
| Path | Purpose |
|------|---------|
| `…/bihu/repos/nemo-rl-async-swe/results/qwen3-30b-thinking-swe1-async-age1-…/step_230_hf` | init checkpoint |
| `…/sdevare/repos/nano/dataset/rl/swe_all_datasets_train_w_agent_ref_r2e_gym_subset.jsonl` | train + val data |
| `…/spanev/swe2-repro/qwen3_swe_chat_template.jinja` | SGLang chat template (Qwen3 thinking) |
| per-instance `swebench_sweb.eval.x86_64.{instance_id}.sif` | SWE-bench sandbox images (resolved by `container_formatter` in the YAML) |

The exact default paths are in the launcher (`MODEL_PATH`, `TRAIN_DATA_PATH`, `SGLANG_CHAT_TEMPLATE`);
override via env if your copies live elsewhere.

### 3.4 Credentials (export yourself — not shipped)
`HF_HOME`, `HF_TOKEN` (gated model), `WANDB_API_KEY` (or `WANDB_MODE=offline`),
optionally `GITHUB_TOKEN` / `GITLAB_TOKEN`.

---

## 4. How to run

The single launcher is **`examples/swe_bench/run_grpo_swe2_scale_gen.sh`**. One knob,
`NUM_VLLM_REPLICAS` (R), derives nodes / batch sizes so per-replica work is constant.

| Mode | Command | Footprint |
|------|---------|-----------|
| **Smoke / parity** (gen-only, no train) | `SKIP_TRAINING=1 NUM_VLLM_REPLICAS=4 BACKEND=sglang bash …/run_grpo_swe2_scale_gen.sh` | 2 nodes (1 gen + 1 train no-op) |
| **Full convergence** (reproduces baseline shape) | `NUM_VLLM_REPLICAS=32 BACKEND=sglang bash …/run_grpo_swe2_scale_gen.sh` | 16 nodes (8 train + 8 gen) |
| **Preview only** | add `DRY_RUN=1` | none (prints derived config) |

Job id is written to `${REPO_ROOT}/latest_scale_gen_job_id.txt`. Logs under
`${REPO_ROOT}/logs/swe_bench_scale/`. wandb project `swe-benchmark`.

### SGLang-specific env toggles
| Var | Default | Effect |
|-----|---------|--------|
| `BACKEND` | `vllm` | set `sglang` to use the SGLang path |
| `SGLANG_DISABLE_CUDA_GRAPH` | `false` | `true` disables full CUDA graph (slower; ~51 tok/s) |
| `SGLANG_CHAT_TEMPLATE` | `…/qwen3_swe_chat_template.jinja` | Qwen3-thinking chat template |
| `TEMPERATURE` | `1.0` | sampling temperature (recipe trains at 1.0) |
| `UV_CACHE_DIR` | `/tmp/uv_cache` | set to **`/root/.cache/uv`** to reuse the container's prebuilt SGLang wheels and skip a ~40-min build |
| `SBATCH_ACCOUNT` / `SBATCH_PARTITION` | `nemotron_agents_dev` / `backfill` | SLURM account / partition |

### What `BACKEND=sglang` injects (the generation overrides)
bf16; `dp=ep=pp=1` with TP via `num_gpus_per_engine=8`; `mem_fraction_static=0.55`;
`disable_piecewise_cuda_graph=true` + `disable_cuda_graph=${SGLANG_DISABLE_CUDA_GRAPH}`;
`tool_call_parser=hermes` + `reasoning_parser=qwen3-thinking`; the chat template;
`pause_generation_mode=retract`; router disabled; and the Gym proxy switched to the SGLang
engine path (`…vllm_model.engine=sglang`).

---

## 5. Expected results (port parity)

| Run | CUDA graph | gen tok/s | rollout | contiguity_fail |
|---|---|---|---|---|
| SGLang | OFF | ~51 | 30:29 | **0** (8/8) |
| **SGLang** | **ON** (default) | **~193** | **13:34** | **0** (8/8) |
| vLLM baseline | default | — | 10–16 min | 0 |

Success markers (same as the vLLM baseline): non-zero `train:total_reward/mean` from step ~1,
logged Gym responses contain real `function_call` items, resolved rate climbs toward ~8%.

---

## 6. (Optional) Reproduce the SGLang↔vLLM logprob parity

The parity hooks are **committed and sentinel-gated** (no effect unless triggered), so a clean
clone can regenerate the parity numbers. They teacher-force both engines on identical token IDs
and compare per-token logprobs. Shared dir + scripts live under
`/lustre/…/spanev/swe2-repro/parity/` (capture `rollouts.jsonl` + `compare_forced.py`).

```bash
P=/lustre/.../swe2-repro/parity        # capture + scripts + sentinels
# Input: 52 recs / 27,493 tokens, derived deterministically from the 864-record capture
#   (filter gen>0 & prompt+gen<=24k, shortest-first) -> rollouts_filtered16.jsonl

# SGLang side: in-proxy /generate teacher-force
touch $P/SGLANG_TF_TRIGGER
SKIP_TRAINING=1 NUM_VLLM_REPLICAS=4 BACKEND=sglang MAX_NUM_STEPS=1 \
  bash examples/swe_bench/run_grpo_swe2_scale_gen.sh     # -> forced_sglang.jsonl, SGLANG_TF_DONE

# vLLM side: in-worker engine teacher-force (fires post-refit on weight update)
touch $P/VLLM_ENGINE_TF_TRIGGER
SKIP_TRAINING=1 NUM_VLLM_REPLICAS=4 BACKEND=vllm MAX_NUM_STEPS=1 \
  bash examples/swe_bench/run_grpo_swe2_scale_gen.sh     # -> forced_vllm.jsonl, VLLM_ENGINE_TF_DONE

python3 $P/compare_forced.py --sglang $P/forced_sglang.jsonl --vllm $P/forced_vllm.jsonl
```

**Result (teacher-forced, all 27,493 tokens, real post-refit weights):**

| metric | value | read |
|---|---|---|
| median \|Δ logprob\| | **1.38e-3** | training-grade at the typical token |
| p95 / p99 / max | 0.140 / 0.245 / 1.01 | real tail at high-entropy tokens |
| top-K KL median | 1.75e-4 | negligible |
| confident-token bucket [0,0.3) nats median | **3.6e-7** | engines essentially identical where confident |
| within-engine baseline (SGLang sampled-vs-forced) median | **1.24e-3** | the model's own bf16/MoE noise floor |

**Verdict:** cross-engine median (1.38e-3) ≈ within-engine noise floor (1.24e-3) — **vLLM differs
from SGLang no more than SGLang differs from itself.** Swapping vLLM→SGLang is safe at the
logprob level. (Re-validated from a fresh clone on 2026-06-25; numbers match to within run-to-run
bf16/MoE noise.)

Notes: vLLM's recipe server is chat-only (no `/v1/completions`), so its teacher-force runs
**in-process inside the worker** (`vllm_worker_async.py`), fired **post-refit** (the engine boots
with dummy weights and gets the real checkpoint via refit). SGLang's `/generate` supports
`logprob_start_len`, so its side runs in the Gym proxy.

---

## 7. Gotchas (already handled, listed so you don't re-hit them)

- **Multi-turn contiguity** — solved by the token-splice fix; do not "fix" it by re-tokenizing.
- **CUDA graph** — full graph works and is ~2× faster; only *piecewise* crashes (kept off).
- **Refit OOM / NCCL hang** — needs `mem_fraction_static=0.55` + refit bucket cap +
  `pause_generation_mode=retract` (all baked into the launcher).
- **Slow first run** — set `UV_CACHE_DIR=/root/.cache/uv` to reuse baked SGLang wheels.
- **Recipe-managed engines are unreachable from the host** (pyxis container netns) — interact
  only via the Gym proxy or in-worker hooks (which is what the parity instrumentation does).

---

## 8. Where things live
| Thing | Location |
|---|---|
| This recipe (RL + launcher + parity hooks) | `Kh4L/NemoRL@swe2-qwen-sglang` (`c88030f`) |
| Splice-fixed + parity-hooked Gym | `Kh4L/NemoGym@swe2-sglang-graft` (`50586ec`, submodule) |
| vLLM baseline guide | [`REPRO_swe2.md`](./REPRO_swe2.md) |
| Parity capture + scripts + results | DFW `/lustre/.../spanev/swe2-repro/parity/` |
| Upstream home of the enhanced SGLang backend | [NVIDIA-NeMo/RL#2447](https://github.com/NVIDIA-NeMo/RL/pull/2447) |
