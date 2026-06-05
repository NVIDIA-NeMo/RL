# Reproducing the baseline SWE2 Async-GRPO run

Step-by-step guide to reproduce baseline's successful SWE2 GRPO run
(wandb `nvidia/binhu-nemo-rl/dc3m70us`) using:

- **Cluster:** `cw-dfw-cs`
- **Branch:** `ruit/SWE_bench` (repo `github.com/NVIDIA-NeMo/RL`)
- **Launcher:** `${REPO_ROOT}/examples/swe_bench/run_grpo_repro_baseline_swe2.sh`
- **Config:** `${REPO_ROOT}/examples/swe_bench/grpo_qwen3_30b_async_swe.yaml` (passed to the launcher via `--config`)

The goal of this run is to confirm that the earlier *zero-reward* failure was
caused by the **container / vLLM** (a broken hermes tool parser that prevented
the agent from emitting real tool calls), **not** by the model or the config.
A correct repro resolves ~8% of SWE-bench instances starting from step 1.

> Run this on **`cw-dfw-cs`**. **Do not run from anyone else's checkout** — clone
> the repo into your own workspace (§2.1). `REPO_ROOT` below means *your* clone;
> the launcher auto-detects it from its own location. The model / data / container
> paths are absolute and world-readable on the `cw-dfw-cs` Lustre.

---

## 1. What this run is

| Item | Value |
|------|-------|
| Algorithm | Async GRPO (non-colocated generation) |
| Model | Qwen3-30B-A3B-Thinking-2507 (MoE, 30B total / 3B active) |
| Init checkpoint | SWE1 `step_230_hf` (the exact checkpoint dc3m70us trained from) |
| Train data | R2E-Gym subset (`swe_all_datasets_train_w_agent_ref_r2e_gym_subset.jsonl`) |
| Eval data | same JSONL (val == train path here) |
| Env | `swe_agents` (OpenHands agent inside an apptainer/singularity sandbox) |
| Entry point | `${REPO_ROOT}/examples/nemo_gym/run_grpo_nemo_gym.py` |
| Scheduler | SLURM (`sbatch` + `ray.sub`) |

---

## 2. Prerequisites

### 2.1 Get the code (clone into your own workspace)

On `cw-dfw-cs`, clone the repo into a directory you own and check out the
`ruit/SWE_bench` branch. Do **not** run from someone else's checkout.

```bash
cd /lustre/<your-own-workspace-on-cw-dfw-cs>
git clone https://github.com/NVIDIA-NeMo/RL.git
cd RL
git checkout ruit/SWE_bench
git submodule update --init --recursive   # needed for the Gym mount (3rdparty/Gym-workspace/Gym)

export REPO_ROOT="$PWD"                     # = your clone; the launcher also auto-detects this
```

> The launcher runs the code **in place** from your clone (`SNAPSHOT_DIR ==
> REPO_ROOT`). Whatever is in the working tree at submit time is what runs, so
> avoid stray local edits.

### 2.2 Container

Uses the **SWE training container** (`ruit-swe_bench`, with mcore + apptainer baked
in), NOT the default NeMo-RL image. Its vLLM has the working hermes tool parser, so
the agent emits real `function_call` items (a broken parser was the original
zero-reward failure mode):

```
/lustre/fsw/portfolios/coreai/users/ruit/enroot-images/docker_images:ruit-swe_bench-6de99f772-x86_64-060326-mcore-apptainer.squashfs
```

It is wired in via the `CONTAINER` env var (overridable). The job mounts:

```
/lustre  ->  /lustre
${REPO_ROOT}  ->  (same path; $PWD)
${REPO_ROOT}/3rdparty/Gym-workspace/Gym  ->  /opt/nemo-rl/3rdparty/Gym-workspace/Gym
```

The last mount overlays the in-repo Gym source over the container's Gym so
your Gym checkout is what runs.

### 2.3 Required files on Lustre

Confirm these absolute paths exist before submitting:

| Path | Purpose |
|------|---------|
| `/lustre/fsw/portfolios/coreai/users/bihu/repos/nemo-rl-async-swe/results/qwen3-30b-thinking-swe1-async-age1-pps64-gpp8-gbs512-lr1e-06/step_230_hf` | init checkpoint |
| `/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sdevare/repos/nano/dataset/rl/swe_all_datasets_train_w_agent_ref_r2e_gym_subset.jsonl` | train + val data |
| `${REPO_ROOT}/ray.sub` | SLURM launcher consumed by `sbatch` |
| `/lustre/fsw/portfolios/coreai/users/ruit/enroot-images/docker_images:ruit-swe_bench-6de99f772-x86_64-060326-mcore-apptainer.squashfs` | training container |

Per-instance SWE-bench `.sif` sandbox images (resolved by `container_formatter`
in the YAML, first match wins):

```
/lustre/fsw/portfolios/llmservice/users/igitman/images/swe-bench/swebench_sweb.eval.x86_64.{instance_id}.sif
/lustre/fsw/portfolios/llmservice/users/sdevare/swe_sweapro/images_train/sweap.{instance_id}.sif
/lustre/fsw/portfolios/llmservice/users/igitman/images/swe-bench/namanjain12_{instance_id}.sif
/lustre/fsw/portfolios/llmservice/users/igitman/images/swe-bench/swebench_sweb.eval.x86_64{instance_id}.sif
```

### 2.4 Tokens / credentials

Credentials were **stripped from this shared copy** of the launcher — it no
longer sources any env script. Before submitting, export these yourself:

- `HF_HOME` — HuggingFace cache root (passed through to the job; also used to
  derive `HF_DATASETS_CACHE`)
- `HF_TOKEN` — required if the model/tokenizer is gated
- `WANDB_API_KEY` — required for wandb logging (`logger.wandb_enabled=True`)
- `GITHUB_TOKEN` — only if your data/repo access needs it

### 2.5 Caches (created automatically, listed for reference)

The launcher seeds vLLM/inductor/triton caches from a persistent dir under your
own `$HOME` (override with the `PERSISTENT_CACHE` env var):

```
Persistent (default ${HOME}/.cache/qwen3_30b_thinking_swe_repro_baseline):
  .../vllm_compile_cache
  .../inductor_cache
  .../triton_cache

Node-local (/tmp, recreated each run):
  /tmp/nemo_rl_vllm_cache
  /tmp/nemo_rl_inductor_cache
  /tmp/nemo_rl_triton_cache
  /tmp/uv_cache
```

---

## 3. Key configuration (what gets reproduced)

The launcher overrides the YAML on the command line. The values that define
the run:

**Cluster / parallelism**
- `NUM_NODES=16` actor nodes, `8` generation nodes (async, non-colocated), 8 GPUs/node
- `TP=4`, `EP=8`, `CP=4`, `PP=2`, `vLLM_TP=2`
- `make_sequence_length_divisible_by = 32` (auto: `CP*2*TP = 4*2*4`)

**GRPO / sampling**
- `num_prompts_per_step=8`, `num_generations_per_prompt=8`, `train_global_batch_size=64`
- `normalize_rewards=True`, `overlong_filtering=True`
- Async: `max_trajectory_age_steps=1`, `in_flight_weight_updates=True`,
  `recompute_kv_cache_after_weight_updates=False`, `force_on_policy_ratio=True`
- `advantage_clip=[-100, 100]`, `truncated_importance_sampling_ratio=5`

**Loss**
- `reference_policy_kl_penalty=0` (no KL), `ratio_clip=[0.2, 0.28]`
- `token_level_loss=True`, `use_importance_sampling_correction=True`,
  `sequence_level_importance_ratios=False`

**Optimizer / model**
- `lr=1e-06` (constant), `weight_decay=0`
- `max_total_sequence_length=131072`, sequence packing on
- MoE: router frozen, `moe_aux_loss_coeff=0`, `alltoall` dispatcher, deepep off

**SWE agent**
- `agent_max_turns=200`, `swebench_agent_timeout=1800`

**Logging**
- wandb project `swe-benchmark`, full Gym responses logged
  (`should_log_nemo_gym_responses=true`) so you can verify `function_call`
  items actually appear.

---

## 4. Command flavor (why it differs from the default)

The training command is **baseline-style**, which is what makes the container work:

- `uv run --frozen --extra mcore` (frozen lockfile)
- `NRL_IGNORE_VERSION_MISMATCH=1` — tolerate the container's vLLM version
- `NEMO_GYM_SKIP_VENV_IF_PRESENT=1` — reuse the container's Gym venv, don't rebuild
- `NRL_FORCE_REBUILD_VENVS=false`, `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`
- vLLM caches seeded from a persistent Lustre cache, then synced back

The `SETUP_COMMAND` (run once per node before training) installs
apptainer/singularity (for the SWE sandbox), clears + seeds the inductor/triton
caches from Lustre, and runs `uv sync --frozen --extra mcore`.

---

## 5. Step-by-step

```bash
# 1. Clone into your own workspace on cw-dfw-cs and check out the branch
cd /lustre/<your-own-workspace-on-cw-dfw-cs>
git clone https://github.com/NVIDIA-NeMo/RL.git
cd RL
git checkout ruit/SWE_bench
git submodule update --init --recursive   # Gym mount (3rdparty/Gym-workspace/Gym)
export REPO_ROOT="$PWD"

# 2. Export your credentials (stripped from this copy — see §2.4)
export HF_HOME=/your/hf/home
export HF_TOKEN=...          # if the model/tokenizer is gated
export WANDB_API_KEY=...     # for wandb logging
export GITHUB_TOKEN=...      # only if your data/repo access needs it

# 3. (Optional) sanity-check the shared assets exist (readable on cw-dfw-cs)
ls "/lustre/fsw/portfolios/coreai/users/ruit/enroot-images/docker_images:ruit-swe_bench-6de99f772-x86_64-060326-mcore-apptainer.squashfs"
ls -d "/lustre/fsw/portfolios/coreai/users/bihu/repos/nemo-rl-async-swe/results/qwen3-30b-thinking-swe1-async-age1-pps64-gpp8-gbs512-lr1e-06/step_230_hf"
ls "/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sdevare/repos/nano/dataset/rl/swe_all_datasets_train_w_agent_ref_r2e_gym_subset.jsonl"

# 4. Submit from your clone. Defaults reproduce dc3m70us; no other args needed.
bash "${REPO_ROOT}/examples/swe_bench/run_grpo_repro_baseline_swe2.sh"
```

The script prints a summary, submits via `sbatch`, and writes the job id to
`${REPO_ROOT}/latest_repro_baseline_job_id.txt`.

### Overridable knobs (env vars)

| Var | Default | Effect |
|-----|---------|--------|
| `MODEL_PATH` (also `$1`) | `/lustre/fsw/portfolios/coreai/users/bihu/repos/nemo-rl-async-swe/results/qwen3-30b-thinking-swe1-async-age1-pps64-gpp8-gbs512-lr1e-06/step_230_hf` | init checkpoint |
| `CONTAINER` | `/lustre/fsw/portfolios/coreai/users/ruit/enroot-images/docker_images:ruit-swe_bench-6de99f772-x86_64-060326-mcore-apptainer.squashfs` | training image |
| `NUM_NODES` | 16 | actor nodes |
| `NUM_GEN_NODES` | 8 | generation nodes (async only) |
| `SKIP_TRAINING` | `0` | `1` = generation-only benchmark: no-op training pinned to 1 node (see §9) |
| `EXP_SUFFIX` | `repro-baseline-swe2-async-age1-pps8-gpp8-gbs64-lr1e-06-tp4` | run + checkpoint dir name (`notrain-` is inserted when `SKIP_TRAINING=1`) |
| `BASE_LOG_DIR` | `${REPO_ROOT}/logs/slurm` | SLURM/Ray logs |

Example — different init checkpoint, smaller cluster:

```bash
NUM_NODES=8 NUM_GEN_NODES=4 \
  bash ${REPO_ROOT}/examples/swe_bench/run_grpo_repro_baseline_swe2.sh \
  /path/to/other/step_X_hf
```

---

## 6. Monitoring

```bash
JOB_ID=$(cat ${REPO_ROOT}/latest_repro_baseline_job_id.txt)

squeue -j "$JOB_ID"                          # queue state
ls ${REPO_ROOT}/logs/slurm/${JOB_ID}-logs/           # Ray + SLURM logs
tail -f ${REPO_ROOT}/logs/slurm/slurm-${JOB_ID}.out  # driver output
```

- **wandb:** project `swe-benchmark`, run name = `EXP_SUFFIX`.
- **Checkpoints:**
  `${REPO_ROOT}/results/${EXP_SUFFIX}/`
  (save every 5 steps, keep top 2 by `train:total_reward/mean`).

### What "success" looks like
- `train:total_reward/mean` is **non-zero from step ~1** (the failure mode was
  identically zero reward).
- Logged Gym responses contain real `function_call` items (proves the hermes
  tool parser is working in this container).
- Resolved rate climbs toward ~8%.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Reward is identically 0 | wrong container — hermes tool parser broken, no tool calls | confirm `CONTAINER` is the `ruit-swe_bench` squashfs, not the default image |
| `version mismatch` abort | strict version check | ensure `NRL_IGNORE_VERSION_MISMATCH=1` is in the command (it is, by default) |
| Gym venv rebuild / slowness | venv rebuilt instead of reused | confirm `NEMO_GYM_SKIP_VENV_IF_PRESENT=1` and the Gym mount are present |
| Agent can't start sandbox | apptainer/singularity missing or `.sif` images missing | check `SETUP_COMMAND` apptainer install succeeded; verify `container_formatter` paths in the YAML |
| Token / auth errors | credentials not exported (this copy ships none) | `export HF_HOME`/`HF_TOKEN`/`WANDB_API_KEY`/`GITHUB_TOKEN` before submitting (see §2.4) |
| OOM / parallelism mismatch | changed `TP`/`EP`/`CP`/`PP` without re-deriving `make_sequence_length_divisible_by` | keep the default parallelism, or recompute `MIN_PAD = CP*2 * TP` |

---

## 8. Reference — exact pinned values

```
Code:        NeMo-RL @ branch ruit/SWE_bench (run in place from your clone)
Compute:     cw-dfw-cs (SLURM)
Repo:        github.com/NVIDIA-NeMo/RL  @  branch ruit/SWE_bench
REPO_ROOT:   your clone (export REPO_ROOT=<clone>; launcher also auto-detects it)
Container:   /lustre/fsw/portfolios/coreai/users/ruit/enroot-images/docker_images:ruit-swe_bench-6de99f772-x86_64-060326-mcore-apptainer.squashfs
Init model:  /lustre/fsw/portfolios/coreai/users/bihu/repos/nemo-rl-async-swe/results/qwen3-30b-thinking-swe1-async-age1-pps64-gpp8-gbs512-lr1e-06/step_230_hf
Train data:  /lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sdevare/repos/nano/dataset/rl/swe_all_datasets_train_w_agent_ref_r2e_gym_subset.jsonl
Config:      ${REPO_ROOT}/examples/swe_bench/grpo_qwen3_30b_async_swe.yaml
Launcher:    ${REPO_ROOT}/examples/swe_bench/run_grpo_repro_baseline_swe2.sh
Mode:        async-age1, colocated=False
Resources:   16 actor nodes + 8 gen nodes, 8 GPU/node
Parallelism: TP=4, EP=8, CP=4, PP=2, vLLM_TP=2, pad=32
Training:    PPS=8, GPP=8, GBS=64, LR=1e-06
Loss:        KL=0, clip=[0.2,0.28], token-level, IS correction on, TIS=5
Agent:       max_turns=200, timeout=1800s
wandb:       project=swe-benchmark
Baseline:    nvidia/binhu-nemo-rl/dc3m70us (~8% resolved from step 1)
```

---

## 9. Generation-only benchmark (skip training)

For **benchmarking generation throughput / scaling** without paying for real
training, the launcher has a no-op-training mode, gated by the
`grpo.gen_benchmark_skip_training` flag (added on `ruit/SWE_bench`). Set
`SKIP_TRAINING=1`:

```bash
SKIP_TRAINING=1 bash "${REPO_ROOT}/examples/swe_bench/run_grpo_repro_baseline_swe2.sh"
```

### What it does
- **`policy.train()` becomes a no-op** — no forward/backward, no optimizer step. The
  weights stay frozen at the init checkpoint and are **still refit to vLLM every
  step**, so the async generation / weight-sync cadence stays realistic.
- **No optimizer is built** (`init_optimizer=False`) — saves memory and startup time.
- A tiny **keep-alive matmul daemon** runs on each training worker so the cluster's
  idle-GPU reaper doesn't kill the (otherwise idle) training node.
- **Checkpoint saving is disabled** (`checkpointing.enabled=false`) — there is no
  optimizer/training state to save.

### What the launcher changes automatically when `SKIP_TRAINING=1`
- Training parallelism → **`TP=8, EP=8, CP=1, PP=1`** (model-parallel = 8, fits one
  node; `train_DP=1`), so training is pinned to a **single node**.
- `NUM_ACTOR_NODES = NUM_GEN_NODES + 1` → total nodes = `gen + 1` (default `8 + 1 = 9`;
  8 generation nodes = 32 vLLM replicas at `vLLM_TP=2`).
- Appends `++grpo.gen_benchmark_skip_training=true checkpointing.enabled=false`.
- `EXP_SUFFIX` gets a `notrain-` tag.

Everything else (model, data, `PPS=8/GPP=8/GBS=64`, agent settings, container) is
unchanged, so the per-replica generation workload (`samples/replica = GBS / replicas
= 64 / 32 = 2`) matches the full run.

### How to verify the scaling is sound (wandb)
Compare runs at different generation sizes (vary `NUM_GEN_NODES`) within one wandb
group. The **per-replica** `generation_metrics/*` timelines should stay **flat**
(invariant) as you add replicas — not grow with scale:

| metric | expectation across scale |
|--------|--------------------------|
| `generation_metrics/*inflight_batch_sizes` | flat, low (≈1–3 per replica) |
| `generation_metrics/*num_pending_samples` | ≈ 0 (no queue backlog) |
| `generation_metrics/*kv_cache_usage_perc` | flat (≈8–10%) |
| `generation_metrics/*generation_tokens` | flat per replica per window |
| worker-trace count | equals the replica count (`gen_gpus / vLLM_TP`) |

> Note: SWE rollouts are **agent / tool-execution-bound** (each sample is a multi-turn
> OpenHands rollout in an apptainer sandbox), so per-replica inflight/KV stay low and
> total throughput scales sub-linearly with GPUs — that is expected, not a regression.
> Weights are frozen, so reward hovers around the init checkpoint's baseline (noisy on
> small per-step sample counts); this mode is for **throughput/scaling**, not learning.
