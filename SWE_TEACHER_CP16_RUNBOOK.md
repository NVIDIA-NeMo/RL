# Run: Nemotron-3-Ultra 550B SWE-Teacher GRPO (cp16, 48× GB200)

Clean runbook for the `swe_teacher_cp16` async-GRPO run on OCI-HSG (GB200, InfiniBand).
Recipe: `examples/configs/ultra/swe_teacher_cp16.yaml` · Launcher: `swe_teacher_cp16_launch.sh`.
(For root-cause / debugging history, see `SWE_TEACHER_CP16_REPRO.md`.)

## TL;DR — run it

```bash
cd <repo-root>            # the RL checkout on lustre
# creds sourced on the LOGIN node only, never inside the compute container:
source /lustre/fs1/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/zhiyul/secrets.sh > >(grep -v HF_TOKEN) 2>&1
bash swe_teacher_cp16_launch.sh          # submits a 48-node job via ultra_launch.sh -> ray.sub
```

Then **immediately register the GPU-idle exemption** (below) or the reaper kills the job during the
~30-min 550B load.

## Working config (committed defaults — use as-is)

`swe_teacher_cp16.yaml`: `max_total_sequence_length: 65536`, **`train_global_batch_size: 32`**
(`num_prompts_per_step: 8` × `num_generations_per_prompt: 4`), `mtp_num_layers: 5`
(`mtp_loss_scaling_factor: 0.3`), CP16 / EP32 / TP8 / PP1, 48 nodes (32 train + 16 gen),
`SEGMENT_SIZE=8`. This config trains from scratch, CP-clean.

> **Do not raise `train_global_batch_size` above 32** — GBS=128 triggers a `CONTEXT_PARALLEL_GROUP`
> `all_to_all` hang (600s watchdog). 65k or 190k seq length both work at GBS ≤ 32.

## Cluster / launch facts

- **Account** `nemotron_sw_post` · **partition** `batch` · **QOS** `short` (2× priority, MaxWall 2h) · **walltime** `1:59:00`.
- **48 nodes** = 32 train + 16 generation (`NUM_TRAIN_NODES=32`, `NUM_GEN_NODES=16`), 4 GPU/node.
- **Container:** `$Z/enroot-images/nvcr.io+nvidian+nemo-rl+nightly.2026-07-13.squashfs`
  where `Z=/lustre/fsw/portfolios/llmservice/users/zhiyul`. The mcore worker venv builds at runtime via
  `uv run --extra mcore` (~5-min transformer-engine compile).
- Model / data / SIFs are all under `$Z` (see the launcher's `MODEL_PATH`, `TRAIN_PATH`, `swe_sifs`).

## GPU-idle reaper exemption (REQUIRED)

The 550B dist-ckpt load sits at SM≈0 for ~30 min; the auto-reaper cancels idle jobs after 30 min. Set
the exemption **while PENDING or right after RUNNING**:

```bash
JOB=<jobid>
scontrol update jobid=$JOB Comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"benchmarking","description":"550B dist-ckpt load ~30min SM~0; SWE rollouts idle GPUs"}}'
squeue -j $JOB -o %k    # verify the JSON is set
```

## Required settings (keep these — the recipe depends on them)

| Setting | Value |
|---|---|
| MoE backend (vLLM) | `moe_backend: triton` |
| Expert parallel | `expert_model_parallel_size: 32` |
| Seq-len divisor | `make_sequence_length_divisible_by: 256` |
| MTP head | `mtp_num_layers: 5`, `mtp_loss_scaling_factor: 0.3` |
| Batch size | `train_global_batch_size: 32` (do not exceed) |
| SWE concurrency | `swe_agents concurrency: 64` |
| Topology | `cluster.segment_size: 8` + `SEGMENT_SIZE=8` (`--segment`), one EP group per NVLink rack |
| NCCL transport | `ray.sub` head+worker env: `NCCL_NET=IB`, `NCCL_NVLS_ENABLE=1`, `NCCL_GIN_*`, `HYBRID_EP_CACHE_DIR` |

`checkpointing.save_period=1` (launcher) writes a 550B checkpoint **every step** — drop it for anything
but short runs.

## Running as a different user (within HSG)

Read paths (model, data, container, SIFs) are world-readable under `.../users/zhiyul` — no copy needed.
Write targets are scoped per-user via `WRITE_ROOT` (default `/lustre/fsw/portfolios/llmservice/users/$USER`:
HF cache, uv caches, per-agent venvs, results/checkpoints/logs). A teammate needs to:
- have Slurm account `nemotron_sw_post` (or override `SLURM_ACCOUNT`);
- optionally `export WRITE_ROOT=<your writable dir>`;
- provide their own creds for `source .../secrets.sh` (the model is local, so `HF_TOKEN` is usually
  unnecessary). Do **not** rely on zhiyul's `secrets.sh`.

## Monitor

```bash
JOB=<jobid>; LD=results/ultra-swe-teacher-cp16/ray_logs/$JOB-logs
grep -aoE "training_step=[0-9]+|Watchdog caught|EngineDeadError" "$LD/ray-driver.log" | sort | uniq -c
```

- Healthy: `number of parameters on` → rollout → `training_step=0` → `1` → … Step 0 can take ~1 h on a
  hard SWE batch (rollout-bound, not a hang — check worker-log mtimes are fresh, not the driver log).
- **Success:** `training_step` advances with **zero `Watchdog caught`** and zero `EngineDeadError`.
- If `Watchdog caught … CONTEXT_PARALLEL_GROUP`: batch size too large — keep GBS ≤ 32.
- If `EngineDeadError` / `Hit N global ClientOSError`: generation overload — lower `concurrency` or raise the `ray.sub` ulimit.

## Security

`HF_TOKEN` is sourced on the login node and must be redacted from all output (`grep -v HF_TOKEN`); never
bake it into logs. The model is local, so the token is typically unneeded.
