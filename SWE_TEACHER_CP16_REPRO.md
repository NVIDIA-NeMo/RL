# Reproduce: Nemotron-3-Ultra 550B SWE-Teacher GRPO (cp16, 48× GB200)

Minimal runbook for the `swe_teacher_cp16` async-GRPO run on OCI-HSG (GB200, InfiniBand).
Recipe: `examples/configs/ultra/swe_teacher_cp16.yaml` · Launcher: `swe_teacher_cp16_launch.sh`.

## TL;DR — run it

```bash
cd <repo-root>            # the RL checkout on lustre
# creds (HF token etc.) sourced on the LOGIN node only; never inside the compute container:
source /lustre/fs1/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/zhiyul/secrets.sh > >(grep -v HF_TOKEN) 2>&1
bash swe_teacher_cp16_launch.sh          # submits a 48-node job via ultra_launch.sh -> ray.sub
```

Then **immediately register the GPU-idle exemption** (see below) or the reaper kills it during the
~30-min 550B load.

## Cluster / launch facts (kept as-is)
- **Account** `nemotron_sw_post` · **partition** `batch` · **QOS** `short` (2× priority, MaxWall 2h) · **walltime** `1:59:00`.
- **48 nodes** = 32 train + 16 generation (`NUM_TRAIN_NODES=32`, `NUM_GEN_NODES=16`), 4 GPU/node.
- **Container (do not change):** `$Z/enroot-images/nvcr.io+nvidian+nemo-rl+nightly.2026-06-22.squashfs`
  where `Z=/lustre/fsw/portfolios/llmservice/users/zhiyul`. The mcore worker venv is built at runtime via
  `uv run --extra mcore` (compiles transformer-engine — the ~5-min setup cost).
- Model / data / SIFs are all under `$Z` (see the launcher's `MODEL_PATH`, `TRAIN_PATH`, `swe_sifs`).

## Running as a different user (within HSG)
Read paths (model, data, container, SIFs, sdevare symlink targets) are world-readable under
`.../users/zhiyul`, so **you do not need to copy them** — the launcher keeps them as defaults.
Only **write** targets are scoped per-user via `WRITE_ROOT`, which defaults to
`/lustre/fsw/portfolios/llmservice/users/$USER` (HF cache, compile/uv caches, per-agent venvs,
results/checkpoints/logs). So a teammate typically only needs to:
- have access to Slurm account `nemotron_sw_post` (or override `SLURM_ACCOUNT`);
- optionally `export WRITE_ROOT=<your writable dir>` if you don't want the `users/$USER` default;
- provide your own creds for the `source .../secrets.sh` line — though the model is local, so
  `HF_TOKEN` is usually unnecessary. Do **not** rely on zhiyul's `secrets.sh`.

Leave `NRL_NCCL_FLIGHT_RECORDER` unset unless you also set `TORCH_NCCL_DEBUG_INFO_TEMP_FILE` to a
writable dump-dir prefix. Note `checkpointing.save_period=1` (in the launcher) writes a 550B
checkpoint **every step** — drop it for anything but short debugging.

## GPU-idle reaper exemption (REQUIRED)
The 550B dist-ckpt load sits at SM≈0 for ~30 min and the auto-reaper (`svc-hwinf-cs-sched`) cancels
idle jobs after 30 min. Set the exemption comment on the job **while PENDING or right after RUNNING**:

```bash
JOB=<jobid>
scontrol update jobid=$JOB Comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"benchmarking","description":"550B dist-ckpt load ~30min SM~0; SWE rollouts idle GPUs"}}'
squeue -j $JOB -o %k    # verify the JSON is set
```

## Fixes already applied (why the recipe runs at all)
This recipe had never run end-to-end; the following are in the branch/config (keep them):

| Area | Setting / change | Why |
|---|---|---|
| NaN | `policy.generation.vllm_kwargs.moe_backend: triton` | vLLM default FlashInfer MoE is refit-incompatible → NaN |
| OOM | `expert_model_parallel_size: 32` | EP16 → 32 experts/rank → grad-buffer OOM |
| fd storm | `ulimit -Sn` → hard cap (`ray.sub`) | concurrency-256 SWE rollouts exhausted 65535 fds |
| vLLM assert | `make_sequence_length_divisible_by: 256` | nemotron_h `minimum_pad_factor=256` |
| KV thrash | `swe_agents concurrency: 64` | 8 vLLM engines (16 gen nodes/TP8); 256 oversubscribed the KV |
| logprobs | `mtp_num_layers: 1` (loss-disabled) | model ships an MTP head; `0` trips a mcore hybrid assert |
| topology | `cluster.segment_size: 8` (= EP/gpus_per_node) + `SEGMENT_SIZE=8` (`--segment`) | keep each EP group in one NVLink rack |
| topology | backport **#2924** (`virtual_cluster.py`: exclude unknown-NVLink-domain nodes) | drop mis-probed nodes from segment selection |
| topology | backport **#3178** (`ray.sub`: unique `TOPO_RANK`, head=1, workers=`PROCID+2`) | value-0 TOPO_RANK → Ray drops it → fragmented/duplicate-GPU placement |
| packing | backport **#3182** (`loss/utils.py::_pack_input_ids` clamp) | inflated last-seq len can exceed row width |
| NCCL transport | `ray.sub` BAKED_ENV: `NCCL_NET=IB`, `NCCL_NVLS_ENABLE=1`, `NCCL_GIN_*`, `HYBRID_EP_CACHE_DIR` | from `gtp_benchmark/nemotron/a55b/hsg/set_env.sh` — stable multi-node MoE collectives on HSG |

> If you **rebase onto main**, verify #2924/#3178/#3182 land (or are already upstream) and re-check
> `cluster.segment_size` and the NCCL env survive. mcore submodule is already at main's latest (554c7b9).

## Known open issue (as of this writing)
Intermittent **training-step NCCL collective stall** (distributed-optimizer `_REDUCE_SCATTER_BASE`/
`_ALLGATHER_BASE`) → 600s watchdog → actor dies. It moves *later* with each fix (step 0 → 1 → 2),
i.e. it's **intermittent**, not a deterministic deadlock. Root cause is topology/rank placement
(addressed by segment=8 + #3178 + #2924 + NCCL GIN/NVLS). Two candidate residual mechanisms:
cross-rack transport flakiness vs async-GRPO straggler. To diagnose, enable the flight recorder and
force a fast abort so it dumps before walltime:

```bash
# already wired in ray.sub BAKED_ENV: TORCH_NCCL_TRACE_BUFFER_SIZE, TORCH_NCCL_DUMP_ON_TIMEOUT=1,
# TORCH_NCCL_DEBUG_INFO_TEMP_FILE=<...>/results/nan-debug/fr_e2e/fr_
# NOTE: keep NCCL_DEBUG=WARN (INFO floods 155MB/876k lines at 48 nodes and throttles the run).
```

## Monitor / success
```bash
JOB=<jobid>; LD=results/ultra-swe-teacher-cp16/ray_logs/$JOB-logs
squeue -j $JOB -o "%T %M"
grep -aoE "step=[0-9]+|Watchdog caught|Duplicate GPU detected|Collecting rollouts" "$LD/ray-driver.log" | sort | uniq -c
```
- Healthy: `number of parameters on` (load, no OOM) → `Collecting rollouts` → `step=0` → `step=1` → …
- **Success = it advances past step 2/3 with zero `Watchdog caught` and zero `Duplicate GPU detected`.**
- Failure signatures: `OutOfMemoryError` (memory), `Out of range float values` (NaN — should be gone),
  `Too many open files` (ulimit), `Watchdog caught` (the open collective-stall issue).

## Security note
`HF_TOKEN` must be sourced on the login node and redacted from all output (`grep -v HF_TOKEN`); do not
bake it into logs. Source secrets on login, pass via env — the model is local so the token is unneeded.
