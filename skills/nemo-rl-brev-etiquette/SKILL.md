---
name: nemo-rl-brev-etiquette
license: Apache-2.0
description: Brev instance operating guidance for NeMo-RL agents working in /home/ubuntu/RL with limited workspace disk, an optional large data volume discovered from actual mounts, and optional /home/ubuntu/RL/.env secrets. Use when running nemo-rl-auto-research campaigns, experiments, training jobs, model or dataset downloads, shared cache-heavy commands, log-producing runs, checkpoint generation, W&B or Hugging Face authenticated workflows, or any workflow that may create large files on Brev.
when_to_use: Running on a Brev instance; launching nemo-rl-auto-research campaigns or long jobs; managing large logs, checkpoints, caches, datasets, Ray temp files, W&B files, or Hugging Face auth on Brev.
---

# Brev Etiquette

Operate as though `/home/ubuntu/RL` is the source checkout. Discover suitable working storage for generated experiment state before launching anything expensive. Keep the repo small, reproducible, and easy to inspect.

## Discover the Data Volume

- Honor a user-specified storage path, but verify its backing filesystem, available capacity, and write access before use. If it is unsuitable, explain why before choosing an alternative.
- When the user has not specified storage, inspect the machine with `df -h`, `lsblk -f`, and `findmnt` (where available). `/data` and `/ephemeral` are candidate mount points; names vary across machines and versions, and a separate data volume may not exist.
- Compare candidates with the checkout filesystem using `findmnt -T /home/ubuntu/RL`, `findmnt -T <candidate>`, and `df -h /home/ubuntu/RL <candidate>`. An existing directory alone does not establish that a data volume is mounted there. Do not create `/data` or `/ephemeral` and assume it provides extra storage.
- Prefer a mounted, writable data filesystem with substantial free space for the expected downloads, caches, checkpoints, and logs, separate from the limited checkout/root filesystem. Inspect its top-level contents to understand existing use; do not assume the volume is empty or available for cleanup. Exclude memory-backed filesystems for persistent run storage. Do not format disks or mount unmounted devices as part of discovery.
- Set `DATA_VOLUME` to the verified absolute storage path. If one candidate clearly fits, proceed without an extra approval step. If the choice is ambiguous or no suitable volume exists, report the findings and ask the user to choose storage before starting a large run; do not silently fall back to the checkout/root disk.
- Before writing bulky data, tell the user which path and backing filesystem you selected, its available space, and why it is suitable (or explain any missing/unsuitable volume). Record the selection in the experiment ledger. Do not imply durability from a mount name; verify retention guarantees if the run depends on them.

## Storage Rules

- Keep code edits, small config changes, committed experiment hypotheses, and concise reproducibility records under `/home/ubuntu/RL`.
- Put generated experiment assets under the selected `$DATA_VOLUME`, including checkpoints, run logs, Ray temp directories, W&B offline files, profiler traces, evaluation dumps, rollout samples, and per-experiment artifacts.
- Keep reusable caches under one shared `$DATA_VOLUME` cache root per user, not under each experiment. This includes Hugging Face models, dataset caches, PyTorch caches, Triton caches, `uv` caches, and pip caches.
- Before a campaign or long run, recheck the mount and capacity with `findmnt -T "$DATA_VOLUME"` and `df -h /home/ubuntu/RL "$DATA_VOLUME"`; avoid starting if the selected volume is no longer mounted or lacks sufficient free space.
- Create a campaign root such as `$DATA_VOLUME/nemo-rl/${USER:-ubuntu}/nemo-rl-auto-research/<campaign>` and use one subdirectory per experiment.
- Do not leave large files, cache directories, or generated outputs in the git checkout. If a tool defaults to the repo, override its output/cache path before running it.

## Environment Secrets

- Treat `/home/ubuntu/RL/.env` as the local secret store. It may contain keys such as `WANDB_API_KEY`, `HF_TOKEN`, or `HUGGING_FACE_HUB_TOKEN`.
- Before any run that may need external auth, load `/home/ubuntu/RL/.env` when it exists. Never print, `cat`, log, commit, or summarize secret values.
- If `/home/ubuntu/RL/.env` is absent, or a required key is still unset after loading it, remind the user to add the needed key to that file before launching authenticated work.

```bash
if [ -f /home/ubuntu/RL/.env ]; then
  set -a
  . /home/ubuntu/RL/.env
  set +a
else
  echo "Missing /home/ubuntu/RL/.env; add required keys such as WANDB_API_KEY or HF_TOKEN before authenticated runs."
fi
```

## Auto-Research Pattern

When using `nemo-rl-auto-research`, keep the git ledger in the repo and heavy evidence on the discovered data volume. Set `DATA_VOLUME` from the discovery step before using this example.

```bash
if [ -f /home/ubuntu/RL/.env ]; then
  set -a
  . /home/ubuntu/RL/.env
  set +a
fi

: "${DATA_VOLUME:?Set DATA_VOLUME to the verified data volume before continuing}"
BREV_ROOT=$DATA_VOLUME/nemo-rl/${USER:-ubuntu}
CACHE_ROOT=$BREV_ROOT/cache
CAMPAIGN_ROOT=$BREV_ROOT/nemo-rl-auto-research/<campaign>
EXP_DIR=$CAMPAIGN_ROOT/<experiment>
mkdir -p "$EXP_DIR"/{logs,checkpoints,artifacts,ray,tmp,wandb}
mkdir -p "$CACHE_ROOT"/{huggingface,torch,triton,uv,pip,xdg,wandb}

export HF_HOME=$CACHE_ROOT/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TORCH_HOME=$CACHE_ROOT/torch
export TRITON_CACHE_DIR=$CACHE_ROOT/triton
export UV_CACHE_DIR=$CACHE_ROOT/uv
export PIP_CACHE_DIR=$CACHE_ROOT/pip
export XDG_CACHE_HOME=$CACHE_ROOT/xdg
export WANDB_CACHE_DIR=$CACHE_ROOT/wandb
export RAY_TMPDIR=$EXP_DIR/ray
export TMPDIR=$EXP_DIR/tmp
export WANDB_DIR=$EXP_DIR/wandb
```

Record the resolved absolute paths on the selected volume in the nemo-rl-auto-research TSV fields for log path, checkpoint path, artifacts, shared cache root, and command. If the TSV itself may grow large, store the full TSV on that volume and keep a small pointer file or summary in the repo.

## Launch Checklist

- Discover or revalidate `$DATA_VOLUME`, check its capacity, and inform the user of the storage selection.
- Choose a unique run root under `$DATA_VOLUME` before editing recipes or launching jobs.
- Reuse a shared cache root such as `$DATA_VOLUME/nemo-rl/${USER:-ubuntu}/cache` across experiments unless a run explicitly requires a clean cache.
- Override recipe output paths, logger paths, checkpoint paths, and temp paths to point under the experiment directory.
- Override cache paths to point under the shared cache root.
- Stream stdout/stderr to `$EXP_DIR/logs/run.log` or an equivalent file on the selected volume.
- Periodically check disk during long runs with `df -h "$DATA_VOLUME"` and stop gracefully if the volume is approaching exhaustion.
- At the end, summarize the important metrics and paths in the repo ledger; do not copy bulky artifacts back into `/home/ubuntu/RL`.

## Cleanup

- Clean only files that belong to the current campaign or experiment.
- Prefer pruning clearly named experiment directories under `$DATA_VOLUME/nemo-rl/...`; never remove shared caches or another user's run directory without an explicit instruction.
- Preserve enough small metadata in the repo to reproduce a result after the data volume is cleaned.
