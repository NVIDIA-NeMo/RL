---
name: dfw-cluster
description: Use when working on the dfw cluster: connecting to login nodes, choosing Slurm partitions, submitting/monitoring jobs, and handling cluster etiquette for NeMo-RL/Nemotron experiments.
---

# DFW Cluster

Use this skill when running commands, submitting jobs, or monitoring jobs on dfw.

## Access

Use the current dfw login node unless the user gives a different one:

```bash
cw-dfw-cs-001-login-02.nvidia.com
```

Prefer a persistent SSH control connection for repeated commands:

```bash
ssh -o ControlMaster=auto -o ControlPath=/tmp/ssh-dfw2-%r@%h:%p -o ControlPersist=4h cw-dfw-cs-001-login-02.nvidia.com
```

Do not run training, conversion, or GPU-heavy work on login nodes. Use Slurm or an allocated interactive node.

## Partitions

- `batch_short`: 2 hour GPU partition. Prefer this for evals, smoke tests, and short sanity jobs because it is usually more available.
- `interactive`: GPU interactive partition. Interactive allocations usually start quickly, so this is a good alternative for smoke tests and debugging when `batch_short` is backed up.
- `batch`: 4 hour default GPU partition. Use this for normal single-node training/debug jobs that fit under 4 hours.
- `batch_large`: 4 hour GPU partition for larger-node jobs. Use only when the job genuinely needs larger allocation behavior.
- `batch_long`: 8 hour GPU partition. Use when 4 hours is not enough and the job does not need the large-job partition.
- `batch_large_long`: long-running large GPU partition. Use sparingly; confirm with the user before submitting.
- `cpu`, `cpu_short`, `cpu_long`, `cpu_interactive`, `cpu_datamover`, `cpu_dataprocessing`: CPU-only partitions for preprocessing, file movement, and lightweight scripts.
- `admin`, `defq`: do not use for user jobs.

Check current availability before making assumptions:

```bash
sinfo -o "%P %.12l %.8a %.10T %D"
```

## Accounts and Fairshare

Check the user's usable Slurm accounts before choosing an account:

```bash
sacctmgr -n -P show assoc where user=$USER format=Account,Partition,QOS,DefaultQOS | sort -u
```

Common accounts for this workspace include `coreai_dlalgo_llm`, `coreai_dlalgo_genai`, `coreai_dlalgo_nemofw`, `coreai_dlalgo_modelopt`, and `nvr_lpr_llm`. Use the full account name in submissions, for example `sbatch -A nvr_lpr_llm ...`, unless the repo's submit wrapper exposes a different account variable.

Before submitting any Slurm job, measure fairshare for the usable accounts and choose the account with the best current fairshare unless the user explicitly requests a specific account:

```bash
sshare -U -l
sprio -u $USER -l
```

In `sshare`, compare accounts using `FairShare` and `LevelFS`. Higher `FairShare` is generally better; `LevelFS > 1` means the account is relatively under-used, while `LevelFS < 1` means it is relatively over-used. Fairshare changes over time, so check it immediately before submission and do not hard-code current values in commands or skills.

