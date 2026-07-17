---
name: slurm
description: Safe Slurm procedures for NeMo-RL on cw-dfw-cs-001. Use when launching, testing, monitoring, stopping, or debugging with srun, sbatch, or ray.sub; choosing accounts, partitions, nodes, or GPUs; or following up on idle-GPU and stranded-GPU alerts.
---

# Slurm

## Resource safety

- Treat the target H100/H200 nodes as 8-GPU nodes unless current cluster
  metadata proves otherwise.
- Never combine `--exclusive` with a partial-GPU request. With `--exclusive`,
  request the entire node using `--gpus-per-node=8`.
- For single-node diagnostics and tests, default to the full-node form below so
  the idle-GPU monitor does not report seven stranded GPUs:

  ```bash
  srun --account=nemotron_sw_post --partition=interactive \
    --nodes=1 --gpus-per-node=8 --exclusive ...
  ```

- If a genuinely shareable partial-GPU job is required, remove `--exclusive`,
  request GPUs with `--gpus-per-node=N`, and omit explicit CPU and memory
  requests so Slurm assigns them proportionally.
- Do not use `--gpus=N` for per-node placement.
- Use the `interactive` partition for jobs using at most two nodes and the
  `nemotron_sw_post` account unless the user explicitly chooses another valid
  account.
- A job may target multiple eligible partitions with a comma-separated list
  (for example, `--partition=interactive,batch`) when that can reduce queueing
  time. Never include the `backfill` partition. Preserve all node/GPU and
  exclusive-safety rules regardless of which listed partition admits the job.
- A multi-partition request must be legal in every listed partition. Inspect
  `AllowAccounts`, `MaxTime`, and the partition QOS before submission. Do not
  assume Slurm will fall back past an ineligible or quota-saturated partition:
  a three-node `interactive,batch` request was rejected by interactive's
  two-node QOS, a full `batch_short` QOS blocked `batch_short,batch`, and
  `batch_large` rejected `nemotron_sw_post` through its account allowlist.

## Launch checks

1. Inspect the resolved submission before launching. Verify account,
   partition, node count, `--exclusive`, and GPUs per node together.
2. Perform the mandatory cache audit from
   @skills/build-and-dependency/SKILL.md. Every applicable package, HF, and
   compiler/JIT cache must have an explicit safe location. Unless the user
   requests a cold-cache run, seed node-local build caches before an expensive
   launch and configure safe persistence for newly built artifacts.
3. Log cache roots and cold/seeded/warm state without printing credentials or
   the full environment. Include cache state in performance-run metadata so
   cold startup cannot be mistaken for steady-state regression.
4. After submission, inspect allocation accounting:

   ```bash
   sacct -j <job-id> --allocations \
     -o JobID,State,Account,Partition,AllocTRES,ReqTRES,NodeList
   ```

5. For an exclusive allocation, require requested and allocated GPU totals to
   equal `nodes * GPUs per node`. Cancel and correct a mismatched job promptly.
6. When investigating an alert, use `sacct --json` or the `SubmitLine` field to
   identify the exact command before changing a shared launcher.
7. Inspect test fixtures as well as model parallel settings when sizing a
   diagnostic. Slurm `14028950` allocated one GPU for
   `test_vllm_http_server`, but its Ray cluster fixture requested two and spent
   a minute retrying placement before failing. That same bare-node run also
   showed a PyTorch CUDA/driver mismatch. Run this test with two GPUs in the
   validated SWE container; use `interactive`, omit `--exclusive`, CPU, and
   memory requests, and preflight CUDA visibility inside the exact container.

## Attached `srun` lifecycle

- Set the client/tool timeout longer than the Slurm `--time` limit for a long
  attached `srun`, or submit with `sbatch` and a persistent output file. A
  short client timeout can detach without terminating the allocation.
- After an interrupted or timed-out `srun`, immediately inspect both `squeue`
  and `sacct`. A running allocation with only an `.extern` step, no workload
  step, and no GPU process is orphaned; cancel it promptly.
- Add a persistent `--output` path for diagnostics. If the client is
  interrupted, terminal-only stderr may be unrecoverable even though Slurm
  retains the exit code.
- In container commands, use `/usr/bin/env` rather than bare `env`; see
  @skills/build-and-dependency/SKILL.md for the known uv/PATH collision.
- Do not dump full `SubmitLine`, environment, or `COMMAND` fields when they can
  contain credentials. Project only the resource or error fields needed for
  the diagnosis and redact sensitive values.
