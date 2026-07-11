# Qwen3-30B-A3B CuTeDSL OCI-HSG 1n4g gate

This experiment is the first Linux/GB200 gate for the CuTeDSL fused grouped-MLP policy-training slice. It runs three synchronous GRPO steps on one OCI-HSG node with four GPUs, including Megatron-to-HF export and rollout refit. No GPU run has been performed from this worktree yet.

## Fixed request and runtime

| Item | Value |
|---|---|
| Account | `nemotron_n3_post` |
| Partition | `batch` |
| Allocation | 1 node, `--gres=gpu:4` |
| Expected GPU | 4x GB200 |
| Image | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260711_4677250.sqsh` |
| Image SHA256 | `af1d2ca2a7b169aa13be4b129a0fad8e206c63576d4941b00ae312bd65d0f3e1` |
| Recipe | `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-1n4g-megatron-mxfp8-cutedsl.yaml` |
| GRPO steps | 3 |
| Profiler | Megatron policy worker, step range `2:3` |
| MXFP8 rollout/refit parity | `max(train/gen_kl_error) < 0.05`; `max(train/token_mult_prob_error) < 2.0` |

The wrapper does not accept a container override. The pinned nightly was staged by job `4677250`. Its baked environment is older than this branch, so the wrapper creates a locked Python 3.13.13 MCore environment in node-local storage before validation. If that compatibility build fails, classify it separately from policy-training failures. Do not overwrite the dated image.

## Preflight and scheduling check

Use a pushed branch on the OCI-HSG checkout. Before asking SLURM to schedule it:

```bash
git pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive
git status --short --branch
git submodule status --recursive
git rev-parse '@{upstream}'
git rev-parse HEAD
sbatch --test-only experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_oci_hsg.sh
```

`sbatch --test-only` parses the fixed resource request without executing the wrapper or consuming GPUs. Resolve account, partition, GRES, or feasibility errors before submitting.

The wrapper requires a configured upstream, requires `HEAD == @{upstream}`, rejects tracked changes and non-ignored untracked files, and requires every recursive submodule to be initialized and clean. Push the parent and Bridge branches before this preflight.

The runtime wrapper rejects tracked parent or submodule changes. It then hashes the image, creates a run-local Linux uv environment, and performs these gates in order:

1. Install the locked MCore, test, and dev groups with `uv sync --locked`, then run `uv lock --check`.
2. Record source/upstream, recursive submodule, image, runtime, topology, and resolved-config provenance.
3. Run the official Bridge parameter-mapping/checkpointing suites and the three focused NeMo-RL suites.
4. Run `pyrefly check`, parent pre-commit over the changed parent files, and Bridge pre-commit over the changed Bridge files.
5. Run the required Cutlass DSL `python -c` import.
6. Execute a Transformer Engine `Linear` forward and backward with finite output/input/parameter-gradient checks on each of the four visible GPUs.
7. Run three GRPO steps, persist Ray logs under the result directory, and capture the step-2 Megatron policy worker with Nsight Systems.
8. Export TensorBoard scalars to `metrics.json` and enforce explicit successful optimizer updates, finite loss/gradients, MXFP8 rollout/refit parity, non-empty post-warmup timing/throughput, and `.mem_gb` peak-memory metrics.

These validations are requirements of the job; they are not claimed as passing until the OCI job executes successfully.

## Submit and monitor

```bash
JOB_ID=$(sbatch --parsable experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_oci_hsg.sh)
RUN_DIR="experiments/cutedsl_qwen3_30ba3b_oci_1n4g/results/${JOB_ID}"
echo "${JOB_ID} ${RUN_DIR}"
squeue -j "${JOB_ID}" -o '%.18i %.2t %.10M %.6D %R'
tail -F "${RUN_DIR}/slurm.out"
```

On requeue, the result directory is `$JOB_ID-r$SLURM_RESTART_COUNT`. Discover every attempt safely with:

```bash
find experiments/cutedsl_qwen3_30ba3b_oci_1n4g/results \
  -maxdepth 1 -type d -name "${JOB_ID}*" -print | sort
```

Poll the queue and tail the run log frequently for at least five minutes after the job starts. Model download and kernel compilation can be slow; cancel only for a confirmed unrecoverable error.

## Run artifacts

All generated output stays under `results/$RUN_ID/` (`$JOB_ID` initially or the requeue suffix documented above), which is ignored by Git.

| Artifact | Contents |
|---|---|
| `metadata.json` | Git/submodule SHAs, image SHA256, runtime versions, GPU topology, SLURM allocation, and embedded effective config |
| `effective_config.yaml` | Fully resolved recipe plus wrapper overrides |
| `slurm.out` | Complete wrapper, test, smoke, and GRPO output |
| `uv_lock_check.log` | In-container `uv lock --check` output |
| `focused_tests.log` | Official Bridge and focused NeMo-RL test output |
| `pyrefly.log` | Parent `pyrefly check` output |
| `parent_precommit.log` | Parent pre-commit output over changed parent files |
| `bridge_precommit.log` | Bridge pre-commit output over changed Bridge files |
| `cutlass_import.log` | Cutlass DSL import path |
| `gpu_smoke.log` | Per-device Transformer Engine forward/backward smoke output |
| `grpo.log` | GRPO driver output |
| `ray_tmp/` and `ray_artifacts.txt` | Persistent Ray logs/temp artifacts and their manifest, retained even when GRPO fails |
| `metrics.json` | TensorBoard scalars from the run |
| `metrics_summary.json` | Explicit successful updates, finite checks, parity thresholds/results, timing, throughput, `.mem_gb` peaks, and Nsight file list |
| `kernel_evidence.txt` | `nsys stats` CUDA kernel summary used to identify the fused CuTeDSL kernel |
| `nsight/*.nsys-rep` | Full step-2 policy-worker capture; keep on Lustre and do not commit |
| `status.json` | Wrapper exit code and completion time |

## Classification and acceptance

Classify missing Cutlass DSL, TE, cuDNN frontend, CUDA, binary symbols, container imports/mounts, or insufficient runtime versions as image/runtime failures. Classify config validation, model shape, refit tensor mismatch, loss divergence, and exceptions in the changed Python path as code failures. Use the systematic-debugging workflow for code failures; do not change the image to mask them.

The gate passes only when the wrapper completes and review confirms all of the following:

- At least two `train/optimizer_update_successful == 1` entries were logged; loss-row count is not used as an optimizer-update proxy.
- Megatron-to-HF export and rollout refit completed without shape or layout errors.
- The repository's existing GB200 MXFP8 rollout/refit parity metrics pass: `max(train/gen_kl_error) < 0.05` and `max(train/token_mult_prob_error) < 2.0`. These exact thresholds come from `tests/functional/grpo_vllm_mxfp8_rollout_gb200.sh`; they are proxy metrics for rollout/refit parity, not a claim of elementwise logit equality.
- Loss and gradient norms are finite.
- `kernel_evidence.txt` or the retained Nsight capture identifies the CuTeDSL fused grouped-MLP kernel, not only the generic op-fuser path.
- Non-empty post-warmup policy timing and policy tokens/s series are present, and peak GPU memory is recorded from Ray logger keys ending in `.mem_gb`.

If functional checks pass but kernel evidence is absent, report the result as functionally passing with performance activation unverified. Do not claim a performance win from this three-update smoke; use at least five measured updates or a repeated microbenchmark for a comparison.

## Result record

Status: **Not run**

| Field | Baseline | CuTeDSL |
|---|---|---|
| Job ID | Pending | Pending |
| NeMo-RL SHA | Pending | Pending |
| Megatron-Bridge SHA | Pending | Pending |
| Image SHA256 | Pending | Pending |
| Feature flags | CuTeDSL off / contiguous GLU | MXFP8, grouped GEMM, TE op fuser, GLU interleave 32, `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1` |
| Correctness | Pending | Pending |
| Kernel evidence | Pending | Pending |
| Median post-warmup policy step | Pending | Pending |
| Policy tokens/s | Pending | Pending |
| Peak memory | Pending | Pending |
| Direct log path | Pending | Pending |
| Failures/resolutions | None recorded | None recorded |

After the run, replace the pending cells with exact job IDs, SHAs, Lustre log/profile paths, correctness results, kernel evidence, and measurements. Commit only the README and small text/JSON summaries; never commit checkpoints, full Nsight captures, model caches, or large logs.
