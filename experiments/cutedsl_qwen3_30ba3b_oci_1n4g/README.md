# Qwen3-30B-A3B CuTeDSL OCI-HSG 1n4g gate

This experiment is the first Linux/GB200 gate for the CuTeDSL fused grouped-MLP policy-training slice. It runs three synchronous GRPO policy updates on one OCI-HSG node with four GPUs, including Megatron-to-HF export and rollout refit. No GPU run has been performed from this worktree yet.

## Fixed request and runtime

| Item | Value |
|---|---|
| Account | `nemotron_n3_post` |
| Partition | `batch` |
| Allocation | 1 node, `--gres=gpu:4` |
| Expected GPU | 4x GB200 |
| Image used first | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo_rl_nightly_20260707.sqsh` |
| Recipe | `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-1n4g-megatron-mxfp8-cutedsl.yaml` |
| Policy updates | 3 |
| Profiler | Megatron policy worker, step range `2:3` |

The wrapper does not accept a container override. If the pinned image has a confirmed runtime failure, stage a new immutable nightly through the container-staging workflow, commit the new provenance, and only then change the wrapper. Do not overwrite the dated image.

## Preflight and scheduling check

Use a pushed branch on the OCI-HSG checkout. Before asking SLURM to schedule it:

```bash
git pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive
git status --short --branch
git submodule status --recursive
sbatch --test-only experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_oci_hsg.sh
```

`sbatch --test-only` parses the fixed resource request without executing the wrapper or consuming GPUs. Resolve account, partition, GRES, or feasibility errors before submitting.

The runtime wrapper rejects tracked parent or submodule changes. It then hashes the image, creates a run-local Linux uv environment, and performs these gates in order:

1. Record source, recursive submodule, image, runtime, topology, and resolved-config provenance.
2. Run the three focused Linux pytest files.
3. Run the required Cutlass DSL `python -c` import.
4. Import PyTorch and Transformer Engine and execute a finite BF16 matmul on each of the four visible GPUs.
5. Run three GRPO updates and capture the step-2 Megatron policy worker with Nsight Systems.
6. Export TensorBoard scalars to `metrics.json` and summarize update count, finite loss/gradients, post-warmup timing/throughput, memory metrics, and profiler artifacts.

## Submit and monitor

```bash
JOB_ID=$(sbatch --parsable experiments/cutedsl_qwen3_30ba3b_oci_1n4g/submit_oci_hsg.sh)
RUN_DIR="experiments/cutedsl_qwen3_30ba3b_oci_1n4g/results/${JOB_ID}"
echo "${JOB_ID} ${RUN_DIR}"
squeue -j "${JOB_ID}" -o '%.18i %.2t %.10M %.6D %R'
tail -F "${RUN_DIR}/slurm.out"
```

Poll the queue and tail the run log frequently for at least five minutes after the job starts. Model download and kernel compilation can be slow; cancel only for a confirmed unrecoverable error.

## Run artifacts

All generated output stays under `results/$JOB_ID/`, which is ignored by Git.

| Artifact | Contents |
|---|---|
| `metadata.json` | Git/submodule SHAs, image SHA256, runtime versions, GPU topology, SLURM allocation, and embedded effective config |
| `effective_config.yaml` | Fully resolved recipe plus wrapper overrides |
| `slurm.out` | Complete wrapper, test, smoke, and GRPO output |
| `focused_tests.log` | Linux focused-test output |
| `cutlass_import.log` | Cutlass DSL import path |
| `gpu_smoke.log` | Per-device PyTorch/TE smoke output |
| `grpo.log` | GRPO driver output |
| `metrics.json` | TensorBoard scalars from the run |
| `metrics_summary.json` | Update count, finite checks, timing, throughput, memory metrics, and Nsight file list |
| `kernel_evidence.txt` | `nsys stats` CUDA kernel summary used to identify the fused CuTeDSL kernel |
| `nsight/*.nsys-rep` | Full step-2 policy-worker capture; keep on Lustre and do not commit |
| `status.json` | Wrapper exit code and completion time |

## Classification and acceptance

Classify missing Cutlass DSL, TE, cuDNN frontend, CUDA, binary symbols, container imports/mounts, or insufficient runtime versions as image/runtime failures. Classify config validation, model shape, refit tensor mismatch, loss divergence, and exceptions in the changed Python path as code failures. Use the systematic-debugging workflow for code failures; do not change the image to mask them.

The gate passes only after review confirms all of the following:

- At least two policy optimizer updates completed.
- Megatron-to-HF export and rollout refit completed without shape or layout errors.
- Post-refit rollout logits match the eager/non-interleaved reference within the agreed MXFP8 tolerance.
- Loss and gradient norms are finite.
- `kernel_evidence.txt` or the retained Nsight capture identifies the CuTeDSL fused grouped-MLP kernel, not only the generic op-fuser path.
- Median post-warmup policy step time, policy tokens/s, and peak memory are recorded.

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
