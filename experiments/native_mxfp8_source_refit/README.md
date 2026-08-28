# Native MXFP8 Source Refit

This task-owned experiment compares BF16 parameter storage (`fp8_param=false`)
with native MXFP8 parameter storage (`fp8_param=true`) on the same
non-colocated NCCL Reshard topology. The launcher does not submit work unless
`ACTION=submit` is selected.

The four configurations are resolved from the repository's baseline GRPO
configuration and retain their task-owned model geometry:

| Model | Configuration | Nodes x GPUs | Segment |
| --- | --- | ---: | ---: |
| Qwen3-30B-A3B | `qwen30-fp8param-false.yaml` | 4 x 4 | 2 |
| Qwen3-30B-A3B | `qwen30-fp8param-true.yaml` | 4 x 4 | 2 |
| Nemotron-3 Nano | `nano-fp8param-false.yaml` | 8 x 4 | 4 |
| Nemotron-3 Nano | `nano-fp8param-true.yaml` | 8 x 4 | 4 |

`te_nano_routed.yaml` orders routed `linear_fc1` and `linear_fc2` MXFP8
matchers before a BF16 catch-all. Nano additionally holds zero leading layers
and the final eight layers in BF16. Its native smoke must verify that only the
routed-expert FC1/FC2 tensors carry E4M3 values plus E8M0 scales; attention,
shared experts, router, QKVO, and `lm_head` remain BF16 misc entries.

The native overlays enable a bounded worker-side assertion over the actual
refit metadata. Before the NCCL refit plan is built, it emits one
`[native-mxfp8-inventory]` JSON record and exits nonzero when the routed scope,
BF16 catch-all scopes, component roles/dtypes, or Nano final-eight boundary do
not match. `RAY_TMPDIR_ROOT` is exported before `ray.sub`; `ray.sub` derives
the job-scoped scratch directory before starting the Ray head or workers.

## Environment

`REPO` must be a clean source checkout under `/home`. `CONTAINER`, `HF_HOME`,
and `RESULT_ROOT` must be durable `/lustre` paths; `WANDB_HOME` contains the
credentials under `/home`. The launcher creates worker environments, UV,
vLLM, Triton, Inductor, Ray temporary files, and staged model caches only
under `/raid/scratch/${USER}`. Build caches are keyed by source SHA and model
cache staging is keyed by model, never job ID or run label. The worker
environment is keyed only by source SHA; compiled vLLM, Triton, and Inductor
caches are additionally scoped by the `fp8_param` arm.

The known model cache directories are staged once on each allocated node with
`rsync --ignore-existing`; the launcher does not scan or copy the entire
Hugging Face cache. Durable Slurm logs and requested run artifacts are the only
per-run output under `RESULT_ROOT`.

## Commands

Render the selected setup without contacting Slurm:

```bash
MODEL=qwen30 FP8_PARAM=true MAX_STEPS=2 ACTION=render \
  ./experiments/native_mxfp8_source_refit/submit_oci_hsg.sh
```

Run `sbatch --test-only` before each smoke or submission:

```bash
MODEL=qwen30 FP8_PARAM=true MAX_STEPS=2 ACTION=test-only \
  ./experiments/native_mxfp8_source_refit/submit_oci_hsg.sh
MODEL=nano FP8_PARAM=true MAX_STEPS=2 ACTION=test-only \
  ./experiments/native_mxfp8_source_refit/submit_oci_hsg.sh
```

The native two-step smoke commands are:

```bash
MODEL=qwen30 FP8_PARAM=true MAX_STEPS=2 ACTION=submit \
  ./experiments/native_mxfp8_source_refit/submit_oci_hsg.sh
MODEL=nano FP8_PARAM=true MAX_STEPS=2 ACTION=submit \
  ./experiments/native_mxfp8_source_refit/submit_oci_hsg.sh
```

Run matched 20-step A/B measurements from one clean source SHA:

```bash
MODEL=qwen30 FP8_PARAM=false MAX_STEPS=20 ACTION=submit ./experiments/native_mxfp8_source_refit/submit_oci_hsg.sh
MODEL=qwen30 FP8_PARAM=true  MAX_STEPS=20 ACTION=submit ./experiments/native_mxfp8_source_refit/submit_oci_hsg.sh
MODEL=nano   FP8_PARAM=false MAX_STEPS=20 ACTION=submit ./experiments/native_mxfp8_source_refit/submit_oci_hsg.sh
MODEL=nano   FP8_PARAM=true  MAX_STEPS=20 ACTION=submit ./experiments/native_mxfp8_source_refit/submit_oci_hsg.sh
```

After a submission, use one filtered `squeue -j <job-id>` or `squeue --me`
query no sooner than 60 seconds after the preceding query, for at least five
minutes. Inspect only the submitted job's logs; do not run unfiltered scheduler
queries or recursive scans on shared storage.

## Smoke Gate

The two-step native smoke is accepted only when both steps have finite loss,
reward, entropy, and generation-KL error, and the logs show the native source
route (`fp8_param=true`, `fp8_recipe=mxfp8`, `refit_transport=nccl_reshard`)
with E4M3 values and `torch.uint8` E8M0 scales. It must also prove one vLLM
reload initialization and finalization per refit, no BF16 receiver
quantization for native components, and a Qwen value-changing second refit.

The worker validates native source tensor dtype, compact scale layout, and
swizzled-scale rejection before the collective. For the true-arm smoke it also
emits `[native-mxfp8-inventory]` with the complete task-owned routed FC1/FC2
and BF16 ignored-scope inventory, and exits nonzero on any mismatch. Preserve
that JSON record in the Task 10 results artifact; no manual reconstruction of
the module storage scope is required.
