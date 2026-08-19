# GB200 deck refresh

Refreshes every B200-backed result in `mxfp8_systems_experiments.pptx` on one
current NeMo-RL integration commit and GB200 GPUs.

The Nano matrix compares BF16 and MoE-only MXFP8 rollout in both colocated Sync
and non-colocated Async modes. Every arm enables CUDA Graphs and computes both
previous-policy and reference-policy logprobs. The Qwen3-235B matrix compares
the legacy packed-broadcast refit with NCCL Reshard for the same MXFP8 rollout.
Report means use completed steps 3-20.

Render a command locally:

```bash
ACTION=render MODEL=nano MODE=sync ARM=mxfp8 ./submit_oci_hsg.sh
```

Run scheduler preflight before submission:

```bash
ACTION=test-only ./prepare_ray_runtime_oci_hsg.sh
ACTION=submit ./prepare_ray_runtime_oci_hsg.sh
ACTION=test-only MODEL=nano MODE=sync ARM=mxfp8 ./submit_oci_hsg.sh
ACTION=submit MODEL=nano MODE=sync ARM=mxfp8 ./submit_oci_hsg.sh
```
