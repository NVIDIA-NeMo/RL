# GB200 deck refresh

Refreshes every B200-backed result in `mxfp8_systems_experiments.pptx` on one
current NeMo-RL integration commit and GB200 GPUs.

The Nano matrix compares BF16 and MoE-only MXFP8 rollout in both colocated Sync
and non-colocated Async modes. Every arm enables CUDA Graphs and computes both
previous-policy and reference-policy logprobs. The Qwen3-235B matrix compares
the legacy packed-broadcast refit with NCCL Reshard for the same non-colocated
Sync MXFP8 rollout used by the B200 result. Report means use completed steps
3-20.

The PR 3294 full ablation reproduces the prior GCP-NRT on/off controls on
commit `313f41a9654cd67e44d783128543fe1638c778da`, which pins vLLM 0.25.1.
It compares prequantization, batched MXFP8 shuffle, and cached loaders disabled
against all three enabled on 4x4 GB200 GPUs.

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
