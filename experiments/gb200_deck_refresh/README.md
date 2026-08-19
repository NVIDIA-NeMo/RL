# GB200 deck refresh

Refreshes every B200-backed result in `mxfp8_systems_experiments.pptx` on one
current NeMo-RL integration commit and GB200 GPUs.

The Nano matrix compares BF16 and MoE-only MXFP8 rollout in both colocated Sync
and non-colocated Async modes. Every arm enables CUDA Graphs and computes both
previous-policy and reference-policy logprobs. The Qwen3-235B matrix compares
the legacy packed-broadcast refit with NCCL Reshard for the same non-colocated
Sync MXFP8 rollout used by the B200 result. Report means use completed steps
3-20.

## 2026-08-18 GB200 run matrix

All runs below use 20 optimization steps. Final values must average completed
steps 3-20 and report the included step count. Nano and Qwen3-235B use source
commit `f8d3514b436cf79bf2e733b83a9ca902d88831c9`.

| Study | Arm | SLURM job |
| --- | --- | ---: |
| Nano Sync | BF16 rollout | 6302542 |
| Nano Sync | MXFP8 rollout | 6302578 |
| Nano Async | BF16 rollout | 6302582 |
| Nano Async | MXFP8 rollout | 6302584 |
| Qwen3-235B Sync | MXFP8 legacy refit | 6302606 |
| Qwen3-235B Sync | MXFP8 NCCL Reshard | 6302692 |
| PR 3294 full ablation | all three disabled | 6302717 |
| PR 3294 full ablation | all three enabled | 6302751 |
| Batched shuffle only | parent `e45e29da` | 6302200 |
| Batched shuffle only | PR 3478 `d5fb8d04` | 6302232 |

The PR 3294 full ablation reproduces the prior GCP-NRT on/off controls on
commit `313f41a9654cd67e44d783128543fe1638c778da`, which pins vLLM 0.25.1.
It compares prequantization, batched MXFP8 shuffle, and cached loaders disabled
against all three enabled on 4x4 GB200 GPUs.

Set `STUDY=shuffle_only` to compare the PR 3478 parent
`e45e29da7266a7a219d2a0bc4adb0a1f78456985` against the merged implementation
`d5fb8d044031420e9170aae66ee0c3166b798381`. Both arms keep prequantization
enabled, so this pair isolates the batched MXFP8 MoE shuffle.

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
