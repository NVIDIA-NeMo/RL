# OCI-HSG HybridEP GRPO launchers

`submit_grpo.sh` is a thin, reproducible wrapper around the repository's
`ray.sub`. Model shape and recipe selection live in `models/*.env`.

Run from a clean, pushed NeMo-RL checkout:

```bash
scripts/experiments/oci-hsg/hybridep/submit_grpo.sh
```

The default Qwen3-30B-A3B profile uses:

- `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml`
- 4 OCI-HSG GB200 nodes, 4 GPUs per node, and `--segment=4`
- HybridEP flex dispatch with 32 SMs
- the recipe's sequence packing and fused loss
- 20 GRPO steps
- current nightly image staged under the user NeMo-RL Lustre area

To run with a custom DeepEP wheel:

```bash
DEEPEP_COMMIT=f725d29699f5bda9ba789456bb9579af69844685 \
DEEPEP_WHEEL=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/deepep-wheels/f725d29699f5bda9ba789456bb9579af69844685-doca-sm100/deep_ep-1.2.1+f725d29-cp313-cp313-linux_aarch64.whl \
scripts/experiments/oci-hsg/hybridep/submit_grpo.sh
```

To run the 32-node Nemotron3 Super 120B async 1-off recipe with the same
HybridEP overrides:

```bash
WANDB_ENABLED=False \
scripts/experiments/oci-hsg/hybridep/submit_grpo.sh \
  scripts/experiments/oci-hsg/hybridep/models/nemotron3-super-120ba12b-32n4g-async-1off.env
```

The launcher selects the highest current user-level FairShare account. Set
`ACCOUNT` only to override that choice. Set `WANDB_ENABLED=True` to enable W&B;
the launcher requires `WANDB_API_KEY` in the environment and never writes its
value to metadata.

Each submission first runs `sbatch --test-only`. It then records the model
profile, all source commits, DeepEP selection, image SHA256, job ID, and log
paths under `exp_logs/hybridep/<model>/<run-name>/submission.env`.
