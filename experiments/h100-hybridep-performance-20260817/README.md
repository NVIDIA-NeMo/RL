# H100 HybridEP Performance Validation

## Objective

Validate that enabling HybridEP in the existing H100 8-GPU-per-node MoE
performance recipes does not introduce out-of-memory failures and improves
end-to-end performance relative to each recipe's previous default
configuration.

## Fixed software stack

Both comparison arms use the same NeMo-RL source revision, container, model
checkpoint, dataset, node count, GPU count, scheduler placement, and random
configuration. The experiment branch combines:

- the latest `main` revision at experiment creation;
- the x86 HybridEP dependency pin from PR #3436;
- the HybridEP sequence-packing compatibility changes from PR #2964; and
- the recipe-only HybridEP configuration changes being validated.

The baseline arm loads the corresponding recipe exactly as it existed at the
fixed `main` revision. The HybridEP arm loads the modified recipe. This keeps
the dispatcher configuration as the intentional A/B variable.

## Recipe inventory

The scope is every `n8g` performance recipe that resolves to the Megatron
backend with expert parallelism greater than one:

| Model family | Recipes | Canonical 20-step A/B recipe |
|---|---:|---|
| DeepSeek V3 | 5 | `grpo-deepseek-v3-32n8g.yaml` |
| Nemotron 3 Super 120B-A12B | 2 | `grpo-nemotron3-super-120BA12B-32n8g.yaml` |
| Qwen3 235B-A22B | 3 | `grpo-qwen3-235b-16n8g.yaml` |
| Qwen3 30B-A3B | 4 | `grpo-qwen3-30ba3b-4n8g.yaml` |

Dense `n8g` recipes with expert parallelism equal to one remain unchanged.

## HybridEP overlay

All in-scope recipes set the flex dispatcher with the HybridEP backend and an
8-GPU H100 NVLink domain:

```yaml
megatron_cfg:
  moe_token_dispatcher_type: flex
  moe_flex_dispatcher_backend: hybridep
  moe_hybridep_num_sms: 32

env_vars:
  NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN: "8"
  NUM_OF_TOKENS_PER_CHUNK_COMBINE_API: "128"
  NVLINK_DOMAIN_SIZE: "8"
  USE_MNNVL: "0"
```

Packed-input pre-padding is enabled only for pipeline-parallel-size-one,
MTP-disabled recipes supported by the NeMo-RL compatibility path. Other
recipes rely on the Megatron-LM uneven-dispatch padding path.

## Experiment matrix

1. Run unit/static validation over all 14 modified recipes.
2. Run matched 20-step baseline and HybridEP jobs for the four canonical sync
   recipes.
3. Run three-step HybridEP smoke jobs for the remaining ten recipes to detect
   OOMs and startup/runtime failures without duplicating every large baseline.
4. Average performance metrics over optimizer steps 2 through 20, inclusive.
   If a job records fewer steps, report the exact observed window and do not
   compare it as a completed 20-step result.

## Recorded metrics

- completion state and highest completed optimizer step;
- CUDA OOM, host OOM, hang, and non-OOM failure classification;
- end-to-end step time;
- end-to-end throughput in tokens/second/GPU;
- policy-training and log-probability time/throughput when logged; and
- W&B run URL for reproducibility.

The primary speedup calculations are:

```text
step-time speedup = baseline E2E time / HybridEP E2E time
throughput gain   = (HybridEP throughput / baseline throughput - 1) * 100
```

## Reproducibility and security

Submission scripts take the container, scheduler account, remote worktree, and
W&B project from environment variables. Reports must not contain private host
aliases, filesystem paths, scheduler metadata, credentials, or job IDs. Public
artifacts may include Git commit SHAs, recipe names, hardware type, aggregate
metrics, and W&B links explicitly approved for sharing.
