# H100 HybridEP Performance Validation

## Objective

Validate that enabling HybridEP in the existing H100 8-GPU-per-node MoE
performance recipes does not introduce out-of-memory failures and improves
end-to-end performance relative to each recipe's previous default
configuration.

## Fixed software stack

Both comparison arms use the same NeMo-RL source revision, container, model
checkpoint, dataset, requested H100 topology, node count, GPU count, and random
configuration. The launcher requires a clean source tree at submission and
rejects a queued job if the source or submodule revision changes before it
starts. The experiment branch combines:

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

Both arms build dependencies with the same H100 environment. In particular,
`HYBRID_EP_MULTINODE=1`, `TORCH_CUDA_ARCH_LIST=9.0`, `NVTE_CUDA_ARCHS=90`,
`RDMA_CORE_HOME=/usr`, and `USE_NIXL=0` make the pinned DeepEP revision
compile its Hopper multi-node DOCA/NCCL path without rebuilding unrelated GPU
architectures. These are fixed software-stack prerequisites, not A/B recipe
variables. Each arm uses a persistent venv directory inside its result
directory with `NRL_FORCE_REBUILD_VENVS=false`. The first builder materializes
each actor environment once on shared storage and later nodes or retries reuse
it, avoiding concurrent multi-node force-rebuild races.

## Experiment matrix

1. Run unit/static validation over all 14 modified recipes.
2. Run matched 20-step baseline and HybridEP jobs for the four canonical sync
   recipes.
3. Run three-step HybridEP smoke jobs for the remaining ten recipes to detect
   startup OOMs and initialization failures without duplicating every large
   baseline. A three-step pass is not reported as long-run OOM clearance.
4. Keep each recipe's inherited validation schedule unchanged so the default
   runtime path, including validation at Steps 10 and 20, is exercised.
5. Compute steady-state training performance over Steps 2 through 20 excluding
   validation Steps 10 and 20. Report validation-inclusive operational step
   time separately. If a job records fewer steps, report the exact observed
   window and do not compare it as a completed 20-step result.

## Recorded metrics

- completion state and highest completed optimizer step;
- CUDA OOM, host OOM, hang, and non-OOM failure classification;
- end-to-end step time;
- end-to-end throughput in tokens/second/GPU;
- policy-training and log-probability time/throughput when logged; and
- valid-token counts, reward, loss, and generation-KL sanity checks; and
- W&B run URL for reproducibility.

The primary speedup calculations are:

```text
aggregate throughput = sum(valid tokens) / sum(E2E step time) / GPU count
step-time speedup     = baseline E2E time / HybridEP E2E time
throughput gain       = (HybridEP throughput / baseline throughput - 1) * 100
```

Per-step throughput means and dispersion are secondary diagnostics. A speedup
is interpreted only when both arms complete the same step window, process
comparable valid-token counts, and show no reward/loss/generation-KL anomaly.
One allocation per arm is directional evidence; allocation-to-allocation
variance is called out rather than inferred from within-run step samples.

## Reproducibility and security

Submission scripts take the container, scheduler account, remote worktree, and
W&B project from environment variables. Reports must not contain private host
aliases, filesystem paths, scheduler metadata, credentials, or job IDs. Public
artifacts may include Git commit SHAs, recipe names, hardware type, aggregate
metrics, and W&B links explicitly approved for sharing.

## Launcher

`matrix.tsv` defines the four A/B pairs and ten additional smoke runs.
`submit.sh` validates the selected recipe and requires all infrastructure values
through the environment:

```bash
export SBATCH_ACCOUNT=...
export CONTAINER=...
export MOUNTS=...
export HF_HOME=...
export RUN_ROOT=...
export WANDB_API_KEY=...
export WANDB_PROJECT=...
# Optional: reuse wheels built for the pinned HybridEP revision across jobs.
export UV_CACHE_DIR_OVERRIDE=...
# Optional: reuse a fully materialized actor environment across matched arms.
export NEMO_RL_VENV_DIR_OVERRIDE=...
# Required only by DeepSeek-V3 rows.
export NRL_DEEPSEEK_V3_BF16_CKPT=...

experiments/h100-hybridep-performance-20260817/submit.sh --list
experiments/h100-hybridep-performance-20260817/submit.sh \
  qwen3-30b-baseline --test-only
experiments/h100-hybridep-performance-20260817/submit.sh \
  qwen3-30b-baseline
```

The launcher requests all eight GPUs on every H100 node. It never embeds a
cluster hostname, account, shared-storage path, or credential. The baseline
configuration tree is materialized from commit
`4a1454bf430624786251d14ba0197169c8e68a5c` inside the run's result directory
and retained with the logs so inherited YAML paths remain valid. An optional
absolute `UV_CACHE_DIR_OVERRIDE`
selects the warmed cache only after Ray has started; this avoids shadowing the
nightly image's bootstrap cache while still reusing the pinned HybridEP wheel
in driver and actor environments. Before launching Ray actors, the launcher
also verifies that the Ray version in `uv.lock` matches the nightly image's
bootstrap Ray version. This prevents a mixed-version Ray cluster from being
mistaken for a model or HybridEP failure. An absolute
`NEMO_RL_VENV_DIR_OVERRIDE` can select a prebuilt, verified actor environment;
using the same immutable environment for both arms avoids concurrent partial
environment construction and keeps the software stack matched. DeepSeek-V3
rows require an absolute
converted BF16 checkpoint path and pass it to both `policy.model_name` and
`policy.tokenizer.name`; without that prerequisite the launcher exits before
submission rather than misclassifying a placeholder-config failure as an OOM.

`validate_recipes.py` provides the same resolved-config gate without depending
on the pytest console entrypoint. Run it inside the selected container before
the GPU model jobs:

```bash
python experiments/h100-hybridep-performance-20260817/validate_recipes.py
```

Before submitting model jobs, run `warm_hybridep_cache.sbatch` once with the
same immutable container and cache path. It builds the pinned HybridEP wheel on
one H100 and makes it reusable by every actor environment:

```bash
export PROJECT_ROOT=...
export UV_CACHE_DIR_OVERRIDE=...

sbatch --test-only warm_hybridep_cache.sbatch
sbatch warm_hybridep_cache.sbatch
```

Container staging itself is CPU-only. Submit `stage_enroot_image.sbatch` to a
CPU partition so a long layer extraction cannot reserve an idle GPU.
